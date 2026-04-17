"""
Qwen3.5 MoE model implementation for nanovllm.

Architecture:
- Hybrid attention: 3 linear attention (Gated DeltaNet) + 1 full attention per 4 layers
- MoE MLP: TopK router + fused experts (3D tensors) + shared expert + shared expert gate
- Full attention has output gating: q_proj outputs 2x dim (query + gate)
- Partial rotary embedding (25% of head_dim)
- Uses (1+w) style RMSNorm (weight initialized to 0, applies as (1+w)*norm(x))
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ============================================================
# Qwen3.5-style RMSNorm: uses (1 + weight) * norm(x)
# Weight is initialized to zeros, so at init the norm is identity-like.
# This is DIFFERENT from standard Qwen3 RMSNorm which uses weight * norm(x).
# ============================================================

class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling, as used in Qwen3.5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # NOTE: initialized to zeros! The checkpoint stores zero-initialized weights.
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        # (1 + weight) scaling: when weight=0, this is identity after normalization
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# ============================================================
# RMSNorm with gating (used by GatedDeltaNet output)
# Also uses (1 + weight) style but additionally applies silu(gate)
# ============================================================

class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating: norm(x) * (1+w) * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ============================================================
# Utility: L2 norm (used in Gated DeltaNet)
# ============================================================

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ============================================================
# Torch fallback: chunk-based gated delta rule
# ============================================================

def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_scores = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_scores @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ============================================================
# Torch fallback: recurrent gated delta rule (for single-step decode)
# ============================================================

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ============================================================
# Gated DeltaNet (Linear Attention)
# ============================================================

class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer.

    Maintains per-sequence recurrent state and conv state in pre-allocated GPU buffers
    for CUDA Graph compatibility. During decode, all sequences are processed in a single
    batched operation with no Python control flow.

    During prefill: processes each sequence independently, writes final states to buffer.
    During decode: reads/writes pre-allocated buffers via slot indices (CUDA Graph safe).
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size

        # Projections
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # Causal conv1d (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        # Time step parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Output norm and projection
        self.norm = RMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Pre-allocated state buffers (set by model_runner.allocate_linear_attn_states)
        # recurrent_state_buf: [max_num_seqs, num_v_heads, head_k_dim, head_v_dim] float32
        # conv_state_buf: [max_num_seqs, conv_dim, kernel_size - 1] model_dtype
        self.recurrent_state_buf: torch.Tensor | None = None
        self.conv_state_buf: torch.Tensor | None = None

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        slot_idx: int | None = None,
    ) -> torch.Tensor:
        """Process a full sequence during prefill. Write final state to buffer slot if provided."""
        seq_len = hidden_states.shape[0]
        batch_size = 1
        hidden_states_3d = hidden_states.unsqueeze(0)  # [1, seq_len, hidden_size]

        # Projections
        mixed_qkv = self.in_proj_qkv(hidden_states_3d)  # [1, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [1, conv_dim, seq_len]

        z = self.in_proj_z(hidden_states_3d)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states_3d)
        a = self.in_proj_a(hidden_states_3d)

        # Save conv state to buffer: last (kernel_size - 1) columns of pre-activation mixed_qkv
        if slot_idx is not None and self.conv_state_buf is not None:
            self.conv_state_buf[slot_idx].copy_(
                mixed_qkv[0, :, -(self.conv_kernel_size - 1):]
            )

        # Causal conv1d + SiLU activation
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Split into Q, K, V
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Chunk-based gated delta rule — output final state for decode
        save_state = (slot_idx is not None and self.recurrent_state_buf is not None)
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=save_state,
            use_qk_l2norm_in_kernel=True,
        )

        if save_state and last_recurrent_state is not None:
            self.recurrent_state_buf[slot_idx].copy_(last_recurrent_state.squeeze(0))

        # Apply gated RMSNorm
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output.squeeze(0)

    def _forward_decode_batched(
        self,
        hidden_states: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched decode: process all tokens simultaneously.
        All tensor ops, no Python control flow → CUDA Graph compatible.

        hidden_states: [B, hidden_size]
        slot_indices: [B] int tensor mapping batch position to buffer slot
        """
        B = hidden_states.shape[0]

        # 1. Linear projections (batched)
        mixed_qkv = self.in_proj_qkv(hidden_states)   # [B, conv_dim]
        z = self.in_proj_z(hidden_states)               # [B, value_dim]
        a = self.in_proj_a(hidden_states)               # [B, num_v_heads]
        b = self.in_proj_b(hidden_states)               # [B, num_v_heads]

        # 2. Conv1d with pre-allocated state buffer
        mixed_qkv_col = mixed_qkv.unsqueeze(-1)                     # [B, conv_dim, 1]
        conv_state = self.conv_state_buf[slot_indices]                # [B, conv_dim, kernel_size-1]
        conv_input = torch.cat([conv_state, mixed_qkv_col], dim=-1)  # [B, conv_dim, kernel_size]
        # Update conv state buffer: sliding window (drop oldest, keep newest kernel_size-1)
        self.conv_state_buf[slot_indices] = conv_input[:, :, 1:]
        # Depthwise conv + activation
        mixed_qkv_act = F.silu(
            F.conv1d(conv_input, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
        )  # [B, conv_dim, 1]
        mixed_qkv_act = mixed_qkv_act.squeeze(-1)  # [B, conv_dim]

        # 3. Split Q, K, V and reshape
        query, key, value = mixed_qkv_act.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.view(B, self.num_k_heads, self.head_k_dim)
        key = key.view(B, self.num_k_heads, self.head_k_dim)
        value = value.view(B, self.num_v_heads, self.head_v_dim)

        # L2 norm on Q, K
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        # GQA expand k_heads → v_heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=1)
            key = key.repeat_interleave(ratio, dim=1)

        # 4. Single-step recurrent delta rule (batched, float32)
        scale = self.head_k_dim ** -0.5
        q = query.float() * scale                   # [B, H, Dk]
        k = key.float()                              # [B, H, Dk]
        v = value.float()                            # [B, H, Dv]
        beta_val = b.sigmoid().float()               # [B, H]
        g_val = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).exp()  # [B, H]

        # Read state from buffer
        state = self.recurrent_state_buf[slot_indices].float()  # [B, H, Dk, Dv]

        # Decay: state *= g
        state = state * g_val.unsqueeze(-1).unsqueeze(-1)       # [B, H, Dk, Dv]
        # Recall: kv_mem = (state * k).sum(Dk)
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)           # [B, H, Dv]
        # Delta update: delta = (v - kv_mem) * beta
        delta = (v - kv_mem) * beta_val.unsqueeze(-1)            # [B, H, Dv]
        # Write: state += outer(k, delta)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)    # [B, H, Dk, Dv]
        # Query: output = (state * q).sum(Dk)
        output = (state * q.unsqueeze(-1)).sum(dim=-2)           # [B, H, Dv]

        # Write state back to buffer (cast float32 → buffer dtype)
        self.recurrent_state_buf[slot_indices] = state.to(self.recurrent_state_buf.dtype)

        # 5. RMSNorm + gate + output projection
        output = output.to(hidden_states.dtype)
        output = output.reshape(B * self.num_v_heads, self.head_v_dim)
        z_flat = z.reshape(B * self.num_v_heads, self.head_v_dim)
        output = self.norm(output, z_flat)
        output = output.view(B, self.value_dim)
        return self.out_proj(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process hidden_states of shape [total_tokens, hidden_size].

        Prefill: each sequence processed independently, state saved to buffer slots.
        Decode: batched processing via pre-allocated buffers (CUDA Graph compatible).
        """
        from nanovllm.utils.context import get_context
        context = get_context()

        if context.is_prefill and context.cu_seqlens_q is not None:
            cu_seqlens = context.cu_seqlens_q
            num_seqs = len(cu_seqlens) - 1
            outputs = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_hidden = hidden_states[start:end]
                # Get buffer slot index for this sequence
                if (context.linear_attn_slot_indices is not None
                        and self.recurrent_state_buf is not None):
                    slot_idx = context.linear_attn_slot_indices[i].item()
                else:
                    slot_idx = None
                seq_out = self._forward_prefill(seq_hidden, slot_idx=slot_idx)
                outputs.append(seq_out)
            return torch.cat(outputs, dim=0)
        else:
            # Decode: batched processing with buffer slot indices
            if (context.linear_attn_slot_indices is not None
                    and self.recurrent_state_buf is not None):
                return self._forward_decode_batched(
                    hidden_states, context.linear_attn_slot_indices
                )
            else:
                # Fallback for warmup (no buffer allocated yet)
                return torch.zeros_like(hidden_states)


# ============================================================
# Full Attention (with output gating, partial rotary)
# ============================================================

class Qwen3_5FullAttention(nn.Module):
    """
    Full attention layer with output gating.

    Reference: q_proj outputs [num_heads * head_dim * 2] which is reshaped to
    [bs, seq, num_heads, head_dim*2], then chunk(2, dim=-1) -> (query, gate).
    gate is then reshaped to [bs, seq, num_heads*head_dim].
    query goes through q_norm, then RoPE.
    After attention, output *= sigmoid(gate).

    Weights:
    - self_attn.q_proj.weight  [num_heads * head_dim * 2, hidden_size]
    - self_attn.k_proj.weight  [num_kv_heads * head_dim, hidden_size]
    - self_attn.v_proj.weight  [num_kv_heads * head_dim, hidden_size]
    - self_attn.o_proj.weight  [hidden_size, num_heads * head_dim]
    - self_attn.q_norm.weight  [head_dim]  (zeros-initialized, (1+w) style)
    - self_attn.k_norm.weight  [head_dim]  (zeros-initialized, (1+w) style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        
        # Determine whether to shard or replicate KV heads
        if num_kv_heads < tp_size:
            # KV heads are less than TP size: replicate them instead of sharding
            self.num_kv_heads = num_kv_heads
            kv_output_size = num_kv_heads * head_dim
            use_replicated_kv = True
        else:
            # Normal sharding case
            self.num_kv_heads = num_kv_heads // tp_size
            kv_output_size = num_kv_heads * head_dim
            use_replicated_kv = False
        
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # q_proj outputs query + gate (2x query size)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * head_dim * 2,  # query + gate
            bias=False,
        )
        
        # K,V projections: use ReplicatedLinear if num_kv_heads < tp_size
        if use_replicated_kv:
            self.k_proj = ReplicatedLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
            self.v_proj = ReplicatedLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
        else:
            self.k_proj = ColumnParallelLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
        
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )

        # Rotary embedding (partial)
        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = rotary_dim
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
        )

        # Q/K norms — Qwen3.5 uses (1+w) style RMSNorm
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)

        # Attention kernel
        self.attn = Attention(
            self.num_heads,
            head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Q projection: outputs [N, num_heads * head_dim * 2 / tp_size]
        q_out = self.q_proj(hidden_states)

        # Reshape to [N, num_heads, head_dim * 2], then split query and gate
        q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
        query, gate = q_out.chunk(2, dim=-1)  # each [N, num_heads, head_dim]
        # gate is flattened: [N, num_heads * head_dim]
        gate = gate.reshape(-1, self.num_heads * self.head_dim)

        # K, V projections
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        # Apply Q/K norms (Qwen3.5 uses (1+w) style)
        query = self.q_norm(query)
        k = self.k_norm(k)

        # Apply partial rotary embedding
        query, k = self.rotary_emb(positions, query, k)

        # Attention
        o = self.attn(query, k, v)

        # Apply output gate: output *= sigmoid(gate)
        o_flat = o.flatten(1, -1)  # [N, q_size]
        o_flat = o_flat * torch.sigmoid(gate)

        # Output projection
        output = self.o_proj(o_flat)
        return output


# ============================================================
# MoE: Top-K Router
# ============================================================

class Qwen3_5TopKRouter(nn.Module):
    """
    Top-K router with softmax normalization.

    Weight: mlp.gate.weight [num_experts, hidden_size]
    """

    def __init__(self, num_experts: int, hidden_size: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [N, hidden_size]
        router_logits = F.linear(hidden_states, self.weight)  # [N, num_experts]
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        top_k_values = top_k_values.to(hidden_states.dtype)
        return top_k_values, top_k_indices


# ============================================================
# MoE: Fused Experts (3D Parameter Tensors)
# ============================================================

class Qwen3_5Experts(nn.Module):
    """
    Fused expert weights stored as 3D tensors.

    Weights:
    - experts.gate_up_proj  [num_experts, 2 * moe_intermediate_size, hidden_size]
    - experts.down_proj     [num_experts, hidden_size, moe_intermediate_size]
    """

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from nanovllm.utils.context import get_context
        context = get_context()
        if context.is_prefill:
            return self._forward_sparse(hidden_states, top_k_indices, top_k_weights)
        else:
            return self._forward_dense(hidden_states, top_k_indices, top_k_weights)

    def _forward_sparse(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse dispatch — used for prefill (not in CUDA Graph).
        Uses nonzero/where to only compute active expert-token pairs."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, N]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dense gather-based dispatch — CUDA Graph compatible for decode.
        Iterates over top_k slots (fixed count), gathers per-token expert weights,
        and uses batched matmul. All tensor shapes are fixed regardless of routing."""
        final = torch.zeros_like(hidden_states)
        top_k = top_k_indices.shape[1]

        for k in range(top_k):
            idx = top_k_indices[:, k]           # [N] expert index per token
            w = top_k_weights[:, k:k+1]         # [N, 1] routing weight

            # Gather expert parameters for each token's selected expert
            # gate_up_proj: [num_experts, 2*inter, hidden] -> [N, 2*inter, hidden]
            # down_proj:    [num_experts, hidden, inter]   -> [N, hidden, inter]
            gate_up_w = self.gate_up_proj[idx]
            down_w = self.down_proj[idx]

            # Batched matmul: gate_up projection
            # [N, 2*inter, hidden] @ [N, hidden, 1] -> [N, 2*inter]
            x = torch.bmm(gate_up_w, hidden_states.unsqueeze(-1)).squeeze(-1)
            gate, up = x.chunk(2, dim=-1)
            x = F.silu(gate) * up               # [N, inter]

            # Batched matmul: down projection
            # [N, hidden, inter] @ [N, inter, 1] -> [N, hidden]
            x = torch.bmm(down_w, x.unsqueeze(-1)).squeeze(-1)

            final = final + w * x

        return final


# ============================================================
# MoE: Sparse MoE Block (Router + Experts + Shared Expert)
# ============================================================

class Qwen3_5SparseMoeBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
    ):
        super().__init__()
        self.gate = Qwen3_5TopKRouter(num_experts, hidden_size, num_experts_per_tok)
        self.experts = Qwen3_5Experts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_expert = Qwen3_5SharedExpertMLP(
            hidden_size, shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_expert_output = self.shared_expert(hidden_states)
        top_k_weights, top_k_indices = self.gate(hidden_states)
        expert_output = self.experts(hidden_states, top_k_indices, top_k_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        output = expert_output + shared_expert_output
        return output


class Qwen3_5SharedExpertMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Decoder Layer
# ============================================================

class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.hidden_size = text_config.hidden_size
        self.layer_type = text_config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=text_config.hidden_size,
                num_k_heads=text_config.linear_num_key_heads,
                num_v_heads=text_config.linear_num_value_heads,
                head_k_dim=text_config.linear_key_head_dim,
                head_v_dim=text_config.linear_value_head_dim,
                conv_kernel_size=text_config.linear_conv_kernel_dim,
                rms_norm_eps=text_config.rms_norm_eps,
            )
        elif self.layer_type == "full_attention":
            rope_params = text_config.rope_parameters
            self.self_attn = Qwen3_5FullAttention(
                hidden_size=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                max_position=text_config.max_position_embeddings,
                rms_norm_eps=text_config.rms_norm_eps,
                rope_theta=rope_params["rope_theta"],
                partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.25),
            )

        self.mlp = Qwen3_5SparseMoeBlock(
            hidden_size=text_config.hidden_size,
            num_experts=text_config.num_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            moe_intermediate_size=text_config.moe_intermediate_size,
            shared_expert_intermediate_size=text_config.shared_expert_intermediate_size,
        )

        # NOTE: Qwen3.5 uses (1+w) style RMSNorm with weights initialized to 0
        self.input_layernorm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Qwen3.5 uses standard residual connection pattern (NOT fused norm+residual):
        #   residual = hidden_states
        #   hidden_states = norm(hidden_states)
        #   hidden_states = attn(hidden_states)
        #   hidden_states = residual + hidden_states
        #
        # We use the residual parameter to carry state between layers.
        # On first layer: residual is None, so we use hidden_states as residual.
        # On subsequent layers: hidden_states is the output from previous mlp,
        #   residual carries the pre-mlp state. We add them first.
        if residual is not None:
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        torch.cuda.nvtx.range_push(f"{self.layer_type}_{self.layer_idx}")
        # Token mixing
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        torch.cuda.nvtx.range_pop()

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_push(f"{self.layer_idx}_mlp")
        hidden_states = self.mlp(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Return hidden_states (mlp output, not yet added to residual)
        # and residual (pre-mlp state). They'll be added at start of next layer.
        return hidden_states, residual


# ============================================================
# Text Model (Language Model)
# ============================================================

class Qwen3_5TextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.embed_tokens = VocabParallelEmbedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i)
            for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # After last layer, need to add final residual
        hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================
# Top-level: Qwen3.5 For Causal LM
# ============================================================

class Qwen3_5ForCausalLM(nn.Module):

    packed_modules_mapping = {}

    # Weight name prefix to strip (VLM structure)
    weight_prefix = "model."

    # Prefixes to skip during weight loading
    skip_prefixes = ("mtp.", "model.visual.", "model.merger.")

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.language_model = Qwen3_5TextModel(config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.language_model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
