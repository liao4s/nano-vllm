# Qwen3 Model Implementation - Deep Dive

**File Location:** `nanovllm/models/qwen3.py` (216 lines)

---

## 1. CLASS HIERARCHY

```
Qwen3ForCausalLM (TOP LEVEL)
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding
│   │   └── weight: [vocab_size // tp_size, hidden_size]
│   │
│   ├── layers: ModuleList[Qwen3DecoderLayer]
│   │   └── (repeat num_hidden_layers times)
│   │       ├── self_attn: Qwen3Attention
│   │       ├── mlp: Qwen3MLP
│   │       ├── input_layernorm: RMSNorm
│   │       └── post_attention_layernorm: RMSNorm
│   │
│   └── norm: RMSNorm
│
└── lm_head: ParallelLMHead
    └── weight: [vocab_size // tp_size, hidden_size]
```

---

## 2. COMPLETE FILE BREAKDOWN

### Lines 1-12: Imports and Setup

```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
```

**Dependencies:**
- **Custom Layers**: All layer implementations are custom-built
- **Distributed**: Uses NCCL for tensor parallelism
- **Transformers**: Uses official Qwen3Config from HuggingFace

---

### Lines 14-87: Qwen3Attention

**Full Implementation:**

```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        # ==================== Tensor Parallel Setup ====================
        tp_size = dist.get_world_size()           # Number of GPUs
        self.total_num_heads = num_heads          # Original number
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # Per-GPU heads
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # Per-GPU KV heads
        
        # ==================== Dimensions ====================
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim        # Per-GPU Q size
        self.kv_size = self.num_kv_heads * self.head_dim    # Per-GPU K/V size
        self.scaling = self.head_dim ** -0.5                # Attention scaling
        self.qkv_bias = qkv_bias
        
        # ==================== Projection Layers ====================
        # QKVParallelLinear outputs: Q (q_size) + K (kv_size) + V (kv_size)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,              # Input size
            self.head_dim,            # Head dimension
            self.total_num_heads,     # Total heads (before TP sharding)
            self.total_num_kv_heads,  # Total KV heads (before TP sharding)
            bias=qkv_bias,
        )
        # Output projection (row parallel for all-reduce)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # Input from attention
            hidden_size,                            # Output back to hidden
            bias=False,
        )
        
        # ==================== Position Embeddings ====================
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,      # Use full head_dim for rotation
            max_position=max_position,     # Max position
            base=rope_theta,               # RoPE base (usually 10000 or 1000000)
            rope_scaling=rope_scaling,     # Scaling (e.g., for long context)
        )
        
        # ==================== Attention ====================
        self.attn = Attention(
            self.num_heads,       # Per-GPU heads
            self.head_dim,
            self.scaling,
            self.num_kv_heads,    # Per-GPU KV heads
        )
        
        # ==================== Optional Q/K Normalization ====================
        # From Qwen3 paper: normalize Q and K if no bias
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,          # [num_tokens]
        hidden_states: torch.Tensor,      # [num_tokens, hidden_size]
    ) -> torch.Tensor:
        # ==================== Project to Q, K, V ====================
        qkv = self.qkv_proj(hidden_states)  # [num_tokens, q_size + kv_size + kv_size]
        
        # Split into Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # ==================== Reshape for Multi-Head Attention ====================
        q = q.view(-1, self.num_heads, self.head_dim)      # [num_tokens, num_heads, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)   # [num_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.num_kv_heads, self.head_dim)   # [num_tokens, num_kv_heads, head_dim]
        
        # ==================== Optional Q/K Normalization ====================
        if not self.qkv_bias:
            q = self.q_norm(q)  # Normalize query heads
            k = self.k_norm(k)  # Normalize key heads
        
        # ==================== Apply Rotary Embeddings ====================
        # RoPE applies rotation based on position
        q, k = self.rotary_emb(positions, q, k)
        
        # ==================== Attention Computation ====================
        # Handles both prefill and decode phases
        # Stores K/V to cache automatically
        o = self.attn(q, k, v)  # [num_tokens, num_heads, head_dim]
        
        # ==================== Output Projection ====================
        # Flatten back to [num_tokens, num_heads * head_dim]
        output = self.o_proj(o.flatten(1, -1))  # [num_tokens, hidden_size]
        
        return output
```

**Key Design Decisions:**

1. **Tensor Parallelism First**: All sizes computed for per-GPU sharding
2. **QKV Projection**: Single projection layer for efficiency
3. **Optional Q/K Norm**: Paper-specific feature (when no bias)
4. **RoPE**: Modern positional encoding
5. **Attention Abstraction**: Uses Attention class (not inline)

---

### Lines 90-116: Qwen3MLP

**Full Implementation:**

```python
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # ==================== Merged Gate and Up Projections ====================
        # Instead of:
        #   gate = gate_proj(x)        [*, intermediate_size]
        #   up = up_proj(x)            [*, intermediate_size]
        # We do:
        #   gate_up = gate_up_proj(x)  [*, intermediate_size*2]
        # This reduces memory bandwidth and kernel calls
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # Two outputs of size intermediate_size
            bias=False,
        )
        
        # ==================== Down Projection ====================
        self.down_proj = RowParallelLinear(
            intermediate_size,  # Input is one of the intermediate outputs
            hidden_size,        # Project back to hidden_size
            bias=False,
        )
        
        # ==================== Activation Function ====================
        # Only SiLU (Swish) is supported
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()  # Implements SiLU(gate) * up

    def forward(self, x):
        # ==================== Gate and Up ====================
        # x: [num_tokens, hidden_size]
        gate_up = self.gate_up_proj(x)  # [num_tokens, intermediate_size*2]
        
        # ==================== Activation ====================
        # SiluAndMul splits input in half and does: SiLU(gate) * up
        x = self.act_fn(gate_up)  # [num_tokens, intermediate_size]
        
        # ==================== Down Project ====================
        # Project back to hidden dimension
        x = self.down_proj(x)  # [num_tokens, hidden_size]
        
        return x
```

**Why This Design?**

1. **Merged Projections**: Reduces kernel calls and bandwidth
2. **SiLU Activation**: Modern activation (better than ReLU)
3. **Gating Mechanism**: Gate * FFN output (from GLU variants)
4. **Row Parallel Down**: All-reduce combines outputs across GPUs

---

### Lines 119-158: Qwen3DecoderLayer

**Full Implementation:**

```python
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # ==================== Attention ====================
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # ==================== MLP ====================
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # ==================== Layer Normalization ====================
        # Pre-norm architecture: normalize before sublayer
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,           # [num_tokens]
        hidden_states: torch.Tensor,       # [num_tokens, hidden_size]
        residual: torch.Tensor | None,    # [num_tokens, hidden_size] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ==================== Attention Block ====================
        # Pre-norm: normalize input
        if residual is None:
            # First layer: no residual yet
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # Other layers: add residual before normalization
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Apply attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # ==================== MLP Block ====================
        # Pre-norm: normalize after attention
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # Apply MLP
        hidden_states = self.mlp(hidden_states)
        
        # Return for next layer
        # Residual will be added in next layer's pre-norm
        return hidden_states, residual
```

**Architecture Pattern: Pre-Norm**

```
Input (x)
  │
  ├─ LayerNorm → Attention → y_attn
  │
  ├─ Add residual: x + y_attn
  │
  ├─ LayerNorm → MLP → y_mlp
  │
  └─ Output: x + y_attn + y_mlp
```

**Key Feature: Fused Residual Addition**

- `RMSNorm.add_rms_forward()` combines addition and normalization
- More efficient than separate operations
- Returns both normalized and residual values

---

### Lines 161-183: Qwen3Model

**Full Implementation:**

```python
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # ==================== Embedding ====================
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,         # Total vocabulary size
            config.hidden_size,        # Embedding dimension
        )
        # Each GPU holds vocab_size // tp_size embeddings
        
        # ==================== Transformer Stack ====================
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # ==================== Final Normalization ====================
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,    # [num_tokens] with token IDs
        positions: torch.Tensor,    # [num_tokens] with absolute positions
    ) -> torch.Tensor:
        # ==================== Embedding ====================
        # Map token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)  # [num_tokens, hidden_size]
        
        # ==================== Transformer Layers ====================
        residual = None  # Accumulates residual across layers
        for layer in self.layers:
            # Each layer adds its contribution to residual
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # ==================== Final Layer Norm ====================
        # Add final residual and normalize
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states  # [num_tokens, hidden_size]
```

**Residual Streaming Architecture**

- Residual is accumulated across layers
- Each layer adds its output to residual
- Final norm adds final residual
- Reduces activation memory usage

---

### Lines 185-216: Qwen3ForCausalLM

**Full Implementation:**

```python
class Qwen3ForCausalLM(nn.Module):
    # ==================== Packed Module Mapping ====================
    # Maps checkpoint weight names to implementation parameter names
    # Handles architectural differences between checkpoint and implementation
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # q_proj → qkv_proj[:,  :q_size]
        "k_proj": ("qkv_proj", "k"),      # k_proj → qkv_proj[:, q_size:q_size+kv_size]
        "v_proj": ("qkv_proj", "v"),      # v_proj → qkv_proj[:, q_size+kv_size:]
        "gate_proj": ("gate_up_proj", 0), # gate_proj → gate_up_proj[:, :intermediate]
        "up_proj": ("gate_up_proj", 1),   # up_proj → gate_up_proj[:, intermediate:]
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # ==================== Model ====================
        self.model = Qwen3Model(config)
        
        # ==================== Language Model Head ====================
        # Projects hidden states to vocabulary logits
        self.lm_head = ParallelLMHead(
            config.vocab_size,      # Number of output classes
            config.hidden_size,     # Input dimension
        )
        
        # ==================== Weight Tying ====================
        # If enabled, share embeddings and output weights
        if config.tie_word_embeddings:
            # Make lm_head weight point to same data as embeddings
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,    # [num_tokens]
        positions: torch.Tensor,    # [num_tokens]
    ) -> torch.Tensor:
        # ==================== Model Forward ====================
        # Returns hidden states for the last layer
        return self.model(input_ids, positions)  # [num_tokens, hidden_size]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
    ) -> torch.Tensor:
        # ==================== Compute Output Logits ====================
        # Project hidden states to vocabulary
        # Only rank 0 returns logits (others return None)
        return self.lm_head(hidden_states)  # [num_tokens (rank 0), vocab_size]
```

**Design Insight: Separation of Concerns**

- `forward()`: Runs transformer model only
- `compute_logits()`: Runs LM head only
- In ModelRunner: called as `self.model.compute_logits(self.model(...))`
- Allows graph capturing at appropriate granularity

---

## 3. WEIGHT LOADING DETAILS

### Checkpoint Format

Original checkpoint (HuggingFace):
```
attention.q_proj.weight    [hidden_size, hidden_size]
attention.k_proj.weight    [num_kv_heads*head_dim, hidden_size]
attention.v_proj.weight    [num_kv_heads*head_dim, hidden_size]
mlp.gate_proj.weight       [intermediate_size, hidden_size]
mlp.up_proj.weight         [intermediate_size, hidden_size]
```

Implementation format:
```
self_attn.qkv_proj.weight         [q_size+kv_size+kv_size, hidden_size] (TP sharded)
mlp.gate_up_proj.weight           [intermediate_size*2, hidden_size] (TP sharded)
```

### Loading Process with packed_modules_mapping

```python
# For each weight file:
for weight_name in file.keys():
    # Check if this matches a packed module pattern
    for pattern_key in packed_modules_mapping.keys():
        if pattern_key in weight_name:
            # Example: weight_name = "layer.0.attention.q_proj.weight"
            # pattern_key = "q_proj"
            # This matches!
            
            mapped_param_name, shard_id = packed_modules_mapping[pattern_key]
            # mapped_param_name = "qkv_proj"
            # shard_id = "q"
            
            # Replace q_proj with qkv_proj in the name
            new_name = weight_name.replace("q_proj", "qkv_proj")
            # new_name = "layer.0.attention.qkv_proj.weight"
            
            param = model.get_parameter(new_name)
            weight_loader = param.weight_loader  # QKVParallelLinear.weight_loader
            
            # Call loader with shard_id to tell it which part to load
            weight_loader(param, loaded_weight, shard_id="q")
```

### Tensor Parallel Sharding During Load

**QKVParallelLinear.weight_loader:**

```python
def weight_loader(param, loaded_weight, shard_id="q"):
    # loaded_weight shape: [hidden_size, hidden_size] from checkpoint
    # param shape: [(num_heads//tp + 2*num_kv_heads//tp)*head_dim, hidden_size]
    
    if shard_id == "q":
        # Take first num_heads*head_dim rows
        shard_size = self.num_heads * self.head_dim
        shard_offset = 0
    elif shard_id == "k":
        # Take middle num_kv_heads*head_dim rows
        shard_size = self.num_kv_heads * self.head_dim
        shard_offset = self.num_heads * self.head_dim
    else:  # "v"
        # Take last num_kv_heads*head_dim rows
        shard_size = self.num_kv_heads * self.head_dim
        shard_offset = self.num_heads * self.head_dim + self.num_kv_heads * self.head_dim
    
    # Chunk loaded_weight across tp_size
    chunks = loaded_weight.chunk(self.tp_size, dim=0)
    
    # Take only this GPU's chunk
    local_weight = chunks[self.tp_rank]
    
    # Store in parameter at correct offset
    param.data[shard_offset:shard_offset+shard_size] = local_weight
```

---

## 4. INFERENCE FLOW

### Prefill Phase (Processing prompt)

```
Qwen3ForCausalLM.forward(input_ids, positions)
  │
  ├─ Qwen3Model
  │   ├─ embed_tokens(input_ids)
  │   │   └─ [num_prompt_tokens, hidden_size]
  │   │
  │   └─ For each Qwen3DecoderLayer:
  │       ├─ input_layernorm(x)
  │       ├─ Qwen3Attention
  │       │   ├─ qkv_proj
  │       │   ├─ Apply RoPE
  │       │   ├─ flash_attn_varlen_func (prefill path)
  │       │   │   └─ store_kvcache to cache
  │       │   └─ o_proj
  │       │
  │       ├─ post_attention_layernorm
  │       ├─ Qwen3MLP
  │       │   ├─ gate_up_proj
  │       │   ├─ SiluAndMul
  │       │   └─ down_proj
  │       │
  │       └─ [hidden_states, residual]
  │
  └─ [hidden_size] tensors for all prompt tokens
```

### Decode Phase (Generating tokens)

```
Qwen3ForCausalLM.forward(last_token_id, position)
  │
  ├─ Qwen3Model
  │   ├─ embed_tokens([token_id])
  │   │   └─ [1, hidden_size]
  │   │
  │   └─ For each Qwen3DecoderLayer:
  │       ├─ RMSNorm
  │       ├─ Qwen3Attention
  │       │   ├─ qkv_proj
  │       │   ├─ Apply RoPE (position = seq_len - 1)
  │       │   ├─ flash_attn_with_kvcache (decode path)
  │       │   │   └─ Use cached K/V, update cache slot
  │       │   └─ o_proj
  │       │
  │       ├─ RMSNorm
  │       ├─ Qwen3MLP
  │       │
  │       └─ [hidden_states, residual]
  │
  └─ [1, hidden_size] hidden state
```

---

## 5. OPTIMIZATION TECHNIQUES

### 1. Tensor Parallelism

- **Attention**: Column-parallel Q,K,V + Row-parallel output
- **MLP**: Merged gate+up + row-parallel down
- All GPUs compute simultaneously, minimal sync

### 2. Packed Modules

- Q, K, V packed into single qkv_proj
- Gate, up packed into single gate_up_proj
- Reduces kernel calls and improves cache utilization

### 3. Residual Streaming

- Residual accumulated across layers
- Saves intermediate activation memory
- Each layer adds its contribution

### 4. RMSNorm Fusion

- Supports in-place addition: `norm(x + residual)`
- Compiled with `@torch.compile`
- Reduces memory bandwidth

### 5. Flash Attention

- Efficient attention with custom kernels
- Reduced memory usage
- Faster computation
- Handles both prefill (varlen) and decode (with_kvcache)

---

**End of Qwen3 Implementation Details**
