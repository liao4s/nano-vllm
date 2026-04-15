# CUDA Graph & GatedDeltaNet Compatibility Analysis

## Executive Summary

GatedDeltaNet (linear attention) in nanovllm is currently **incompatible with CUDA Graph** due to its stateful nature. The engine forces `enforce_eager=True` when loading Qwen3.5 (lines 45-53 of `model_runner.py`). This document provides an exhaustive analysis of:

1. How CUDA Graph currently works in the engine
2. Why GatedDeltaNet breaks graph capture
3. Exact design requirements to make it compatible

---

## Part 1: CUDA Graph Implementation in nanovllm

### 1.1 Overview: Graph Capture Flow

**File**: `nanovllm/engine/model_runner.py`

The engine uses CUDA graphs to cache kernel sequences for faster replay during decode. The pattern is:
```
Pre-allocate tensors → Warmup run (not captured) → Capture run (with CUDAGraph) → Store graph
```

### 1.2 Graph Capture Code (Lines 264-299)

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    
    # PRE-ALLOCATE: All tensors at maximum size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        
        # Set context for this batch size (used by Attention kernels)
        set_context(False, slot_mapping=slot_mapping[:bs], 
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        
        # WARMUP: Not captured, ensures all kernels are compiled
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        # CAPTURE: Record all kernel calls
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # Store references to pre-allocated tensors
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

**Key Points:**
- Captures for multiple batch sizes: [1, 2, 4, 8, 16, ..., max_bs]
- All tensors are **pre-allocated at maximum size** and reused
- The graph records which GPU memory addresses will be read/written
- Context (slot_mapping, context_lens, block_tables) is set during capture


### 1.3 Graph Replay Code (Lines 223-239)

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # UPDATE VALUES IN PRE-ALLOCATED TENSORS
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        
        # CRITICAL: Reset slot_mapping to -1 (cache miss marker)
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        
        # UPDATE context info
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        # REPLAY: Execute cached kernel sequence
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**Critical Sequence:**
1. Select pre-captured graph for batch size
2. **Update tensor values** at same GPU addresses
3. **Call `graph.replay()`** which executes kernels without re-reading parameters
4. Extract outputs from pre-allocated tensor

---

## Part 2: How Attention Works with CUDA Graph

### 2.1 Attention Layer Structure

**File**: `nanovllm/layers/attention.py` (76 lines)

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])  # Allocated by ModelRunner

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel():
            # STORE: Write k,v to pre-allocated cache blocks
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:    # decode
            # flash_attn_with_kvcache reads from k_cache/v_cache at positions
            # specified by context_lens and block_tables
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, 
                                        block_table=context.block_tables, ...)
        return o
```

**CUDA Graph Compatibility:**
- ✅ `k_cache` and `v_cache` are **pre-allocated at fixed GPU addresses** (ModelRunner.allocate_kv_cache)
- ✅ `store_kvcache` kernel reads GPU address once, writes to fixed cache slots
- ✅ `context.slot_mapping` tells kernel which slots to write
- ✅ `flash_attn_with_kvcache` reads from fixed k_cache/v_cache addresses

### 2.2 KV Cache Allocation (Lines 123-149)

```python
def allocate_kv_cache(self):
    # Calculate cache size
    num_kv_heads = ...
    head_dim = ...
    num_attn_layers = ...
    block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * itemsize
    
    # Pre-allocate ONCE
    self.kv_cache = torch.empty(2, num_attn_layers, config.num_kvcache_blocks, 
                                 self.block_size, num_kv_heads, head_dim)
    
    # Assign to each Attention module
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]  # Fixed GPU address
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**Why This Works:**
- Tensors never reallocated: same GPU addresses throughout inference
- Kernel operations record these addresses at capture time
- During replay, kernels access same addresses with updated values

---

## Part 3: GatedDeltaNet Architecture & Current Implementation

### 3.1 Configuration Values

**From qwen3.5 config** (lines 60-64):

```json
"linear_conv_kernel_dim": 4,           // Convolution kernel size
"linear_key_head_dim": 128,            // Per-head key dimension
"linear_num_key_heads": 16,            // Number of key heads
"linear_num_value_heads": 32,          // Number of value heads
"linear_value_head_dim": 128,          // Per-head value dimension
```

**Derived Dimensions:**
```
key_dim = linear_num_key_heads × linear_key_head_dim = 16 × 128 = 2048
value_dim = linear_num_value_heads × linear_value_head_dim = 32 × 128 = 4096
conv_dim = key_dim × 2 + value_dim = 2048 × 2 + 4096 = 8192
```

### 3.2 Layer Structure (Lines 228-273)

```python
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, hidden_size: int, num_k_heads: int, num_v_heads: int,
                 head_k_dim: int, head_v_dim: int, conv_kernel_size: int = 4):
        super().__init__()
        self.key_dim = num_k_heads * head_k_dim                    # 2048
        self.value_dim = num_v_heads * head_v_dim                  # 4096
        self.conv_dim = self.key_dim * 2 + self.value_dim          # 8192
        self.conv_kernel_size = conv_kernel_size                   # 4
        
        # Projections
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)  # 2048 -> 8192
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)   # 2048 -> 4096
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)      # 2048 -> 32
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)      # 2048 -> 32
        
        # Depthwise conv1d: each of 8192 channels has its own 1D convolution
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim,           # 8192
                                out_channels=self.conv_dim,          # 8192
                                kernel_size=conv_kernel_size,        # 4
                                groups=self.conv_dim,                # 8192 (depthwise)
                                padding=conv_kernel_size - 1,        # 3
                                bias=False)
        
        # Time step parameters
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))        # [32]
        A = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))                     # [32]
        
        # Output processing
        self.norm = RMSNormGated(head_v_dim, eps=1e-6)              # Per head_v_dim
        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False)  # 4096 -> 2048
        
        # STATEFUL: Per-sequence cache (THIS IS THE PROBLEM)
        self._recurrent_states: dict[int, torch.Tensor] = {}  # seq_id -> recurrent_state
        self._conv_states: dict[int, torch.Tensor] = {}       # seq_id -> conv_state
```

### 3.3 Prefill Forward (Lines 285-347)

```python
def _forward_prefill(self, hidden_states: torch.Tensor, seq_id: int | None = None) -> torch.Tensor:
    """Process full sequence during prefill. Save final state if seq_id provided."""
    seq_len = hidden_states.shape[0]
    batch_size = 1
    hidden_states_3d = hidden_states.unsqueeze(0)  # [1, seq_len, hidden_size]

    # LINEAR PROJECTIONS
    mixed_qkv = self.in_proj_qkv(hidden_states_3d)  # [1, seq_len, 8192]
    mixed_qkv = mixed_qkv.transpose(1, 2)           # [1, 8192, seq_len]

    z = self.in_proj_z(hidden_states_3d)            # [1, seq_len, 4096]
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [1, seq_len, 32, 128]

    b = self.in_proj_b(hidden_states_3d)            # [1, seq_len, 32]
    a = self.in_proj_a(hidden_states_3d)            # [1, seq_len, 32]

    # SAVE CONV STATE FOR DECODE: last (kernel_size - 1) = 3 columns
    if seq_id is not None:
        self._conv_states[seq_id] = mixed_qkv[:, :, -(self.conv_kernel_size - 1):].clone()
        # Shape: [1, 8192, 3]

    # CAUSAL CONV1D + ACTIVATION
    mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])  # [1, 8192, seq_len]
    mixed_qkv = mixed_qkv.transpose(1, 2)                        # [1, seq_len, 8192]

    # SPLIT INTO Q, K, V
    query, key, value = torch.split(mixed_qkv, [2048, 2048, 4096], dim=-1)
    query = query.reshape(batch_size, seq_len, 16, 128)   # [1, seq_len, 16, 128]
    key = key.reshape(batch_size, seq_len, 16, 128)       # [1, seq_len, 16, 128]
    value = value.reshape(batch_size, seq_len, 32, 128)   # [1, seq_len, 32, 128]

    beta = b.sigmoid()                                      # [1, seq_len, 32]
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [1, seq_len, 32]

    # EXPAND KEY HEADS IF NEEDED (16 -> 32)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(2, dim=2)          # [1, seq_len, 32, 128]
        key = key.repeat_interleave(2, dim=2)              # [1, seq_len, 32, 128]

    # CHUNK-BASED GATED DELTA RULE
    core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
        query, key, value,
        g=g, beta=beta,
        initial_state=None,
        output_final_state=(seq_id is not None),
        use_qk_l2norm_in_kernel=True,
    )
    # core_attn_out: [seq_len, 32, 128]
    # last_recurrent_state: [1, 32, 128, 128] (if seq_id provided)

    # SAVE RECURRENT STATE FOR DECODE
    if seq_id is not None and last_recurrent_state is not None:
        self._recurrent_states[seq_id] = last_recurrent_state

    # GATED RMSNORM + OUTPUT PROJECTION
    core_attn_out = core_attn_out.reshape(-1, 128)        # [seq_len*32, 128]
    z = z.reshape(-1, 128)                                # [seq_len*32, 128]
    core_attn_out = self.norm(core_attn_out, z)           # [seq_len*32, 128]
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)  # [1, seq_len, 4096]

    output = self.out_proj(core_attn_out)                 # [1, seq_len, 2048]
    return output.squeeze(0)                              # [seq_len, 2048]
```

### 3.4 Decode Forward (Lines 349-415)

```python
def _forward_decode_one(self, hidden_states: torch.Tensor, seq_id: int) -> torch.Tensor:
    """Process single token during decode using cached state."""
    batch_size = 1
    hidden_states_3d = hidden_states.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]

    # LINEAR PROJECTIONS
    mixed_qkv = self.in_proj_qkv(hidden_states_3d)        # [1, 1, 8192]
    mixed_qkv = mixed_qkv.transpose(1, 2)                 # [1, 8192, 1]

    z = self.in_proj_z(hidden_states_3d)                  # [1, 1, 4096]
    z = z.reshape(batch_size, 1, -1, self.head_v_dim)    # [1, 1, 32, 128]

    b = self.in_proj_b(hidden_states_3d)                  # [1, 1, 32]
    a = self.in_proj_a(hidden_states_3d)                  # [1, 1, 32]

    # CONV1D WITH CACHED STATE
    conv_state = self._conv_states.get(seq_id)            # [1, 8192, 3] (or None)
    if conv_state is not None:
        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)  # [1, 8192, 4]
    else:
        conv_input = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))  # [1, 8192, 4]
    
    # UPDATE CONV STATE: last (kernel_size - 1) = 3 columns
    self._conv_states[seq_id] = conv_input[:, :, -(self.conv_kernel_size - 1):].clone()
    # Updated state shape: [1, 8192, 3]

    # APPLY DEPTHWISE CONV (requires full kernel_size window)
    mixed_qkv_conv = F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias,
                              padding=0, groups=self.conv_dim)  # [1, 8192, 1]
    mixed_qkv = F.silu(mixed_qkv_conv)                    # [1, 8192, 1]
    mixed_qkv = mixed_qkv.transpose(1, 2)                 # [1, 1, 8192]

    # SPLIT INTO Q, K, V
    query, key, value = torch.split(mixed_qkv, [2048, 2048, 4096], dim=-1)
    query = query.reshape(batch_size, 1, 16, 128)        # [1, 1, 16, 128]
    key = key.reshape(batch_size, 1, 16, 128)            # [1, 1, 16, 128]
    value = value.reshape(batch_size, 1, 32, 128)        # [1, 1, 32, 128]

    beta = b.sigmoid()                                     # [1, 1, 32]
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [1, 1, 32]

    # EXPAND KEY HEADS IF NEEDED
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(2, dim=2)         # [1, 1, 32, 128]
        key = key.repeat_interleave(2, dim=2)             # [1, 1, 32, 128]

    # RECURRENT GATED DELTA RULE (single step)
    recurrent_state = self._recurrent_states.get(seq_id)   # [1, 32, 128, 128] (or None)
    core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
        query, key, value,
        g=g, beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # core_attn_out: [1, 32, 128]
    # last_recurrent_state: [1, 32, 128, 128]

    # UPDATE RECURRENT STATE
    self._recurrent_states[seq_id] = last_recurrent_state

    # GATED RMSNORM + OUTPUT PROJECTION
    core_attn_out = core_attn_out.reshape(-1, 128)        # [32, 128]
    z = z.reshape(-1, 128)                                # [32, 128]
    core_attn_out = self.norm(core_attn_out, z)           # [32, 128]
    core_attn_out = core_attn_out.reshape(batch_size, 1, -1)  # [1, 1, 4096]

    output = self.out_proj(core_attn_out)                 # [1, 1, 2048]
    return output.squeeze(0).squeeze(0)                   # [2048]
```

### 3.5 Main Forward (Lines 417-452)

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [total_tokens, hidden_size]
    
    Prefill: split by sequence, process each with saved state
    Decode: process each token with its sequence's saved state
    """
    from nanovllm.utils.context import get_context
    context = get_context()

    if context.is_prefill and context.cu_seqlens_q is not None:
        # PREFILL: Process multiple sequences independently
        cu_seqlens = context.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1
        outputs = []
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_hidden = hidden_states[start:end]
            seq_id = context.seq_ids[i] if context.seq_ids else i
            seq_out = self._forward_prefill(seq_hidden, seq_id=seq_id)
            outputs.append(seq_out)
        return torch.cat(outputs, dim=0)
    else:
        # DECODE: Process each token with its sequence's state
        num_tokens = hidden_states.shape[0]
        outputs = []
        for i in range(num_tokens):
            token_hidden = hidden_states[i]
            seq_id = context.seq_ids[i] if context.seq_ids else i
            out = self._forward_decode_one(token_hidden, seq_id=seq_id)
            outputs.append(out)
        return torch.stack(outputs, dim=0)
```

---

## Part 4: Why GatedDeltaNet Breaks CUDA Graph

### 4.1 The Core Problem

CUDA graphs replay **cached kernel sequences** without re-executing Python code. GatedDeltaNet violates this fundamental constraint:

```python
# This is Python dictionary access — NOT recorded in CUDA graph
recurrent_state = self._recurrent_states.get(seq_id)
conv_state = self._conv_states.get(seq_id)

# These dictionaries are updated AFTER each forward pass
self._recurrent_states[seq_id] = last_recurrent_state
self._conv_states[seq_id] = conv_state_updated
```

**Why This Fails During Capture:**

1. **Capture Phase**: For batch size 4, Python code executes:
   ```
   Token 0 (seq_id=0): recurrent_state = None → compute initial → save to dict
   Token 1 (seq_id=1): recurrent_state = None → compute initial → save to dict
   Token 2 (seq_id=2): recurrent_state = None → compute initial → save to dict
   Token 3 (seq_id=3): recurrent_state = None → compute initial → save to dict
   ```

2. **Replay Phase**: CUDA graph cannot re-execute the Python code that:
   - Retrieves states from dict
   - Updates states in dict
   - Creates new tensors for states
   
   The graph only knows: "execute kernels at GPU addresses X, Y, Z"
   
   But the Python variables that reference these addresses are lost!

### 4.2 Breaking the Graph Invariant

CUDA graphs require:
- **Fixed GPU memory addresses** for all tensors read/written by kernels
- **No dynamic tensor creation** during replay
- **No Python-side state mutations** during replay

GatedDeltaNet violates all three:

| Requirement | Attention ✓ | GatedDeltaNet ✗ |
|---|---|---|
| Fixed GPU addresses | `k_cache` pre-allocated | `recurrent_state` created per token |
| No dynamic creation | Uses fixed cache blocks | Creates new state tensors |
| No Python mutations | Reads `context_lens` (updated before replay) | Reads/updates `_recurrent_states` dict |

### 4.3 Current Workaround (Lines 45-53)

```python
# Force enforce_eager for models with stateful layers (e.g., GatedDeltaNet)
# CUDA Graph captures kernel calls but doesn't re-execute Python code,
# so dict-based state management in linear attention breaks during graph.replay()
model_type = getattr(hf_config, 'model_type', '')
if model_type == 'qwen3_5_moe' and not self.enforce_eager:
    print("[model_runner] WARNING: Qwen3.5 uses stateful linear attention layers "
          "incompatible with CUDA Graph. Forcing enforce_eager=True.")
    self.enforce_eager = True
    config.enforce_eager = True
```

When `enforce_eager=True`, the engine always takes the first branch of `run_model()`:
```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
    # No CUDA graph used, Python code executes normally
```

---

## Part 5: Design Specification for CUDA Graph Compatibility

### 5.1 High-Level Approach

To make GatedDeltaNet compatible with CUDA graphs, we must **pre-allocate state tensors** just like `k_cache`/`v_cache`, so they exist at **fixed GPU addresses**. Instead of using a Python dict, state is stored in a pre-allocated buffer indexed by seq_id.

### 5.2 Required Changes

#### 5.2.1 State Storage Structure

**Current (dict-based, incompatible):**
```python
self._recurrent_states: dict[int, torch.Tensor] = {}
self._conv_states: dict[int, torch.Tensor] = {}
```

**Proposed (pre-allocated, compatible):**
```python
# Pre-allocate in ModelRunner during setup (like k_cache, v_cache)
# One copy per GatedDeltaNet layer

# Recurrent state buffer:
# [max_num_seqs, num_v_heads, head_k_dim, head_v_dim]
# For each seq_id, maps to buffer[seq_id]
recurrent_state_buffer: torch.Tensor  # [512, 32, 128, 128] for Qwen3.5

# Conv state buffer:
# [max_num_seqs, conv_dim, conv_kernel_size - 1]
# For each seq_id, maps to buffer[seq_id]
conv_state_buffer: torch.Tensor      # [512, 8192, 3] for Qwen3.5
```

#### 5.2.2 State Initialization

During prefill, states must be **explicitly allocated and zeroed**:

```python
def _forward_prefill_graph_compatible(self, hidden_states, seq_id):
    # ... existing code ...
    
    # Initialize state buffers for this sequence
    if seq_id >= 0:  # seq_id is valid
        # Zero-init recurrent state: [1, num_v_heads, head_k_dim, head_v_dim]
        self.recurrent_state_buffer[seq_id].zero_()
        
        # Conv state initialization: depends on conv kernel
        # Option 1: Zero pad with (kernel_size-1) zeros
        # Option 2: Pre-compute and cache
        self.conv_state_buffer[seq_id].zero_()
    
    # ... rest of prefill ...
```

#### 5.2.3 State Access Pattern

**Current (dict, reads from Python):**
```python
recurrent_state = self._recurrent_states.get(seq_id)  # None or tensor
```

**Proposed (buffer, reads from GPU):**
```python
# Map seq_id to buffer index at setup time (in prepare_decode)
# This mapping is passed in context (like slot_mapping)
seq_id_to_buffer_idx = ...  # from context

# Access is now deterministic
recurrent_state = self.recurrent_state_buffer[seq_id_to_buffer_idx]
conv_state = self.conv_state_buffer[seq_id_to_buffer_idx]
```

#### 5.2.4 State Update Pattern

**Current (dict, mutates Python state):**
```python
self._recurrent_states[seq_id] = last_recurrent_state.clone()
self._conv_states[seq_id] = conv_state_updated.clone()
```

**Proposed (buffer, in-place update):**
```python
# In-place update at fixed GPU address
self.recurrent_state_buffer[seq_id_to_buffer_idx].copy_(last_recurrent_state.squeeze(0))
self.conv_state_buffer[seq_id_to_buffer_idx].copy_(conv_state_updated.squeeze(0))

# These updates happen OUTSIDE the graph capture
# During graph replay, these are pre-updated just like context_lens and slot_mapping
```

### 5.3 Integration with ModelRunner

#### 5.3.1 Pre-allocation Phase (modify allocate_kv_cache)

```python
def allocate_linear_attn_states(self):
    """Pre-allocate GatedDeltaNet recurrent and conv states."""
    config = self.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    
    layer_id = 0
    for module in self.model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            # Recurrent state: [max_bs, num_v_heads, head_k_dim, head_v_dim]
            recurrent_buffer = torch.zeros(
                max_bs,
                module.num_v_heads,
                module.head_k_dim,
                module.head_v_dim,
                dtype=torch.float32  # Or model's dtype
            )
            module.recurrent_state_buffer = recurrent_buffer
            
            # Conv state: [max_bs, conv_dim, conv_kernel_size - 1]
            conv_buffer = torch.zeros(
                max_bs,
                module.conv_dim,
                module.conv_kernel_size - 1,
                dtype=torch.float32
            )
            module.conv_state_buffer = conv_buffer
            
            layer_id += 1
```

#### 5.3.2 Graph Capture Modification

```python
def capture_cudagraph(self):
    # ... existing code ...
    
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        
        # Initialize linear attention states for all sequences in batch
        set_context(False, 
                   slot_mapping=slot_mapping[:bs], 
                   context_lens=context_lens[:bs], 
                   block_tables=block_tables[:bs],
                   seq_id_to_buffer_idx={i: i for i in range(bs)})  # NEW
        
        # PREFILL: Initialize recurrent/conv states to zero
        for i in range(bs):
            for module in self.model.modules():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    module.recurrent_state_buffer[i].zero_()
                    module.conv_state_buffer[i].zero_()
        
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        
        # ... rest unchanged ...
```

#### 5.3.3 Graph Replay Modification (run_model)

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # Update all pre-allocated tensors as before
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        # NEW: Update linear attention state buffers
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                for i in range(bs):
                    seq_id = context.seq_ids[i]
                    if seq_id in self._seq_id_to_state:
                        recurrent_state, conv_state = self._seq_id_to_state[seq_id]
                        module.recurrent_state_buffer[i].copy_(recurrent_state)
                        module.conv_state_buffer[i].copy_(conv_state)
                    else:
                        # First time seeing this sequence: zero init
                        module.recurrent_state_buffer[i].zero_()
                        module.conv_state_buffer[i].zero_()
        
        graph.replay()
        
        # NEW: Save updated state after graph replay (still outside graph)
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                for i in range(bs):
                    seq_id = context.seq_ids[i]
                    recurrent_state = module.recurrent_state_buffer[i].clone()
                    conv_state = module.conv_state_buffer[i].clone()
                    self._seq_id_to_state[seq_id] = (recurrent_state, conv_state)
        
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 5.4 GatedDeltaNet Implementation Changes

#### 5.4.1 Decode Forward (Modified)

```python
def _forward_decode_one(self, hidden_states: torch.Tensor, seq_id: int, 
                        buffer_idx: int) -> torch.Tensor:
    """Process single token during decode using pre-allocated state buffer."""
    batch_size = 1
    hidden_states_3d = hidden_states.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]

    # ... projections as before ...
    
    # CHANGED: Access pre-allocated buffer instead of dict
    recurrent_state = self.recurrent_state_buffer[buffer_idx].unsqueeze(0)  # [1, 32, 128, 128]
    conv_state = self.conv_state_buffer[buffer_idx].unsqueeze(0)           # [1, 8192, 3]

    # Conv with state (unchanged logic, but state comes from buffer)
    if conv_state.abs().sum() > 0:  # Check if state is non-zero
        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
    else:
        conv_input = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))

    # Update conv state in buffer (in-place or copy back)
    self.conv_state_buffer[buffer_idx].copy_(
        conv_input[:, :, -(self.conv_kernel_size - 1):].squeeze(0)
    )

    # ... rest of computation ...

    # Update recurrent state in buffer
    self.recurrent_state_buffer[buffer_idx].copy_(
        last_recurrent_state.squeeze(0)
    )

    return output.squeeze(0).squeeze(0)
```

#### 5.4.2 Context Extension

```python
# In utils/context.py, add seq_id to buffer index mapping
@dataclass
class Context:
    is_prefill: bool = True
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_ids: list[int] | None = None
    seq_id_to_buffer_idx: dict[int, int] | None = None  # NEW
    seq_id_to_buffer_idx_tensor: torch.Tensor | None = None  # GPU version (optional)
```

### 5.5 State Initialization During Prefill

When a new sequence starts prefill, its recurrent and conv states must be **explicitly cleared**:

```python
def _forward_prefill(self, hidden_states, seq_id, buffer_idx):
    """Prefill with pre-allocated state buffers."""
    seq_len = hidden_states.shape[0]
    batch_size = 1
    hidden_states_3d = hidden_states.unsqueeze(0)

    # Zero-init state for this sequence
    self.recurrent_state_buffer[buffer_idx].zero_()
    self.conv_state_buffer[buffer_idx].zero_()

    # ... projections as before ...

    # Save conv state at end of prefill
    self.conv_state_buffer[buffer_idx].copy_(
        mixed_qkv[:, :, -(self.conv_kernel_size - 1):].squeeze(0)
    )

    # ... compute output ...

    # Save recurrent state
    self.recurrent_state_buffer[buffer_idx].copy_(
        last_recurrent_state.squeeze(0)
    )

    return output.squeeze(0)
```

---

## Part 6: State Tensor Shapes and Memory Requirements

### 6.1 Qwen3.5 Configuration

```
hidden_size: 2048
linear_num_key_heads: 16
linear_num_value_heads: 32
linear_key_head_dim: 128
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
max_num_seqs: 512
```

### 6.2 Per-Layer State Tensors

#### Recurrent State
- **Shape**: `[max_num_seqs, num_v_heads, head_k_dim, head_v_dim]`
- **For Qwen3.5**: `[512, 32, 128, 128]`
- **Size**: 512 × 32 × 128 × 128 × 4 bytes (float32) = **1 GB per layer**
- **For 10 linear layers**: **10 GB**

#### Conv State
- **Shape**: `[max_num_seqs, conv_dim, conv_kernel_size - 1]`
- **For Qwen3.5**: `[512, 8192, 3]`
- **Size**: 512 × 8192 × 3 × 4 bytes = **50 MB per layer**
- **For 10 linear layers**: **500 MB**

#### Total Memory
- **Recurrent states only**: ~10 GB (dominant cost)
- **Conv states only**: ~500 MB
- **Per linear layer total**: ~1.05 GB

### 6.3 Memory Optimization Strategies

#### Option 1: Reduce max_num_seqs in CUDA Graph Path
Only pre-allocate states for the batch sizes actually used in graphs (1-512):
```python
# Use adaptive batch-sized state buffers
max_graph_bs = 512
recurrent_buffer = torch.zeros(max_graph_bs, 32, 128, 128)
```

#### Option 2: Share State Buffer Across Layers (Unsafe)
Use single shared buffer reshaped per layer (risky, complicates cleanup).

#### Option 3: Allocate States On-Demand Per Sequence
Keep dict-based state but also pre-allocate a "graph state buffer" just for sequences in graph replay.

#### Option 4: Use Reduced Precision (fp16/bf16)
Store states in bfloat16 instead of float32:
- Qwen3.5 model already uses bfloat16
- Recurrent state: 512 × 32 × 128 × 128 × 2 bytes = **500 MB per layer**
- Total for 10 layers: **5 GB** (50% reduction)

---

## Part 7: Exact Tensor Operation Breakdown

### 7.1 Recurrent Gated Delta Rule (Single Step)

**File**: Lines 170-212

```python
def torch_recurrent_gated_delta_rule(
    query,          # [batch=1, 1, num_v_heads=32, head_k_dim=128]
    key,            # [batch=1, 1, num_v_heads=32, head_k_dim=128]
    value,          # [batch=1, 1, num_v_heads=32, head_v_dim=128]
    g,              # [batch=1, 1, num_v_heads=32] decay factor (log scale)
    beta,           # [batch=1, 1, num_v_heads=32] update gate
    initial_state,  # [batch=1, num_v_heads=32, head_k_dim=128, head_v_dim=128]
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
):
    # Convert to float32
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(float32) for x in ...]
    # After transpose: [batch=1, num_v_heads=32, seq_len=1, head_k_dim/v_dim]

    batch_size = 1
    num_heads = 32
    sequence_length = 1
    k_head_dim = 128
    v_head_dim = 128
    
    # LOOP OVER TOKENS (only 1 iteration for decode)
    for i in range(1):
        q_t = query[:, :, i]                # [1, 32, 128]
        k_t = key[:, :, i]                  # [1, 32, 128]
        v_t = value[:, :, i]                # [1, 32, 128]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [1, 32, 1, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)               # [1, 32, 1]

        # OPERATION 1: Decay recurrent state
        last_recurrent_state = last_recurrent_state * g_t
        # [1, 32, 128, 128] * [1, 32, 1, 1] → [1, 32, 128, 128]

        # OPERATION 2: Compute memory value
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        # [1, 32, 128, 128] * [1, 32, 128, 1] → [1, 32, 128, 128]
        # sum dim=-2 → [1, 32, 128]

        # OPERATION 3: Compute delta
        delta = (v_t - kv_mem) * beta_t
        # [1, 32, 128] - [1, 32, 128] → [1, 32, 128]
        # * [1, 32, 1] → [1, 32, 128]

        # OPERATION 4: Update recurrent state
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # [1, 32, 128, 128] + ([1, 32, 128, 1] * [1, 32, 1, 128])
        # → [1, 32, 128, 128] + [1, 32, 128, 128] → [1, 32, 128, 128]

        # OPERATION 5: Compute output
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        # [1, 32, 128, 128] * [1, 32, 128, 1] → [1, 32, 128, 128]
        # sum dim=-2 → [1, 32, 128]

    # Final output shape: [1, 1, 32, 128]
    # Final state shape: [1, 32, 128, 128]
    return core_attn_out, last_recurrent_state
```

**Tensor Memory Layout:**
- **Recurrent state**: `[batch, num_v_heads, head_k_dim, head_v_dim]`
  - Stores k×v outer product for linear recurrence
  - Must be updated in-place for CUDA graph compatibility
  - Size: 1 × 32 × 128 × 128 × 4 bytes = 2 MB per token

### 7.2 Chunk-Based Gated Delta Rule (Prefill)

**File**: Lines 93-163

For prefill with sequence length L:
- Chunks query/key/value into size 64
- For each chunk, uses gated delta rule with inter-chunk residuals
- Final state saved for next decode step

---

## Part 8: Comparison Table: Current vs. Proposed

| Aspect | Current (Dict-Based) | Proposed (Pre-Allocated) |
|---|---|---|
| **State Storage** | `dict[int, Tensor]` | `Tensor[max_bs, ...]` |
| **GPU Addresses** | Dynamic (new tensor each forward) | Fixed (same address per seq slot) |
| **CUDA Graph Compatibility** | ❌ Incompatible | ✅ Compatible |
| **Prefill Requirement** | Dict initialization | Zero-init buffer at seq_id |
| **Decode Requirement** | Fetch from dict | Copy from buffer pre-replay, update post-replay |
| **Memory Overhead** | Minimal (only active seqs) | ~10 GB for Qwen3.5 (512 max_seqs × 10 layers) |
| **State Cleanup** | Call `clear_state()` per seq | Zero buffer slot on seq finish |
| **Code Complexity** | Simple (dict access) | More complex (buffer indexing, copies) |

---

## Part 9: Summary: Required Implementation Steps

### Phase 1: Setup (ModelRunner)
1. [ ] Add `allocate_linear_attn_states()` method
   - Pre-allocate recurrent/conv state buffers
   - Attach to each GatedDeltaNet layer
   - Size: `[max_bs, ...]` for each tensor

2. [ ] Extend `Context` class with `seq_id_to_buffer_idx` mapping

3. [ ] Update `prepare_decode()` to populate seq_id→buffer_idx mapping

### Phase 2: Graph Capture (ModelRunner.capture_cudagraph)
1. [ ] Initialize state buffers to zero before warmup/capture
2. [ ] Create mapping for test batch during capture

### Phase 3: Graph Replay (ModelRunner.run_model)
1. [ ] Pre-update all state buffers BEFORE `graph.replay()`
   - Load saved state from `_seq_id_to_state` dict
   - Copy into buffer at current token positions

2. [ ] Post-update state storage AFTER `graph.replay()`
   - Read final state from buffer
   - Save to `_seq_id_to_state` dict

### Phase 4: GatedDeltaNet Refactor
1. [ ] Add `recurrent_state_buffer` and `conv_state_buffer` tensors
2. [ ] Modify `_forward_decode_one()` to:
   - Accept `buffer_idx` parameter
   - Read/write state from buffer instead of dict
3. [ ] Update `_forward_prefill()` to zero-init buffers
4. [ ] Remove dict-based state, keep only buffer-based

### Phase 5: Testing & Optimization
1. [ ] Verify graph capture with linear attention layers
2. [ ] Benchmark: CUDA graph vs eager execution
3. [ ] Optimize state tensor precision (fp16/bf16)
4. [ ] Profile memory usage

---

## Part 10: Code Example: Minimal Working Implementation

Here's pseudocode for a minimal implementation:

```python
# In model_runner.py
class ModelRunner:
    def allocate_linear_attn_states(self):
        """Pre-allocate state buffers for GatedDeltaNet."""
        for module in self.model.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                max_bs = 512
                dtype = torch.bfloat16  # Match model dtype
                
                module.recurrent_state_buffer = torch.zeros(
                    max_bs, module.num_v_heads, module.head_k_dim, module.head_v_dim,
                    dtype=dtype, device='cuda'
                )
                module.conv_state_buffer = torch.zeros(
                    max_bs, module.conv_dim, module.conv_kernel_size - 1,
                    dtype=dtype, device='cuda'
                )
        
        # Track saved state per seq_id
        self._seq_id_to_state: dict[int, tuple] = {}

    def run_model(self, input_ids, positions, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # Update tensor values
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        # UPDATE STATE BUFFERS PRE-REPLAY (new)
        for i in range(bs):
            seq_id = context.seq_ids[i]
            for module in self.model.modules():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    if seq_id in self._seq_id_to_state:
                        rec_state, conv_state = self._seq_id_to_state[seq_id]
                        module.recurrent_state_buffer[i].copy_(rec_state)
                        module.conv_state_buffer[i].copy_(conv_state)
                    else:
                        module.recurrent_state_buffer[i].zero_()
                        module.conv_state_buffer[i].zero_()
        
        # Replay graph
        graph.replay()
        
        # SAVE STATE BUFFERS POST-REPLAY (new)
        for i in range(bs):
            seq_id = context.seq_ids[i]
            for module in self.model.modules():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    rec_state = module.recurrent_state_buffer[i].clone()
                    conv_state = module.conv_state_buffer[i].clone()
                    self._seq_id_to_state[seq_id] = (rec_state, conv_state)
        
        return self.model.compute_logits(graph_vars["outputs"][:bs])

# In qwen3_5.py
class Qwen3_5GatedDeltaNet(nn.Module):
    def _forward_decode_one(self, hidden_states, seq_id, buffer_idx):
        # ... compute mixed_qkv, z, a, b ...
        
        # Access pre-allocated buffer
        recurrent_state = self.recurrent_state_buffer[buffer_idx].unsqueeze(0)
        conv_state_cached = self.conv_state_buffer[buffer_idx].unsqueeze(0)
        
        # Conv with cached state
        if conv_state_cached.abs().sum() > 1e-6:
            conv_input = torch.cat([conv_state_cached, mixed_qkv], dim=-1)
        else:
            conv_input = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        
        # Update buffer in-place
        self.conv_state_buffer[buffer_idx].copy_(
            conv_input[:, :, -(self.conv_kernel_size - 1):].squeeze(0)
        )
        
        # ... rest of computation ...
        
        # Update recurrent state buffer
        self.recurrent_state_buffer[buffer_idx].copy_(
            last_recurrent_state.squeeze(0)
        )
        
        return output.squeeze(0).squeeze(0)
```

---

## Conclusion

Making GatedDeltaNet compatible with CUDA Graph requires replacing Python dict-based state management with **pre-allocated GPU buffers at fixed addresses**. This is non-trivial due to:

1. **Memory overhead**: ~1 GB per linear attention layer
2. **Code complexity**: Requires careful state indexing and buffer management
3. **Prefill/decode coordination**: State must be zeroed at sequence start, saved after prefill, loaded before decode

The upside is **significant throughput improvement** for decode batches, since kernel launch overhead is eliminated. With proper implementation, decode batches can be 2-3× faster with graphs enabled.

