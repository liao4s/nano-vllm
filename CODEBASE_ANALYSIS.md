# Nano-vLLM Codebase Analysis

## 1. Project Overview

**Nano-vLLM** is a lightweight vLLM implementation built from scratch (~1,200 lines of Python). It's comparable in performance to vLLM while maintaining a clean, readable codebase.

**Key Metrics:**
- Performance: ~1434 tok/s (vs vLLM ~1362 tok/s) on Qwen3-0.6B
- Models supported: Qwen3, Qwen3.5 Dense, Qwen3.5 MoE
- Hardware optimizations: CUDA graphs, prefix caching, tensor parallelism, Triton kernels

## 2. Project Structure

```
nano-vllm/
├── nanovllm/
│   ├── config.py                 # Config loading and validation
│   ├── sampling_params.py        # Sampling parameters
│   ├── llm.py                    # Main LLM interface
│   ├── server.py                 # Web server/API
│   ├── engine/
│   │   ├── llm_engine.py         # Main inference loop
│   │   ├── scheduler.py          # Sequence scheduling (prefill/decode)
│   │   ├── model_runner.py       # Model execution, KV cache management
│   │   ├── block_manager.py      # Block-based KV cache management
│   │   └── sequence.py           # Sequence state tracking
│   ├── layers/
│   │   ├── attention.py          # Attention module + KV cache storage
│   │   ├── linear.py             # Tensor parallel linear layers
│   │   ├── rotary_embedding.py   # RoPE implementation
│   │   ├── embed_head.py         # Embedding + LM head
│   │   ├── layernorm.py          # RMSNorm
│   │   ├── sampler.py            # Token sampling
│   │   ├── activation.py         # Activations (SiLU+gating)
│   │   └── fla_ops/              # Triton kernels for linear attention
│   │       ├── chunk.py          # Chunk-based GatedDeltaNet
│   │       ├── decode_kernel.py  # Single-step GatedDeltaNet decode
│   │       ├── chunk_delta_h.py
│   │       ├── chunk_o.py
│   │       ├── chunk_scaled_dot_kkt.py
│   │       ├── cumsum.py
│   │       ├── solve_tril.py
│   │       └── ...
│   ├── models/
│   │   ├── qwen3.py              # Standard Qwen3 (full attention only)
│   │   ├── qwen3_5.py            # Qwen3.5 MoE (hybrid attention + MoE)
│   │   └── qwen3_5_dense.py      # Qwen3.5 Dense (hybrid attention + dense MLP)
│   └── utils/
│       ├── context.py            # Thread-local context for inference
│       ├── loader.py             # Weight loading
│       └── logger.py
├── example.py                     # Usage example
└── bench.py                       # Benchmark script
```

## 3. Attention Mechanisms

### 3.1 Full Attention (Qwen3_5FullAttention)

**File:** `nanovllm/models/qwen3_5.py` (lines 487-633)

**Features:**
- Multi-head attention with GQA (Grouped Query Attention)
- **Output gating**: q_proj outputs 2x the dimension (query + gate)
- **Partial rotary embedding**: Only 25% of head_dim uses RoPE
- Tensor parallel support with KV head replication when needed
- Q/K norms using (1+w)-style RMSNorm

**Key Code:**
```python
class Qwen3_5FullAttention(nn.Module):
    def forward(self, positions, hidden_states):
        # Q projection outputs [query, gate] (2x dims)
        q_out = self.q_proj(hidden_states)
        q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
        query, gate = q_out.chunk(2, dim=-1)
        
        # K, V projections
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Q/K norms (Qwen3.5 style: (1+w)*norm(x))
        query = self.q_norm(query)
        k = self.k_norm(k)
        
        # Partial RoPE (25% of head_dim)
        query, k = self.rotary_emb(positions, query, k)
        
        # Attention
        o = self.attn(query, k, v)
        
        # Output gating: output *= sigmoid(gate)
        o_flat = o.flatten(1, -1)
        o_flat = o_flat * torch.sigmoid(gate)
        
        return self.o_proj(o_flat)
```

### 3.2 Linear Attention (Qwen3_5GatedDeltaNet)

**File:** `nanovllm/models/qwen3_5.py` (lines 220-481)

**Architecture:** Gated DeltaNet (linear attention) with recurrent state and convolutional components

**Key Features:**
- **Linear complexity**: Uses recurrent state instead of quadratic attention matrix
- **Recurrent state buffer**: Pre-allocated GPU buffers for CUDA Graph compatibility
  - `recurrent_state_buf`: [num_layers, max_slots, num_v_heads, head_k_dim, head_v_dim]
  - `conv_state_buf`: [num_layers, max_slots, conv_dim, kernel_size-1]
- **Two modes:**
  - **Prefill:** Chunk-based processing for long sequences (Triton kernel)
  - **Decode:** Single-step recurrent processing (batched via Triton kernel)
- **Components:**
  - Conv1d (depthwise, causal, kernel_size=4)
  - Gated delta rule for state updates
  - Time step parameters (A_log, dt_bias)
  - RMSNorm with SiLU gating

**Prefill Process (`_forward_prefill`):**
```python
def _forward_prefill(self, hidden_states, slot_idx=None):
    # 1. Linear projections
    mixed_qkv = self.in_proj_qkv(hidden_states)  # [1, seq_len, conv_dim]
    z = self.in_proj_z(hidden_states)             # gating
    a = self.in_proj_a(hidden_states)             # decay
    b = self.in_proj_b(hidden_states)             # beta
    
    # 2. Causal conv1d + SiLU
    mixed_qkv = F.silu(self.conv1d(mixed_qkv))
    
    # 3. Split Q, K, V and reshape
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim])
    
    # 4. Chunk-based gated delta rule (Triton kernel - fast for long sequences)
    core_attn_out, last_recurrent_state = triton_chunk_gated_delta_rule(...)
    
    # 5. RMSNorm + gate + output projection
    core_attn_out = self.norm(core_attn_out, z)
    return self.out_proj(core_attn_out)
```

**Decode Process (`_forward_decode_batched`):**
```python
def _forward_decode_batched(self, hidden_states, slot_indices):
    # Batched processing (no Python loops = CUDA Graph compatible)
    # 1. Linear projections (batched)
    mixed_qkv = self.in_proj_qkv(hidden_states)   # [B, conv_dim]
    z = self.in_proj_z(hidden_states)
    a, b = self.in_proj_a(hidden_states), self.in_proj_b(hidden_states)
    
    # 2. Conv1d with pre-allocated state buffer
    conv_state = self.conv_state_buf[slot_indices]  # [B, conv_dim, kernel_size-1]
    conv_input = torch.cat([conv_state, mixed_qkv.unsqueeze(-1)], dim=-1)
    # Update: sliding window (drop oldest, keep newest kernel_size-1)
    self.conv_state_buf[slot_indices] = conv_input[:, :, 1:]
    
    # 3. Split Q, K, V
    query, key, value = mixed_qkv_act.split([key_dim, key_dim, value_dim])
    
    # 4. Single-step recurrent delta rule (Triton kernel - updates state in-place)
    output = gdn_decode_batched(q, k, v, g_log, beta_val,
                                self.recurrent_state_buf, slot_indices)
    
    # 5. RMSNorm + gate + output projection
    output = self.norm(output, z)
    return self.out_proj(output)
```

### 3.3 Attention Module (Standard KV Cache)

**File:** `nanovllm/layers/attention.py`

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store KV in cache (using Triton kernel for efficiency)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill: use flash attention with prefix caching support
            if context.block_tables is not None:  # prefix cache hit
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                      max_seqlen_q=context.max_seqlen_q,
                                      cu_seqlens_q=context.cu_seqlens_q,
                                      max_seqlen_k=context.max_seqlen_k,
                                      cu_seqlens_k=context.cu_seqlens_k,
                                      softmax_scale=self.scale, causal=True,
                                      block_table=context.block_tables)
        else:
            # Decode: single token, use flash attention with KV cache
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                       cache_seqlens=context.context_lens,
                                       block_table=context.block_tables,
                                       softmax_scale=self.scale, causal=True)
        return o
```

### 3.4 Hybrid Attention Pattern

**File:** `nanovllm/config.py` (lines 7-96) and model files

Qwen3.5 models use **hybrid attention**: 3 linear attention layers + 1 full attention per 4 layers

**Configuration in JSON:**
```
"layer_types": ["linear_attention", "linear_attention", "linear_attention", 
                "full_attention", "linear_attention", "linear_attention", ...]
```

**Dynamic selection in decoder layer:**
```python
class Qwen3_5DenseDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
```

## 4. Inference & Scheduling

### 4.1 LLM Engine Flow

**File:** `nanovllm/engine/llm_engine.py`

```python
class LLMEngine:
    def generate(self, prompts, sampling_params):
        # 1. Add requests
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        # 2. Inference loop
        while not self.is_finished():
            seqs, is_prefill = self.scheduler.schedule()
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
```

### 4.2 Scheduling (Prefill vs Decode)

**File:** `nanovllm/engine/scheduler.py`

**Two-phase scheduling:**

```python
class Scheduler:
    def schedule(self):
        scheduled_seqs = []
        
        # PHASE 1: Prefill (first request in sequence)
        while self.waiting and num_seqs < max_num_seqs:
            seq = self.waiting[0]
            # Check if can allocate blocks and fit in batched tokens
            if num_batched_tokens + len(seq) > max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq):
                break
            
            # Move to running
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # is_prefill=True
        
        # PHASE 2: Decode (continue existing sequences)
        while self.running and num_seqs < max_num_seqs:
            seq = self.running.popleft()
            
            # Preempt if can't append
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        
        return scheduled_seqs, False  # is_prefill=False
```

### 4.3 Model Runner: Prefill Preparation

**File:** `nanovllm/engine/model_runner.py` (lines 294-333)

```python
def prepare_prefill(self, seqs):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # Cumulative sequence lengths for queries
    cu_seqlens_k = [0]  # Cumulative sequence lengths for keys
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        # Only add non-cached tokens
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        
        seqlen_q = seqlen - seq.num_cached_tokens  # new tokens
        seqlen_k = seqlen  # full sequence (for attention context)
        
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        
        # Map each token to KV cache slot
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            end = start + (self.block_size if i < seq.num_blocks - 1 
                          else seq.last_block_num_tokens)
            slot_mapping.extend(list(range(start, end)))
    
    # Set context for attention layers
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables, 
                linear_attn_slot_indices=...)
    
    return input_ids, positions
```

### 4.4 Model Runner: Decode Preparation

**File:** `nanovllm/engine/model_runner.py` (lines 335-354)

```python
def prepare_decode(self, seqs):
    input_ids = []    # Last token of each sequence
    positions = []    # Position index for each token
    slot_mapping = [] # KV cache slot for each token
    context_lens = [] # Context length for each sequence
    
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        # Map to KV cache: last position in last block
        slot_mapping.append(seq.block_table[-1] * self.block_size + 
                           seq.last_block_num_tokens - 1)
    
    # Set context for single-token attention
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens,
                block_tables=block_tables,
                linear_attn_slot_indices=...)
    
    return input_ids, positions
```

### 4.5 Model Execution

**File:** `nanovllm/engine/model_runner.py` (lines 363-392)

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    # Use CUDA graphs for decode if available
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Use pre-captured CUDA graph for decode
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        
        # Update graph variables with current batch
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs] = context.block_tables
        
        # Update linear attention slot indices
        if context.linear_attn_slot_indices is not None:
            graph_vars["linear_attn_slot_indices"][:bs] = context.linear_attn_slot_indices
        
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

## 5. KV Cache Management

### 5.1 Block-Based KV Cache

**File:** `nanovllm/engine/block_manager.py`

```python
class BlockManager:
    def __init__(self, num_blocks, block_size, enable_prefix_caching=True):
        self.block_size = block_size  # 256 tokens per block
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}  # Prefix caching: hash -> block_id
        self.free_block_ids = deque(range(num_blocks))
        
    def allocate(self, seq):
        # Allocate blocks for sequence with prefix caching
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            
            # Hash each block for prefix caching
            h = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id == -1:  # Cache miss
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:  # Cache hit - reuse block
                seq.num_cached_tokens += self.block_size
                block.ref_count += 1
            
            seq.block_table.append(block_id)
```

### 5.2 Allocation in ModelRunner

**File:** `nanovllm/engine/model_runner.py` (lines 184-213)

```python
def allocate_kv_cache(self):
    # Calculate KV cache requirements
    num_kv_heads = hf_config.num_key_value_heads // world_size
    head_dim = hf_config.head_dim
    num_attn_layers = sum(1 for m in model.modules() 
                         if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
    
    # Block size: 2 (k,v) * num_layers * block_size * num_kv_heads * head_dim
    block_bytes = 2 * num_attn_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    
    # Reserve memory for linear attention and CUDA graphs
    reserved = linear_attn_budget + cuda_graph_reserve
    kv_budget = int(total_memory * gpu_util) - reserved
    
    num_kvcache_blocks = kv_budget // block_bytes
    
    # Create unified KV cache tensor
    self.kv_cache = torch.empty(2, num_attn_layers, num_kvcache_blocks, 
                                block_size, num_kv_heads, head_dim)
    
    # Assign to attention layers
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 5.3 Linear Attention State Buffers

**File:** `nanovllm/engine/model_runner.py` (lines 134-280)

```python
def allocate_linear_attn_states(self):
    """Pre-allocate recurrent/conv state buffers for GatedDeltaNet layers."""
    linear_layers = [m for m in model.modules() 
                    if isinstance(m, Qwen3_5GatedDeltaNet)]
    
    # Per-sequence slot management
    max_slots = min(32, max_num_seqs)
    
    # Recurrent state: [num_layers, max_slots, num_v_heads, head_k_dim, head_v_dim]
    self.linear_attn_recurrent_buf = torch.zeros(
        num_layers, max_slots, num_v_heads, head_k_dim, head_v_dim,
        dtype=dtype, device="cuda"
    )
    
    # Conv state: [num_layers, max_slots, conv_dim, kernel_size-1]
    self.linear_attn_conv_buf = torch.zeros(
        num_layers, max_slots, conv_dim, conv_kernel_size - 1,
        dtype=dtype, device="cuda"
    )
    
    # Assign buffers to layers
    for i, module in enumerate(linear_layers):
        module.recurrent_state_buf = self.linear_attn_recurrent_buf[i]
        module.conv_state_buf = self.linear_attn_conv_buf[i]
    
    # Slot management
    self._linear_attn_slot_map = {}  # seq_id -> slot_idx
    self._linear_attn_free_slots = deque(range(max_slots))

def allocate_linear_attn_slot(self, seq_id):
    """Allocate buffer slot when sequence starts."""
    slot_idx = self._linear_attn_free_slots.popleft()
    self._linear_attn_slot_map[seq_id] = slot_idx
    # Zero out slot
    self.linear_attn_recurrent_buf[:, slot_idx].zero_()
    self.linear_attn_conv_buf[:, slot_idx].zero_()
    return slot_idx

def free_linear_attn_slot(self, seq_id):
    """Free buffer slot when sequence finishes."""
    slot_idx = self._linear_attn_slot_map.pop(seq_id, None)
    if slot_idx is not None:
        self._linear_attn_free_slots.append(slot_idx)
```

### 5.4 KV Cache Storage (Triton Kernel)

**File:** `nanovllm/layers/attention.py` (lines 10-40)

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                        k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    
    # Load key/value for token idx
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # Store at slot in cache
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

## 6. Linear Attention Inference (FLA Operations)

### 6.1 Prefill: Chunk-Based Gated Delta Rule

**File:** `nanovllm/layers/fla_ops/chunk.py` (lines 24-219)

The chunk-based approach processes sequences in fixed-size chunks (typically 64 tokens) for efficiency:

```python
def chunk_gated_delta_rule(q, k, v, g, beta, ...):
    """
    Processes sequences in chunks for linear attention.
    
    Args:
        q: [B, T, H, K] - queries
        k: [B, T, H, K] - keys
        v: [B, T, H, V] - values
        g: [B, T, H] - forget gates (log-space)
        beta: [B, T, H] - update gates
    """
    # Uses several Triton kernels for efficiency:
    g = chunk_local_cumsum(g, chunk_size=64)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g)
    A = solve_tril(A)
    w, u = recompute_w_u_fwd(k, v, beta, A, g)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, ...)
    o = chunk_fwd_o(q, k, v_new, h, g)
    return o, final_state
```

**Key Components:**
- **chunk_local_cumsum**: Local cumulative sum of gates within chunks
- **chunk_scaled_dot_kkt_fwd**: Compute KKT matrix (block-wise attention scores)
- **solve_tril**: Solve triangular system (invert KKT)
- **recompute_w_u_fwd**: WY representation for efficient computation
- **chunk_gated_delta_rule_fwd_h**: Forward pass for hidden states
- **chunk_fwd_o**: Compute output

### 6.2 Decode: Single-Step Recurrent

**File:** `nanovllm/layers/fla_ops/decode_kernel.py` (lines 21-146)

```python
@triton.jit
def gdn_decode_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
                     state_ptr, slot_indices_ptr, ...):
    """Triton kernel for batched single-step GatedDeltaNet decode."""
    
    # Get batch and head indices from thread block
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # Get slot index for this sequence
    slot_idx = tl.load(slot_indices_ptr + i_b)
    
    # Load scalars: g, beta, q, k, v
    g_exp = tl.exp(g_val)
    
    # FIRST PASS: Decay state and compute kv_mem
    # state *= g (decay)
    # kv_mem = sum_k(state[k,v] * k[k])
    for i_k in range(K):
        s = tl.load(state[slot_idx, h, k, v_block])
        s = s * g_exp  # Decay
        kv_mem += s * k[k]
        tl.store(state, s)  # Store decayed state
    
    # Compute delta = (v - kv_mem) * beta
    delta = (v - kv_mem) * beta
    
    # SECOND PASS: Update state and compute output
    # state += k^T @ delta
    # output = state @ q
    for i_k in range(K):
        s = tl.load(state[slot_idx, h, k, v_block])
        s = s + k[k] * delta  # Update state with rank-1 update
        output += s * q[k]
        tl.store(state, s)  # Store final state
```

**Process:**
1. **State decay**: Multiply previous state by exp(g) (forgetting)
2. **Recall**: Compute kv_mem = state @ k (retrieve from memory)
3. **Update**: delta = (v - kv_mem) * beta; state += k ⊗ delta
4. **Output**: output = state @ q (query the updated state)

## 7. CUDA Graph Capture

**File:** `nanovllm/engine/model_runner.py` (lines 394-439)

```python
def capture_cudagraph(self):
    max_bs = min(max_num_seqs, 512)
    
    # Pre-allocate input/output tensors for graph capture
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    linear_attn_slot_indices = torch.zeros(max_bs, dtype=torch.int64)
    
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], ...)
        
        # Warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        # Capture
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        
        self.graphs[bs] = graph
```

**Benefits:**
- Eliminates CPU overhead during decode
- All attention computations fused
- Supports multiple batch sizes (1, 2, 4, 8, 16, 32, ...)

## 8. Context & Threading

**File:** `nanovllm/utils/context.py`

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor  # [num_seqs+1] cumulative query seqlen
    cu_seqlens_k: torch.Tensor  # [num_seqs+1] cumulative key seqlen
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: torch.Tensor  # [total_tokens] -> cache slot
    context_lens: torch.Tensor  # [batch] -> length for each seq
    block_tables: torch.Tensor  # [batch, max_blocks] -> block indices
    seq_ids: list[int]
    linear_attn_slot_indices: torch.Tensor  # [batch] -> linear attn buffer slot
```

Used to pass inference context through attention layers without explicit parameters.

## 9. Key Summary Tables

### Attention Comparison

| Aspect | Full Attention | Linear Attention |
|--------|---|---|
| Complexity | O(T²) | O(T) |
| State | KV cache | Recurrent state + conv state |
| Prefill | Flash Attention (Triton) | Chunk-based (Triton) |
| Decode | Flash Attention + KV cache | Single-step recurrent (Triton) |
| CUDA Graph | Supported | Supported (batch slots) |

### Layer Types (Hybrid Attention)

For Qwen3.5 models:
- **Pattern**: 3 linear + 1 full (repeating)
- **Layer 0-2**: Linear Attention (GatedDeltaNet)
- **Layer 3**: Full Attention
- **Layer 4-6**: Linear Attention
- **Layer 7**: Full Attention
- etc.

### Memory Allocation

```
Total GPU Memory
├── Model weights
├── KV Cache
│   ├── Allocated based on: (total - used - peak) * gpu_util - reserved
│   └── Block size: 256 tokens
├── Linear Attention Buffers
│   ├── Recurrent: [num_layers, max_slots, HV, K, V]
│   ├── Conv: [num_layers, max_slots, conv_dim, kernel_size-1]
│   └── ~1GB for 32 concurrent sequences (Qwen3.5-35B)
└── CUDA Graph Reserve (~2MB per layer)
```

## 10. File Reference Map

| Functionality | Files |
|---|---|
| Attention mechanisms | `attention.py`, `qwen3_5.py` (lines 220-633) |
| Linear attention kernels | `fla_ops/chunk.py`, `fla_ops/decode_kernel.py` |
| KV cache | `block_manager.py`, `attention.py` (Triton kernel) |
| Scheduling | `scheduler.py` |
| Inference loop | `llm_engine.py`, `model_runner.py` |
| Models | `qwen3.py`, `qwen3_5.py`, `qwen3_5_dense.py` |
| Tensor parallelism | `linear.py` (ColumnParallel, RowParallel, etc.) |
| Context passing | `utils/context.py` |
| CUDA graphs | `model_runner.py` (lines 394-439) |

