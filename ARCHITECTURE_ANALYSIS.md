# Nano-vLLM Architecture Analysis

## Project Overview

**Nano-vLLM** is a lightweight, production-focused vLLM implementation built from scratch in ~1,200 lines of Python code. It achieves comparable inference speeds to vLLM while maintaining a clean, readable codebase.

### Key Stats
- **Version**: 0.2.0
- **Supported Models**: Qwen3 (standard), Qwen3.5-MoE (with hybrid attention)
- **Performance**: 1434.13 tok/s on RTX 4070 (vs vLLM's 1361.84 tok/s)
- **Features**: Prefix caching, tensor parallelism, torch compilation, CUDA graphs

---

## 1. PROJECT STRUCTURE

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py              # Entry point exports (LLM, SamplingParams)
│   ├── config.py                # Configuration and model loading
│   ├── llm.py                   # Main LLM class (wrapper around LLMEngine)
│   ├── sampling_params.py       # Sampling configuration
│   ├── server.py                # Server implementation
│   ├── engine/
│   │   ├── llm_engine.py       # Main inference engine orchestrator
│   │   ├── model_runner.py     # Model execution and CUDA graph management
│   │   ├── scheduler.py        # Batch scheduling and prefill/decode phases
│   │   ├── sequence.py         # Sequence state management
│   │   └── block_manager.py    # KV cache block allocation (prefix caching)
│   ├── layers/
│   │   ├── attention.py        # Flash attention wrapper
│   │   ├── linear.py           # Tensor parallel linear layers
│   │   ├── rotary_embedding.py # Rotary position embeddings
│   │   ├── layernorm.py        # RMSNorm with fused operations
│   │   ├── activation.py       # SiLU + MUL gating
│   │   ├── sampler.py          # Gumbel-max sampling
│   │   └── embed_head.py       # Parallel embeddings/output heads
│   ├── models/
│   │   ├── qwen3.py            # Qwen3 model implementation
│   │   └── qwen3_5.py          # Qwen3.5-MoE with hybrid attention
│   └── utils/
│       ├── context.py          # ThreadLocal context for batch info
│       └── loader.py           # SafeTensors weight loading
├── example.py                   # Basic usage example
├── bench.py                     # Performance benchmark script
└── requirements.txt
```

---

## 2. OVERALL ARCHITECTURE & DATA FLOW

### High-Level Inference Pipeline

```
User Code
    ↓
LLM (wrapper)
    ↓
LLMEngine
    ├─ add_request() → Creates Sequence, allocates linear attention slot
    ├─ step() → Scheduler → ModelRunner → Sample → Postprocess
    └─ generate() → Loop until all sequences finish
         ↓
    Scheduler (Prefill/Decode phase selection)
         ↓
    ModelRunner (Executes model with CUDA graphs)
         ↓
    Sampler (Gumbel-max temperature sampling)
         ↓
    Return token_ids
```

### Memory Management Architecture

```
GPU Memory Allocation (in ModelRunner)
├─ KV Cache (pre-allocated tensors)
│  ├─ [2, num_layers, num_kv_cache_blocks, block_size, num_kv_heads, head_dim]
│  └─ Blocks are allocated by BlockManager (prefix caching via hash-based lookup)
│
├─ Linear Attention State (for Qwen3.5-MoE)
│  ├─ Recurrent state: [num_layers, max_slots, num_v_heads, head_k_dim, head_v_dim]
│  ├─ Conv state: [num_layers, max_slots, conv_dim, kernel_size-1]
│  └─ Managed per-sequence with slot indices
│
├─ CUDA Graphs (pre-captured execution)
│  └─ One graph per batch size: [1, 2, 4, 8, 16, 32, ...] tokens
│
└─ Computation Pool (graph memory reuse)
```

---

## 3. CONFIGURATION SYSTEM (config.py)

### Config Dataclass
```python
@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384    # Max prefill batch size
    max_num_seqs: int = 512                # Max concurrent sequences
    max_model_len: int = 4096              # Context window
    gpu_memory_utilization: float = 0.9    # % of GPU memory to use
    tensor_parallel_size: int = 1          # TP degree
    enforce_eager: bool = False            # Disable CUDA graphs if True
    hf_config: object = None               # HuggingFace config (loaded on init)
    eos: int = -1                          # EOS token ID
    kvcache_block_size: int = 256          # Block size for prefix caching
    num_kvcache_blocks: int = -1           # Computed based on GPU memory
```

### Model Config Loading
- **Standard Models**: Uses `transformers.AutoConfig`
- **Qwen3.5-MoE**: Custom `Qwen3_5MoeConfig` parser (JSON-based)
  - Handles special fields: `linear_num_key_heads`, `num_experts`, `moe_intermediate_size`, etc.
  - Supports both standard model and text_config structure

### Key Post-Init Logic
- Validates `model` is a directory
- Asserts `kvcache_block_size % 256 == 0`
- Caps `max_model_len` to model's `max_position_embeddings`

---

## 4. ENGINE & INFERENCE PIPELINE

### 4.1 LLMEngine (llm_engine.py)

**Main Responsibilities**:
1. Coordinate inference across TP ranks (multiprocessing)
2. Manage tokenization
3. Add requests and drive scheduling
4. Return decoded outputs

**Key Methods**:

#### `__init__(model_path, **kwargs)`
- Creates `Config` from kwargs
- Spawns `ModelRunner` processes for TP ranks > 0 (rank 0 in main process)
- Creates `Scheduler` and tokenizer

#### `add_request(prompt, sampling_params)`
- Tokenizes prompt if string
- Creates `Sequence` object
- **Allocates linear attention slot** (critical for Qwen3.5)
- Adds to scheduler's waiting queue

#### `step()` → Returns `(outputs, num_tokens)`
- Calls `scheduler.schedule()` → Returns scheduled sequences + is_prefill flag
- Calls `model_runner.run(seqs, is_prefill)` → Returns sampled token_ids
- Calls `scheduler.postprocess()` → Updates sequences, marks finished
- **Frees linear attention slots** for finished sequences
- Returns `[(seq_id, completion_token_ids), ...]` for finished sequences
- Returns `num_tokens`: positive if prefill (total tokens), negative if decode (-batch_size)

#### `generate(prompts, sampling_params, use_tqdm=True)`
- Main user-facing loop
- Adds all requests, runs `step()` until finished
- Tracks prefill/decode throughput in progress bar
- Returns `[{"text": str, "token_ids": list[int]}, ...]`

---

### 4.2 Scheduler (scheduler.py)

**Two-Phase Scheduling**:

1. **Prefill Phase** (high parallelism, low latency)
   - Schedules as many waiting sequences as possible
   - Constraints:
     - Max `max_num_seqs` sequences
     - Max `max_num_batched_tokens` total tokens
     - Each sequence must fit in KV cache (via `block_manager.can_allocate`)

2. **Decode Phase** (high throughput)
   - Pops from running sequences
   - For each sequence, checks if KV cache can append one more token
   - If no space, preempts (saves state, moves back to waiting)
   - Returns scheduled batch

**State Transitions**:
```
WAITING → (prefill) → RUNNING → (decode loop) → FINISHED
           ↑___________(preempt)________________↓
```

**Key Attributes**:
- `waiting`: deque of WAITING sequences
- `running`: deque of RUNNING sequences
- `block_manager`: Manages KV cache block allocation

**Key Methods**:

#### `schedule() -> (list[Sequence], bool)`
- Returns scheduled sequences and `is_prefill` flag
- Ensures at least one sequence in output (assertion)

#### `postprocess(seqs, token_ids)`
- Appends sampled token to each sequence
- Marks as FINISHED if: reached max_tokens OR sampled EOS
- Deallocates KV cache blocks
- Removes from running queue

---

### 4.3 BlockManager (block_manager.py) - Prefix Caching

**Purpose**: Implement semantic token-level prefix caching via hash-based block sharing

**Block Structure**:
```python
class Block:
    block_id: int           # Unique block identifier
    ref_count: int          # Reference count for sharing
    hash: int               # xxhash64 of token sequence (with prefix hash)
    token_ids: list[int]    # Cached tokens for verification
```

**Hash Computation**:
```
hash(tokens) = xxhash64(
    [prefix_hash.to_bytes(8)] + 
    np.array(tokens).tobytes()
)
```
- Enables chaining: each block hash includes previous block hash
- Ensures exact token sequence match (guards against collisions)

**Key Operations**:

#### `allocate(seq: Sequence)`
- Iterates over blocks in sequence
- For each block:
  1. Compute hash (with prefix)
  2. Check if hash exists in `hash_to_block_id`
  3. If found AND token_ids match:
     - **Cache hit**: Reuse block, increment ref_count
     - Add `block_size` tokens to sequence's `num_cached_tokens`
  4. If not found:
     - **Cache miss**: Allocate new block from free list
     - Update block hash/token_ids
     - Add to hash_to_block_id mapping
  5. Append block_id to sequence's block_table

#### `deallocate(seq)`
- Decrement ref_count for each block
- If ref_count == 0, return block to free list
- Clear block's hash and token_ids
- Reset sequence's num_cached_tokens and block_table

#### `can_allocate(seq) / can_append(seq)`
- Check if enough free blocks available

**Prefix Caching Semantics**:
- Multiple sequences with identical prefixes share blocks
- Example: Two prompts "Explain X" and "Explain Y" share the "Explain" block
- Reduces memory by ~20-40% on typical workloads

---

### 4.4 Sequence (sequence.py)

**State Management**:
```python
class Sequence:
    seq_id: int                    # Unique ID (auto-increment)
    status: SequenceStatus         # WAITING, RUNNING, FINISHED
    token_ids: list[int]           # All tokens (prompt + completion)
    num_tokens: int                # len(token_ids)
    num_prompt_tokens: int         # Initial prompt length
    num_cached_tokens: int         # Tokens in KV cache blocks
    block_table: list[int]         # Block IDs (indices into kv_cache tensor)
    last_token: int                # Last token ID (for decode)
    temperature: float             # Sampling temperature
    max_tokens: int                # Max completion length
    ignore_eos: bool               # If True, ignore EOS token
```

**Block Calculation Properties**:
```
num_cached_blocks = num_cached_tokens // block_size
num_blocks = (num_tokens + block_size - 1) // block_size
last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size
```

**Serialization** (`__getstate__`/`__setstate__`):
- Reduces pickling overhead for inter-process communication
- During prefill: only save token_ids
- During decode: only save last_token

---

### 4.5 ModelRunner (model_runner.py) - Core Execution & CUDA Graphs

**The Most Complex Component** - Handles:
1. Model initialization and weight loading
2. Memory allocation (KV cache + linear attention states + CUDA graphs)
3. CUDA graph capture and replay
4. Batch preparation (prefill vs decode)
5. Multi-GPU communication (TP rank 0 broadcasts via shared memory)

#### **Initialization** (`__init__`)

**Step 1: Distributed Setup**
```python
dist.init_process_group("nccl", "tcp://localhost:2333", 
                        world_size=tensor_parallel_size, rank=rank)
torch.cuda.set_device(rank)
```

**Step 2: Model Loading**
- Sets default dtype and device to GPU
- Instantiates model via `get_model_class(hf_config)`
- Loads weights via `load_model()` with SafeTensors

**Step 3: Memory Budget Estimation**
- **Linear attention budget**: Pre-computed for Qwen3.5-MoE
  - Per-sequence buffer size: `num_layers * (num_v_heads * head_k_dim * head_v_dim + conv_dim * (kernel_size-1))`
  - Max slots: 32 (conservative for 1GB)
- **CUDA graph reserve**: 2-3MB per layer + extra for MoE

**Step 4: KV Cache Allocation**
```python
free, total = torch.cuda.mem_get_info()
available_budget = total * gpu_memory_utilization - reserved_memory
num_kvcache_blocks = available_budget // block_bytes
kv_cache = torch.empty(2, num_layers, num_kvcache_blocks, block_size, 
                       num_kv_heads, head_dim)
# Assign to attention modules
for module in model.modules():
    if hasattr(module, "k_cache"):
        module.k_cache = kv_cache[0, layer_id]
        module.v_cache = kv_cache[1, layer_id]
```

**Step 5: Linear Attention State Allocation** (Qwen3.5 only)
```python
recurrent_state_buf = torch.zeros(num_layers, max_slots, 
                                  num_v_heads, head_k_dim, head_v_dim)
conv_state_buf = torch.zeros(num_layers, max_slots, 
                             conv_dim, kernel_size - 1)
# Attach to linear attention modules
for module in linear_layers:
    module.recurrent_state_buf = recurrent_state_buf[i]
    module.conv_state_buf = conv_state_buf[i]
```

**Step 6: CUDA Graph Capture**
- See "CUDA Graphs" section below

**Step 7: Multi-GPU Sync** (if TP > 1)
- Rank 0 creates shared memory for IPC
- Rank 0+ execute `barrier()` then rank 0+ enter `loop()` for RPC

#### **Memory Reservation Model**

```python
def _compute_linear_attn_budget():
    # Conservative: 32 concurrent sequences × state size
    bytes_per_slot = num_layers * (recurrent_size + conv_size)
    budget = bytes_per_slot * 32
    
    # CUDA Graph reserve: peak activation memory
    if moe_model:
        # Per-token peak during dense expert dispatch
        per_token_peak = (2 * moe_inter * hidden + hidden * moe_inter)
        graph_reserve = max_decode_bs * per_token_peak + 2MB/layer
    else:
        graph_reserve = 2MB * num_layers
```

#### **KV Cache Slot Management**

```python
def allocate_linear_attn_slot(seq_id):
    """Called when sequence is added."""
    slot_idx = free_slots.popleft()
    slot_map[seq_id] = slot_idx
    # Zero out buffers
    recurrent_buf[:, slot_idx].zero_()
    conv_buf[:, slot_idx].zero_()
    return slot_idx

def free_linear_attn_slot(seq_id):
    """Called when sequence finishes."""
    slot_idx = slot_map.pop(seq_id)
    free_slots.append(slot_idx)
```

#### **Batch Preparation**

##### Prefill Path (`prepare_prefill`)

```
Input: seqs = [seq1, seq2, ...]
       each seq might have num_cached_tokens > 0 (prefix cache hit)

Goal: Prepare tensors for flash_attn_varlen_func

Output:
- input_ids: concatenated new tokens [total_new_tokens]
- positions: absolute positions in sequence
- cu_seqlens_q: cumulative lengths for queries (new tokens only)
- cu_seqlens_k: cumulative lengths for keys (including prefix cache)
- max_seqlen_q, max_seqlen_k: max lengths in batch
- slot_mapping: maps each new token to KV cache slot
- block_tables: if prefix cache, maps sequences to block IDs
- linear_attn_slots: maps batch position to buffer slot index
```

**Key Detail: Prefix Cache Integration**
```python
# If sequence has cached prefix (num_cached_tokens > 0):
#   cu_seqlens_k = cumsum including all tokens (prefix + new)
#   block_tables passed to flash_attn for cached KV reads
#   cu_seqlens_q = only new tokens (attention computed only on new)
```

##### Decode Path (`prepare_decode`)

```
Input: seqs = [seq1, seq2, ..., seqN] (N ≤ max_num_seqs)
       Each sequence is processing one token

Output:
- input_ids: last token of each sequence [N]
- positions: absolute position for each seq
- slot_mapping: where to store KV for this token
- context_lens: current sequence length
- block_tables: maps seq to its blocks
- linear_attn_slots: maps batch pos → buffer slot
```

#### **Model Execution** (`run_model`)

**CUDA Graph Decision Tree**:
```
if is_prefill or enforce_eager or batch_size > 512:
    # Use eager execution (no graph)
    return model(input_ids, positions)
else:
    # Use CUDA graph (decode phase, small batch)
    bs = input_ids.shape[0]
    # Find smallest graph that fits this batch
    graph_idx = next(x for x in graph_bs if x >= bs)
    graph = graphs[graph_idx]
    
    # Update graph input tensors (pinned memory)
    graph_vars["input_ids"][:bs] = input_ids
    graph_vars["positions"][:bs] = positions
    graph_vars["slot_mapping"][:bs] = context.slot_mapping
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][:bs] = context.block_tables
    if "linear_attn_slot_indices" in graph_vars:
        graph_vars["linear_attn_slot_indices"][:bs] = context.linear_attn_slot_indices
    
    # Replay
    graph.replay()
    return graph_vars["outputs"][:bs]
```

#### **CUDA Graph Capture** (`capture_cudagraph`)

```python
def capture_cudagraph():
    # Batch sizes to capture: [1, 2, 4, 8, 16, 32, 48, 64, ..., max_bs]
    graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    graphs = {}
    graph_pool = None
    
    # Allocate graph input/output tensors
    input_ids = torch.zeros(max_bs, dtype=int64)
    outputs = torch.zeros(max_bs, hidden_size)
    [other tensors...]
    
    for bs in reversed(graph_bs):  # Reverse order for memory locality
        graph = torch.cuda.CUDAGraph()
        
        # Setup context for this batch size
        set_context(is_prefill=False, 
                   slot_mapping=slot_mapping[:bs],
                   block_tables=block_tables[:bs],
                   ...)
        
        # Warmup (outside graph)
        outputs[:bs] = model(input_ids[:bs], positions[:bs])
        
        # Capture (inside graph)
        with torch.cuda.graph(graph, graph_pool):
            outputs[:bs] = model(input_ids[:bs], positions[:bs])
        
        # Setup pool on first graph for memory reuse
        if graph_pool is None:
            graph_pool = graph.pool()
        
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
    
    # Store graph input tensors for later update+replay
    graph_vars = {
        "input_ids": input_ids,
        "positions": positions,
        "outputs": outputs,
        ...
    }
```

**CUDA Graph Benefits**:
1. **Overhead reduction**: Single kernel launch vs multiple launches
2. **Memory efficiency**: Fixed memory footprint across replays
3. **Deterministic scheduling**: Same GPU schedule each time
4. **Dynamic batching**: Different batch sizes have separate graphs

---

### 4.6 Sampler (layers/sampler.py)

**Temperature Sampling via Gumbel-max trick**:

```python
def forward(logits, temperatures):
    # Scale by temperature
    logits = logits / temperatures.unsqueeze(1)
    
    # Gumbel-max: argmax(logits + gumbel_noise)
    # Equivalent to: argmax(exp(logits/temp) / exp(gumbel))
    gumbel_noise = -log(-log(uniform()))
    samples = argmax(logits / clamp(gumbel_noise, min=1e-10))
    
    return samples
```

**Why Gumbel-max**:
- Produces proper categorical samples from temperature-scaled logits
- More numerically stable than softmax + sampling
- Compiled with `@torch.compile` for speed

---

## 5. MODEL IMPLEMENTATIONS

### 5.1 Qwen3 (Standard Transformer)

**Architecture**:
```
Embedding → 
  LayerNorm → Attention → Residual →
  LayerNorm → MLP → Residual → 
  (repeat for num_hidden_layers)
→ Final LayerNorm → LMHead
```

**Attention Module** (`Qwen3Attention`):
- QKV parallel projection (tensor sharded)
- Partial rotary embeddings (full head_dim)
- Flash-attention variant
- RMS norm on Q/K (with learnable weight)

**MLP** (`Qwen3MLP`):
```
gate_up = Linear(hidden, 2*intermediate)
split into gate, up
activation(gate) * up
down_proj = Linear(intermediate, hidden)
```

**Tensor Parallelism Support**:
- `QKVParallelLinear`: Shards Q/K/V outputs across ranks
- `MergedColumnParallelLinear`: Fuses gate+up projections
- `RowParallelLinear`: Shards input to output projection

---

### 5.2 Qwen3.5-MoE (Hybrid Attention + MoE)

**Hybrid Layer Structure** (every 4 layers = 1 cycle):
- **Layers 0-2**: Gated DeltaNet (linear attention)
- **Layer 3**: Full attention with output gating
- **All layers**: MoE MLP with shared expert

#### **1. Gated DeltaNet** (Linear Attention)

**Purpose**: Efficient sequence modeling via recurrent state instead of full attention

**Architecture**:
```
Hidden_t →
  proj_qkv (conv1d kernel_size=4) → SiLU →
  split → Q, K, V → reshape to heads →
  Gated Delta Rule (recurrent) →
  reshape →
  proj_output →
Output_t
```

**Gated Delta Rule** (single-token decode):
```
For each token t:
  beta_t = sigmoid(b_t)              # Update gate
  g_t = exp(decay coefficient)       # Decay rate
  
  state *= g_t                       # Decay existing state
  kv_mem = (state * k_t).sum()       # Recall from state
  delta = (v_t - kv_mem) * beta_t    # Update
  state += k_t ⊗ delta               # Write new state
  output = (state * q_t).sum()       # Query state
```

**Key Design for CUDA Graphs**:
- State stored in **pre-allocated GPU buffers** (one slot per sequence)
- Slot index passed via context at runtime
- No Python control flow during decode (all batched tensor ops)
- Conv state uses sliding window (kernel_size-1 history)

**Prefill vs Decode**:

**Prefill** (`_forward_prefill`):
- Process full sequence with chunk-based rule
- Save final state to buffer for decode
- Save conv window for next token

**Decode** (`_forward_decode_batched`):
- Read state from buffer[slot_idx]
- Single recurrent step on all tokens simultaneously
- Update buffer in-place
- All operations vectorized

#### **2. Full Attention with Output Gating** (Every 4th layer)

**Key Difference from Qwen3**:
```
Q projection outputs 2x dimensions:
  qkv = q_proj(hidden)  # [N, num_heads * head_dim * 2]
  reshape to [N, num_heads, head_dim*2]
  query, gate = chunk(2)
  gate = reshape to [N, num_heads*head_dim]

After attention:
  output *= sigmoid(gate)
```

**Partial Rotary Embedding** (25% of head_dim):
```
head_dim = 128
rotary_dim = 32

In RoPE:
  q_rot, q_pass = split(query, rotary_dim)
  q_rot = apply_rope(q_rot)
  q = concat(q_rot, q_pass)
```

#### **3. MoE MLP** (All Layers)

**Components**:
- **TopK Router**: Selects top-2 experts per token
- **Sparse Experts**: 3D parameter tensors `[num_experts, 2*inter, hidden]`
- **Shared Expert**: Always active, gated by sigmoid

**Dispatch Strategy**:

**Prefill** (Sparse):
```python
# Only compute expert-token pairs with non-zero routing weights
expert_mask = one_hot(top_k_indices, num_experts)
for expert_idx in active_experts:
    token_idx = where(expert_mask[expert_idx])
    output[token_idx] += expert(hidden[token_idx]) * weight[token_idx]
```

**Decode** (Dense, CUDA Graph Safe):
```python
# Iterate over top_k slots (fixed), gather per-token expert params
for k in range(top_k):
    idx = top_k_indices[:, k]           # [N] expert indices
    gate_up_w = gate_up_proj[idx]       # [N, 2*inter, hidden]
    down_w = down_proj[idx]             # [N, hidden, inter]
    
    output += bmm(gate_up_w, hidden) @ down_w * weights[:, k]
```

**Why Dense for Decode**:
- Consistent batch shape regardless of routing distribution
- All tensor shapes fixed: enables CUDA graph capture
- Prefill doesn't need graph (high overhead of control flow acceptable)

#### **Custom RMSNorm** (Qwen3.5 Style)

```python
class Qwen3_5RMSNorm:
    weight = Parameter(torch.zeros(dim))  # Zero-initialized!
    
    def forward(x):
        norm_x = normalize(x)
        return norm_x * (1 + weight)  # (1 + w) scaling
```

**Different from Standard**:
- Standard: `weight * norm(x)` (weight ≈ 1.0)
- Qwen3.5: `(1 + weight) * norm(x)` (weight ≈ 0)
- Both equivalent, different initialization

---

## 6. CACHE MECHANISMS

### 6.1 KV Cache (Attention)

**Storage**:
```python
kv_cache = torch.empty(2, num_layers, num_blocks, block_size, 
                       num_kv_heads, head_dim)
# kv_cache[0] = K cache
# kv_cache[1] = V cache
```

**Indexing**:
```
block_id = seq.block_table[i]  # Which physical block
offset = seq.last_block_num_tokens  # Position within block

kv_flat = kv_cache[0, layer_id, block_id]  # [block_size, num_kv_heads, head_dim]
```

**Storage via Triton** (`store_kvcache_kernel`):
```
For each output token (from model forward):
  slot = slot_mapping[token_idx]
  k_cache[slot] = key[token_idx]
  v_cache[slot] = value[token_idx]
```

**Retrieval via Flash-Attn**:
- Flash-attn handles indirect indexing internally
- Passes `block_table` to decoder variant
- Handles page-faulting efficiently

### 6.2 Linear Attention State (Qwen3.5-MoE)

**Recurrent State**:
- Shape: `[num_layers, num_seqs, num_v_heads, head_k_dim, head_v_dim]`
- Stores accumulated key-value correlation
- Updated in-place each decode step

**Conv State**:
- Shape: `[num_layers, num_seqs, conv_dim, kernel_size-1]`
- Stores last (kernel_size-1) conv inputs
- Implements causal convolution sliding window

**Lifetime**:
```
allocate_linear_attn_slot(seq_id):
  slot_idx = pop_free_slots()
  slots_map[seq_id] = slot_idx
  recurrent_buf[:, slot_idx].zero_()
  conv_buf[:, slot_idx].zero_()

[During prefill & decode]
  Read/write via slot indices

free_linear_attn_slot(seq_id):
  slot_idx = slots_map.pop(seq_id)
  recurrent_buf[:, slot_idx].zero_()  # Optional
  conv_buf[:, slot_idx].zero_()
  push_free_slots(slot_idx)
```

### 6.3 Prefix Caching (Block-Level)

**Mechanism**:
- Hash-based block sharing (described in BlockManager section)
- Multiple sequences with identical prefix → shared KV cache
- Reduces memory bandwidth and capacity

**Example**:
```
Sequence 1: "Explain deep learning" [token_ids]
Sequence 2: "Explain quantum" [token_ids]

Common prefix "Explain" (say 10 tokens)
  → Hashed as one block
  → Stored once in KV cache
  → Both sequences reference same block
  → Saves 10 × num_kv_heads × head_dim × (block_size + value_cache)
```

---

## 7. CUDA GRAPH SUPPORT

### Capture Strategy

**Multi-size Graphs**:
```
graph_bs = [1, 2, 4, 8, 16, 32, 48, 64, ...]
```
- Small graphs (1-8): Better latency hiding
- Large graphs (16+): Amortize kernel launch overhead

**Capture Process**:
```
for each bs in graph_bs:
  Create graph
  Prepare input tensors [bs, ...]
  Warmup: model(input_ids[:bs], positions[:bs])
  Capture: with torch.cuda.graph():
             model(input_ids[:bs], positions[:bs])
  Store graph and input tensors
```

### Replay Strategy

**Runtime**:
```
batch_size = actual_batch
graph_idx = find_smallest_graph(batch_size)
graph = graphs[graph_idx]

# Update inputs (pinned memory, fast H2D)
graph_vars["input_ids"][:batch_size] = input_ids
graph_vars["positions"][:batch_size] = positions
graph_vars["slot_mapping"][:batch_size] = context.slot_mapping
[update other fields...]

# Replay (entire forward pass replayed from graph)
graph.replay()
```

### CUDA Graph + Linear Attention

**Challenge**: Slot indices change per batch (which sequence is which)

**Solution**: Slot indices as **graph variable**
```python
# Pre-allocate in graph
graph_vars["linear_attn_slot_indices"] = torch.zeros(max_bs, dtype=int64)

# Update at replay time
if "linear_attn_slot_indices" in context:
    graph_vars["linear_attn_slot_indices"][:batch_size] = context.linear_attn_slot_indices

# During replay, model reads: context.linear_attn_slot_indices[:batch_size]
#   → Maps batch position to buffer slot
```

### Performance Impact

**Measured** (per paper):
- Prefill: ~1434 tok/s (eager mode, high overhead acceptable)
- Decode: CUDA graph reduces per-token latency by ~10-20%

**Limitations**:
- `if enforce_eager`: Skip graphs entirely (testing/debugging)
- `if batch_size > 512`: Use eager (graph capture memory overhead)
- `if is_prefill`: Always eager (high control flow, cache misses)

---

## 8. CONTEXT MANAGEMENT (utils/context.py)

**Problem**: Pass batch information through model forward without global state

**Solution**: ThreadLocal context dictionary

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Tensor | None = None      # Prefill query lengths
    cu_seqlens_k: Tensor | None = None      # Prefill key lengths
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None      # KV cache slot indices
    context_lens: Tensor | None = None      # Decode context lengths
    block_tables: Tensor | None = None      # Prefill block mappings
    seq_ids: list[int] = field(default_factory=list)
    linear_attn_slot_indices: Tensor | None = None  # Linear attention slots

_CONTEXT = Context()

def set_context(...): global _CONTEXT = Context(...)
def get_context(): return _CONTEXT
def reset_context(): global _CONTEXT = Context()
```

**Usage**:
```python
# Before model forward
set_context(is_prefill=True, cu_seqlens_q=..., ...)

# Inside model.forward()
context = get_context()
if context.is_prefill:
    # Use flash_attn_varlen_func with context.cu_seqlens_q
else:
    # Use flash_attn_with_kvcache with context.context_lens

# After model forward
reset_context()
```

---

## 9. WEIGHT LOADING (utils/loader.py)

**SafeTensors Format**:
- Supports packed modules (fused Q/K/V projections)
- Custom weight loaders for sharded parameters

```python
def load_model(model, path):
    for safetensors_file in glob(path / "*.safetensors"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip prefixes (e.g., "mtp.", "visual.")
                if any(weight_name.startswith(p) for p in skip_prefixes):
                    continue
                
                # Strip prefix (e.g., "model." for VLMs)
                param_name = weight_name
                if param_name.startswith(weight_prefix):
                    param_name = param_name[len(weight_prefix):]
                
                # Check packed module mapping
                # Example: "q_proj" → ("qkv_proj", "q")
                for k, v in packed_modules_mapping.items():
                    if k in param_name:
                        param_name = param_name.replace(k, v[0])
                        param = model.get_parameter(param_name)
                        weight_loader = param.weight_loader
                        weight_loader(param, f.get_tensor(weight_name), v[1])
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = param.weight_loader
                    weight_loader(param, f.get_tensor(weight_name))
```

**Key Feature**: Tensor-parallel weight sharding during load (no intermediate copies)

---

## 10. TENSOR PARALLELISM

### Linear Layer Types

| Layer | Input | Output | Purpose |
|-------|-------|--------|---------|
| `ReplicatedLinear` | `[N, in]` | `[N, out]` | Same weights on all ranks |
| `ColumnParallelLinear` | `[N, in]` | `[N, out/tp_size]` | Shard output dimension |
| `RowParallelLinear` | `[N, in/tp_size]` | `[N, out]` | Shard input dimension, AllReduce |
| `QKVParallelLinear` | `[N, in]` | `[N, (q+2*kv)/tp_size]` | Fused Q/K/V projection |

### TP Communication Pattern

```
Prefill:
  Embedding (broadcast via AllReduce)
    ↓
  ColumnParallel(QKV) × num_layers
    ↓
  Attention (compute happens locally)
    ↓
  RowParallel(out_proj) (AllReduce)
    ↓
  ColumnParallel(gate_up) × num_layers
    ↓
  RowParallel(down) (AllReduce)
    ↓
  ParallelLMHead (Gather on rank 0)

Decode: Same pattern, smaller batch
```

### Collective Operations

- **AllReduce**: After RowParallel outputs (sum across ranks)
- **Gather**: At LM head (collect logits from all ranks to rank 0)

### KV Head Handling

```python
if num_kv_heads < tp_size:
    # KV heads are replicated (not sharded)
    # Use ReplicatedLinear for K/V projections
    num_kv_heads_local = num_kv_heads
else:
    # Normal sharding
    num_kv_heads_local = num_kv_heads // tp_size
```

---

## 11. TEST & EXAMPLE SCRIPTS

### `example.py`
```python
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [prompt1, prompt2, ...]
outputs = llm.generate(prompts, sampling_params)
# outputs[i]["text"], outputs[i]["token_ids"]
```

### `bench.py`
```python
# 256 sequences, random lengths 100-1024, random output 100-1024
# Measures throughput (tokens/second)
llm.generate(prompts, sampling_params, use_tqdm=False)
```

### `example/test_qwen3_5_load.py` and `test_server.py`
- Likely test-specific configurations for Qwen3.5-MoE
- Server API endpoints

---

## 12. KEY DESIGN PATTERNS & OPTIMIZATIONS

### 1. **Batch Splitting & Preemption**
- Sequences preempted if KV cache full
- Move back to waiting queue
- Resume from cached KV state
- Enables oversubscription with deterministic behavior

### 2. **Flash-Attention Integration**
- `flash_attn_varlen_func`: Variable-length prefill (efficient for ragged batches)
- `flash_attn_with_kvcache`: KV cache decode (handles indirect indexing)
- `block_table`: Maps sequences to physical KV cache blocks

### 3. **CUDA Graph Determinism**
- Fixed batch sizes per graph
- Fixed memory footprint
- Replayed on same GPU memory addresses
- Enables cycle-level reproducibility

### 4. **Linear Attention for Hybrid Models**
- Recurrent state per-sequence (vs dense attention QK matrix)
- Enables long context without quadratic memory
- Compatible with CUDA graphs via slot-based state buffers

### 5. **Multi-Process TP**
- Rank 0 in main Python process (I/O, scheduling)
- Ranks 1+ in child processes (computation)
- IPC via shared memory + pickle serialization
- Synchronization via events + barriers

### 6. **Memory-Safe Weight Loading**
- Custom weight loaders for sharded params
- No intermediate full-precision copies
- Handles packed modules (fused projections)
- SafeTensors format for safety

### 7. **Eager Eager First, Graph Second**
- Default: Eager execution (straightforward, debuggable)
- Opt-in: CUDA graphs for 10-20% latency reduction
- Fallback: Always works without graphs

---

## 13. DATA FLOW EXAMPLE: Simple Request

```
User: "Hello"
  ↓
LLMEngine.add_request()
  → Tokenize: "Hello" → [token_ids]
  → Create Sequence
  → allocate_linear_attn_slot(seq_id=0)  [Qwen3.5 only]
  → Scheduler.add(seq)  [state=WAITING]
  
LLMEngine.step() [Iteration 1: PREFILL]
  → Scheduler.schedule()
    → Moving seq 0 from WAITING → RUNNING
    → BlockManager.allocate() [Prefill, assign KV cache blocks]
  → ModelRunner.prepare_prefill()
    → input_ids = [token_ids]
    → positions = [0, 1, 2, ...]
    → cu_seqlens_q = [0, len(token_ids)]
    → slot_mapping = [cache_slots]
  → set_context(is_prefill=True, ...)
  → ModelRunner.run_model()
    → model.forward(input_ids, positions)
      → Embedding
      → Layer 0: Attention (reads KV cache, computes attention)
               + LinearAttention (writes recurrent state to buffer)
               + MoE MLP
      → ...
      → LMHead → logits [batch, vocab]
    → [CUDA graph captured for decode]
  → Sampler(logits, temperatures) → next_token_id
  → Scheduler.postprocess()
    → seq.append_token(next_token_id)
    → seq.status = RUNNING (not finished)
  
LLMEngine.step() [Iteration 2-N: DECODE]
  → Scheduler.schedule()
    → seq still in RUNNING
    → BlockManager.may_append()  [Check if room for one more token]
  → ModelRunner.prepare_decode()
    → input_ids = [last_token]
    → context_lens = [current_len]
    → block_tables = [seq.block_table]
    → linear_attn_slot_indices = [slot_map[seq_id]]
  → set_context(is_prefill=False, ...)
  → ModelRunner.run_model()
    → Use CUDA graph (if batch_size in [1,2,4,...])
    → graph_vars["input_ids"][0] = input_ids[0]
    → graph.replay()
      → Model processes via captured graph
      → KV cache updated via slot_mapping
      → Linear attention state updated via slot indices
  → Sampler() → next_token_id
  → Scheduler.postprocess()
    → seq.append_token(next_token_id)
    → If next_token_id == EOS: seq.status = FINISHED
  
LLMEngine.step() [When finished]
  → Scheduler.schedule()
    → seq 0 is FINISHED, removed from running
    → scheduler.is_finished() == True
  
LLMEngine.generate() returns
  → [{"text": "Hello, I'm fine.", "token_ids": [...]}]
```

---

## 14. SUMMARY TABLE: File-by-File Roles

| File | Lines | Key Responsibility |
|------|-------|-------------------|
| `config.py` | 130 | Config loading, HF model detection |
| `llm.py` | 5 | Wrapper (LLM = LLMEngine) |
| `sampling_params.py` | 12 | Sampling configuration |
| `llm_engine.py` | 100 | Orchestration, tokenization, main loop |
| `model_runner.py` | 421 | Model loading, CUDA graphs, batch prep |
| `scheduler.py` | 72 | Prefill/decode phase, preemption |
| `sequence.py` | 85 | Per-sequence state, serialization |
| `block_manager.py` | 113 | KV cache block allocation, prefix caching |
| `attention.py` | 76 | Flash attention wrapper, KV cache store |
| `layers/linear.py` | 160 | TP-aware linear layers |
| `layers/rotary_embedding.py` | 70 | Rotary position embeddings |
| `layers/layernorm.py` | 51 | RMSNorm with fused residual |
| `layers/activation.py` | 15 | SiLU + Mul gating |
| `layers/embed_head.py` | 67 | Parallel embeddings/LM head |
| `layers/sampler.py` | 16 | Gumbel-max sampling |
| `models/qwen3.py` | 216 | Standard Qwen3 architecture |
| `models/qwen3_5.py` | 953 | Hybrid attention + MoE |
| `utils/context.py` | 34 | Batch context management |
| `utils/loader.py` | 58 | SafeTensors weight loading |

**Total**: ~1,200 lines of core logic

