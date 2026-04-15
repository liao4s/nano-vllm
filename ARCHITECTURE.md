# Nano-vLLM: Comprehensive Architecture Overview

## Project Summary

**Nano-vLLM** is a lightweight, high-performance vLLM implementation built from scratch in ~1,200 lines of Python code. It achieves comparable inference speeds to vLLM (1,434.13 tok/s vs vLLM's 1,361.84 tok/s on RTX 4070) while maintaining a clean, readable codebase. The architecture implements state-of-the-art optimizations including prefix caching, tensor parallelism, CUDA graph capture, and flash attention.

**Project Dependencies:**
- torch>=2.4.0
- triton>=3.0.0
- transformers>=4.51.0
- flash-attn
- xxhash

---

## High-Level Architecture

```
LLMEngine (Main Interface)
├── Config (Configuration parameters)
├── Tokenizer (HF AutoTokenizer)
├── Scheduler (Request scheduling & preemption)
│   └── BlockManager (KV cache memory management)
├── ModelRunner (Model execution on GPU, with tensor parallelism)
│   ├── Qwen3ForCausalLM (Model)
│   ├── Sampler (Token sampling)
│   └── KV Cache Manager
└── Sequence (Request state tracking)
```

---

## 1. Core Components

### 1.1 LLM / LLMEngine (`nanovllm/llm.py`, `nanovllm/engine/llm_engine.py`)

**Class: `LLM` (extends `LLMEngine`)**
- Thin wrapper providing public API
- Inherits all functionality from LLMEngine

**Class: `LLMEngine`**
The main orchestrator of the inference system. Manages the entire request lifecycle.

**Key Attributes:**
- `config: Config` - System configuration
- `model_runner: ModelRunner` - GPU compute runner
- `tokenizer: AutoTokenizer` - HF tokenizer
- `scheduler: Scheduler` - Request scheduling logic
- `ps: list[Process]` - Tensor parallel worker processes
- `events: list[Event]` - Sync events for multi-GPU coordination

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__init__(model, **kwargs)` | Initialize engine, spawn worker processes (tensor parallelism), create scheduler/tokenizer |
| `add_request(prompt, sampling_params)` | Queue a new generation request |
| `step()` | Execute one generation step: schedule seqs → run model → postprocess tokens |
| `is_finished()` | Check if all requests are complete |
| `generate(prompts, sampling_params, use_tqdm)` | **Main entry point**: Batch process multiple prompts |
| `exit()` | Cleanup: terminate workers, deallocate resources |

**Process Management:**
- Uses `multiprocessing.spawn` context for tensor parallelism
- Rank 0 (main process) handles scheduling and I/O
- Ranks 1+ run model forward passes and communicate via shared memory (SHM)
- Each rank sets its CUDA device independently

**Usage:**
```python
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=256))
```

---

### 1.2 Configuration (`nanovllm/config.py`)

**Class: `Config` (dataclass)**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `model` | - | Model directory path (required) |
| `max_num_batched_tokens` | 16384 | Max tokens per forward pass |
| `max_num_seqs` | 512 | Max requests in flight |
| `max_model_len` | 4096 | Max sequence length |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction for KV cache |
| `tensor_parallel_size` | 1 | Number of GPUs (tensor parallelism) |
| `enforce_eager` | False | Disable CUDA graph optimization |
| `kvcache_block_size` | 256 | Tokens per KV cache block |
| `num_kvcache_blocks` | -1 | Auto-calculated blocks |
| `eos` | -1 | EOS token ID (set after tokenizer load) |

**Post-initialization validation:**
- Model directory must exist
- KV cache block size must be % 256 == 0
- Tensor parallel size must be 1-8
- Loads HF config and enforces constraints

---

### 1.3 Sampling Parameters (`nanovllm/sampling_params.py`)

**Class: `SamplingParams` (dataclass)**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `temperature` | 1.0 | Sampling temperature (must be > 1e-10) |
| `max_tokens` | 64 | Max generation tokens per request |
| `ignore_eos` | False | Continue past EOS token if True |

**Note:** Greedy sampling is explicitly disabled (temperature > 1e-10).

---

## 2. Request/Sequence Management

### 2.1 Sequence (`nanovllm/engine/sequence.py`)

**Class: `SequenceStatus` (Enum)**
- `WAITING` - Queued, not yet allocated KV cache
- `RUNNING` - Currently in batch
- `FINISHED` - Generation complete

**Class: `Sequence`**
Tracks state of a single generation request. Designed for efficient serialization (custom `__getstate__`/`__setstate__`).

**Key Attributes:**
- `seq_id: int` - Unique ID (static counter)
- `status: SequenceStatus` - Current state
- `token_ids: list[int]` - All tokens (prompt + completion)
- `last_token: int` - Last generated token (for efficiency)
- `num_tokens: int` - Total tokens
- `num_prompt_tokens: int` - Original prompt length
- `num_cached_tokens: int` - Tokens cached in KV cache (with prefix caching)
- `block_table: list[int]` - KV cache block IDs assigned to sequence
- `temperature: float` - Sampling temperature
- `max_tokens: int` - Max generation length
- `ignore_eos: bool` - Whether to stop at EOS

**Derived Properties:**
- `num_completion_tokens` - Tokens generated so far
- `is_finished` - Status check
- `num_cached_blocks` - `num_cached_tokens // 256`
- `num_blocks` - Total blocks needed
- `last_block_num_tokens` - Tokens in final block
- `block(i)` - Get tokens for block i

**Key Methods:**
- `append_token(token_id)` - Add next generated token
- `__getstate__`/`__setstate__` - Optimized pickling (avoids large token_ids after prefill)

---

### 2.2 Scheduler (`nanovllm/engine/scheduler.py`)

Implements request scheduling with preemption and prefix caching support.

**Class: `Scheduler`**

**Key Attributes:**
- `waiting: deque[Sequence]` - Queued requests
- `running: deque[Sequence]` - Active requests
- `block_manager: BlockManager` - KV cache allocator
- `max_num_seqs: int` - Max concurrent requests
- `max_num_batched_tokens: int` - Max tokens per batch
- `eos: int` - EOS token ID

**Scheduling Algorithm (Two-Phase):**

1. **Prefill Phase** (if no running requests):
   - Dequeue waiting requests while:
     - Don't exceed `max_num_seqs`
     - Don't exceed `max_num_batched_tokens`
     - BlockManager can allocate KV cache
   - Allocate KV cache blocks
   - Mark as RUNNING
   - Return sequences with `is_prefill=True`

2. **Decode Phase**:
   - Process running requests one token at a time
   - Handle preemption: if request can't fit, evict other requests
   - Allocate new blocks as needed (one token = one slot)
   - Return sequences with `is_prefill=False`

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `schedule()` → `(seqs, is_prefill)` | Select which sequences to run this step |
| `postprocess(seqs, token_ids)` | Update sequences with generated tokens, check for completion |
| `preempt(seq)` | Move sequence back to WAITING, deallocate KV cache |
| `is_finished()` | Check if all work done |
| `add(seq)` | Queue new request |

**Preemption Strategy:**
- When decode batch is full, evict last sequence in batch
- If all sequences must run, evict the current sequence and retry later
- Preempted sequences go back to waiting queue for later scheduling

---

### 2.3 Block Manager (`nanovllm/engine/block_manager.py`)

Implements KV cache memory management with prefix caching (multi-request token caching).

**Class: `Block`**
Represents one KV cache block (256 tokens worth).

**Attributes:**
- `block_id: int` - Physical block index
- `ref_count: int` - Reference count (for sharing across sequences)
- `hash: int` - xxHash64 of token_ids (for deduplication)
- `token_ids: list[int]` - Actual token IDs (for verification)

**Class: `BlockManager`**

**Key Attributes:**
- `blocks: list[Block]` - All physical blocks
- `free_block_ids: deque[int]` - Available blocks
- `used_block_ids: set[int]` - Allocated blocks
- `hash_to_block_id: dict[hash → block_id]` - Prefix cache lookup

**Prefix Caching Algorithm:**
During allocation, for each block in sequence:
1. Compute hash of token_ids + previous block's hash
2. Check if this exact prefix exists in `hash_to_block_id`
3. If hit: increment ref_count, mark tokens as cached
4. If miss: allocate new block, update hash table
5. Partial blocks (< 256 tokens) are never cached (hash = -1)

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `allocate(seq)` | Assign blocks to sequence with prefix caching |
| `deallocate(seq)` | Release blocks, decrement ref_counts, update hash table |
| `can_allocate(seq)` | Check if enough free blocks |
| `can_append(seq)` | Check if room for one more token |
| `may_append(seq)` | Allocate new block if crossing 256-token boundary |
| `compute_hash(token_ids, prefix)` | Static method for incremental hash |

**Example:**
```
Request A: [a,b,c,...] (256 tokens) → Block 0
Request B: [a,b,c,...,d] (257 tokens) → Block 0 (cached!), Block 1 (new)
```

---

## 3. Model Execution

### 3.1 ModelRunner (`nanovllm/engine/model_runner.py`)

Executes model forward passes on GPU. Handles tensor parallelism via shared memory IPC.

**Class: `ModelRunner`**

**Initialization:**
- Rank 0: Allocates shared memory, loads model, captures CUDA graphs
- Ranks 1+: Attaches to shared memory, enters event loop

**Key Attributes:**
- `model: Qwen3ForCausalLM` - The LLM
- `sampler: Sampler` - Token sampler
- `kv_cache: torch.Tensor` - Pre-allocated GPU memory for KV cache
- `graphs: dict[bs → CUDAGraph]` - Captured graphs for decode
- `graph_bs: list[int]` - Batch sizes: [1,2,4,8,16,...,max_bs]
- `rank: int`, `world_size: int` - Tensor parallel rank/size

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `run(seqs, is_prefill)` | Main execution: prefill/decode, sample tokens |
| `run_model(input_ids, positions, is_prefill)` | Forward pass (uses CUDA graph if possible) |
| `prepare_prefill(seqs)` | Format prefill batch with prefix caching |
| `prepare_decode(seqs)` | Format decode batch (one token per seq) |
| `prepare_sample(seqs)` | Extract temperatures for sampler |
| `warmup_model()` | Initialize KV cache, check memory |
| `allocate_kv_cache()` | Pre-allocate all KV cache (using 90% GPU mem) |
| `capture_cudagraph()` | Pre-record model forward for decode |

**Tensor Parallelism (IPC):**
- Rank 0 writes method name + args to shared memory
- Sets event for each rank
- Ranks 1+ wait on event, read/execute, synchronize with barrier
- All ranks call model forward (distributed ops sync internally)

**CUDA Graph Optimization:**
- Captures for batch sizes: 1, 2, 4, 8, 16, 32, ..., max_bs
- Decode only: prefill too dynamic due to varying sequence lengths
- Graphs use a pool to minimize memory

**Memory Layout Example (with 3 sequences, 256-token block):**
```
Input phase (prefill):
  seq0: 100 tokens (50 new, 50 cached)
  seq1: 150 tokens (all new)
  seq2: 200 tokens (all new)
  → input_ids = [50 new from seq0, 150 from seq1, 200 from seq2]
  → cu_seqlens_q = [0, 50, 200, 400]
  → cu_seqlens_k = [0, 100, 250, 450]  (k includes cached)

Decode phase:
  seq0, seq1, seq2: 1 token each
  → input_ids = [token0, token1, token2]
  → positions = [100, 149, 199]
  → slot_mapping = [block_id * 256 + offset for each]
```

---

### 3.2 Context (`nanovllm/utils/context.py`)

**Class: `Context` (dataclass)**
Thread-local state passed to layers during forward pass. Enables efficient attention and memory management.

| Field | Purpose |
|-------|---------|
| `is_prefill: bool` | Prefill (varlen) vs decode (single token) |
| `cu_seqlens_q: Tensor` | Cumulative sequence lengths for query (varlen attention) |
| `cu_seqlens_k: Tensor` | Cumulative sequence lengths for key (varlen attention) |
| `max_seqlen_q: int` | Max query length in batch |
| `max_seqlen_k: int` | Max key length in batch |
| `slot_mapping: Tensor` | Where to store KV in cache (1D: flat offsets) |
| `context_lens: Tensor` | Length of context for each sequence (decode) |
| `block_tables: Tensor` | 2D: block indices per sequence (for sparse KV) |

**Functions:**
- `get_context()` - Retrieve global context
- `set_context(...)` - Update global context
- `reset_context()` - Clear (set to defaults)

---

### 3.3 Model Loader (`nanovllm/utils/loader.py`)

**Function: `load_model(model, path)`**
Loads weights from safetensors files in `path` directory.

**Special Handling:**
- Supports `packed_modules_mapping` (e.g., separate Q/K/V → merged QKV)
- Calls `weight_loader` method on each parameter for custom loading logic
- Handles tensor parallel sharding automatically

**Example:** Qwen3 unpacks q_proj/k_proj/v_proj into qkv_proj with shard IDs.

---

## 4. Qwen3 Model Implementation

### 4.1 Qwen3ForCausalLM (`nanovllm/models/qwen3.py`)

**Class: `Qwen3ForCausalLM`**

**Attributes:**
- `model: Qwen3Model` - Backbone
- `lm_head: ParallelLMHead` - Output projection

**Key Methods:**
- `forward(input_ids, positions)` - Returns hidden states
- `compute_logits(hidden_states)` - Returns logits (for sampling)

**Packed Modules Mapping** (for weight loading):
```python
{
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

---

### 4.2 Qwen3Model

**Class: `Qwen3Model`**

**Architecture:**
```
input_ids (int64)
  ↓
VocabParallelEmbedding → hidden_size
  ↓
[Qwen3DecoderLayer] × num_hidden_layers
  ↓
RMSNorm
  ↓
hidden_states (float)
```

**Key Methods:**
- `forward(input_ids, positions)` - Runs all layers with residual connections

---

### 4.3 Qwen3DecoderLayer

**Class: `Qwen3DecoderLayer`**

**Architecture:**
```
input (hidden_states)
  ↓
RMSNorm (input_layernorm)
  ↓
Qwen3Attention
  ↓ (residual add)
RMSNorm (post_attention_layernorm)
  ↓
Qwen3MLP
  ↓
output (hidden_states, residual)
```

**Key Feature:** Pre-norm with residual pass-through for efficiency.

---

### 4.4 Qwen3Attention

**Class: `Qwen3Attention`**

**Components:**
- `QKVParallelLinear`: Projects hidden_states → [Q, K, V]
- `RotaryEmbedding`: RoPE encoding
- `Attention` (custom ops): FlashAttention with block tables
- `RowParallelLinear`: Projects attention output back to hidden_size

**Forward Pass:**
```
hidden_states (B*L, hidden_size)
  ↓
QKVParallelLinear → (B*L, heads, head_dim) each for Q, K, V
  ↓
RMSNorm (if no bias) on Q and K
  ↓
RotaryEmbedding: Apply RoPE
  ↓
Attention: Flash attention with KV cache
  ↓
RowParallelLinear: (B*L, heads, head_dim) → (B*L, hidden_size)
```

**Tensor Parallelism:**
- Each rank gets `num_heads // world_size` heads
- Attention computed locally
- Output aggregated via all_reduce

---

### 4.5 Qwen3MLP

**Class: `Qwen3MLP`**

**Architecture:**
```
hidden_states (B*L, hidden_size)
  ↓
MergedColumnParallelLinear → 2 × intermediate_size
  ↓
SiluAndMul: Chunk in half, apply SiLU to first, multiply by second
  ↓
RowParallelLinear → hidden_size
```

**SwiGLU variant** with tensor parallelism.

---

## 5. Layer Implementations

### 5.1 Attention (`nanovllm/layers/attention.py`)

**Triton Kernel: `store_kvcache_kernel`**
Efficiently stores K/V into pre-allocated cache buffers using slot mapping (for scatter operations).

**Class: `Attention`**

**Methods:**
- `forward(q, k, v)` - Main attention forward pass

**Logic:**
1. Store K/V to cache using Triton kernel + slot_mapping
2. If prefill:
   - If prefix cache available: use cached K/V for known tokens
   - Call `flash_attn_varlen_func` with block tables for sparse access
3. If decode:
   - Call `flash_attn_with_kvcache` for single-token attention
   - Use block_table for scattered KV lookup

**Optimizations:**
- Triton kernel avoids Python loops
- Block tables enable efficient sparse KV cache access
- Flash attention handles variable-length sequences

---

### 5.2 Linear Layers (`nanovllm/layers/linear.py`)

Implements tensor parallelism patterns:

**Class: `LinearBase`**
Abstract base with common initialization.

**Class: `ReplicatedLinear`**
Standard linear layer (replicated across all TP ranks).

**Class: `ColumnParallelLinear`**
Output dimension parallelized.
```
Input: (B, in_features)
  ↓
Linear [tp_rank] → (B, out_features // tp_size)
```

**Class: `QKVParallelLinear` (extends ColumnParallelLinear)**
Specialized for Q/K/V projection with optional KV head scaling.
- Q: `num_heads // tp_size * head_dim`
- K: `num_kv_heads // tp_size * head_dim`
- V: `num_kv_heads // tp_size * head_dim`

**Class: `RowParallelLinear`**
Input dimension parallelized.
```
Inputs: (B, in_features // tp_size) per rank
  ↓
Linear → (B, out_features)
  ↓
AllReduce (sum across ranks)
```

**All-Reduce Sync:** Required on rank 0 to aggregate outputs.

---

### 5.3 Layer Normalization (`nanovllm/layers/layernorm.py`)

**Class: `RMSNorm`**

**Methods:**
- `rms_forward(x)` - Standard RMSNorm
- `add_rms_forward(x, residual)` - RMSNorm + residual addition (fused)
- `forward(x, residual=None)` - Dispatches to appropriate method

**Compilation:** Both forward methods are `@torch.compile`d for efficiency.

**Formula:**
```
RMSNorm(x) = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x²) + eps)
```

---

### 5.4 Rotary Embeddings (`nanovllm/layers/rotary_embedding.py`)

**Function: `apply_rotary_emb(x, cos, sin)`**
Applies 2D rotation matrices to embedding pairs.

**Class: `RotaryEmbedding`**
Pre-computes cos/sin tables for all positions.

**Methods:**
- `forward(positions, query, key)` - Apply RoPE to Q and K tensors

**Caching:** `@lru_cache(1)` singleton via `get_rope()` function.

**Formula:**
```
For each pair (x1, x2):
  y1 = x1*cos(θ) - x2*sin(θ)
  y2 = x2*cos(θ) + x1*sin(θ)
where θ = base^(2i/d) for dimension i
```

---

### 5.5 Activation Functions (`nanovllm/layers/activation.py`)

**Class: `SiluAndMul`**

**Method:** `forward(x)` → `SiLU(x1) * x2` where x is split in half

**Use Case:** SwiGLU activation in FFN (gate and value projections).

---

### 5.6 Embedding & Head (`nanovllm/layers/embed_head.py`)

**Class: `VocabParallelEmbedding`**
Input embeddings with vocabulary parallelization.

**Tensor Parallelism:**
- Vocab split across ranks: `[vocab_start, vocab_end)` per rank
- Mask invalid indices, compute locally, all_reduce sum

**Class: `ParallelLMHead` (extends VocabParallelEmbedding)**
Output logits with special handling for prefill.

**Prefill Optimization:**
- Extract logits only for last token of each sequence (for loss computation)
- Indices computed from `cu_seqlens_q[1:] - 1`

**Decode:**
- Extract logits for all single tokens (one per sequence)

---

### 5.7 Sampler (`nanovllm/layers/sampler.py`)

**Class: `Sampler`**

**Method:** `forward(logits, temperatures)` → token_ids

**Algorithm (Gumbel-Max trick):**
1. Scale logits by temperature: `logits / temp`
2. Softmax to probabilities
3. Sample Gumbel noise: `u ~ Exponential(1)`
4. Compute: `logits' = log(p) - log(-log(u))`
5. Argmax (equivalent to sampling from categorical)

**Advantage:** Differentiable and GPU-efficient.

---

## 6. Data Flow & Execution Example

### Request: Generate 2 completions (256 tokens each) with 100-token prefix

**Batch 1 - Prefill:**
```
Input:
  Seq0: [p0, p1, ..., p99, (start)]  100 tokens → generate 256 tokens
  Seq1: [q0, q1, ..., q99, (start)]  100 tokens → generate 256 tokens

Scheduler: can_allocate? Yes (need 1 block each, have plenty)
  Allocate seq0 block 0 (no cache hit)
  Allocate seq1 block 0 (prefix cache hit!)
  seq0.num_cached_tokens = 0
  seq1.num_cached_tokens = 100  ← cached!

ModelRunner.prepare_prefill([seq0, seq1]):
  cu_seqlens_q = [0, 100, 100]  (new tokens to process)
  cu_seqlens_k = [0, 100, 200]  (total context)
  input_ids = [p0...p99, q0...q99, start_token0, start_token1]
  slot_mapping = [block0 offsets, block1 offsets] (100 slots per seq)
  → Flash attention with block_tables for sparse KV

ModelRunner.run_model(..., is_prefill=True):
  Forward pass with varlen attention
  Output: logits (100 + 100 + 2) = 202 predictions

Sampler: sample_token([logits_seq0_last, logits_seq1_last])
  → [token_a, token_b]

Postprocess:
  seq0.append_token(token_a)  → 101 tokens
  seq1.append_token(token_b)  → 101 tokens

Batch 2-257 - Decode (each step):
  Scheduler: schedule decode phase
    → [seq0, seq1]
    → is_prefill=False

  ModelRunner.prepare_decode([seq0, seq1]):
    input_ids = [last_token_seq0, last_token_seq1]
    positions = [100, 100]
    slot_mapping = [block0 * 256 + 100, block1 * 256 + 100]
    context_lens = [101, 101]
    block_tables = [[0, 1, ...], [0, 1, ...]]

  Run CUDA graph (pre-captured for bs=2):
    Input already in graph variables
    Replay graph → forward pass
    Output: logits (2 predictions)

  Sample → [token_c, token_d]
  Postprocess: seq0.num_completion_tokens = 1, seq1.num_completion_tokens = 1

After 256 decode steps:
  seq0.is_finished = True (256 completion tokens)
  seq1.is_finished = True (256 completion tokens)
  Return: [decoded_completion_0, decoded_completion_1]
```

---

## 7. Optimization Techniques

### 7.1 Prefix Caching
- **What:** Reuse KV cache blocks across sequences with identical token prefixes
- **Implementation:** Hash-based lookup (xxHash64) with ref counting
- **Benefit:** Reduce GPU memory usage, speed up prefill for repeated prompts

### 7.2 Tensor Parallelism
- **Pattern:** Column-parallel (Q proj) + Row-parallel (output proj)
- **Sync:** All-reduce on row-parallel outputs
- **IPC:** Shared memory for multi-process synchronization
- **Benefit:** Scale to multi-GPU with minimal communication

### 7.3 CUDA Graph Capture
- **What:** Pre-record decode forward passes for batch sizes 1-512
- **Benefit:** Skip CPU overhead, pure GPU compute
- **Limitation:** Only for decode (prefill too dynamic)
- **Implementation:** Store both graph and variable buffers

### 7.4 Flash Attention
- **What:** Optimized attention implementation (from flash-attn library)
- **Variants:**
  - `flash_attn_varlen_func`: Variable-length sequences (prefill)
  - `flash_attn_with_kvcache`: Single token with existing KV (decode)
- **Block Tables:** Sparse KV access via physical block mappings

### 7.5 KV Cache Management
- **Pre-allocation:** Allocate all KV cache upfront (90% GPU memory)
- **Block-based:** 256 tokens per block (flexible) for efficient memory layout
- **Slot Mapping:** 1D flat offsets enable Triton scatter/gather

### 7.6 Request Scheduling with Preemption
- **Prefill-first:** Process new requests' full prompt in one batch
- **Preemption:** Evict running requests if decode batch full
- **Benefit:** Higher GPU utilization, prioritize faster completion

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Process per rank (not threads)** | Avoid GIL, clean GPU independence |
| **Shared memory IPC** | Lower overhead than socket communication |
| **Prefix caching** | Critical for repeated prompt prefixes |
| **Block-based KV cache** | Variable-size sequences, efficient allocation |
| **Slot mapping (1D)** | Enables Triton scatter/gather operations |
| **CUDA graphs** | 10x+ speedup for decode (no CPU overhead) |
| **FlashAttention v2** | State-of-art attention performance |
| **Compiled RMSNorm** | torch.compile for micro-optimizations |
| **Gumbel-max sampler** | Differentiable, GPU-native |
| **Preemption** | Handle requests larger than GPU memory |

---

## 9. File Structure Summary

```
nano-vllm/
├── README.md                          # Project overview
├── pyproject.toml                     # Dependencies & metadata
├── example.py                         # Usage example
├── bench.py                           # Benchmark script
└── nanovllm/
    ├── __init__.py                    # Public API (LLM, SamplingParams)
    ├── llm.py                         # LLM wrapper
    ├── config.py                      # Config dataclass
    ├── sampling_params.py             # SamplingParams dataclass
    ├── engine/
    │   ├── llm_engine.py              # Main orchestrator
    │   ├── model_runner.py            # GPU execution + TP
    │   ├── scheduler.py               # Request scheduling
    ├── sequence.py                    # Sequence state tracking
    ├── block_manager.py               # KV cache management + prefix caching
    ├── layers/
    │   ├── attention.py               # Flash attention + KV cache storage
    │   ├── linear.py                  # Tensor parallel linear layers
    │   ├── layernorm.py               # RMSNorm + fusion
    │   ├── rotary_embedding.py        # RoPE embeddings
    │   ├── activation.py              # SiLU + gating
    │   ├── embed_head.py              # Embeddings + LM head
    │   └── sampler.py                 # Token sampling
    ├── models/
    │   └── qwen3.py                   # Qwen3 model architecture
    └── utils/
        ├── context.py                 # Thread-local inference context
        └── loader.py                  # Weight loading from safetensors
```

---

## 10. Performance Characteristics

**Benchmark (RTX 4070, Qwen3-0.6B, 256 sequences):**

| Metric | Nano-vLLM | vLLM |
|--------|-----------|------|
| Total Output Tokens | 133,966 | 133,966 |
| Time (s) | 93.41 | 98.37 |
| Throughput (tok/s) | 1434.13 | 1361.84 |
| **Speedup** | **1.053x** | - |

**Factors enabling this:**
1. Efficient KV cache management (prefix caching + block-based)
2. CUDA graph capture (no CPU scheduling per step)
3. Minimal scheduling overhead (simpler preemption)
4. Flash attention optimizations
5. Lean codebase (less overhead than full vLLM)

---

## 11. Example Usage

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Load model
path = "/path/to/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

# Single prompt
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])

# Batch
prompts = ["Tell me a joke", "Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output['text']}\n")

# With token IDs
token_ids = tokenizer.encode("What is AI?")
outputs = llm.generate([token_ids], sampling_params)
```

---

## 12. Limitations & Future Work

**Current Limitations:**
- Qwen3 model only (architecture specific)
- Single-machine (no distributed training)
- No speculative decoding
- No quantization support

**Potential Extensions:**
- Add more models (LLaMA, Mistral, etc.)
- LoRA/fine-tuning support
- vLLM-style LoRA serving
- INT8/FP8 quantization
- Speculative decoding for faster sampling

