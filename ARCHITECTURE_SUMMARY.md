# Nano-vLLM Architecture Quick Reference

## Component Overview

| Component | File | Purpose | Key Classes |
|-----------|------|---------|------------|
| **Core API** | `llm.py` | Public inference interface | `LLM` |
| **Engine** | `engine/llm_engine.py` | Main orchestrator | `LLMEngine` |
| **Configuration** | `config.py` | System parameters | `Config` |
| **Sampling** | `sampling_params.py` | Generation parameters | `SamplingParams` |
| **Model Execution** | `engine/model_runner.py` | GPU forward pass + TP | `ModelRunner` |
| **Scheduling** | `engine/scheduler.py` | Request scheduling | `Scheduler` |
| **Sequences** | `engine/sequence.py` | Request state | `Sequence` |
| **KV Cache** | `engine/block_manager.py` | Memory management + prefix cache | `BlockManager`, `Block` |
| **Model** | `models/qwen3.py` | Qwen3 architecture | `Qwen3ForCausalLM` |
| **Layers** | `layers/` | Neural network components | `Attention`, `RMSNorm`, `Linear*`, etc. |
| **Utils** | `utils/` | Helpers | `Context`, loader functions |

---

## Execution Flow

```
User: llm.generate([prompts], sampling_params)
  ↓
LLMEngine.generate()
  ├─ Tokenize prompts
  ├─ Add requests to scheduler
  ├─ Loop until finished:
  │  ├─ Scheduler.schedule() → get batch [seqs], is_prefill
  │  ├─ ModelRunner.run(seqs, is_prefill)
  │  │  ├─ prepare_prefill() or prepare_decode()
  │  │  ├─ run_model() [with CUDA graph for decode]
  │  │  ├─ Sampler() → sample tokens
  │  │  └─ return token_ids
  │  ├─ Scheduler.postprocess(seqs, token_ids)
  │  └─ Collect finished outputs
  ├─ Decode token_ids to strings
  └─ Return [{"text": ..., "token_ids": ...}]
```

---

## Key Data Structures

### Sequence (Request State)
- **Mutable:** `token_ids`, `status`, `num_cached_tokens`, `block_table`
- **Immutable:** `seq_id`, `num_prompt_tokens`, `temperature`, `max_tokens`
- **Derived:** `num_completion_tokens`, `is_finished`, `num_blocks`

### Block (KV Cache)
- **block_id:** Physical location in GPU memory
- **hash:** xxHash64 of token_ids (for prefix caching)
- **ref_count:** For multi-sequence sharing
- **token_ids:** Stored tokens (for verification)

### Context (Thread-local State)
- **is_prefill:** Bool (varlen attn vs single-token)
- **cu_seqlens_q/k:** Cumulative lengths for attention
- **slot_mapping:** 1D flat offsets for KV cache storage
- **block_tables:** 2D sparse layout for KV lookup
- **context_lens:** Per-sequence context length (decode)

---

## Scheduling Algorithm

### Phase 1: Prefill (New Requests)
```
while waiting_requests and slots_available and memory_available:
    seq = waiting.popleft()
    BlockManager.allocate(seq)  # with prefix caching
    seq.status = RUNNING
    running.append(seq)
return running, is_prefill=True
```

### Phase 2: Decode (Running Requests)
```
while running_requests:
    seq = running.popleft()
    if not BlockManager.can_append(seq):
        preempt(seq)  # back to waiting
    else:
        BlockManager.may_append(seq)
        scheduled.append(seq)
return scheduled, is_prefill=False
```

---

## Prefix Caching Algorithm

**Goal:** Reuse KV cache blocks with identical token sequences.

**Hash Computation:** `hash = xxHash64(token_ids || prev_hash)`

**Allocation:**
1. For each block in sequence:
   - Compute incremental hash
   - Check `hash_to_block_id` table
   - **Hit:** Increment ref_count, mark cached
   - **Miss:** Allocate new block, update table

**Example:**
```
Seq A: [token_0 ... token_255] → Block 0
Seq B: [token_0 ... token_255, token_256]
       → Block 0 (reused, ref_count=2) + Block 1 (new)
```

---

## Tensor Parallelism Patterns

### Column-Parallel Linear
- Output dimension split across ranks
- Example: Q/K/V projections
- Local compute, no sync needed

### Row-Parallel Linear
- Input dimension split across ranks
- Example: Attention output projection
- **Requires:** `all_reduce(output)` to aggregate

### Vocabulary Parallel
- Embedding table split across ranks
- Mask invalid vocab range per rank
- **Requires:** `all_reduce(embedding)` for all_reduce sum

**IPC Sync:** Rank 0 writes to shared memory → events signal ranks 1+ → barrier sync

---

## Optimization Techniques

### 1. Prefix Caching
- Hash-based deduplication of KV blocks
- Multi-request token sharing via ref counting
- Saves memory for repeated prefixes

### 2. Block-Based KV Cache
- 256 tokens per block (configurable)
- Flexible allocation for variable-length sequences
- Slot mapping for efficient Triton scatter/gather

### 3. CUDA Graph Capture
- Pre-record decode forward passes (batch sizes 1-512)
- Replay at runtime (no CPU scheduling overhead)
- ~10x faster than eager execution

### 4. Flash Attention
- `flash_attn_varlen_func()` for prefill (variable lengths)
- `flash_attn_with_kvcache()` for decode (single token)
- Block tables enable sparse KV access

### 5. Pre-norm with Residual Fusion
- `RMSNorm` + residual addition in one kernel
- `@torch.compile` for micro-optimizations

### 6. Gumbel-Max Sampling
- Differentiable sampling on GPU
- Avoids CPU roundtrip
- Temperature scaling

---

## Memory Layout Example

### Prefill Batch (3 sequences, 256-token blocks)
```
Seq0: 100 tokens (prefix cached after block 0)
Seq1: 150 tokens (all new)
Seq2: 200 tokens (all new)

cu_seqlens_q: [0, 100, 150, 150]    # new tokens to compute
cu_seqlens_k: [0, 100, 250, 450]    # total context (includes cache)

slot_mapping:
  [block0_offsets] (Seq0 new: pos 0-99)
  [block1_offsets, block2_offsets] (Seq1 new: pos 0-149)
  [block3_offsets, block4_offsets] (Seq2 new: pos 0-199)

block_tables:
  Seq0: [0, 1, ...]
  Seq1: [1, 2, ...]
  Seq2: [3, 4, ...]
```

### Decode Batch (same 3 sequences, 1 token each)
```
input_ids: [last_token_seq0, last_token_seq1, last_token_seq2]
positions: [100, 150, 200]
slot_mapping: [block0*256+100, block1*256+150, block3*256+200]
context_lens: [101, 151, 201]

block_tables:
  [[0, 1, ...], [1, 2, ...], [3, 4, ...]]
```

---

## Qwen3 Model Architecture

```
Input: input_ids (B*L,), positions (B*L,)
  ↓
Embedding: VocabParallelEmbedding
  ↓
[24 Layers × Qwen3DecoderLayer]:
  ├─ Self-Attention (QKV → Flash Attention)
  ├─ Residual Add
  ├─ MLP (SwiGLU + SiLU)
  └─ Residual Add
  ↓
RMSNorm
  ↓
LMHead: ParallelLMHead
  ↓
Logits (vocab_size,)
```

**Key Layers:**
- `Qwen3Attention`: QKV projection → RoPE → FlashAttention → output projection
- `Qwen3MLP`: Gate-up projection → SiLU-mul → down projection
- `RMSNorm`: Root-mean-square normalization with residual

---

## Configuration Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_num_batched_tokens` | 16384 | Max tokens processed per step |
| `max_num_seqs` | 512 | Max concurrent requests |
| `max_model_len` | 4096 | Max sequence length |
| `gpu_memory_utilization` | 0.9 | Fraction of GPU for KV cache |
| `tensor_parallel_size` | 1 | Number of GPUs |
| `enforce_eager` | False | Disable CUDA graphs |
| `kvcache_block_size` | 256 | Tokens per block |

---

## Files by Category

### Core Engine
- `llm.py`, `engine/llm_engine.py`, `engine/model_runner.py`

### Scheduling & Memory
- `engine/scheduler.py`, `engine/block_manager.py`, `engine/sequence.py`

### Configuration
- `config.py`, `sampling_params.py`

### Layers
- `layers/attention.py` (Flash attention + KV cache)
- `layers/linear.py` (TP-aware linear layers)
- `layers/layernorm.py` (RMSNorm)
- `layers/rotary_embedding.py` (RoPE)
- `layers/activation.py` (SiLU + mul)
- `layers/embed_head.py` (Embeddings + LM head)
- `layers/sampler.py` (Gumbel-max sampling)

### Model
- `models/qwen3.py` (Qwen3 architecture)

### Utilities
- `utils/context.py` (Thread-local state)
- `utils/loader.py` (Weight loading)

---

## Performance Insights

**Nano-vLLM vs vLLM (RTX 4070, Qwen3-0.6B):**
- Nano-vLLM: **1,434.13 tok/s** (93.41s for 133,966 tokens)
- vLLM: 1,361.84 tok/s (98.37s)
- **Speedup: 1.053x** ✓

**Why Faster:**
1. Prefix caching + block-based memory (efficient allocation)
2. CUDA graph capture (decode is pure GPU)
3. Simpler scheduling (lower CPU overhead)
4. Flash attention (IO-optimized kernels)
5. Clean codebase (less Python overhead)

---

## Common Workflows

### Single Prompt
```python
llm = LLM(model_path)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=256))
print(outputs[0]["text"])
```

### Batch Processing
```python
prompts = ["prompt1", "prompt2", ...]
outputs = llm.generate(prompts, SamplingParams(max_tokens=256))
```

### Token IDs Input
```python
token_ids = tokenizer.encode("prompt")
outputs = llm.generate([token_ids], SamplingParams())
```

### Tensor Parallelism
```python
llm = LLM(model_path, tensor_parallel_size=2)  # Use 2 GPUs
```

---

## Key Insights

1. **Prefix Caching is Critical:** Shared prefixes across requests reuse KV cache blocks
2. **CUDA Graphs Save Time:** Decode without CPU scheduling gives 10x+ speedup
3. **Block-Based Layout:** Enables flexible allocation and sparse KV access
4. **Minimal Scheduling:** Simple prefill-first + preemption approach
5. **Process-per-Rank:** Clean isolation for tensor parallelism
6. **Shared Memory IPC:** Low-overhead multi-GPU communication

