# NanoVLLM - Quick Start Architecture Guide

## What is NanoVLLM?

A ~1,200 line Python implementation of a production-grade LLM inference engine (comparable to vLLM).

## Core Architecture

```
User Code
    ↓
LLM (wrapper)
    ↓
LLMEngine (orchestrator)
    ├─ Tokenizer
    ├─ Scheduler (batching, KV cache allocation)
    └─ ModelRunner (execution)
        ├─ Qwen3ForCausalLM (model)
        └─ CUDA graphs, KV cache
```

## Key Files & What They Do

| File | Purpose | Key Classes |
|------|---------|-------------|
| `llm.py` | Entry point | `LLM` |
| `engine/llm_engine.py` | Orchestration | `LLMEngine` |
| `engine/scheduler.py` | Batching logic | `Scheduler` |
| `engine/model_runner.py` | GPU execution | `ModelRunner` |
| `engine/sequence.py` | Per-request state | `Sequence` |
| `engine/block_manager.py` | KV cache mgmt + prefix caching | `BlockManager` |
| `models/qwen3.py` | Model architecture | `Qwen3ForCausalLM` |
| `layers/attention.py` | Flash Attention + KV cache | `Attention` |
| `layers/linear.py` | Tensor parallel layers | `ColumnParallelLinear`, `RowParallelLinear`, etc. |
| `utils/loader.py` | Weight loading | `load_model()` |
| `utils/context.py` | Global inference state | `Context`, `get_context()`, `set_context()` |

## Generation Flow (High Level)

```python
llm = LLM("model_path")
outputs = llm.generate(["hello"], SamplingParams(max_tokens=10))
```

### Step 1: Add Requests
- Tokenize prompt → `Sequence` object
- Add to Scheduler's waiting queue

### Step 2: Prefill Phase
- Scheduler loads sequences from waiting queue
- BlockManager allocates KV cache blocks
  - ✨ **Prefix Caching**: Detects matching token prefixes, reuses blocks
- ModelRunner prepares batched inputs for FlashAttn
- Model forward pass + sample tokens

### Step 3: Decode Phase (repeated)
- Process each sequence one token at a time
- Use cached K,V from prefill
- Append new K,V to cache
- Sample next token
- Repeat until EOS or max_tokens reached

### Step 4: Detokenize
- Convert token IDs → text
- Return to user

## Qwen3 Model Structure

```
Qwen3ForCausalLM
├─ Qwen3Model
│  ├─ VocabParallelEmbedding (token embeddings)
│  ├─ Qwen3DecoderLayer × N
│  │  ├─ Qwen3Attention
│  │  │  ├─ QKVParallelLinear (fused Q, K, V projection)
│  │  │  ├─ RowParallelLinear (output projection)
│  │  │  ├─ RoPE (rotary embeddings)
│  │  │  ├─ Attention (FlashAttn + KV cache)
│  │  │  └─ RMSNorm (Q/K normalization, if no bias)
│  │  ├─ Qwen3MLP
│  │  │  ├─ MergedColumnParallelLinear (gate + up)
│  │  │  ├─ SiluAndMul (gate * up)
│  │  │  └─ RowParallelLinear (down projection)
│  │  ├─ RMSNorm (input, pre-attention)
│  │  └─ RMSNorm (post-attention)
│  └─ RMSNorm (final)
└─ ParallelLMHead (vocabulary prediction)
```

## Key Optimizations

### 1. Prefix Caching
```
Prompt 1: "Hello world, how are you"
Prompt 2: "Hello world, what's the weather"
          ↓
          Shared prefix: "Hello world" is cached!
          → Save KV cache computation + memory
```

**How**: BlockManager hashes token blocks, reuses matching prefixes

### 2. CUDA Graphs
```
Decode loop normally:
  ├─ CPU→GPU copy inputs
  ├─ GPU kernel launches
  ├─ GPU→CPU copy outputs
  └─ Repeat (overhead!)

With CUDA graphs:
  ├─ Record once
  └─ Replay (just kernel execution!)
  → ~10% faster
```

### 3. Tensor Parallelism
```
GPU 0                GPU 1
├─ Weight[0:2048]    ├─ Weight[2048:4096]
├─ Q projection      └─ K,V projections
└─ Sync via all-reduce
```

**Patterns**:
- **Column Parallel**: Split output dimension (no communication needed)
- **Row Parallel**: Split input dimension (needs all-reduce)
- **Vocab Parallel**: Split vocabulary (embedding + LM head)

### 4. Torch Compilation
- `@torch.compile` on hot functions (RMSNorm, activation, sampler, RoPE)
- Reduces Python overhead, fuses kernels

### 5. Flash Attention
- Faster attention kernel (IO-aware)
- Supports variable sequence lengths (prefill)
- Integrated KV cache access (decode)

## Model Loading: Packed Weights

Hugging Face saves separate Q, K, V projections:
```
model.safetensors:
├─ q_proj: [4096, 1024]
├─ k_proj: [4096, 1024]
├─ v_proj: [4096, 1024]
```

NanoVLLM fuses them:
```
class Qwen3ForCausalLM:
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

During loading, custom `weight_loader()` methods combine weights into fused layers.

## Execution Context

Global state passed to layers during forward:

```python
@dataclass
class Context:
    is_prefill: bool              # Prefill vs decode?
    cu_seqlens_q: Tensor          # Cumulative seq lengths (query)
    cu_seqlens_k: Tensor          # Cumulative seq lengths (key/value)
    slot_mapping: Tensor          # KV cache slot indices
    context_lens: Tensor          # Context length per sequence (decode)
    block_tables: Tensor          # Prefix cache block mappings
```

**Who uses it**:
- `Attention`: Knows prefill vs decode, accesses KV cache
- `ParallelLMHead`: Knows which tokens need logits (decode: all, prefill: last per seq)

## Configuration

```python
Config(
    model="path/to/model",
    max_num_batched_tokens=16384,  # Max tokens per batch
    max_num_seqs=512,              # Max sequences per batch
    max_model_len=4096,            # Max sequence length
    gpu_memory_utilization=0.9,    # Use 90% GPU memory for KV cache
    tensor_parallel_size=1,        # Number of GPUs
    enforce_eager=False,           # Disable CUDA graphs (slower but simpler)
    kvcache_block_size=256,        # KV cache block size (must be 256+)
)
```

## Sampling

```python
SamplingParams(
    temperature=0.6,      # Temperature scaling (must be > 1e-10!)
    max_tokens=256,       # Max generation length
    ignore_eos=False,     # Keep generating past EOS?
)
```

**Why temperature > 1e-10**? Constraint to force sampling (no greedy).

## Example Usage

```python
from nanovllm import LLM, SamplingParams

# Load model
llm = LLM(
    "~/huggingface/Qwen3-0.6B",
    enforce_eager=True,        # Disable CUDA graphs (simpler)
    tensor_parallel_size=1     # Single GPU
)

# Prepare inputs
prompts = ["Hello", "How are you"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Generate
outputs = llm.generate(prompts, sampling_params)
# outputs = [
#   {"text": "Hello ...", "token_ids": [...]},
#   {"text": "How are you ...", "token_ids": [...]}
# ]
```

## Multi-GPU Tensor Parallelism

```python
llm = LLM("model_path", tensor_parallel_size=4)  # 4 GPUs

# Internally:
# - Main process (rank 0): Orchestrates
# - 3 spawned processes (ranks 1-3): Model workers
# - Communication: SharedMemory + NCCL for all-reduce
```

## Performance Characteristics

- **Prefill throughput**: ~1000-1500 tok/s (GPU-dependent)
- **Decode throughput**: ~100 tok/s (memory-limited)
- **Memory usage**: ~90% GPU VRAM (configurable)
- **Compared to vLLM**: ~5-10% faster (tighter codebase)

## Extension Points

### Add a new model:
1. Create `nanovllm/models/mymodel.py`
2. Implement `MyForCausalLM(nn.Module)` with:
   - `forward(input_ids, positions) → hidden_states`
   - `compute_logits(hidden_states) → logits`
   - `packed_modules_mapping` (if using weight fusion)

### Add a new optimization:
- Custom kernels: Add to layer files
- Better scheduling: Modify `Scheduler.schedule()`
- New layer types: Add to `layers/`

## File Size Summary

| Component | LOC | Purpose |
|-----------|-----|---------|
| Engine | ~600 | Orchestration, scheduling, execution |
| Models | ~200 | Qwen3 architecture |
| Layers | ~300 | Attention, linear, embeddings |
| Utils | ~100 | Loading, context |
| **Total** | **~1200** | Production-grade inference |

## Debugging Tips

1. **Set `enforce_eager=True`** to disable CUDA graphs (simpler debugging)
2. **Check `context.is_prefill`** to understand current phase
3. **Monitor `config.num_kvcache_blocks`** if memory issues
4. **Use `torch.cuda.memory_stats()`** to track GPU memory
5. **Print layer shapes** during `warmup_model()` to verify architecture

## References

- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **vLLM**: https://github.com/lm-sys/vllm
- **Tensor Parallelism**: https://arxiv.org/abs/2104.04473
- **Prefix Caching**: Similar to vLLM's RadixAttention

