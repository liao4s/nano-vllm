# NanoVLLM - Quick Reference Guide

---

## 1. FILE STRUCTURE AT A GLANCE

```
nanovllm/
├── __init__.py                    → exports: LLM, SamplingParams
├── llm.py                         → class LLM(LLMEngine)
├── config.py                      → @dataclass Config
├── sampling_params.py             → @dataclass SamplingParams
│
├── models/
│   └── qwen3.py                   → Qwen3ForCausalLM, Qwen3Attention, Qwen3MLP
│
├── engine/
│   ├── llm_engine.py              → class LLMEngine (main API)
│   ├── model_runner.py            → class ModelRunner (execution)
│   ├── scheduler.py               → class Scheduler (batching)
│   ├── sequence.py                → class Sequence (request)
│   └── block_manager.py           → class BlockManager (KV cache)
│
├── layers/
│   ├── linear.py                  → LinearBase, ColumnParallelLinear, etc.
│   ├── attention.py               → Attention class
│   ├── layernorm.py               → RMSNorm
│   ├── rotary_embedding.py        → RotaryEmbedding, get_rope()
│   ├── activation.py              → SiluAndMul
│   ├── embed_head.py              → VocabParallelEmbedding, ParallelLMHead
│   └── sampler.py                 → Sampler (token sampling)
│
└── utils/
    ├── loader.py                  → load_model() function
    └── context.py                 → Context, get_context(), set_context()
```

---

## 2. KEY DATA STRUCTURES

### Sequence

```python
Sequence:
  .seq_id: int                          # Unique ID
  .token_ids: list[int]                 # All tokens so far
  .status: SequenceStatus               # WAITING, RUNNING, FINISHED
  .temperature: float                   # Sampling temperature
  .max_tokens: int                      # Max completion tokens
  .block_table: list[int]               # KV cache block IDs
  .num_cached_tokens: int               # Cached prefix length
  
  Properties:
  .is_finished: bool
  .num_completion_tokens: int
  .num_blocks: int                      # (num_tokens + block_size - 1) // block_size
  .last_block_num_tokens: int
```

### Config

```python
Config:
  model: str                            # Path to model dir
  max_num_batched_tokens: int = 16384
  max_num_seqs: int = 512
  max_model_len: int = 4096
  gpu_memory_utilization: float = 0.9
  tensor_parallel_size: int = 1
  enforce_eager: bool = False
  hf_config: AutoConfig                 # Loaded from model
  eos: int                              # Set from tokenizer
  kvcache_block_size: int = 256
  num_kvcache_blocks: int               # Computed from GPU memory
```

### SamplingParams

```python
SamplingParams:
  temperature: float = 1.0              # > 1e-10 (no greedy)
  max_tokens: int = 64
  ignore_eos: bool = False
```

---

## 3. EXECUTION PATHS

### User Code to Generate

```python
llm = LLM(model_path)
outputs = llm.generate(prompts, sampling_params)
```

### Initialization

```
LLM.__init__
  ├─ Config.__post_init__ (load HF config)
  ├─ ModelRunner(rank 0)
  │   ├─ dist.init_process_group()
  │   ├─ Qwen3ForCausalLM(config)
  │   ├─ load_model() ← WEIGHT LOADING
  │   ├─ warmup_model()
  │   ├─ allocate_kv_cache()
  │   └─ capture_cudagraph() [if not enforce_eager]
  ├─ Worker ModelRunners (rank > 0)
  ├─ AutoTokenizer.from_pretrained()
  └─ Scheduler(config)
```

### Generation Loop

```
llm.generate(prompts, sampling_params)
  ├─ add_request() for each prompt
  └─ While not finished:
      ├─ scheduler.schedule()
      │   └─ Returns: seqs, is_prefill
      ├─ model_runner.call("run", seqs, is_prefill)
      │   ├─ prepare_prefill() or prepare_decode()
      │   ├─ run_model()
      │   │   ├─ forward pass (eager or graph replay)
      │   │   └─ compute_logits()
      │   └─ sampler() → token_ids
      └─ scheduler.postprocess(seqs, token_ids)
          ├─ append_token()
          └─ check finish condition
```

---

## 4. MODEL LOADING

### Entry Point

```python
load_model(model: nn.Module, path: str)
```

### Weight Loader Pattern

```python
# Each parameter has:
param.weight_loader(param, loaded_weight, shard_id=None)

# Different for each layer type:
LinearBase.weight_loader(param, loaded_weight)
ColumnParallelLinear.weight_loader(param, loaded_weight)
QKVParallelLinear.weight_loader(param, loaded_weight, shard_id)
RowParallelLinear.weight_loader(param, loaded_weight)
VocabParallelEmbedding.weight_loader(param, loaded_weight)
```

### Packed Module Mapping

```python
Qwen3ForCausalLM.packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),      # shard_id = "q"
    "k_proj": ("qkv_proj", "k"),      # shard_id = "k"
    "v_proj": ("qkv_proj", "v"),      # shard_id = "v"
    "gate_proj": ("gate_up_proj", 0), # shard_id = 0
    "up_proj": ("gate_up_proj", 1),   # shard_id = 1
}
```

---

## 5. TENSOR PARALLEL TYPES

| Type | Input Sharded | Output Sharded | Requires AllReduce |
|------|---------------|----------------|--------------------|
| ColumnParallel | No | Yes | No (at next layer) |
| RowParallel | Yes | No | Yes (each layer) |
| Replicated | No | No | No |
| Vocab | N/A | Yes | Yes (AllGather) |

---

## 6. KEY ALGORITHMS

### Scheduler.schedule()

```
Phase 1 - Prefill:
  For each waiting sequence:
    If fits in batch AND blocks available:
      Allocate blocks
      Move to running
      Add to scheduled_seqs
  
  If scheduled_seqs not empty:
    Return scheduled_seqs, True (prefill phase)

Phase 2 - Decode:
  For each running sequence:
    While NOT can_append():
      Preempt another sequence to free blocks
    Add to scheduled_seqs
  
  Return scheduled_seqs, False (decode phase)
```

### BlockManager.allocate()

```
For each block in sequence:
  Compute hash(token_ids, prefix_hash)
  
  If hash in cache AND tokens match:
    Cache hit - reuse block
    Increment ref_count
    num_cached_tokens += block_size
  Else:
    Cache miss - allocate new block
    Update hash_to_block_id mapping
  
  Add block_id to sequence.block_table
```

### Attention Flow

```python
# Prefill (processing prompt)
flash_attn_varlen_func(q, k, v,
  cu_seqlens_q=cu_seqlens,
  max_seqlen_q=max_len,
  ...)

# Decode (generating tokens)
flash_attn_with_kvcache(q,
  k_cache, v_cache,
  cache_seqlens=context_lens,
  ...)
```

---

## 7. IMPORTANT CONSTANTS & DEFAULTS

```python
# KV Cache
block_size = 256                # Tokens per cache block
num_kvcache_blocks = computed   # From GPU memory

# Attention
head_dim = hidden_size // num_heads
rope_theta = 10000              # or 1000000
rope_scaling = None             # or custom

# Sampling
temperature > 1e-10             # No greedy sampling allowed
```

---

## 8. COMMON PATTERNS

### Pattern 1: Custom Weight Loader

```python
class MyLayer(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(...)
        self.weight.weight_loader = self.weight_loader
    
    def weight_loader(self, param, loaded_weight, shard_id=None):
        # Custom loading logic
        param.data.copy_(processed_weight)
```

### Pattern 2: Tensor Parallel Sharding

```python
# Column Parallel: outputs sharded
output_per_gpu = output_size // tp_size

# Row Parallel: inputs sharded  
input_per_gpu = input_size // tp_size
# Then all-reduce to sum

# Replica: same on all GPUs
```

### Pattern 3: Context-Aware Module

```python
from nanovllm.utils.context import get_context

class ContextAwareModule(nn.Module):
    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            # Prefill logic
        else:
            # Decode logic
```

---

## 9. DEBUGGING CHECKLIST

- [ ] Check `model.packed_modules_mapping` exists
- [ ] Verify all weight files are in model directory
- [ ] Check weight names match packed module keys
- [ ] Verify tensor parallel size matches weight loading
- [ ] Check KV cache is allocated before inference
- [ ] Verify CUDA graphs are captured (if not enforce_eager)
- [ ] Check block table is initialized for each sequence
- [ ] Verify context is set before attention layer
- [ ] Check scheduler has available blocks
- [ ] Verify sequences are properly deallocated

---

## 10. PERFORMANCE TIPS

1. **Tensor Parallelism**: Use multiple GPUs with `tensor_parallel_size > 1`
2. **CUDA Graphs**: Disable with `enforce_eager=True` only for debugging
3. **Batch Size**: Increase `max_num_seqs` for higher throughput
4. **KV Cache**: Monitor with `num_kvcache_blocks` calculation
5. **Memory**: Adjust `gpu_memory_utilization` based on needs
6. **Prefill/Decode**: Natural batching prevents head-of-line blocking

---

## 11. ERROR MESSAGES & SOLUTIONS

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `assert param not in loaded` | Weight already loaded | Check for duplicates |
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch size or lower memory util |
| `shape mismatch in load` | TP mismatch | Check tensor_parallel_size |
| `block_id not in blocks` | KV cache issue | Check block allocation |
| `seq_id not found` | Sequence cleanup issue | Check deallocate logic |

---

## 12. MODULE IMPORTS REFERENCE

```python
# Public API
from nanovllm import LLM, SamplingParams

# Config
from nanovllm.config import Config

# Model
from nanovllm.models.qwen3 import Qwen3ForCausalLM

# Engine
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# Layers
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm

# Utils
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import get_context, set_context
```

---

**End of Quick Reference**
