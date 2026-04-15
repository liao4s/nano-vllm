# Nano-vLLM Quick Start & Architecture Guide

## 🚀 Quick Start

### Installation
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### Basic Usage
```python
from nanovllm import LLM, SamplingParams

# Initialize model
llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)

# Create sampling parameters
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Generate
prompts = ["Hello, how are you?", "Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params)

# Access results
for output in outputs:
    print(output['text'])
    print(output['token_ids'])
```

---

## 📁 Project Structure Quick Reference

```
nanovllm/
├── engine/           # Request scheduling & execution
│   ├── llm_engine.py      → Main orchestrator, generate() entry point
│   ├── model_runner.py    → GPU model execution, CUDA graphs
│   ├── scheduler.py       → Prefill/decode scheduling
│   ├── sequence.py        → Single request state tracking
│   └── block_manager.py   → KV cache block allocation with prefix caching
│
├── models/           # Model architectures
│   └── qwen3.py           → Complete Qwen3 transformer
│
├── layers/           # Neural network layers
│   ├── attention.py       → Flash-Attention + KV cache storage
│   ├── linear.py          → Tensor parallel linear layers
│   ├── embed_head.py      → Vocab parallel embedding & LM head
│   ├── rotary_embedding.py → RoPE implementation
│   ├── layernorm.py       → RMSNorm
│   ├── activation.py      → SiLU + multiply
│   └── sampler.py         → Token sampling
│
└── utils/            # Utilities
    ├── loader.py          → Weight loading from safetensors
    └── context.py         → Thread-local execution context
```

---

## 🔑 Key Concepts

### 1. Two-Phase Scheduling

**Prefill Phase** (Initial prompt processing):
- Process entire prompt at once
- All tokens in parallel (high throughput)
- Example: 1024-token prompt processed in 1 pass

**Decode Phase** (Token generation):
- Generate 1 token per sequence per step
- Sequential (latency-optimized)
- Repeated until all sequences reach max_tokens

### 2. KV Cache Management

**Block-based Storage**:
- Split cache into 256-token blocks
- Track via `block_table`: list of physical block IDs
- Enables dynamic memory reuse

**Prefix Caching**:
- Hash-based block deduplication
- If two sequences share same prefix tokens → share blocks
- Saves memory automatically

### 3. Tensor Parallelism

**How it works**:
- Model split across N GPUs
- Each GPU computes N-th fraction of heads
- Attention heads sharded (36 heads → 18 per GPU with TP=2)
- Embedding vocab sharded by range

**Weight Loading**:
- Each tensor parallel layer has `weight_loader` callback
- Handles slicing weights for each rank during loading

### 4. CUDA Graphs

**What**: Pre-recorded GPU kernel sequences

**When Used**: Decode phase (small batches, fixed structure)

**Benefit**: Eliminates Python interpreter overhead
- Standard execution: Python → CUDA (overhead ~5-10ms)
- CUDA graph replay: Just GPU (overhead ~0.1ms)

**Batch Sizes**: 1, 2, 4, 8, 16, 32, ... (up to max_num_seqs)

---

## 🔄 Request Lifecycle

```
1. User adds prompt
   └→ tokenize() → Sequence object → waiting queue

2. Prefill phase starts (if space)
   ├→ Allocate KV cache blocks (with prefix caching)
   ├→ Run model on entire prompt
   └→ Move to running queue

3. Decode loop (each step)
   ├→ Get 1 token from each running sequence
   ├→ Run model (using CUDA graph if possible)
   ├→ Sample next token
   ├→ Append to sequence
   └→ Check if finished (EOS or max_tokens)

4. Request complete
   └→ Return generated tokens + text
```

---

## 📊 Model Loading Flow

```
Model Path
    ↓
AutoConfig.from_pretrained(path)  → Get config
    ↓
Qwen3ForCausalLM(config)  → Create empty model
    ↓
load_model(model, path)  → Load weights from .safetensors files
    ├─ For each .safetensors file:
    │  ├─ Check if weight name in packed_modules_mapping
    │  │  ├─ YES: Call custom weight_loader with shard_id
    │  │  │    (e.g., load q_proj into qkv_proj[0])
    │  │  └─ NO: Standard copy via default_weight_loader
    │  └─ Move weight to GPU
    ↓
model.to_gpu()
    ↓
Ready for inference!
```

---

## 🎯 Qwen3 Model Architecture

### Layer Stack

```
Input IDs → Embedding
           ↓
        [ResBlock ×32]  ← Repeated 32 times
           ├─ RMSNorm
           ├─ Attention (QKV fused)
           │  ├─ Q, K, V projections
           │  ├─ RoPE (rotary position embeddings)
           │  └─ Flash-Attention + KV cache
           ├─ RMSNorm
           └─ MLP (gate+up fused)
              ├─ Gate-Up projection
              ├─ SiLU activation
              └─ Down projection
           ↓
        RMSNorm
           ↓
        LM Head (vocab projection)
           ↓
        Logits (vocab_size,)
           ↓
        Sampler
           ↓
        Next Token ID
```

### Weight Fusion (Packed Modules)

**Why**: Reduce memory bandwidth, improve cache locality

**Original**: 5 separate projections
- Q projection (Q = hidden @ W_q.T)
- K projection (K = hidden @ W_k.T)
- V projection (V = hidden @ W_v.T)
- Gate projection (G = hidden @ W_gate.T)
- Up projection (U = hidden @ W_up.T)

**Fused**: 2 projections
- QKV projection (QKV = hidden @ W_qkv.T) ← concatenate W_q, W_k, W_v
- Gate-Up projection (GU = hidden @ W_gu.T) ← concatenate W_gate, W_up

**Loading Process**:
```python
# In Qwen3ForCausalLM
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),       # Map source → (target, shard_id)
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),  # 0 = first half
    "up_proj": ("gate_up_proj", 1),    # 1 = second half
}

# During loading, weight_loader() handles the mapping
```

---

## ⚙️ Configuration Parameters

```python
Config(
    model="/path/to/model",            # Model directory
    max_num_batched_tokens=16384,      # Max tokens per batch
    max_num_seqs=512,                  # Max concurrent requests
    max_model_len=4096,                # Max sequence length
    gpu_memory_utilization=0.9,        # GPU memory % for KV cache
    tensor_parallel_size=1,            # Number of GPUs (1-8)
    enforce_eager=False,               # Skip CUDA graphs (for debugging)
    kvcache_block_size=256,            # Tokens per KV block
    num_kvcache_blocks=-1,             # Auto-computed
)
```

### Tuning Tips

**Throughput (batch processing)**:
- Increase `max_num_batched_tokens` (if GPU memory allows)
- Increase `max_num_seqs`
- Disable `enforce_eager` (use CUDA graphs)

**Memory Usage**:
- Decrease `gpu_memory_utilization` (e.g., 0.7)
- Decrease `kvcache_block_size` (e.g., 128)
- Decrease `max_model_len` (e.g., 2048)

**Single Request Latency**:
- Enable `enforce_eager=True` (cold start, then improve)
- Use CUDA graphs (default, off by default only for debugging)
- Reduce `max_num_seqs` if needed

---

## 🔍 Debugging & Profiling

### Enable Eager Execution (No CUDA Graphs)
```python
llm = LLM(model_path, enforce_eager=True)  # Slower but easier to debug
```

### Monitor Memory
```python
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
# ... run inference ...
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
print(f"Peak memory: {peak / 1e9:.2f} GB")
```

### Inspect Sequences
```python
# Inside llm_engine.py
scheduler.running  # List of Sequence objects
for seq in scheduler.running:
    print(f"Seq {seq.seq_id}: {len(seq)} tokens, status={seq.status}")
```

---

## 🚀 Performance Benchmarks

**Test Setup**:
- Hardware: RTX 4070 Laptop (8GB VRAM)
- Model: Qwen3-0.6B
- Requests: 256 sequences
- Input: 100-1024 tokens
- Output: 100-1024 tokens (avg ~524 tokens)

**Results**:
| Engine | Total Tokens | Time | Throughput |
|--------|-------------|------|-----------|
| vLLM | 133,966 | 98.37s | 1361.84 tok/s |
| Nano-vLLM | 133,966 | **93.41s** | **1434.13 tok/s** ✅ |

**Speedup**: +5.3% vs vLLM on same hardware

---

## 📚 Example Code Snippets

### Basic Generation
```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "~/huggingface/Qwen3-0.6B/"
llm = LLM(model_path, enforce_eager=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "What is machine learning?"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(
    [inputs],
    SamplingParams(temperature=0.7, max_tokens=256)
)

print(outputs[0]['text'])
```

### Batch Generation with Different Params
```python
prompts = [
    "Explain AI",
    "What is Python?",
    "List 5 fruits",
]
sampling_params = [
    SamplingParams(temperature=0.7, max_tokens=128),
    SamplingParams(temperature=0.5, max_tokens=256),
    SamplingParams(temperature=1.0, max_tokens=100),
]

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt}")
    print(f"A: {output['text']}\n")
```

### Using Tensor Parallelism
```python
# Requires 2 GPUs
llm = LLM(model_path, tensor_parallel_size=2)
outputs = llm.generate(prompts, sampling_params)
# Model automatically sharded across 2 GPUs
```

### Disable Prefix Caching (if needed)
```python
# Currently always enabled, but can modify block_manager.py:
# if h != -1:  # Skip hash update to disable prefix caching
#     block.update(h, token_ids)
#     self.hash_to_block_id[h] = block_id
```

---

## 🤝 How to Extend

### Add Support for New Model (e.g., Llama3)

1. **Create model file**: `nanovllm/models/llama3.py`
```python
class Llama3Attention(nn.Module): ...
class Llama3MLP(nn.Module): ...
class Llama3DecoderLayer(nn.Module): ...
class Llama3Model(nn.Module): ...
class Llama3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        # Define weight fusion mappings
    }
```

2. **Update model_runner.py**:
```python
def create_model(hf_config):
    if hf_config.model_type == "qwen":
        return Qwen3ForCausalLM(hf_config)
    elif hf_config.model_type == "llama":
        return Llama3ForCausalLM(hf_config)
    else:
        raise ValueError(f"Unknown model: {hf_config.model_type}")

# Then use:
self.model = create_model(config.hf_config)
```

3. **Test**: Run example.py with new model

---

## ❓ FAQ

**Q: Why is prefix caching enabled by default?**
A: It transparently saves memory when multiple requests share prompts (e.g., same system prompt). No downside.

**Q: Can I use this with smaller models?**
A: Yes! Works with any size. Memory footprint scales with model size.

**Q: How do I use quantization?**
A: Current version loads full precision. To add quantization, modify weight loading in utils/loader.py.

**Q: Does it support multi-turn conversations?**
A: Yes! Pass chat templates through tokenizer.apply_chat_template(), then generate().

**Q: What about streaming?**
A: Current API returns full outputs. Streaming can be added by modifying llm_engine.generate() to yield progressively.

---

## 📖 Further Reading

- **CUDA Graphs**: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- **Flash Attention**: https://arxiv.org/abs/2205.14135
- **Prefix Caching**: https://github.com/vllm-project/vllm/blob/main/docs/source/index.md
- **Tensor Parallelism**: https://arxiv.org/abs/2104.04473
- **vLLM**: https://arxiv.org/abs/2309.06180

---

**Happy inferencing! 🚀**
