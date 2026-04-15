# Nano-vLLM Codebase Exploration Report

## 1. OVERALL PROJECT STRUCTURE

### Directory Tree
```
nano-vllm/
├── nanovllm/                    # Main package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration dataclass
│   ├── llm.py                   # Main LLM class (wrapper around LLMEngine)
│   ├── sampling_params.py       # Sampling parameters dataclass
│   ├── engine/                  # Core inference engine
│   │   ├── llm_engine.py        # LLMEngine - main orchestrator
│   │   ├── model_runner.py      # ModelRunner - model execution
│   │   ├── scheduler.py         # Scheduler - request scheduling
│   │   ├── sequence.py          # Sequence - request tracking
│   │   └── block_manager.py     # BlockManager - KV-cache management
│   ├── layers/                  # Neural network layers
│   │   ├── attention.py         # Attention with FlashAttention
│   │   ├── linear.py            # Distributed linear layers
│   │   ├── activation.py        # SiLU + multiplication
│   │   ├── rotary_embedding.py  # RoPE (Rotary Positional Embeddings)
│   │   ├── layernorm.py         # RMSNorm
│   │   ├── embed_head.py        # Embedding & LM Head (distributed)
│   │   └── sampler.py           # Token sampling
│   ├── models/                  # Model architectures
│   │   └── qwen3.py             # Qwen3 model implementation
│   └── utils/                   # Utilities
│       ├── loader.py            # Model weight loader
│       └── context.py           # Thread-local context manager
├── qwen3.5/                     # Qwen3.5 model config/weights
│   ├── qwen3.5-35B-A3B-config   # Model config (JSON)
│   ├── generation_config.json   # Generation parameters
│   ├── tokenizer_config.json    # Tokenizer config
│   ├── preprocessor_config.json # Preprocessor config
│   ├── tokenizer.json           # Tokenizer vocabulary
│   ├── merges.txt               # BPE merges
│   ├── qwen3.5-35B-A3B_model.safetensors.index.json  # Model index
│   ├── qwen_roofline.py         # Roofline analysis script
│   └── README.md                # Model card
├── example.py                   # Usage example
├── bench.py                     # Benchmark script
├── pyproject.toml               # Project metadata
└── README.md                    # Main documentation
```

### Total File Count
- **Python files**: 13 core files + examples
- **Lines of core code**: ~1,200 LOC (as mentioned in README)

---

## 2. MODEL LOADING INFRASTRUCTURE

### Entry Point: `nanovllm/llm.py` (5 lines)
```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```
**Purpose**: Simple wrapper providing public API

### Main Orchestrator: `nanovllm/engine/llm_engine.py` (94 lines)

**Key Components**:
1. **Initialization**:
   - Creates `Config` object from model path and kwargs
   - Initializes distributed processes for tensor parallelism
   - Creates `ModelRunner` processes for multi-GPU support
   - Loads tokenizer via `AutoTokenizer.from_pretrained()`

2. **Request Flow**:
   ```
   add_request() → Sequence created → Scheduler adds to queue
           ↓
   step() → scheduler.schedule() → ModelRunner.run()
           ↓
   scheduler.postprocess() → token_id update, check EOS/max_tokens
   ```

3. **Key Methods**:
   - `__init__()`: Sets up engine, tokenizer, scheduler
   - `add_request()`: Tokenizes prompt, creates Sequence
   - `step()`: Single inference step (prefill or decode)
   - `generate()`: Main inference loop with tqdm progress

### Configuration: `nanovllm/config.py` (27 lines)

```python
@dataclass
class Config:
    model: str                          # Model path
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None # Loaded from model
    eos: int = -1                       # Filled from tokenizer
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1        # Auto-calculated
```

**Validation** (`__post_init__`):
- Model path must exist
- KV-cache block size must be multiple of 256
- Loads HuggingFace config via `AutoConfig.from_pretrained()`
- Caps max_model_len to config's max_position_embeddings

### Model Weight Loader: `nanovllm/utils/loader.py` (29 lines)

```python
def load_model(model: nn.Module, path: str):
    """Load weights from safetensors files"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Check for packed module mapping (e.g., qkv_proj)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # Default: direct loading
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

**Features**:
- Loads from all `*.safetensors` files in model directory
- Supports packed module mapping (q/k/v → qkv_proj)
- Uses custom `weight_loader` method on each parameter
- Supports tensor parallelism via shard IDs

---

## 3. QWEN3 MODEL IMPLEMENTATION

### Complete File Path
`nanovllm/models/qwen3.py` (216 lines)

### Architecture Overview

**Hierarchy**:
```
Qwen3ForCausalLM (Main entry point)
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding
│   └── layers: ModuleList[Qwen3DecoderLayer] × num_hidden_layers
│       ├── self_attn: Qwen3Attention
│       │   ├── qkv_proj: QKVParallelLinear
│       │   ├── o_proj: RowParallelLinear
│       │   ├── rotary_emb: RotaryEmbedding
│       │   └── attn: Attention (with FlashAttention)
│       ├── mlp: Qwen3MLP
│       │   ├── gate_up_proj: MergedColumnParallelLinear
│       │   ├── act_fn: SiluAndMul
│       │   └── down_proj: RowParallelLinear
│       ├── input_layernorm: RMSNorm
│       └── post_attention_layernorm: RMSNorm
└── lm_head: ParallelLMHead
```

### 3.1 Qwen3ForCausalLM (Lines 185-216)

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

**Key Features**:
- Packed module mapping for efficient weight loading
- Q, K, V projections fused into single qkv_proj
- Gate and Up projections fused into gate_up_proj
- Tied embeddings (lm_head shares weights with embed_tokens if configured)

**Methods**:
- `forward(input_ids, positions)`: Returns hidden states
- `compute_logits(hidden_states)`: Returns logits for sampling

### 3.2 Qwen3Model (Lines 161-183)

```python
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
    
    def forward(input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**Key Features**:
- Parallel embedding for distributed inference
- Residual connections carried through all layers
- Final RMSNorm before logits

### 3.3 Qwen3DecoderLayer (Lines 119-159)

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def forward(positions, hidden_states, residual=None):
        # Pre-norm architecture
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**Key Features**:
- Pre-norm architecture (norm before each sub-layer)
- Residual connections for gradient flow
- Efficient fused norm-add operation

### 3.4 Qwen3Attention (Lines 14-88)

```python
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
    
    def forward(positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # Reshape for multi-head
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)
        
        # Optional normalization
        if not qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

**Configuration Parameters**:
- `hidden_size`: Model hidden dimension
- `num_heads`: Total attention heads (distributed across TP)
- `num_kv_heads`: KV heads for GQA (Grouped Query Attention)
- `max_position`: Maximum sequence length (4096 * 32 = 131072 default)
- `head_dim`: Dimension per head
- `qkv_bias`: Whether to use bias (configurable)
- `rope_theta`: Base for RoPE
- `rope_scaling`: For extending context (not yet supported)

**Key Features**:
- Multi-query attention (MQA) support via separate KV heads
- Q/K normalization (if no bias)
- RoPE for position encoding
- FlashAttention backend via custom Attention layer

### 3.5 Qwen3MLP (Lines 90-117)

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # Gate and Up
            bias=False
        )
        self.down_proj = RowParallelLinear(...)
        self.act_fn = SiluAndMul()
    
    def forward(x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)  # SiLU(gate) * up
        x = self.down_proj(x)
        return x
```

**Key Features**:
- Gated Linear Unit (GLU) architecture: SiLU(gate) * up
- Gate and Up projections fused for efficiency
- Only supports SiLU activation

---

## 4. MODEL REGISTRATION & LOADING

### How Models Are Loaded

**Current Design**: **Hardcoded single model**

The architecture currently only supports **Qwen3**, with no model registry:

1. **In `ModelRunner.__init__()` (model_runner.py, line 31)**:
```python
self.model = Qwen3ForCausalLM(hf_config)
```
- Directly instantiates Qwen3ForCausalLM
- Uses `hf_config` from `AutoConfig.from_pretrained(model_path)`

2. **Model Loading Flow**:
```
LLMEngine.__init__()
  ↓ Creates Config(model_path)
  ↓ config.hf_config = AutoConfig.from_pretrained(model_path)
  ↓ ModelRunner(config, rank, ...)
  ↓ self.model = Qwen3ForCausalLM(hf_config)
  ↓ load_model(self.model, config.model)
```

### Integration Point: `nanovllm/engine/model_runner.py` (Line 31)

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    ...
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    ...
```

### Packed Modules Mapping

The `Qwen3ForCausalLM.packed_modules_mapping` tells the loader how to unpack weights:

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),      # q_proj from weights → qkv_proj, shard "q"
    "k_proj": ("qkv_proj", "k"),      # k_proj from weights → qkv_proj, shard "k"
    "v_proj": ("qkv_proj", "v"),      # v_proj from weights → qkv_proj, shard "v"
    "gate_proj": ("gate_up_proj", 0), # gate_proj → gate_up_proj, shard 0
    "up_proj": ("gate_up_proj", 1),   # up_proj → gate_up_proj, shard 1
}
```

**How it works**:
1. Loader searches for these keys in weight names
2. If found, uses custom `weight_loader(param, tensor, shard_id)` from the layer
3. Each layer type (QKVParallelLinear, MergedColumnParallelLinear) implements shard-aware loading

---

## 5. EXAMPLE DIRECTORY STRUCTURE

### `example.py` (34 lines)

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # Model path
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Initialize engine
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    
    # Setup sampling
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # Prepare prompts with chat template
    prompts = ["introduce yourself", "list all prime numbers within 100"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```

**Usage**:
- Expects model at `~/huggingface/Qwen3-0.6B/`
- Uses chat template from tokenizer
- Returns list of dicts with `text` and `token_ids` keys

### `bench.py` (33 lines)

```python
from nanovllm import LLM, SamplingParams

def main():
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    
    # Generate random token sequences
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]
    
    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())
    
    # Benchmark
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Throughput: {throughput:.2f} tok/s")
```

**Features**:
- 256 sequences with variable input/output lengths
- Warmup run before benchmarking
- Measures token/second throughput

---

## 6. KEY COMPONENTS SUMMARY

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Public API** | `llm.py` | 5 | Wrapper |
| **Main Engine** | `engine/llm_engine.py` | 94 | Orchestrates inference |
| **Config** | `config.py` | 27 | Configuration |
| **Model Runner** | `engine/model_runner.py` | 252 | Model execution |
| **Scheduler** | `engine/scheduler.py` | 72 | Request scheduling |
| **Sequence** | `engine/sequence.py` | 84 | Request tracking |
| **Block Manager** | `engine/block_manager.py` | 113 | KV-cache management |
| **Model** | `models/qwen3.py` | 216 | Qwen3 architecture |
| **Attention** | `layers/attention.py` | 76 | Attention + FlashAttention |
| **Linear** | `layers/linear.py` | 154 | Distributed linear layers |
| **Activation** | `layers/activation.py` | 15 | SiLU + mul |
| **RoPE** | `layers/rotary_embedding.py` | 62 | Rotary embeddings |
| **RMSNorm** | `layers/layernorm.py` | 51 | Normalization |
| **Embed/Head** | `layers/embed_head.py` | 67 | Embeddings & LM head |
| **Sampler** | `layers/sampler.py` | 16 | Token sampling |
| **Loader** | `utils/loader.py` | 29 | Weight loading |
| **Context** | `utils/context.py` | 28 | Thread-local context |

**Total**: ~1,200 lines core code

---

## 7. CONFIGURATION HANDLING

### Config Sources

1. **Model Config** (`AutoConfig.from_pretrained(model_path)`):
   - Loaded from model's `config.json`
   - For Qwen3: `Qwen3Config` from transformers
   - Contains: hidden_size, num_heads, vocab_size, etc.

2. **Runtime Config** (`Config` dataclass in config.py):
   - Passed as kwargs to LLM()
   - Includes: batch_size, sequence_length, GPU memory utilization
   - Auto-calculates: num_kvcache_blocks based on GPU memory

3. **Sampling Config** (`SamplingParams`):
   - temperature, max_tokens, ignore_eos
   - Per-request configuration

### Example Configuration

For Qwen3-0.6B (from example):
```python
llm = LLM(
    "~/huggingface/Qwen3-0.6B/",
    enforce_eager=True,              # No CUDA graph capture
    tensor_parallel_size=1           # Single GPU
)
```

For Qwen3.5-35B (from qwen3.5-35B-A3B-config):
```python
{
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "num_hidden_layers": 40,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rope_theta": 10000000,
    # ... MoE, linear attention configs
}
```

---

## 8. DATA FLOW

### Inference Pipeline

```
User calls: llm.generate(prompts, sampling_params)
    ↓
LLMEngine.generate()
    ↓ [For each prompt]
    LLMEngine.add_request()
        ↓ tokenizer.encode()
        ↓ Sequence(tokens, sampling_params)
        ↓ scheduler.add()
    ↓ [Main loop]
    LLMEngine.step()
        ↓
        Scheduler.schedule()
            ├→ Prefill: Load new sequences from waiting queue
            └→ Decode: Continue running sequences
        ↓
        ModelRunner.run(seqs, is_prefill)
            ├→ prepare_prefill() or prepare_decode()
            ├→ run_model() [with optional CUDA graph]
            ├→ sampler() [sample next tokens]
            └→ return token_ids
        ↓
        Scheduler.postprocess()
            ├→ seq.append_token()
            ├→ Check if finished (EOS or max_tokens)
            └→ Move to finished or continue
    ↓ [Until all sequences finished]
    Return: List of {text, token_ids}
```

### Tensor Parallelism Flow

```
rank=0 (main process)          rank>0 (distributed processes)
    ↓                                  ↓
LLMEngine (rank 0 only)        ModelRunner (rank i)
    ↓                                  ↓
Tokenizer                      Setup NCCL + CUDA
Add requests                   Initialize dist.process_group
    ↓                                  ↓
Scheduler (rank 0)             Loop: wait for SharedMemory
    ↓                                  ↓
ModelRunner.call()  ──pickle──→ Read SharedMemory
Write to SharedMemory                 ↓
Signal events                  Execute model (distributed)
    ↓                                  ↓
Wait for results   ←──return── all_reduce/gather (NCCL)
```

---

## 9. IMPORTANT NOTES

### Qwen3 Specifics
- **Packed modules**: Q/K/V and gate/up projections are fused
- **GQA**: Uses fewer KV heads than Q heads (e.g., 16 Q heads, 2 KV heads)
- **Pre-norm**: Applies normalization before each sub-layer
- **RoPE**: Uses theta=1000000 by default for Qwen3

### Tensor Parallelism Details
- Uses PyTorch distributed (NCCL on GPU, gloo on CPU)
- TCP localhost:2333 for process communication
- SharedMemory for rank 0 to rank>0 communication
- Each layer implements:
  - `ColumnParallelLinear`: Output distributed across ranks
  - `RowParallelLinear`: Input distributed across ranks
  - Vocabulary parallelism for embeddings

### KV-Cache Management
- Block-based (256 token blocks)
- Supports prefix caching via hashing
- Auto-calculated based on GPU memory utilization
- Allocated contiguously on GPU

### Optimizations
1. **FlashAttention**: Via `flash_attn_varlen_func` and `flash_attn_with_kvcache`
2. **CUDA Graphs**: Capture decode phase for faster execution
3. **Torch Compilation**: `@torch.compile` on hot functions (SiLU, RMSNorm, RoPE)
4. **Prefix Caching**: Hash-based block reuse across requests
5. **Batch Processing**: Packs multiple sequences for efficiency

---

## 10. FILES CHECKLIST

### Model Files
- [x] `nanovllm/models/qwen3.py` - Complete Qwen3 implementation
- [x] `nanovllm/config.py` - Configuration
- [x] `nanovllm/utils/loader.py` - Weight loading with packed module support

### Engine Files  
- [x] `nanovllm/engine/llm_engine.py` - Main orchestrator
- [x] `nanovllm/engine/model_runner.py` - Model execution + CUDA graphs
- [x] `nanovllm/engine/scheduler.py` - Request scheduling
- [x] `nanovllm/engine/sequence.py` - Sequence tracking
- [x] `nanovllm/engine/block_manager.py` - KV-cache + prefix caching

### Layer Files
- [x] `nanovllm/layers/attention.py` - Attention with FlashAttention
- [x] `nanovllm/layers/linear.py` - Distributed linear layers (Col, Row, QKV)
- [x] `nanovllm/layers/activation.py` - SiLU + mul
- [x] `nanovllm/layers/rotary_embedding.py` - RoPE
- [x] `nanovllm/layers/layernorm.py` - RMSNorm with fused ops
- [x] `nanovllm/layers/embed_head.py` - Parallel embeddings & LM head
- [x] `nanovllm/layers/sampler.py` - Token sampling

### Utility Files
- [x] `nanovllm/utils/context.py` - Thread-local context for forward passes
- [x] `nanovllm/__init__.py` - Public API (LLM, SamplingParams)
- [x] `nanovllm/sampling_params.py` - Sampling configuration

### Example Files
- [x] `example.py` - Basic usage example
- [x] `bench.py` - Benchmark script

### Configuration Files
- [x] `qwen3.5/qwen3.5-35B-A3B-config` - Model config (JSON)
- [x] `qwen3.5/generation_config.json` - Generation parameters
- [x] `qwen3.5/tokenizer_config.json` - Tokenizer config
- [x] Various tokenizer and model metadata files

