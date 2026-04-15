# Nano-vLLM Codebase Analysis

## Executive Summary

**Nano-vLLM** is a lightweight vLLM implementation built from scratch (~1,200 lines of Python code). It provides fast offline LLM inference with support for:
- Prefix caching (KV cache block sharing)
- Tensor parallelism (for multi-GPU inference)
- CUDA graph capture (for decode performance)
- Flash Attention integration

The codebase currently supports **Qwen3 models** with a clean, extensible architecture for adding more models.

---

## 1. PROJECT STRUCTURE

### Directory Layout
```
nano-vllm/
├── nanovllm/                          # Main package (~1,358 lines)
│   ├── __init__.py                    # Public API: LLM, SamplingParams
│   ├── llm.py                         # LLM class (thin wrapper around LLMEngine)
│   ├── config.py                      # Config dataclass
│   ├── sampling_params.py             # SamplingParams dataclass
│   ├── models/
│   │   └── qwen3.py                   # Qwen3ForCausalLM implementation (216 lines)
│   ├── engine/
│   │   ├── llm_engine.py              # Main orchestration engine (94 lines)
│   │   ├── model_runner.py            # GPU model execution (252 lines)
│   │   ├── scheduler.py               # Request scheduling & block allocation (72 lines)
│   │   ├── sequence.py                # Sequence/request management (84 lines)
│   │   └── block_manager.py           # KV cache block management (113 lines)
│   ├── layers/
│   │   ├── attention.py               # Attention with flash-attn + Triton KV cache (76 lines)
│   │   ├── linear.py                  # Tensor parallelism linear layers (154 lines)
│   │   ├── layernorm.py               # RMSNorm with torch.compile (51 lines)
│   │   ├── embed_head.py              # VocabParallelEmbedding, ParallelLMHead (67 lines)
│   │   ├── activation.py              # SiluAndMul (torch.compile) (15 lines)
│   │   ├── rotary_embedding.py        # Rotary embeddings (62 lines)
│   │   └── sampler.py                 # Temperature sampling (16 lines)
│   └── utils/
│       ├── context.py                 # Global context for prefill/decode (28 lines)
│       └── loader.py                  # Model weight loading from safetensors (29 lines)
├── qwen3.5/                           # Qwen3.5-35B model weights & config
│   ├── qwen3.5-35B-A3B-config         # HuggingFace config
│   ├── qwen_roofline.py               # Performance analysis script
│   ├── tokenizer.json                 # BPE tokenizer
│   └── *.safetensors                  # Model weights (sharded)
├── example.py                         # Usage example
├── bench.py                           # Benchmark script
└── pyproject.toml                     # Package metadata
```

### Key Statistics
- **Total lines**: ~1,358 lines of Python
- **Models**: Qwen3ForCausalLM (only Qwen3 currently)
- **Dependencies**: torch, transformers, flash-attn, triton, xxhash, safetensors

---

## 2. MODEL LOADING INFRASTRUCTURE

### Entry Point: LLM Initialization Flow

```
LLM(model_path, **kwargs)
  ↓
LLMEngine.__init__(model_path, **kwargs)
  ├─→ Config(model_path, **config_kwargs)
  │   └─→ AutoConfig.from_pretrained(model_path)
  ├─→ ModelRunner(config, rank=0, events=[])  # Main process
  │   ├─→ Qwen3ForCausalLM(hf_config)
  │   ├─→ load_model(model, config.model)
  │   ├─→ allocate_kv_cache()
  │   └─→ capture_cudagraph()
  ├─→ AutoTokenizer.from_pretrained(config.model)
  └─→ Scheduler(config)
```

### Config Management (`config.py`)

**File**: `/Users/water/work/code/LALearning/nano-vllm/nanovllm/config.py`

```python
@dataclass
class Config:
    model: str                          # HuggingFace model path
    max_num_batched_tokens: int = 16384 # Max tokens per batch
    max_num_seqs: int = 512             # Max sequences per batch
    max_model_len: int = 4096           # Max sequence length
    gpu_memory_utilization: float = 0.9 # GPU memory usage target
    tensor_parallel_size: int = 1       # TP degree (1-8)
    enforce_eager: bool = False         # Disable CUDA graphs
    hf_config: AutoConfig | None = None # Cached HF config
    eos: int = -1                       # EOS token ID (set at runtime)
    kvcache_block_size: int = 256       # KV cache block size
    num_kvcache_blocks: int = -1        # Num KV cache blocks (auto-allocated)
```

**Key features**:
- Validates `model` is a directory (not a model name)
- `kvcache_block_size` must be divisible by 256
- `max_num_batched_tokens >= max_model_len`
- Automatically loads HuggingFace config and caps `max_model_len` to model's max

### Model Weight Loading (`utils/loader.py`)

**File**: `/Users/water/work/code/LALearning/nano-vllm/nanovllm/utils/loader.py` (29 lines)

```python
def load_model(model: nn.Module, path: str):
    """Load safetensors weights from path to model."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Handle packed weights (Q, K, V combined into qkv_proj)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # Standard weight loading
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

**Key features**:
- Loads from **safetensors** files (not PyTorch pickle)
- Supports **packed modules** via `packed_modules_mapping` on model
- Each parameter can have a custom `weight_loader` method
- CPU loading then device transfer handled by parameter's `weight_loader`

---

## 3. QWEN3 MODEL IMPLEMENTATION

### File: `nanovllm/models/qwen3.py` (216 lines)

#### Qwen3ForCausalLM Architecture

```
Input IDs (batch_size, seq_len)
  ↓
Qwen3ForCausalLM
  ├─→ Qwen3Model
  │   ├─→ VocabParallelEmbedding (vocab_parallel)
  │   ├─→ [Qwen3DecoderLayer × num_hidden_layers]
  │   │   ├─→ Qwen3Attention
  │   │   │   ├─→ QKVParallelLinear (q, k, v projection)
  │   │   │   ├─→ RMSNorm (q, k normalization if no bias)
  │   │   │   ├─→ Rotary Embedding
  │   │   │   ├─→ Flash Attention + KV cache storage
  │   │   │   └─→ RowParallelLinear (output projection)
  │   │   ├─→ RMSNorm (input layernorm)
  │   │   ├─→ Qwen3MLP
  │   │   │   ├─→ MergedColumnParallelLinear (gate_up_proj)
  │   │   │   ├─→ SiluAndMul activation
  │   │   │   └─→ RowParallelLinear (down_proj)
  │   │   └─→ RMSNorm (post-attention layernorm)
  │   └─→ RMSNorm (final norm)
  └─→ ParallelLMHead (lm_head)
       └─→ Logits (batch_size, seq_len, vocab_size)
```

#### Class Implementations

##### 1. Qwen3Attention (lines 14-87)

```python
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position=4096*32,
                 head_dim=None, rms_norm_eps=1e-6, qkv_bias=False, 
                 rope_theta=10000, rope_scaling=None):
        # Tensor parallelism: divide heads by TP size
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads // tp_size
        
        # Projections
        self.qkv_proj = QKVParallelLinear(...)  # Combined Q, K, V
        self.o_proj = RowParallelLinear(...)    # Output projection
        
        # Optional normalization (if no bias)
        if not qkv_bias:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        
        # Rotary embeddings
        self.rotary_emb = get_rope(...)
        
        # Flash attention + KV cache
        self.attn = Attention(...)
    
    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size])
        q, k, v = reshape_to_heads(q, k, v)
        
        if not qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

**Key features**:
- **Tensor parallel**: Q, K, V heads divided by TP size
- **QKV projection**: Combined into single `qkv_proj` for efficiency
- **Q, K normalization**: Applied only when `qkv_bias=False`
- **Rotary embeddings**: Rope theta, scaling support
- **Flash attention**: Integrates with KV cache

##### 2. Qwen3MLP (lines 90-116)

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        # Fused gate + up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size])
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()  # Fused SiLU * gate
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)    # [B, N, 2*intermediate]
        x = self.act_fn(gate_up)          # SiLU(gate) * up
        x = self.down_proj(x)             # Project back
        return x
```

**Key features**:
- **Merged gate+up**: Single `MergedColumnParallelLinear` with 2 outputs
- **SiluAndMul**: Fused activation (chunk, apply SiLU to first, multiply by second)
- **RowParallelLinear**: All-reduce for TP gather

##### 3. Qwen3DecoderLayer (lines 119-158)

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
    
    def forward(self, positions, hidden_states, residual):
        # Pre-norm residual stream
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        
        # Post-attention norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**Key features**:
- **Pre-norm architecture**: LayerNorm applied before each sublayer
- **Residual stream tracking**: Maintains separate `residual` for efficiency
- **RMSNorm supports residual add**: `norm(x, residual)` returns `(normed, x+residual)`

##### 4. Qwen3Model (lines 161-182)

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)
    
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

##### 5. Qwen3ForCausalLM (lines 185-215)

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        
        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**Key features**:
- **packed_modules_mapping**: Maps HF weight names to packed layer structure
- **Tied embeddings**: Optionally shares embedding/head weights
- **Separate compute_logits**: Called after model forward for decoding

---

## 4. MODEL REGISTRATION & LOADING MECHANISM

### Current Model Registration

**Location**: `nanovllm/engine/model_runner.py` (lines 31-32)

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    ...
    self.model = Qwen3ForCausalLM(hf_config)  # HARDCODED
    load_model(self.model, config.model)
    ...
```

**Issue**: Currently **HARDCODED** to only use `Qwen3ForCausalLM`. To add other models, you must:

1. Create model class in `nanovllm/models/{model_name}.py`
2. Add to `model_registry` or `if/elif` logic in `ModelRunner.__init__`

### Recommended Extensibility Pattern

To make model registration extensible without code changes:

```python
# nanovllm/models/__init__.py
MODEL_REGISTRY = {
    "qwen3": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
    "qwen3_5": ("nanovllm.models.qwen3_5", "Qwen3_5ForCausalLM"),
}

def get_model_class(hf_config):
    model_type = hf_config.model_type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_type} not supported")
    module_path, class_name = MODEL_REGISTRY[model_type]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

# In model_runner.py
model_class = get_model_class(hf_config)
self.model = model_class(hf_config)
```

---

## 5. HOW OTHER MODELS ARE LOADED

### Model Autodetection

The system uses **HuggingFace model config** to determine model type:

```python
hf_config = AutoConfig.from_pretrained(model_path)
# Reads from config.json in model_path
# Contains: model_type, hidden_size, num_hidden_layers, etc.
```

### What's Required to Add a New Model

1. **Create model class** in `nanovllm/models/{model_name}.py`
   - Must match HuggingFace architecture (same config format)
   - Implement `forward(input_ids, positions)`
   - Implement `compute_logits(hidden_states)`
   - Define `packed_modules_mapping` if weights are packed differently

2. **Register in model_runner.py** (currently hardcoded)
   ```python
   model_type = hf_config.model_type
   if model_type == "qwen3":
       from nanovllm.models.qwen3 import Qwen3ForCausalLM
       self.model = Qwen3ForCausalLM(hf_config)
   elif model_type == "llama":
       from nanovllm.models.llama import LlamaForCausalLM
       self.model = LlamaForCausalLM(hf_config)
   ```

3. **Ensure tensor parallel layers** use provided layer classes:
   - `QKVParallelLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`
   - `RowParallelLinear` for gather operations
   - Layers automatically handle TP sharding via `dist.get_rank()` and `dist.get_world_size()`

---

## 6. EXAMPLE DIRECTORY STRUCTURE

### Usage Examples

#### 1. Basic Usage (`example.py`)

**File**: `/Users/water/work/code/LALearning/nano-vllm/example.py` (34 lines)

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["introduce yourself", "list all prime numbers within 100"]

# Apply chat template
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")
```

**Key points**:
- `LLM.__init__` takes model path and config kwargs
- `llm.generate()` returns list of dicts with `text` and `token_ids`
- Compatible with HuggingFace tokenizer chat templates

#### 2. Benchmark (`bench.py`)

**File**: `/Users/water/work/code/LALearning/nano-vllm/bench.py` (33 lines)

```python
from nanovllm import LLM, SamplingParams

num_seqs = 256
max_input_len = 1024
max_output_len = 1024

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(path, enforce_eager=False, max_model_len=4096)

prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, max_input_len))]
    for _ in range(num_seqs)
]
sampling_params = [
    SamplingParams(temperature=0.6, ignore_eos=True, 
                   max_tokens=randint(100, max_output_len))
    for _ in range(num_seqs)
]

llm.generate(["Benchmark: "], SamplingParams())  # Warmup
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = time.time() - t

total_tokens = sum(sp.max_tokens for sp in sampling_params)
throughput = total_tokens / t
print(f"Throughput: {throughput:.2f}tok/s")
```

**Performance** (RTX 4070 Laptop, Qwen3-0.6B):
- Nano-vLLM: 1434.13 tok/s
- vLLM: 1361.84 tok/s

#### 3. Model Config (qwen3.5/)

**Files in directory**:
- `qwen3.5-35B-A3B-config`: HuggingFace model config (JSON)
- `tokenizer.json`: BPE tokenizer vocab
- `tokenizer_config.json`: Tokenizer settings
- `generation_config.json`: Default generation parameters
- `qwen3.5-35B-A3B_model.safetensors.index.json`: Weight manifest

---

## 7. COMPLETE FILE INVENTORY

### Core Files by Purpose

#### Main Entry Points
| File | Lines | Purpose |
|------|-------|---------|
| `nanovllm/__init__.py` | 3 | Public API exports |
| `nanovllm/llm.py` | 5 | LLM class (wrapper) |
| `example.py` | 34 | Usage example |

#### Configuration & Initialization
| File | Lines | Purpose |
|------|-------|---------|
| `nanovllm/config.py` | 27 | Config dataclass |
| `nanovllm/sampling_params.py` | 12 | SamplingParams dataclass |
| `nanovllm/utils/context.py` | 28 | Global context for execution |
| `nanovllm/utils/loader.py` | 29 | Weight loading from safetensors |

#### Engine & Orchestration
| File | Lines | Purpose |
|------|-------|---------|
| `nanovllm/engine/llm_engine.py` | 94 | Main orchestration, generate() |
| `nanovllm/engine/model_runner.py` | 252 | GPU inference, CUDA graph capture |
| `nanovllm/engine/scheduler.py` | 72 | Request scheduling, preemption |
| `nanovllm/engine/sequence.py` | 84 | Request/sequence state |
| `nanovllm/engine/block_manager.py` | 113 | KV cache block management |

#### Model Architecture
| File | Lines | Purpose |
|------|-------|---------|
| `nanovllm/models/qwen3.py` | 216 | Qwen3 model implementation |

#### Layers (Tensor Parallel + Optimized)
| File | Lines | Purpose |
|------|-------|---------|
| `nanovllm/layers/attention.py` | 76 | Flash attn + KV cache (Triton) |
| `nanovllm/layers/linear.py` | 154 | TP linear layers |
| `nanovllm/layers/layernorm.py` | 51 | RMSNorm (torch.compile) |
| `nanovllm/layers/embed_head.py` | 67 | Vocab parallel embeddings |
| `nanovllm/layers/activation.py` | 15 | SiluAndMul (torch.compile) |
| `nanovllm/layers/rotary_embedding.py` | 62 | Rotary positional embeddings |
| `nanovllm/layers/sampler.py` | 16 | Temperature sampling |

#### Benchmarking & Analysis
| File | Lines | Purpose |
|------|-------|---------|
| `bench.py` | 33 | Performance benchmark |
| `qwen3.5/qwen_roofline.py` | 138 | Roofline analysis (H100/A100) |

**Total Code**: ~1,358 lines in main package

---

## 8. KEY ARCHITECTURAL PATTERNS

### Pattern 1: Tensor Parallelism via dist.get_rank()

All TP layers call `dist.get_rank()` and `dist.get_world_size()`:

```python
class QKVParallelLinear(ColumnParallelLinear):
    def weight_loader(self, param, loaded_weight, shard_id):
        # Auto-slice based on rank
        shard_offset = ... // dist.get_world_size()
        shard_size = ... // dist.get_world_size()
        param.data.copy_(loaded_weight[shard_offset:shard_offset+shard_size])
```

**Design**: Makes TP automatic—no config needed beyond `tensor_parallel_size`.

### Pattern 2: Packed Module Mapping

Model defines how its weights map to packed layer structure:

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
    }
```

**Design**: Loader inspects weight names, matches to this mapping, calls per-param `weight_loader`.

### Pattern 3: Global Context for Execution Mode

Prefill vs decode handled via global context:

```python
# model_runner.py
def run(self, seqs, is_prefill):
    if is_prefill:
        input_ids, positions = self.prepare_prefill(seqs)
    else:
        input_ids, positions = self.prepare_decode(seqs)
    
    set_context(is_prefill, ...)  # Global context
    logits = self.run_model(input_ids, positions, is_prefill)
    
# layers/attention.py
def forward(self, q, k, v):
    context = get_context()
    if context.is_prefill:
        o = flash_attn_varlen_func(...)
    else:
        o = flash_attn_with_kvcache(...)
```

**Design**: Avoids passing `is_prefill` through many layers; layers query context.

### Pattern 4: CUDA Graph Capture for Decode

Batch sizes 1-512 captured; decode uses graph replay:

```python
# model_runner.py
def capture_cudagraph(self):
    # Capture for each batch size
    for bs in [1, 2, 4, 8, 16, ..., max_bs]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool):
            outputs = self.model(input_ids[:bs], positions[:bs])
        self.graphs[bs] = graph

def run_model(self, input_ids, positions, is_prefill):
    if is_prefill:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        # Replay graph for this batch size
        self.graph_vars["input_ids"][:bs] = input_ids
        self.graphs[bs].replay()
        return self.model.compute_logits(self.graph_vars["outputs"][:bs])
```

**Design**: Eliminates GPU launch overhead for decode; significant speedup.

---

## 9. QUICK START FOR EXTENDING

### To Add a New Model (e.g., Llama)

1. **Create `nanovllm/models/llama.py`**:
   ```python
   from torch import nn
   
   class LlamaForCausalLM(nn.Module):
       packed_modules_mapping = {...}  # Map HF weights to your packed structure
       
       def __init__(self, config):
           self.model = LlamaModel(config)
           self.lm_head = ParallelLMHead(...)
       
       def forward(self, input_ids, positions):
           return self.model(input_ids, positions)
       
       def compute_logits(self, hidden_states):
           return self.lm_head(hidden_states)
   ```

2. **Update `nanovllm/engine/model_runner.py`** (line 31):
   ```python
   model_type = hf_config.model_type
   if model_type == "llama":
       from nanovllm.models.llama import LlamaForCausalLM
       self.model = LlamaForCausalLM(hf_config)
   elif model_type == "qwen3":
       from nanovllm.models.qwen3 import Qwen3ForCausalLM
       self.model = Qwen3ForCausalLM(hf_config)
   else:
       raise ValueError(f"Model {model_type} not supported")
   ```

3. **Use TP layers** from `nanovllm.layers`:
   - `QKVParallelLinear` for Q, K, V
   - `ColumnParallelLinear` for projections with column parallelism
   - `RowParallelLinear` for all-reduce gather

4. **Test with**:
   ```python
   llm = LLM("/path/to/llama", tensor_parallel_size=1)
   outputs = llm.generate(["Hello"], SamplingParams(max_tokens=256))
   ```

---

## 10. SUMMARY TABLE: Key Components

| Component | File | Lines | Purpose | Key Class |
|-----------|------|-------|---------|-----------|
| **Public API** | `__init__.py` | 3 | Exports LLM, SamplingParams | - |
| **Config** | `config.py` | 27 | Model + inference config | `Config` |
| **Sampling** | `sampling_params.py` | 12 | Sampling settings | `SamplingParams` |
| **Engine** | `llm_engine.py` | 94 | Main generate() orchestration | `LLMEngine` |
| **Runner** | `model_runner.py` | 252 | GPU execution, CUDA graphs | `ModelRunner` |
| **Scheduler** | `scheduler.py` | 72 | Request scheduling, preemption | `Scheduler` |
| **Sequence** | `sequence.py` | 84 | Request state management | `Sequence` |
| **Block Manager** | `block_manager.py` | 113 | KV cache block allocation | `BlockManager` |
| **Qwen3 Model** | `models/qwen3.py` | 216 | Qwen3 architecture | `Qwen3ForCausalLM` |
| **Attention** | `layers/attention.py` | 76 | Flash attn + KV cache (Triton) | `Attention` |
| **Linear** | `layers/linear.py` | 154 | TP linear layers | `QKVParallelLinear`, etc. |
| **LayerNorm** | `layers/layernorm.py` | 51 | RMSNorm | `RMSNorm` |
| **Embeddings** | `layers/embed_head.py` | 67 | Vocab parallel embeddings | `VocabParallelEmbedding` |
| **Activation** | `layers/activation.py` | 15 | Fused SiLU*gate | `SiluAndMul` |
| **RoPE** | `layers/rotary_embedding.py` | 62 | Rotary embeddings | `RotaryEmbedding` |
| **Sampler** | `layers/sampler.py` | 16 | Temperature sampling | `Sampler` |
| **Context** | `utils/context.py` | 28 | Global execution context | `Context` |
| **Loader** | `utils/loader.py` | 29 | Weight loading from safetensors | - |

---

## Code Snippets for Reference

### Snippet 1: How inference flows

```
llm.generate(prompts, sampling_params)
  ↓
LLMEngine.generate()
  ├─→ for prompt, sp in zip(prompts, sampling_params):
  │     self.add_request(prompt, sp)
  │       ↓ Tokenize + create Sequence
  │       ↓ Add to scheduler.waiting
  │
  └─→ while not self.is_finished():
        seqs, is_prefill = self.scheduler.schedule()
          ↓ Allocate/deallocate blocks, move seqs to/from running
        token_ids = self.model_runner.call("run", seqs, is_prefill)
          ↓ If TP > 1, runs in subprocess via shared memory
          ↓ Else, runs in main process
          ↓ Returns sampled token IDs
        self.scheduler.postprocess(seqs, token_ids)
          ↓ Append tokens, check finish conditions
```

### Snippet 2: Qwen3 forward pass

```
input_ids (B, L)  →  Qwen3ForCausalLM.forward(input_ids, positions)
  ↓
embed_tokens(input_ids)  →  (B, L, H)
  ↓
for layer in layers:
    layer(positions, hidden_states, residual)
      ├─→ input_layernorm(hidden_states, residual)
      ├─→ self_attn(positions, normed_hidden_states)
      │   ├─→ qkv_proj  →  (B, L, H + 2*num_kv_heads*head_dim)
      │   ├─→ rotary_emb(q, k)
      │   ├─→ flash_attn + kv_cache_store
      │   └─→ o_proj
      ├─→ post_attention_layernorm(attn_out, residual)
      └─→ mlp(normed_attn_out)
          ├─→ gate_up_proj  →  (B, L, 2*intermediate)
          ├─→ act_fn (SiLU*gate)
          └─→ down_proj
  ↓
norm(hidden_states, residual)
  ↓
compute_logits(hidden_states)  →  (B, L, vocab_size)  or  (B, vocab_size)  if prefill
```

---

**Last Updated**: April 12, 2026
**Codebase Version**: nano-vllm 0.2.0
