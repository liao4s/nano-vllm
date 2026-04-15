# Nano-vLLM Comprehensive Codebase Exploration

**Project**: Nano-vLLM - A lightweight vLLM implementation built from scratch  
**Version**: 0.2.0  
**Author**: Xingkai Yu  
**Repository**: https://github.com/GeeeekExplorer/nano-vllm  
**Total Lines of Code**: ~1,560 lines of production Python code  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Model Loading Infrastructure](#model-loading-infrastructure)
4. [Qwen3 Model Implementation](#qwen3-model-implementation)
5. [Example Scripts](#example-scripts)
6. [Model Registry & Factory Pattern](#model-registry--factory-pattern)
7. [Key Components Deep Dive](#key-components-deep-dive)
8. [Data Flow & Architecture](#data-flow--architecture)

---

## Project Overview

### Purpose
Nano-vLLM is a lightweight, high-performance inference engine for large language models. It's designed to achieve vLLM-comparable throughput while maintaining a clean, readable codebase (~1,200 lines).

### Key Features
- ✅ Fast offline inference (1434 tok/s vs vLLM's 1361 tok/s on RTX 4070)
- ✅ Clean, readable Python implementation
- ✅ Optimization suite: prefix caching, tensor parallelism, torch compilation, CUDA graphs
- ✅ Support for Qwen3 models (0.6B currently tested)

### Dependencies
```
torch>=2.4.0
triton>=3.0.0
transformers>=4.51.0
flash-attn
xxhash
```

---

## Directory Structure

```
nano-vllm/
├── nanovllm/                          # Main package
│   ├── __init__.py                   # Package exports: LLM, SamplingParams
│   ├── config.py                     # Config dataclass (26 lines)
│   ├── llm.py                        # High-level LLM interface (5 lines)
│   ├── sampling_params.py            # Sampling parameters dataclass (11 lines)
│   │
│   ├── engine/                       # Core inference engine
│   │   ├── llm_engine.py            # Main engine orchestrator (93 lines)
│   │   ├── model_runner.py          # Model execution & GPU management (251 lines)
│   │   ├── scheduler.py             # Request scheduling & preemption (71 lines)
│   │   ├── sequence.py              # Sequence state management (83 lines)
│   │   └── block_manager.py         # KV-cache block allocation (112 lines)
│   │
│   ├── models/                        # Model architectures
│   │   └── qwen3.py                 # Complete Qwen3 implementation (215 lines)
│   │
│   ├── layers/                        # Neural network layers
│   │   ├── attention.py             # Flash-attn based attention (75 lines)
│   │   ├── linear.py                # Tensor parallel linear layers (153 lines)
│   │   ├── activation.py            # SiLU activation (14 lines)
│   │   ├── embed_head.py            # Embedding & LM head (66 lines)
│   │   ├── layernorm.py             # RMSNorm with residual (50 lines)
│   │   ├── rotary_embedding.py      # RoPE implementation (61 lines)
│   │   └── sampler.py               # Token sampling (15 lines)
│   │
│   └── utils/                         # Utility functions
│       ├── loader.py                # Model weight loading (28 lines)
│       └── context.py               # Thread-local execution context (27 lines)
│
├── qwen3.5/                          # Qwen3.5 model resources
│   ├── qwen_roofline.py             # Roofline model analysis
│   ├── config                       # Model config
│   ├── generation_config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── merges.txt
│   └── README.md
│
├── example.py                        # Quick start example (34 lines)
├── bench.py                          # Benchmark script (33 lines)
├── pyproject.toml                    # Package configuration
├── README.md                         # Project documentation
└── LICENSE                           # MIT License
```

---

## Model Loading Infrastructure

### 1. **Configuration Loading** (`config.py`)

```python
@dataclass
class Config:
    model: str                           # Model directory path
    max_num_batched_tokens: int = 16384  # Max tokens per batch
    max_num_seqs: int = 512              # Max concurrent sequences
    max_model_len: int = 4096            # Max sequence length
    gpu_memory_utilization: float = 0.9  # GPU memory usage %
    tensor_parallel_size: int = 1        # TP size
    enforce_eager: bool = False          # Skip CUDA graphs
    hf_config: AutoConfig | None = None  # HuggingFace config
    eos: int = -1                        # End-of-sequence token
    kvcache_block_size: int = 256        # KV cache block size
    num_kvcache_blocks: int = -1         # Auto-computed
```

**Post-init behavior**:
- Validates model directory exists
- Loads HuggingFace config from model
- Caps max_model_len to config's max_position_embeddings
- Validates tensor_parallel_size (1-8)

### 2. **Weight Loading** (`utils/loader.py`)

```python
def load_model(model: nn.Module, path: str):
    """
    Load model weights from safetensors files.
    
    Supports:
    - Standard weight loading
    - Packed modules mapping for weight fusion
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Check packed modules (weight fusion mappings)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
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

**Key Features**:
- Loads all `.safetensors` files from model directory
- Supports packed modules mapping for weight fusion
- Each parameter has a `weight_loader` callback for custom loading logic
- Default loader uses direct data copy

### 3. **Model Runner Initialization** (`engine/model_runner.py`)

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 1. Initialize distributed training (NCCL)
        dist.init_process_group("nccl", "tcp://localhost:2333", 
                               world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 2. Create model
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        
        # 3. Allocate KV cache
        self.allocate_kv_cache()
        
        # 4. Setup optimizations
        self.warmup_model()
        if not enforce_eager:
            self.capture_cudagraph()
```

**Workflow**:
1. Each tensor-parallel rank initializes via NCCL
2. Model created and weights loaded
3. KV cache allocated based on GPU memory
4. Model warmup and CUDA graph capture for optimization

### 4. **KV Cache Allocation** (`engine/model_runner.py`)

```python
def allocate_kv_cache(self):
    """
    Dynamically allocates KV cache blocks.
    Calculates num blocks based on:
    - Total GPU memory
    - Peak memory usage during warmup
    - GPU memory utilization target (default 0.9)
    """
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    num_kv_heads = hf_config.num_key_value_heads // world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_size
    
    num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    
    # Allocate contiguous buffer
    self.kv_cache = torch.empty(
        2,                              # K and V
        hf_config.num_hidden_layers,   # Number of layers
        num_kvcache_blocks,            # Number of blocks
        block_size,                    # Tokens per block
        num_kv_heads,                  # KV heads per GPU
        head_dim                       # Head dimension
    )
    
    # Assign to attention modules
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

---

## Qwen3 Model Implementation

### Complete Qwen3 Architecture (215 lines in `models/qwen3.py`)

#### 1. **Qwen3Attention** - Multi-head Self-Attention with RoPE

```python
class Qwen3Attention(nn.Module):
    """
    Multi-head attention with:
    - Tensor parallelism support
    - RoPE (Rotary Position Embeddings)
    - Optional QKV bias with normalization
    - Flash attention with KV cache
    """
    
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size          # Per-GPU heads
        self.num_kv_heads = num_kv_heads // tp_size    # Per-GPU KV heads
        self.head_dim = hidden_size // num_heads
        
        # QKV projection (fused)
        self.qkv_proj = QKVParallelLinear(...)
        # Output projection
        self.o_proj = RowParallelLinear(...)
        # Rotary embeddings
        self.rotary_emb = get_rope(...)
        # Flash attention
        self.attn = Attention(...)
        
        # Optional normalization (when no bias)
        if not qkv_bias:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
    
    def forward(self, positions, hidden_states):
        # 1. Project to QKV
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # 2. Reshape for multi-head
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)
        
        # 3. Apply normalization
        if not qkv_bias:
            q = q_norm(q)
            k = k_norm(k)
        
        # 4. Apply RoPE
        q, k = rotary_emb(positions, q, k)
        
        # 5. Attention
        o = attn(q, k, v)
        
        # 6. Output projection
        output = o_proj(o.flatten(1, -1))
        return output
```

**Key Components**:
- **QKVParallelLinear**: Fused Q, K, V projection with tensor parallelism
- **RowParallelLinear**: Output projection with reduction across TP ranks
- **RoPE**: Rotary position embeddings applied before attention
- **Attention**: Flash-Attention-based implementation with KV cache storage

#### 2. **Qwen3MLP** - Feed-Forward Network

```python
class Qwen3MLP(nn.Module):
    """
    Gated Linear Unit MLP (SwiGLU variant)
    
    Structure:
    x -> [Linear * 2] -> SiLU ⊙ identity -> Linear -> output
    """
    
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        # Fused gate & up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # [gate, up]
            bias=False
        )
        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )
        self.act_fn = SiluAndMul()  # SiLU activation with element-wise multiply
    
    def forward(self, x):
        gate_up = gate_up_proj(x)       # [batch, seq, 2*intermediate]
        x = act_fn(gate_up)             # Apply SiLU and multiply
        x = down_proj(x)                # Project back to hidden_size
        return x
```

**Optimization**: Gate and up projections are merged for efficiency.

#### 3. **Qwen3DecoderLayer** - Transformer Block

```python
class Qwen3DecoderLayer(nn.Module):
    """
    Single transformer layer with:
    - Pre-norm residual connections
    - Attention
    - MLP
    """
    
    def __init__(self, config: Qwen3Config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
    
    def forward(self, positions, hidden_states, residual=None):
        # Norm + Attention (pre-norm)
        if residual is None:
            hidden_states, residual = input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = input_layernorm(hidden_states, residual)
        hidden_states = self_attn(positions, hidden_states)
        
        # Norm + MLP (pre-norm)
        hidden_states, residual = post_attention_layernorm(hidden_states, residual)
        hidden_states = mlp(hidden_states)
        
        return hidden_states, residual
```

**Features**:
- Pre-norm architecture with residual connections
- RMSNorm with fused add (for efficiency)
- Careful residual tracking for flash-attn compatibility

#### 4. **Qwen3Model** - Transformer Stack

```python
class Qwen3Model(nn.Module):
    """
    Stacked transformer layers
    """
    
    def __init__(self, config: Qwen3Config):
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)
    
    def forward(self, input_ids, positions):
        hidden_states = embed_tokens(input_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = norm(hidden_states, residual)
        return hidden_states
```

#### 5. **Qwen3ForCausalLM** - Complete Model with LM Head

```python
class Qwen3ForCausalLM(nn.Module):
    """
    Causal language model for inference.
    
    Defines packed_modules_mapping for weight fusion:
    - q_proj, k_proj, v_proj -> qkv_proj
    - gate_proj, up_proj -> gate_up_proj
    """
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # Map q_proj to qkv_proj shard "q"
        "k_proj": ("qkv_proj", "k"),      # Map k_proj to qkv_proj shard "k"
        "v_proj": ("qkv_proj", "v"),      # Map v_proj to qkv_proj shard "v"
        "gate_proj": ("gate_up_proj", 0), # Map gate_proj to gate_up_proj[0]
        "up_proj": ("gate_up_proj", 1),   # Map up_proj to gate_up_proj[1]
    }
    
    def __init__(self, config: Qwen3Config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        
        # Weight tying (optional)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**Weight Fusion**:
- The `packed_modules_mapping` enables loading separate Q, K, V projections into a single fused QKV projection
- Same for gate & up projections merging into gate_up_proj
- This is handled by the custom `weight_loader` callbacks in each layer

### Tensor Parallelism in Qwen3

All layers support tensor parallelism:

1. **QKVParallelLinear**: Each rank gets `num_heads // tp_size` heads
2. **RowParallelLinear**: Input is sharded across ranks, outputs reduced via all-reduce
3. **VocabParallelEmbedding**: Vocab is sharded, output masked and reduced
4. **ParallelLMHead**: Similar to embedding, with gather on rank 0

---

## Example Scripts

### 1. **example.py** - Quick Start

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]

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

**Usage**:
```bash
python example.py
```

**Output Format**:
```python
[
    {"text": "...", "token_ids": [...]},
    {"text": "...", "token_ids": [...]},
]
```

### 2. **bench.py** - Benchmark Script

```python
from nanovllm import LLM, SamplingParams
from random import randint, seed

seed(0)
num_seqs = 256
max_input_len = 1024
max_output_len = 1024

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(path, enforce_eager=False, max_model_len=4096)

# Generate random prompt tokens
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
elapsed = time.time() - t

total_tokens = sum(sp.max_tokens for sp in sampling_params)
throughput = total_tokens / elapsed
print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f}tok/s")
```

**Benchmark Results**:
| Engine    | Output Tokens | Time (s) | Throughput (tok/s) |
|-----------|---------------|----------|-------------------|
| vLLM      | 133,966       | 98.37    | 1361.84            |
| Nano-vLLM | 133,966       | 93.41    | **1434.13** ✅     |

**Test Configuration**:
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Sequences: 256
- Input Length: 100-1024 tokens
- Output Length: 100-1024 tokens

---

## Model Registry & Factory Pattern

### Current Implementation

**There is NO explicit model registry or factory pattern** in the codebase.

Instead, the design uses:

1. **Direct Model Class Instantiation**:
```python
# In model_runner.py
self.model = Qwen3ForCausalLM(hf_config)
```

2. **Packed Modules Mapping for Weight Fusion**:
```python
# In models/qwen3.py
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

This mapping is used in `loader.py` to handle weight conversion from original format to fused format.

3. **HuggingFace Config Detection**:
```python
# In config.py
self.hf_config = AutoConfig.from_pretrained(self.model)
```

### Design Rationale

The codebase is intentionally **minimalist** and currently focuses on Qwen3 models only. To support additional models, one would need to:

1. Add new model class (e.g., `Llama3ForCausalLM` in `models/llama3.py`)
2. Define its `packed_modules_mapping`
3. Update model_runner to dispatch based on config:

```python
# Proposed extension
def create_model(hf_config):
    model_type = hf_config.model_type
    if model_type == "qwen":
        return Qwen3ForCausalLM(hf_config)
    elif model_type == "llama":
        return Llama3ForCausalLM(hf_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

---

## Key Components Deep Dive

### 1. **Request Scheduling** (`engine/scheduler.py`)

```python
class Scheduler:
    """
    Manages request lifecycle with two-phase scheduling:
    1. Prefill phase: Process new requests (high latency, high throughput)
    2. Decode phase: Generate tokens (low latency, throughput-limited)
    """
    
    def __init__(self, config: Config):
        self.block_manager = BlockManager(...)
        self.waiting = deque()  # Not started
        self.running = deque()  # In progress
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        # Phase 1: Prefill - move from waiting to running
        scheduled_seqs = []
        while self.waiting and num_seqs < max_num_seqs:
            seq = self.waiting[0]
            if can_allocate_blocks(seq):
                block_manager.allocate(seq)
                seq.status = RUNNING
                scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # Prefill phase
        
        # Phase 2: Decode - continue running sequences
        while self.running and num_seqs < max_num_seqs:
            seq = self.running.popleft()
            while not can_append_block(seq):
                # Preempt lowest priority sequence
                preempt(self.running.pop())
            block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        
        return scheduled_seqs, False  # Decode phase
```

**Two-Phase Scheduling**:
- **Prefill**: Process entire prompt (parallelizable)
- **Decode**: Generate one token at a time (sequential)

### 2. **Block Management & KV Cache** (`engine/block_manager.py`)

```python
class BlockManager:
    """
    Manages KV cache blocks with:
    - Hash-based prefix caching
    - Reference counting
    - Block allocation/deallocation
    """
    
    def allocate(self, seq: Sequence):
        """
        Allocate blocks for sequence.
        Uses prefix caching: if block hash matches previous blocks,
        reuse the cached block without duplicating KV data.
        """
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = compute_hash(token_ids, prefix=h)
            block_id = hash_to_block_id.get(h, -1)
            
            if block_id == -1 or blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Allocate new block
                block_id = free_block_ids[0]
                allocate_block(block_id)
            else:
                # Reuse cached block
                seq.num_cached_tokens += block_size
                blocks[block_id].ref_count += 1
            
            seq.block_table.append(block_id)
```

**Prefix Caching**:
- Each block is hashed with its predecessor's hash (chain hashing)
- Identical sequences share blocks, saving memory
- Reference counting enables safe deallocation

### 3. **Model Runner Execution** (`engine/model_runner.py`)

```python
class ModelRunner:
    """
    Executes model forward pass with:
    - Prefill/decode phase handling
    - KV cache management
    - CUDA graph optimization
    - Tensor parallelism
    """
    
    def run(self, seqs: list[Sequence], is_prefill: bool):
        # 1. Prepare inputs based on phase
        if is_prefill:
            input_ids, positions = prepare_prefill(seqs)
        else:
            input_ids, positions = prepare_decode(seqs)
        
        # 2. Run model
        logits = run_model(input_ids, positions, is_prefill)
        
        # 3. Sample tokens (only on rank 0)
        if rank == 0:
            token_ids = sampler(logits, temperatures)
        
        return token_ids
    
    def run_model(self, input_ids, positions, is_prefill):
        if is_prefill or enforce_eager or input_ids.size(0) > 512:
            # Eager execution for prefill
            return model.compute_logits(model(input_ids, positions))
        else:
            # CUDA graph replay for decode
            bs = input_ids.size(0)
            graph = graphs[next(x for x in graph_bs if x >= bs)]
            # Update graph variables and replay
            graph.replay()
            return model.compute_logits(graph_vars["outputs"][:bs])
```

**CUDA Graphs**:
- Pre-recorded graphs for batch sizes: [1, 2, 4, 8] + multiples of 16
- Decode phase replays graph, avoiding Python overhead
- Dramatically reduces latency for small batches

### 4. **Sequence Management** (`engine/sequence.py`)

```python
class Sequence:
    """
    Tracks state of a single generation sequence.
    """
    
    def __init__(self, token_ids: list[int], sampling_params):
        self.seq_id = next(counter)
        self.status = WAITING
        self.token_ids = token_ids.copy()
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []  # Physical block IDs for KV cache
    
    @property
    def num_blocks(self):
        """Number of KV cache blocks needed"""
        return (self.num_tokens + block_size - 1) // block_size
    
    @property
    def completion_token_ids(self):
        """Generated tokens (excluding prompt)"""
        return self.token_ids[self.num_prompt_tokens:]
    
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_tokens += 1
    
    def __getstate__(self):
        """Efficient serialization for multiprocessing"""
        return (self.num_tokens, self.num_prompt_tokens, 
                self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

### 5. **Tensor Parallelism Details** (`layers/linear.py`)

```python
class QKVParallelLinear(ColumnParallelLinear):
    """
    Fused QKV projection with tensor parallelism.
    
    Handles three cases:
    - Single GPU: standard projection
    - Multi-GPU: split output across GPUs, gather gradients
    - Weight loading: map separate q_proj, k_proj, v_proj to fused QKV
    """
    
    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        """
        Load a single component (q, k, or v) into its position
        in the fused QKV parameter.
        """
        param_data = param.data
        
        # Determine position in fused tensor
        if loaded_shard_id == "q":
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # Extract this rank's portion
        shard_size = ...  # depends on component
        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

### 6. **Flash Attention Integration** (`layers/attention.py`)

```python
class Attention(nn.Module):
    """
    Uses flash-attention for efficient attention computation.
    """
    
    def forward(self, q, k, v):
        context = get_context()
        
        # Store KV cache
        if k_cache and v_cache:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Prefill phase: variable length sequences
        if context.is_prefill:
            if context.block_tables:  # prefix cache enabled
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # Decode phase: single token per sequence
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        
        return o
```

**Key Optimizations**:
- Flash-Attention v2 for efficient attention computation
- Fused KV cache storage via Triton kernel
- Block table support for non-contiguous cache layouts

---

## Data Flow & Architecture

### End-to-End Inference Flow

```
User Input Prompts
    ↓
LLM.generate(prompts, sampling_params)
    ↓
[Tokenize] → add_request() for each prompt
    ↓
Scheduler.waiting ← [Sequence, Sequence, ...]
    ↓
[LOOP] while not is_finished():
    ├─ Scheduler.schedule()
    │   ├─ Prefill Phase: waiting → running
    │   │   • Allocate KV cache blocks (with prefix caching)
    │   │   • Batch all tokens from prompt
    │   │
    │   └─ Decode Phase: generate next tokens
    │       • Take one token from each running sequence
    │       • Use CUDA graphs for efficiency
    │
    ├─ ModelRunner.run(seqs, is_prefill)
    │   ├─ prepare_prefill/decode() - format inputs
    │   ├─ run_model() - forward pass (eager or CUDA graph)
    │   ├─ sampler() - sample next tokens
    │   └─ return token_ids
    │
    ├─ Scheduler.postprocess()
    │   ├─ Append tokens to sequences
    │   ├─ Check EOS or max_tokens
    │   └─ Move finished seqs to output
    │
    └─ Collect outputs
    ↓
[Detokenize] → Return text outputs
```

### Memory Layout During Prefill

```
Request 1: [tok1, tok2, tok3, ..., tokN1]  (length N1)
Request 2: [tok1, tok2, ..., tokN2]        (length N2)
Request 3: [tok1, ..., tokN3]              (length N3)

↓ Flattened for batch processing

input_ids:    [tok1, tok2, tok3, ..., tokN1, tok1, tok2, ..., tokN2, tok1, ..., tokN3]
positions:    [0,    1,    2,   ..., N1-1,  0,    1,   ..., N2-1,  0,   ..., N3-1]
cu_seqlens_q: [0, N1, N1+N2, N1+N2+N3]  (cumulative lengths for query)
cu_seqlens_k: [0, N1, N1+N2, N1+N2+N3]  (cumulative lengths for key/value)

These are passed to flash_attn_varlen_func() for efficient attention.
```

### Memory Layout During Decode

```
Only last tokens of each sequence:
input_ids:    [token_seq1, token_seq2, token_seq3]  (batch_size=3)
positions:    [len(seq1)-1, len(seq2)-1, len(seq3)-1]
context_lens: [len(seq1), len(seq2), len(seq3)]
block_tables: [[blk1, blk2, ...], [blk4, blk5, ...], ...]  (physical block IDs)

These are used with flash_attn_with_kvcache() to efficiently attend to cached KVs.
```

---

## Architecture Highlights

### 1. **Prefix Caching**
- Hash-based deduplication of KV cache blocks
- Saves memory when multiple requests share prompt prefixes
- Transparent to user

### 2. **Tensor Parallelism**
- Splits model across GPUs
- Each GPU computes subset of heads/neurons
- All-reduce operations for collective computation
- Scales up to 8 GPUs

### 3. **CUDA Graphs**
- Pre-recorded compute graphs for decode phase
- Replayed without Python overhead
- Batch sizes: 1, 2, 4, 8, 16, 32, ... (up to max_num_seqs)
- Dramatic latency reduction

### 4. **Flash Attention**
- Reduces attention complexity from O(N²) memory to O(N)
- ~3x speedup vs. standard attention
- Supports KV cache and prefix caching

### 5. **Packed Modules**
- Fuses separate projections into single operations
- Q, K, V → QKV (3 ops → 1)
- Gate, Up → Gate-Up (2 ops → 1)
- Reduced memory bandwidth

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_num_batched_tokens` | 16384 | Max tokens processed per batch |
| `max_num_seqs` | 512 | Max concurrent sequences |
| `max_model_len` | 4096 | Max sequence length |
| `gpu_memory_utilization` | 0.9 | GPU memory % for KV cache |
| `tensor_parallel_size` | 1 | Number of GPUs |
| `enforce_eager` | False | Disable CUDA graphs |
| `kvcache_block_size` | 256 | Tokens per KV cache block |
| `num_kvcache_blocks` | -1 (auto) | Number of KV cache blocks |

---

## Summary Statistics

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| Engine | 5 | 610 | Scheduling, execution, memory management |
| Models | 1 | 215 | Qwen3 architecture |
| Layers | 7 | 404 | Neural network components |
| Utils | 2 | 55 | Model loading, context |
| Examples | 2 | 67 | Usage demonstrations |
| **Total** | **17** | **1,560** | Complete inference engine |

---

## Key Design Principles

1. **Minimalism**: ~1,200 lines of clean Python code
2. **Readability**: Clear variable names and structure
3. **Performance**: CUDA graphs, flash attention, tensor parallelism
4. **Correctness**: Careful handling of sequences, blocks, and residuals
5. **Extensibility**: Support for tensor parallelism and new models (with additions)

---

## Conclusion

Nano-vLLM demonstrates that a high-performance inference engine can be built in minimal lines of code through:
- Careful memory management (prefix caching, block allocation)
- Strategic kernel usage (flash-attention, CUDA graphs)
- Clean abstractions (Sequence, BlockManager, Scheduler)
- Pragmatic optimizations (tensor parallelism, weight fusion)

The codebase serves as both a practical tool and an educational resource for understanding modern LLM serving infrastructure.
