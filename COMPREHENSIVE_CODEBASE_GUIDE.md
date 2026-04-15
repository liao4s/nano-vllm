# NanoVLLM - Comprehensive Codebase Exploration

**Last Updated:** April 12, 2026

---

## 1. PROJECT OVERVIEW

**Project Name:** nano-vllm (Nano Vision Language Model)  
**Version:** 0.2.0  
**Description:** A lightweight vLLM implementation built from scratch  
**Author:** Xingkai Yu  
**License:** MIT  
**Homepage:** https://github.com/GeeeekExplorer/nano-vllm

### Key Technologies:
- PyTorch >= 2.4.0
- Triton >= 3.0.0
- Transformers >= 4.51.0
- Flash-Attention (custom kernels for efficient attention)
- xxhash (for block hashing in KV cache)
- Tensor Parallelism (distributed training)
- CUDA Graphs (for inference optimization)

---

## 2. COMPLETE PROJECT STRUCTURE

```
nano-vllm/
├── nanovllm/                          # Main package
│   ├── __init__.py                    # Package entry point (exports LLM, SamplingParams)
│   ├── llm.py                         # LLM class (wrapper around LLMEngine)
│   ├── config.py                      # Config dataclass for model configuration
│   ├── sampling_params.py             # SamplingParams dataclass for generation
│   │
│   ├── models/                        # Model implementations
│   │   └── qwen3.py                   # Complete Qwen3 model implementation
│   │
│   ├── engine/                        # Core inference engine
│   │   ├── llm_engine.py              # LLMEngine - main orchestrator
│   │   ├── model_runner.py            # ModelRunner - handles model execution
│   │   ├── scheduler.py               # Scheduler - batching and sequence management
│   │   ├── sequence.py                # Sequence - individual request representation
│   │   └── block_manager.py           # BlockManager - KV cache block management
│   │
│   ├── layers/                        # Custom layer implementations
│   │   ├── linear.py                  # Linear layers (ColumnParallel, RowParallel, etc.)
│   │   ├── attention.py               # Attention mechanism with flash-attn
│   │   ├── layernorm.py               # RMSNorm implementation
│   │   ├── rotary_embedding.py        # Rotary Position Embeddings (RoPE)
│   │   ├── activation.py              # SiluAndMul activation
│   │   ├── embed_head.py              # VocabParallelEmbedding, ParallelLMHead
│   │   └── sampler.py                 # Token sampling for generation
│   │
│   └── utils/                         # Utility functions
│       ├── loader.py                  # Model weight loading infrastructure
│       └── context.py                 # Global context for attention/inference
│
├── example.py                         # Example usage script
├── bench.py                           # Benchmarking script
├── pyproject.toml                     # Project configuration
└── README.md                          # Project documentation
```

---

## 3. KEY CLASSES AND THEIR RELATIONSHIPS

### 3.1 Public API

```
LLM (llm.py)
  └─ extends LLMEngine
  
SamplingParams (sampling_params.py)
  └─ Dataclass for generation parameters
```

### 3.2 Core Engine Architecture

```
LLMEngine (engine/llm_engine.py)
  ├─ ModelRunner (engine/model_runner.py)          [Handles model execution]
  ├─ Scheduler (engine/scheduler.py)               [Manages sequences batching]
  ├─ AutoTokenizer (from transformers)             [Tokenization]
  └─ Config (config.py)                            [Configuration]

ModelRunner
  ├─ Qwen3ForCausalLM (models/qwen3.py)           [The actual model]
  ├─ Sampler (layers/sampler.py)                   [Token sampling]
  └─ KV Cache (torch.Tensor)                       [For attention]

Scheduler
  ├─ BlockManager (engine/block_manager.py)        [Manages KV cache blocks]
  ├─ Sequence (engine/sequence.py) [*many]         [Individual requests]
  ├─ waiting: deque[Sequence]                      [Waiting queue]
  └─ running: deque[Sequence]                      [Running queue]
```

---

## 4. MODEL LOADING INFRASTRUCTURE

### 4.1 Load Model Function (`nanovllm/utils/loader.py`)

**Purpose:** Load pretrained model weights from safetensors files

```python
def load_model(model: nn.Module, path: str):
    """
    Load model weights from safetensors files in a directory.
    
    Features:
    - Supports packed module mappings (e.g., q_proj, k_proj -> qkv_proj)
    - Uses custom weight loaders on parameters for tensor parallel sharding
    - Loads all *.safetensors files in the directory
    
    Process:
    1. Check if model has packed_modules_mapping attribute
    2. For each safetensors file:
        - For each weight in file:
            - If weight name matches a packed module key:
                - Get the mapped parameter name
                - Call custom weight_loader with shard_id
            - Else:
                - Call weight_loader or default_weight_loader
    """
```

### 4.2 Packed Module Mapping (Qwen3ForCausalLM)

Located in `nanovllm/models/qwen3.py`:

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # q_proj weight -> qkv_proj[q]
        "k_proj": ("qkv_proj", "k"),      # k_proj weight -> qkv_proj[k]
        "v_proj": ("qkv_proj", "v"),      # v_proj weight -> qkv_proj[v]
        "gate_proj": ("gate_up_proj", 0), # gate_proj weight -> gate_up_proj[0]
        "up_proj": ("gate_up_proj", 1),   # up_proj weight -> gate_up_proj[1]
    }
```

**Why packed modules?**
- Model checkpoint has separate q, k, v projections
- Implementation combines them into single qkv_proj for efficiency
- Mapping handles the unpacking during weight loading

### 4.3 Weight Loader Infrastructure

Each parameter can have a custom `weight_loader` attribute:

```python
class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # Custom loader
```

**Loading Hierarchy:**
1. Check if parameter has custom `weight_loader`
2. If yes, call it with appropriate shard information
3. If no, use `default_weight_loader`

---

## 5. COMPLETE QWEN3 MODEL IMPLEMENTATION

### 5.1 Model Hierarchy

```
Qwen3ForCausalLM (top-level model)
  ├─ model: Qwen3Model
  │   ├─ embed_tokens: VocabParallelEmbedding
  │   ├─ layers: ModuleList[Qwen3DecoderLayer] (num_hidden_layers)
  │   └─ norm: RMSNorm
  └─ lm_head: ParallelLMHead
```

### 5.2 Qwen3DecoderLayer

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        self.self_attn: Qwen3Attention
        self.mlp: Qwen3MLP
        self.input_layernorm: RMSNorm
        self.post_attention_layernorm: RMSNorm
    
    def forward(positions, hidden_states, residual=None):
        # Pre-normalization with residual connection
        hidden_states, residual = input_layernorm(hidden_states, residual)
        
        # Attention
        hidden_states = self_attn(positions, hidden_states)
        
        # Post-attention normalization
        hidden_states, residual = post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = mlp(hidden_states)
        
        return hidden_states, residual
```

### 5.3 Qwen3Attention

**Features:**
- Multi-Head Attention with support for Multi-Query Attention (MQA)
- Tensor Parallelism support
- Rotary Position Embeddings (RoPE)
- Optional Q/K normalization
- QKV parallel linear layers

```python
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, 
                 max_position, head_dim=None, rms_norm_eps=1e-6,
                 qkv_bias=False, rope_theta=10000, rope_scaling=None):
        
        # Tensor parallel division
        tp_size = dist.get_world_size()
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        
        # Projection layers
        self.qkv_proj: QKVParallelLinear
        self.o_proj: RowParallelLinear
        
        # Embeddings
        self.rotary_emb: RotaryEmbedding
        
        # Attention
        self.attn: Attention (uses flash-attn)
        
        # Normalization
        self.q_norm: RMSNorm (optional)
        self.k_norm: RMSNorm (optional)
```

### 5.4 Qwen3MLP

**Gate-Up-Down architecture:**

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        # Merged gate and up projections
        self.gate_up_proj: MergedColumnParallelLinear
        self.down_proj: RowParallelLinear
        self.act_fn: SiluAndMul  # SiLU(gate) * up
    
    def forward(self, x):
        gate_up = gate_up_proj(x)           # Shape: (*, intermediate_size*2)
        x = act_fn(gate_up)                 # SiLU(gate) * up
        x = down_proj(x)                    # Back to hidden_size
        return x
```

### 5.5 Complete File: `nanovllm/models/qwen3.py`

**Lines 1-90: Qwen3Attention**
- Implements multi-head attention with MQA support
- Tensor parallel aware (divides heads across GPUs)
- Uses QKVParallelLinear to project input to Q, K, V
- Applies rotary embeddings to Q and K
- Calls flash-attn for efficient attention computation

**Lines 90-117: Qwen3MLP**
- Gate-up-down FFN architecture
- gate_up_proj: combines gate and up projections
- SiluAndMul: applies SiLU(gate) * up
- down_proj: projects back to hidden_size

**Lines 119-159: Qwen3DecoderLayer**
- Combines attention and MLP with residual connections
- Pre-normalization architecture
- Takes positions, hidden_states, and residual as input

**Lines 161-183: Qwen3Model**
- Embedding layer
- Stack of decoder layers
- Final RMSNorm

**Lines 185-216: Qwen3ForCausalLM**
- Top-level model class
- Contains Qwen3Model and ParallelLMHead
- Implements packed_modules_mapping for weight loading
- Has compute_logits method for inference

---

## 6. MODEL LOADING FLOW

### 6.1 Complete Loading Process

```
LLMEngine.__init__()
  │
  ├─ Create ModelRunner (rank 0)
  │   └─ ModelRunner.__init__()
  │       ├─ Initialize distributed group
  │       ├─ Create model: Qwen3ForCausalLM(hf_config)
  │       ├─ load_model(model, config.model)  [<-- MODEL LOADING]
  │       │   ├─ For each *.safetensors file:
  │       │   │   ├─ Load weights
  │       │   │   ├─ Check packed_modules_mapping
  │       │   │   ├─ Call custom weight_loader
  │       │   │   └─ Handle tensor parallel sharding
  │       │   └─ [All weights loaded]
  │       ├─ Warmup model
  │       ├─ Allocate KV cache
  │       └─ Capture CUDA graphs (if not enforce_eager)
  │
  └─ Create additional ModelRunner processes (rank > 0)
      └─ For tensor parallel > 1
```

### 6.2 Config Class

```python
@dataclass
class Config:
    model: str                           # Path to model directory
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None  # Loaded from model
    eos: int = -1                        # Set during init
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1         # Computed during init
    
    def __post_init__(self):
        # Validation and auto-loading
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, 
                                 self.hf_config.max_position_embeddings)
```

---

## 7. LAYER IMPLEMENTATIONS

### 7.1 Linear Layers (`nanovllm/layers/linear.py`)

**Class Hierarchy:**

```
LinearBase (abstract base)
  ├─ ReplicatedLinear (replicated across GPUs)
  ├─ ColumnParallelLinear (column-wise sharded)
  │   ├─ MergedColumnParallelLinear (multiple outputs, e.g., gate+up)
  │   └─ QKVParallelLinear (special case for Q, K, V)
  └─ RowParallelLinear (row-wise sharded)
```

**Key Features:**
- Custom `weight_loader` methods for each class
- Tensor parallel aware sharding logic
- Handles weight slicing based on `tp_rank` and `tp_size`

**Tensor Parallel Examples:**

```python
# Column Parallel: Each GPU gets subset of output channels
class ColumnParallelLinear:
    def __init__(self, input_size, output_size, bias=False):
        # Each GPU gets output_size // tp_size outputs
        self.weight = nn.Parameter(
            torch.empty(output_size // tp_size, input_size)
        )
    
    def weight_loader(self, param, loaded_weight):
        shard_size = param.data.size(0)  # output_size // tp_size
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param.data.copy_(loaded_weight)

# Row Parallel: Each GPU gets subset of input channels
class RowParallelLinear:
    def __init__(self, input_size, output_size, bias=False):
        # Each GPU gets input_size // tp_size inputs
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size // tp_size)
        )
    
    def forward(self, x):
        y = F.linear(x, self.weight, bias)
        if tp_size > 1:
            dist.all_reduce(y)  # Sum across GPUs
        return y
```

### 7.2 Attention (`nanovllm/layers/attention.py`)

**Custom Kernels:**
- `store_kvcache_kernel`: Triton kernel for storing K/V to cache slots
- Uses flash-attn for efficient attention computation

**Context-Aware:**
- Adapts behavior based on Context (prefill vs decode)
- Stores and retrieves KV cache efficiently

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        
        # Store K/V to cache
        if k_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Compute attention
        if context.is_prefill:
            if context.block_tables:  # Prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:  # Decode
            o = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
        
        return o
```

### 7.3 RMSNorm (`nanovllm/layers/layernorm.py`)

**Features:**
- Compiled with `@torch.compile` for performance
- Supports both simple normalization and residual connection
- Uses torch.rsqrt for numerical stability

```python
class RMSNorm(nn.Module):
    @torch.compile
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + eps))
        return x.to(orig_dtype).mul_(weight)
    
    @torch.compile
    def add_rms_forward(self, x, residual):
        # Fused: x = (x + residual) / RMS(x + residual) * weight
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        # ... normalization ...
        return x, residual
```

### 7.4 Rotary Embeddings (`nanovllm/layers/rotary_embedding.py`)

**RoPE (Rotary Position Embeddings):**
- Precomputes cos/sin cache at init
- Applies rotation matrix to Q and K
- Cached for efficiency

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position, base):
        # Precompute frequency
        inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
        
        # Build cache: [max_position, 1, rotary_dim*2]
        t = arange(max_position)
        freqs = einsum('i,j -> ij', t, inv_freq)  # [max_position, dim/2]
        cos_sin_cache = cat([freqs.cos(), freqs.sin()], dim=-1)
    
    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
```

### 7.5 Embeddings (`nanovllm/layers/embed_head.py`)

**VocabParallelEmbedding:**
- Each GPU holds subset of vocabulary
- Masks tokens outside local vocabulary
- Uses all-reduce to gather outputs

**ParallelLMHead:**
- Extends VocabParallelEmbedding
- Implements distributed gather for logits
- Only rank 0 returns logits

### 7.6 Sampling (`nanovllm/layers/sampler.py`)

**Gumbel-Max Trick for Sampling:**

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        # Scale logits by temperature
        logits = logits.float() / temperatures.unsqueeze(1)
        
        # Softmax to get probabilities
        probs = softmax(logits, dim=-1)
        
        # Gumbel-Max trick: sample ~ argmax(logits + Gumbel(0,1))
        # Equivalent to: argmax(log(probs) + Gumbel(0,1))
        # Approximated by: argmax(probs / Exp(1))
        gumbel_noise = empty_like(probs).exponential_(1).clamp_min(1e-10)
        sample_tokens = (probs / gumbel_noise).argmax(dim=-1)
        
        return sample_tokens
```

### 7.7 Activation (`nanovllm/layers/activation.py`)

```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        # Split input: x = [gate, up]
        gate, up = x.chunk(2, -1)
        # Apply: SiLU(gate) * up
        return F.silu(gate) * up
```

---

## 8. ENGINE COMPONENTS

### 8.1 LLMEngine (`nanovllm/engine/llm_engine.py`)

**Main Orchestrator:**

```python
class LLMEngine:
    def __init__(self, model: str, **kwargs):
        # Create config
        config = Config(model, **kwargs)
        
        # Create ModelRunner (rank 0 + worker processes)
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        
        # Create scheduler
        self.scheduler = Scheduler(config)
    
    def add_request(self, prompt, sampling_params):
        prompt_ids = self.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids, sampling_params)
        self.scheduler.add(seq)
    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        return outputs, num_tokens
    
    def generate(self, prompts, sampling_params, use_tqdm=True):
        # Add all requests
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # Run scheduler loop until all finished
        while not self.is_finished():
            output, num_tokens = self.step()
        
        return outputs
```

### 8.2 ModelRunner (`nanovllm/engine/model_runner.py`)

**Execution Engine:**

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # Distributed setup
        dist.init_process_group("nccl", ...)
        
        # Create model
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        
        # Warmup and allocate KV cache
        self.warmup_model()
        self.allocate_kv_cache()
        
        # Capture CUDA graphs for decode phase
        if not enforce_eager:
            self.capture_cudagraph()
    
    def run(self, seqs: list[Sequence], is_prefill: bool):
        # Prepare inputs
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill 
            else self.prepare_decode(seqs)
        )
        
        # Run model
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # Sample tokens
        temperatures = self.prepare_sample(seqs)
        token_ids = self.sampler(logits, temperatures)
        
        return token_ids
```

**Key Methods:**

1. **prepare_prefill()**: Prepares batched prefill phase
   - Handles variable length sequences
   - Computes cumulative sequence lengths
   - Prepares slot mapping for KV cache storage

2. **prepare_decode()**: Prepares single-token decode phase
   - Simple concatenation of last tokens
   - Single position per sequence
   - Context length tracking

3. **run_model()**: Executes model forward pass
   - Prefill: eager execution
   - Decode: CUDA graph replay (if available)

4. **allocate_kv_cache()**: Manages GPU memory for KV cache
   - Calculates number of blocks based on GPU memory
   - Attaches cache to attention modules

5. **capture_cudagraph()**: Records inference graphs
   - Records graphs for batch sizes: [1, 2, 4, 8, 16, ...]
   - Replays graphs for efficient decode phase

### 8.3 Scheduler (`nanovllm/engine/scheduler.py`)

**Batching Logic:**

```python
class Scheduler:
    def schedule(self) -> tuple[list[Sequence], bool]:
        # Phase 1: Prefill new requests
        for seq in self.waiting:
            if can_fit_in_batch(seq):
                self.block_manager.allocate(seq)
                self.running.append(seq)
                scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # Prefill phase
        
        # Phase 2: Decode existing requests
        for seq in self.running:
            while can_append(seq) is False:
                # Preempt another sequence to free blocks
                self.preempt(self.running.pop())
            
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        
        return scheduled_seqs, False  # Decode phase
```

**Scheduling Strategy:**
1. Prioritize prefilling new requests
2. If no room, decode existing requests
3. If can't append to decode, preempt other sequences
4. Preempted sequences go back to waiting queue

### 8.4 Sequence (`nanovllm/engine/sequence.py`)

```python
class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(counter)  # Unique ID
        self.token_ids = token_ids
        self.status = SequenceStatus.WAITING
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.block_table: list[int] = []  # KV cache block IDs
        self.num_cached_tokens = 0  # How many tokens already cached
    
    @property
    def num_blocks(self):
        return (len(self.token_ids) + block_size - 1) // block_size
    
    @property
    def is_finished(self):
        return (status == FINISHED or 
                num_completion_tokens >= max_tokens)
```

### 8.5 BlockManager (`nanovllm/engine/block_manager.py`)

**KV Cache Block Management:**

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}  # For deduplication
        self.free_block_ids: deque[int] = deque()
        self.used_block_ids: set[int] = set()
    
    @staticmethod
    def compute_hash(token_ids, prefix=-1):
        """Hash tokens for deduplication"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def allocate(self, seq):
        """Allocate blocks for a sequence, deduplicating common prefixes"""
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            
            # Compute hash
            h = compute_hash(token_ids, h) if len(token_ids) == block_size else -1
            
            # Check if we've seen this block before
            block_id = hash_to_block_id.get(h, -1)
            
            if block_id == -1 or blocks[block_id].token_ids != token_ids:
                # Cache miss: need new block
                cache_miss = True
                block_id = free_block_ids[0]
                _allocate_block(block_id)
            else:
                # Cache hit: reuse existing block
                seq.num_cached_tokens += block_size
                blocks[block_id].ref_count += 1
            
            seq.block_table.append(block_id)
    
    def deallocate(self, seq):
        """Free blocks when sequence is finished"""
        for block_id in reversed(seq.block_table):
            blocks[block_id].ref_count -= 1
            if blocks[block_id].ref_count == 0:
                _deallocate_block(block_id)
```

**Key Features:**
- **Prefix Caching**: Multiple sequences can share cache blocks
- **Hash-Based Deduplication**: Detects common prefixes
- **Reference Counting**: Tracks which sequences use which blocks
- **Block Reuse**: Same block used by multiple sequences if safe

---

## 9. EXECUTION FLOW

### 9.1 Initialization

```
User code:
  llm = LLM(model_path, ...)
    └─ LLMEngine.__init__()
        ├─ Config.__post_init__()
        │   └─ AutoConfig.from_pretrained(model_path)
        ├─ ModelRunner.__init__(config, rank=0)
        │   ├─ dist.init_process_group("nccl", ...)
        │   ├─ torch.set_default_device("cuda")
        │   ├─ model = Qwen3ForCausalLM(hf_config)
        │   ├─ load_model(model, config.model)
        │   │   └─ [Load weights from safetensors]
        │   ├─ warmup_model()
        │   │   └─ Run dummy forward pass
        │   ├─ allocate_kv_cache()
        │   │   └─ Allocate KV cache tensors
        │   └─ capture_cudagraph()
        │       └─ Record inference graphs
        ├─ Create worker ModelRunners (if tp_size > 1)
        ├─ AutoTokenizer.from_pretrained(model_path)
        └─ Scheduler(config)
```

### 9.2 Generation

```
User calls: outputs = llm.generate(prompts, sampling_params)
  │
  ├─ Add requests to scheduler
  │   for prompt, sp in zip(prompts, sampling_params):
  │       tokenizer.encode(prompt)
  │       Sequence(token_ids, sp)
  │       scheduler.add(seq)
  │
  ├─ While not finished:
  │   │
  │   ├─ Scheduler.schedule()
  │   │   ├─ Try to prefill new sequences
  │   │   └─ Or decode existing sequences
  │   │   └─ Return scheduled_seqs, is_prefill
  │   │
  │   ├─ ModelRunner.run(seqs, is_prefill)
  │   │   ├─ prepare_prefill() or prepare_decode()
  │   │   ├─ run_model(input_ids, positions, is_prefill)
  │   │   │   ├─ If prefill: eager
  │   │   │   └─ If decode: CUDA graph replay
  │   │   ├─ sampler(logits, temperatures)
  │   │   └─ return token_ids
  │   │
  │   └─ Scheduler.postprocess()
  │       ├─ Append token to sequences
  │       ├─ Check finish conditions
  │       └─ Deallocate finished sequences
  │
  └─ Return results
```

### 9.3 Weight Loading Sequence

```
load_model(model, model_path)
  │
  ├─ Get packed_modules_mapping from model
  │   = {
  │       "q_proj": ("qkv_proj", "q"),
  │       "k_proj": ("qkv_proj", "k"),
  │       ...
  │     }
  │
  ├─ For each *.safetensors file in model_path:
  │   │
  │   ├─ safe_open(file, "pt", "cpu")
  │   │
  │   └─ For each weight_name in file:
  │       │
  │       ├─ Check if weight_name matches packed module key:
  │       │   │
  │       │   ├─ If match:
  │       │   │   ├─ k, (v, shard_id) = packed_modules_mapping[k]
  │       │   │   ├─ param_name = weight_name.replace(k, v)
  │       │   │   ├─ param = model.get_parameter(param_name)
  │       │   │   ├─ weight_loader = param.weight_loader
  │       │   │   └─ weight_loader(param, loaded_weight, shard_id)
  │       │   │
  │       │   └─ Else: (not packed)
  │       │       ├─ param = model.get_parameter(weight_name)
  │       │       ├─ weight_loader = getattr(param, "weight_loader", 
  │       │       │                           default_weight_loader)
  │       │       └─ weight_loader(param, loaded_weight)
  │
  └─ [All weights loaded]
```

---

## 10. EXAMPLE USAGE

### example.py

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Load model and tokenizer
path = "~/huggingface/Qwen3-0.6B/"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

# Create sampling params
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

---

## 11. KEY DESIGN PATTERNS

### 11.1 Distributed Communication

```python
# Used in all parallel layers
import torch.distributed as dist

# Initialize
dist.init_process_group("nccl", "tcp://localhost:2333", 
                        world_size=tp_size, rank=rank)

# Synchronization primitives
dist.all_reduce(tensor)      # Sum across all ranks
dist.barrier()               # Wait for all ranks
dist.gather(local, global)   # Gather to rank 0
```

### 11.2 Context Threading

```python
# Global context for attention
from nanovllm.utils.context import get_context, set_context

# Set context for current forward pass
set_context(is_prefill=True, cu_seqlens_q=..., ...)

# Attention reads from context
context = get_context()
if context.is_prefill:
    # Use prefill-specific logic
```

### 11.3 Custom Parameter Loaders

```python
# Each parameter can have custom loader
param.weight_loader = custom_load_function

def custom_load_function(param: nn.Parameter, loaded_weight, shard_id=None):
    # Shard loaded_weight and copy to param
    param.data.copy_(sharded_weight)
```

### 11.4 CUDA Graphs

```python
# Record computation graph once
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input)

# Replay many times
graph.replay()
```

---

## 12. TENSOR PARALLELISM

### 12.1 How It Works

```
GPU 0                  GPU 1
┌────────────────┐    ┌────────────────┐
│  Q proj (half) │    │  Q proj (half) │
│  K proj (half) │    │  K proj (half) │
│  V proj (half) │    │  V proj (half) │
└────────────────┘    └────────────────┘
       │                      │
    Linear.forward() ──── AllReduce
       │                      │
    [attention computation]
       │                      │
    Linear.forward() ──── AllReduce
```

### 12.2 Column Parallel (Q, K, V projections)

- Each GPU: output_size // tp_size features
- Load: Shard checkpoint weight
- Forward: Output already sharded, no sync needed
- (Sync happens at next RowParallel layer)

### 12.3 Row Parallel (Output projections)

- Each GPU: input_size // tp_size inputs
- Load: Shard checkpoint weight
- Forward: Local matmul + AllReduce sum
- Output: Same on all GPUs

### 12.4 AllGather (LM Head)

- Output logits must be on GPU 0 only
- AllGather: Collect all vocab chunks from all GPUs
- Concat: Full vocabulary logits

---

## 13. PREFILL VS DECODE

### 13.1 Prefill Phase

**When:** Processing new requests (initial prompt)
**Characteristics:**
- Variable sequence lengths
- Many tokens processed in parallel
- One forward pass processes many tokens
- Uses cumulative sequence length trick

**Optimizations:**
- Batch multiple sequences
- Use varlen attention (flash-attn)
- No CUDA graphs (variable input shapes)

### 13.2 Decode Phase

**When:** Generating one token at a time
**Characteristics:**
- Fixed batch size
- One token per sequence
- Many small forward passes
- Reuse KV cache

**Optimizations:**
- CUDA graphs (same shapes repeatedly)
- In-place KV cache updates
- Minimal memory movement
- High throughput (tokens/sec)

---

## 14. KV CACHE MANAGEMENT

### 14.1 Memory Layout

```
K Cache:
  Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
  [0] = K cache
  [1] = V cache

Per sequence:
  - block_table: list of block IDs
  - num_cached_tokens: how many tokens already cached
  - Blocks are referenced by multiple sequences (prefix cache)
```

### 14.2 Slot Mapping

```
slot_mapping[token_i] = block_id * block_size + offset_in_block

During prefill:
  - Compute which slot each token should write to
  - Triton kernel stores K/V at exact slots

During decode:
  - Each sequence has one slot in last block
  - Update KV at that slot
```

### 14.3 Prefix Caching

```
Request 1:  "hello world" → cache blocks [0, 1]
Request 2:  "hello there" → reuse block 0 (same "hello")
                            → allocate new block [0, 2]

BlockManager detects prefix match via hashing
Only allocates new blocks for diverging tokens
```

---

## 15. SUMMARY

### Project Statistics

| Category | Count |
|----------|-------|
| Python Files | 19 |
| Total LOC | ~1500 |
| Models | 1 (Qwen3) |
| Core Classes | 12 |
| Tensor Parallel Support | Yes |
| Distributed Support | Yes (NCCL) |
| CUDA Optimization | Yes (Graphs) |
| Attention Backend | Flash-Attention |

### Key Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| models/qwen3.py | 216 | Qwen3 model implementation |
| engine/model_runner.py | 252 | Model execution |
| engine/scheduler.py | 72 | Batching and scheduling |
| layers/linear.py | 154 | Tensor parallel linear layers |
| engine/block_manager.py | 113 | KV cache management |
| layers/attention.py | 76 | Attention with flash-attn |
| utils/loader.py | 29 | Weight loading |

### Architecture Highlights

1. **Model Loading**: SafeTensors → Custom weight loaders → Tensor parallel sharding
2. **Inference**: LLMEngine → Scheduler → ModelRunner → Qwen3ForCausalLM
3. **Batching**: Mix prefill (variable length) and decode (fixed length)
4. **Memory**: Block-based KV cache with prefix caching and deduplication
5. **Performance**: CUDA graphs for decode, flash-attn for attention
6. **Distribution**: NCCL with ColumnParallel and RowParallel layers

---

## 16. ENTRY POINTS

### Public API

```python
# Main entry point
from nanovllm import LLM, SamplingParams

llm = LLM(model_path, **config_kwargs)
outputs = llm.generate(prompts, sampling_params)
```

### Configuration

```python
Config(
    model="path/to/model",           # Model directory
    max_num_batched_tokens=16384,    # Max tokens in batch
    max_num_seqs=512,                # Max sequences
    tensor_parallel_size=1,          # Number of GPUs
    enforce_eager=False,             # Skip CUDA graphs
    gpu_memory_utilization=0.9,      # GPU memory fraction
)
```

### Model Loading

```python
# Automatic loading via:
1. LLMEngine.__init__()
2. ModelRunner.__init__()
3. load_model(model, config.model)
4. For each *.safetensors file:
   - Extract weights
   - Apply packed_modules_mapping
   - Call custom weight_loader
   - Handle tensor parallel sharding
```

---

**End of Comprehensive Codebase Guide**
