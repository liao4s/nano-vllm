# NanoVLLM - Complete Codebase Exploration

## 1. PROJECT OVERVIEW

**NanoVLLM** is a lightweight vLLM implementation built from scratch in approximately 1,200 lines of Python code. It's a minimal but production-grade LLM inference engine with:
- Fast offline inference comparable to vLLM
- Readable, educational codebase
- Advanced optimization suite (prefix caching, tensor parallelism, CUDA graphs, torch compilation)

**Target Model**: Qwen3-0.6B (though architecture supports any model)

---

## 2. DIRECTORY STRUCTURE

```
nano-vllm/
├── nanovllm/                          # Main package
│   ├── __init__.py                   # Exports: LLM, SamplingParams
│   ├── llm.py                        # LLM class (wrapper around LLMEngine)
│   ├── config.py                     # Config dataclass
│   ├── sampling_params.py            # SamplingParams dataclass
│   ├── layers/                       # Core neural network layers
│   │   ├── __init__.py
│   │   ├── attention.py              # Attention + Flash Attention + KV cache storage
│   │   ├── linear.py                 # Parallel linear layers (TP support)
│   │   ├── rotary_embedding.py       # RoPE implementation
│   │   ├── activation.py             # SiLU activation
│   │   ├── layernorm.py              # RMSNorm implementation
│   │   ├── embed_head.py             # Embedding & LM head (TP-aware)
│   │   └── sampler.py                # Token sampling
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   └── qwen3.py                  # Qwen3 model (Attention, MLP, DecoderLayer, Model, ForCausalLM)
│   ├── engine/                       # Inference engine
│   │   ├── __init__.py
│   │   ├── llm_engine.py             # LLMEngine (main entry point, orchestrates inference)
│   │   ├── model_runner.py           # ModelRunner (model execution, KV cache, CUDA graph)
│   │   ├── scheduler.py              # Scheduler (manages sequence scheduling & block allocation)
│   │   ├── sequence.py               # Sequence (represents a single generation sequence)
│   │   └── block_manager.py          # BlockManager (KV cache block management & prefix caching)
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── loader.py                 # Model weight loading from safetensors
│       └── context.py                # Execution context (thread-local inference state)
├── example.py                        # Usage example script
├── bench.py                          # Benchmark script
├── README.md                         # Project README
└── qwen3.5/                          # Model weights directory (Qwen3-0.6B)
```

---

## 3. MODEL LOADING INFRASTRUCTURE

### Entry Point: LLMEngine.__init__()

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # 1. Parse config from kwargs
        config = Config(model, **config_kwargs)
        
        # 2. Create multi-process setup for tensor parallelism
        for i in range(1, config.tensor_parallel_size):
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
        
        # 3. Initialize rank-0 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 4. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
```

### ModelRunner Initialization

**Location**: `nanovllm/engine/model_runner.py`

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 1. Initialize NCCL process group for distributed inference
        dist.init_process_group("nccl", "tcp://localhost:2333", 
                               world_size=self.world_size, rank=rank)
        
        # 2. Set CUDA device and dtype
        torch.cuda.set_device(rank)
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 3. Create model (Qwen3ForCausalLM)
        self.model = Qwen3ForCausalLM(hf_config)
        
        # 4. Load model weights
        load_model(self.model, config.model)
        
        # 5. Create sampler for token generation
        self.sampler = Sampler()
        
        # 6. Warmup model
        self.warmup_model()
        
        # 7. Allocate KV cache
        self.allocate_kv_cache()
        
        # 8. Capture CUDA graphs (if not enforce_eager)
        self.capture_cudagraph()
```

### Model Weight Loading

**Location**: `nanovllm/utils/loader.py`

The loader handles:
1. **Packed weights mapping**: Maps single loaded weights to multiple parameter positions
   - `q_proj`, `k_proj`, `v_proj` → `qkv_proj` (packed into one linear layer)
   - `gate_proj`, `up_proj` → `gate_up_proj` (fused MLP)

2. **Safetensors loading**: Loads weights from multiple `.safetensors` files
3. **Custom weight loaders**: Uses `weight_loader` attribute on parameters for custom loading logic (important for tensor parallelism)

```python
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Check if this weight needs to be mapped/fused
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param = model.get_parameter(...)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
```

---

## 4. QWEN3 MODEL IMPLEMENTATION

**File**: `nanovllm/models/qwen3.py`

Complete Qwen3 model implementation with full tensor parallelism support:

### A. Qwen3Attention

```python
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        # Tensor-parallel aware head division
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size  # Divide for TP
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads // tp_size
        
        # Fused QKV projection
        self.qkv_proj = QKVParallelLinear(...)
        
        # Output projection with all-reduce for TP
        self.o_proj = RowParallelLinear(...)
        
        # RoPE (Rotary Position Embeddings)
        self.rotary_emb = get_rope(...)
        
        # Optional Q/K normalization (if no bias)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(...)
            self.k_norm = RMSNorm(...)
        
        # Flash attention wrapper
        self.attn = Attention(...)
    
    def forward(self, positions, hidden_states):
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size])
        
        # 2. Reshape to attention heads
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 3. Apply normalization if needed
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 4. Apply RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # 5. Flash Attention (with KV cache)
        o = self.attn(q, k, v)
        
        # 6. Output projection
        output = self.o_proj(o.flatten(1, -1))
        return output
```

### B. Qwen3MLP

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        # Fused gate and up projections (SwiGLU)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], ...
        )
        # Output projection
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, ...)
        # Activation: SiLU with element-wise multiplication
        self.act_fn = SiluAndMul()  # Applies SiLU to gate, element-wise mul with up
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # [batch, seq_len, 2*intermediate_size]
        x = self.act_fn(gate_up)  # Split, SiLU(gate) * up
        x = self.down_proj(x)
        return x
```

### C. Qwen3DecoderLayer

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def forward(self, positions, hidden_states, residual=None):
        # Pre-norm with residual connection
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # Post-attention norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

### D. Qwen3Model

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(...)
    
    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

### E. Qwen3ForCausalLM

```python
class Qwen3ForCausalLM(nn.Module):
    # Maps loaded weights to fused parameter positions
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(...)
        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

---

## 5. INFERENCE ENGINE

### A. LLMEngine (Orchestrator)

**File**: `nanovllm/engine/llm_engine.py`

Main orchestration class that:

1. **Initialization**:
   - Creates Config from kwargs
   - Spawns multi-process ModelRunner instances for tensor parallelism
   - Initializes tokenizer
   - Creates Scheduler

2. **Request Management**:
   ```python
   def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
       if isinstance(prompt, str):
           prompt = self.tokenizer.encode(prompt)
       seq = Sequence(prompt, sampling_params)
       self.scheduler.add(seq)
   ```

3. **Generation Loop**:
   ```python
   def generate(self, prompts, sampling_params, use_tqdm=True):
       # Add all requests
       for prompt, sp in zip(prompts, sampling_params):
           self.add_request(prompt, sp)
       
       # Run inference steps until done
       while not self.is_finished():
           outputs, num_tokens = self.step()  # Returns finished sequences
       
       # Decode and return
       return [{"text": tokenizer.decode(ids), "token_ids": ids} for ids in outputs]
   ```

4. **Step Function**:
   ```python
   def step(self):
       seqs, is_prefill = self.scheduler.schedule()  # Get batched sequences
       token_ids = self.model_runner.call("run", seqs, is_prefill)  # Run model
       self.scheduler.postprocess(seqs, token_ids)  # Update sequences
       outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
       return outputs, num_tokens
   ```

### B. Scheduler

**File**: `nanovllm/engine/scheduler.py`

Manages sequence scheduling with two phases:

1. **Prefill Phase**:
   - Load sequences from waiting queue
   - Check if block manager can allocate KV cache blocks
   - Allocate blocks and move to running queue
   - Returns sequences for prefill computation

2. **Decode Phase**:
   - Process running sequences one decode step
   - Check if block manager can append (extend KV cache)
   - Preempt sequences if memory constrained
   - Returns sequences for decode computation

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # Phase 1: Prefill - load new sequences
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
            not self.block_manager.can_allocate(seq)):
            break
        self.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    
    if scheduled_seqs:
        return scheduled_seqs, True  # True = prefill phase
    
    # Phase 2: Decode - process running sequences
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            # Memory pressure: preempt lower priority sequences
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    
    return scheduled_seqs, False  # False = decode phase
```

### C. Sequence

**File**: `nanovllm/engine/sequence.py`

Represents a single generation sequence:

```python
class Sequence:
    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)  # Unique ID
        self.status = SequenceStatus.WAITING
        self.token_ids = token_ids.copy()  # All tokens (prompt + completion)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0  # For prefix caching
        self.block_table = []  # KV cache block IDs
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
```

### D. Block Manager (KV Cache + Prefix Caching)

**File**: `nanovllm/engine/block_manager.py`

Manages KV cache blocks with **prefix caching** for memory efficiency:

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}  # Prefix cache: hash → block_id
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
    
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # Hash tokens to detect matching prefixes
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # Include previous block's hash
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def allocate(self, seq: Sequence):
        # During prefill, detect and reuse matching prefix blocks
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # No matching prefix cache
            
            if cache_miss:
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size  # Use cached block
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                else:
                    self._allocate_block(block_id)
            
            seq.block_table.append(block_id)
```

### E. ModelRunner (Execution)

**File**: `nanovllm/engine/model_runner.py`

Handles actual model execution:

#### Initialization
1. Initialize NCCL process group (distributed training setup)
2. Create model (Qwen3ForCausalLM)
3. Load model weights
4. Warmup model (dummy forward pass)
5. Allocate KV cache tensors
6. Optionally capture CUDA graphs

#### KV Cache Allocation
```python
def allocate_kv_cache(self):
    # Calculate available GPU memory
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    
    # Calculate block bytes
    num_kv_heads = hf_config.num_key_value_heads // world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    
    # Allocate blocks (2 for K and V, per layer, per block)
    config.num_kvcache_blocks = int(total * gpu_utilization - ...) // block_bytes
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    
    # Attach to attention modules
    for layer in self.model.layers:
        layer.attn.k_cache = self.kv_cache[0, layer_id]
        layer.attn.v_cache = self.kv_cache[1, layer_id]
```

#### Run Function
```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    # Prepare inputs
    input_ids, positions = (self.prepare_prefill(seqs) if is_prefill 
                           else self.prepare_decode(seqs))
    
    # Get temperatures for sampling (only on rank 0)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    
    # Forward pass
    logits = self.run_model(input_ids, positions, is_prefill)
    
    # Sample tokens (only on rank 0)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    
    return token_ids
```

#### Prefill Preparation
```python
def prepare_prefill(self, seqs: list[Sequence]):
    # Pack all sequences with flash attention varlen format
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # Cumulative sequence lengths (for new tokens)
    cu_seqlens_k = [0]  # Cumulative sequence lengths (including cached)
    slot_mapping = []   # KV cache slot mappings
    
    for seq in seqs:
        seqlen = len(seq)
        # Only include new tokens (after cached tokens)
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, seqlen))
        
        # Update cumulative lengths
        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        
        # Map tokens to KV cache slots
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            block_id = seq.block_table[i]
            start = block_id * block_size
            if i != seq.num_blocks - 1:
                end = start + block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(range(start, end))
    
    # Convert to tensors and set global context
    set_context(True, cu_seqlens_q, cu_seqlens_k, ...)
    return input_ids_tensor, positions_tensor
```

#### Decode Preparation
```python
def prepare_decode(self, seqs: list[Sequence]):
    # For decode, process each sequence's last token
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.last_slot)
    
    set_context(False, slot_mapping, context_lens, block_tables)
    return input_ids_tensor, positions_tensor
```

#### CUDA Graph Capture
```python
def capture_cudagraph(self):
    # Capture graphs for different batch sizes (1, 2, 4, 8, 16, 32, ...)
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, ...)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # Warmup
        
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # Capture
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        
        self.graphs[bs] = graph
    
    # Store variables for graph replay
    self.graph_vars = dict(input_ids=..., positions=..., ...)
```

---

## 6. EXECUTION CONTEXT

**File**: `nanovllm/utils/context.py`

Global thread-local context for inference state:

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None  # Cumulative seq lens (Q)
    cu_seqlens_k: torch.Tensor | None = None  # Cumulative seq lens (K/V)
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None  # KV cache slot indices
    context_lens: torch.Tensor | None = None  # Context lengths for decode
    block_tables: torch.Tensor | None = None  # Block table for prefix caching

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, ...):
    global _CONTEXT
    _CONTEXT = Context(...)
```

This context is accessed by:
- `Attention` layer: Determines prefill vs decode path, uses block_tables for cached KVs
- `ParallelLMHead`: Selects which tokens to compute logits for (prefill: last per seq, decode: all)

---

## 7. TENSOR PARALLELISM

**Files**: `nanovllm/layers/linear.py`, `nanovllm/layers/embed_head.py`

Implements 4 parallelization patterns:

### A. Replicated Linear
- Replicates weight across all ranks
- Simple forward: `F.linear(x, W, b)`
- Usage: Rare

### B. Column Parallel Linear
- Splits output dimension across ranks
- Each rank computes `output_size / tp_size` dimensions
- Forward: Local linear, no communication
- Used in: Attention QKV projection, MLP gate+up

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, output_size // tp_size, bias)
    
    def weight_loader(self, param, loaded_weight):
        # Load only my shard
        shard_size = param.size(0)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight[start_idx:start_idx+shard_size])
```

### C. Row Parallel Linear
- Splits input dimension across ranks
- Each rank operates on `input_size / tp_size` dimensions
- All-reduce to combine outputs
- Used in: Attention output, MLP down projection

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size // tp_size, output_size, bias)
    
    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # Sum across all ranks
        return y
```

### D. Merged Column Parallel Linear
- Multiple outputs split across ranks
- Usage: MLP gate+up projection (2 outputs)
- Custom weight loader for fused weights

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes, bias=False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
    
    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # loaded_shard_id indicates which output (0 for gate, 1 for up)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param.data = param.data[shard_offset:shard_offset+shard_size]
        param.data.copy_(loaded_weight.chunk(self.tp_size)[self.tp_rank])
```

### E. Vocabulary Parallel Embedding & LM Head
- Each rank handles `vocab_size / tp_size` tokens
- Outputs masked (valid only for rank with token's embedding)
- All-reduce to sum contributions

```python
class VocabParallelEmbedding(nn.Module):
    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

---

## 8. LAYER IMPLEMENTATIONS

### A. Attention (Flash Attention + KV Cache)

**File**: `nanovllm/layers/attention.py`

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.k_cache = self.v_cache = torch.tensor([])  # Attach KV cache
    
    def forward(self, q, k, v):
        context = get_context()
        
        # 1. Store KV to cache (via Triton kernel)
        if k_cache and v_cache:
            store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        
        if context.is_prefill:
            # Prefill: use flash_attn_varlen_func with optional prefix cache
            if context.block_tables is not None:
                k, v = k_cache, v_cache  # Use cached KVs from prefix
            o = flash_attn_varlen_func(q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                block_table=context.block_tables, causal=True, ...)
        else:
            # Decode: use flash_attn_with_kvcache
            o = flash_attn_with_kvcache(q, k_cache, v_cache,
                cache_seqlens=context_lens,
                block_table=block_tables,
                causal=True, ...)
        
        return o
```

Includes Triton kernel for efficient KV cache storage:
```python
@triton.jit
def store_kvcache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr,
                        slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key = tl.load(key_ptr + idx * D + tl.arange(0, D))
    value = tl.load(value_ptr + idx * D + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

### B. RMSNorm with Residual

**File**: `nanovllm/layers/layernorm.py`

Pre-norm architecture with fused add + norm:

```python
class RMSNorm(nn.Module):
    @torch.compile
    def rms_forward(self, x):
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight)
    
    @torch.compile
    def add_rms_forward(self, x, residual):
        x = x.float().add_(residual.float())  # Fused add
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight), residual
```

### C. Rotary Embeddings (RoPE)

**File**: `nanovllm/layers/rotary_embedding.py`

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position, base):
        # Pre-compute cos/sin cache
        inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
        t = arange(max_position)
        freqs = einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = cat((cos, sin), dim=-1).unsqueeze(1)
        self.register_buffer("cos_sin_cache", cache)
    
    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return cat((y1, y2), dim=-1).to(x.dtype)
```

### D. SiLU + Mul (Gating)

**File**: `nanovllm/layers/activation.py`

```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y  # gate * up
```

### E. Sampler

**File**: `nanovllm/layers/sampler.py`

Temperature-scaled sampling:

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick for efficient sampling
        sample_tokens = (probs / torch.exp(torch.rand_like(probs).log())
                        .clamp_min(1e-10)).argmax(dim=-1)
        return sample_tokens
```

---

## 9. CONFIGURATION

**File**: `nanovllm/config.py`

```python
@dataclass
class Config:
    model: str  # Path to model directory
    max_num_batched_tokens: int = 16384  # Max tokens in batch
    max_num_seqs: int = 512  # Max sequences in batch
    max_model_len: int = 4096  # Max sequence length
    gpu_memory_utilization: float = 0.9  # GPU memory fraction for KV cache
    tensor_parallel_size: int = 1  # Number of GPUs
    enforce_eager: bool = False  # Disable CUDA graphs
    hf_config: AutoConfig | None = None  # HF config (loaded in __post_init__)
    eos: int = -1  # EOS token ID (set by tokenizer)
    kvcache_block_size: int = 256  # KV cache block size
    num_kvcache_blocks: int = -1  # Calculated based on GPU memory
```

---

## 10. SAMPLING PARAMETERS

**File**: `nanovllm/sampling_params.py`

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0  # Sampling temperature (must be > 1e-10)
    max_tokens: int = 64  # Max tokens to generate
    ignore_eos: bool = False  # Whether to generate past EOS token
```

---

## 11. EXAMPLE SCRIPTS

### A. example.py - Basic Usage

```python
from nanovllm import LLM, SamplingParams

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["introduce yourself", "list all prime numbers within 100"]

# Apply chat template
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
) for prompt in prompts]

outputs = llm.generate(prompts, sampling_params)
# outputs[i] = {"text": str, "token_ids": list[int]}
```

### B. bench.py - Benchmark

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=False, max_model_len=4096)

# Generate random token sequences
prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))] 
                    for _ in range(256)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, 
                                  max_tokens=randint(100, 1024))
                   for _ in range(256)]

llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
```

---

## 12. MULTI-GPU TENSOR PARALLELISM

### Process Architecture

```
Main Process (Rank 0)
├── ModelRunner (GPU 0) - Controls execution
├── IPC SharedMemory - Communicates with other ranks
└── Spawned Processes
    ├── ModelRunner (Rank 1, GPU 1)
    ├── ModelRunner (Rank 2, GPU 2)
    └── ModelRunner (Rank 3, GPU 3)

Communication:
- Main loop: Rank 0 sends method calls via SharedMemory
- Worker ranks: Wait for event, read from SharedMemory, execute
- NCCL: All-reduce operations for TP computations
```

### Initialization Flow

```python
# Rank 0 initialization
for i in range(1, tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)

self.model_runner = ModelRunner(config, 0, self.events)

# Worker rank initialization (in spawned process)
if rank > 0:
    dist.init_process_group(...)
    # ... model loading ...
    self.loop()  # Wait for commands from rank 0
```

### Method Invocation

```python
# Rank 0 calls method on all ranks
token_ids = self.model_runner.call("run", seqs, is_prefill)

# ModelRunner.call() implementation
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)  # Send to workers
    method = getattr(self, method_name)
    return method(*args)
```

---

## 13. KEY OPTIMIZATIONS

### 1. **Prefix Caching**
- Blocks with matching token sequences reuse KV cache
- Hash-based prefix detection using xxhash
- Reduces memory and compute for long contexts

### 2. **CUDA Graphs**
- Captures decode kernels for batch sizes 1, 2, 4, 8, 16+
- Graph replay eliminates CPU overhead
- ~10% speedup vs eager execution

### 3. **Flash Attention**
- Uses flash-attn library for efficient attention
- Variadic length support (flash_attn_varlen_func)
- KV cache-aware variant (flash_attn_with_kvcache)

### 4. **Torch Compilation**
- @torch.compile on:
  - RMSNorm
  - SiluAndMul activation
  - Sampler
  - RoPE forward
- Reduces overhead and fuses operations

### 5. **Tensor Parallelism**
- Replicated input embeddings with masking
- Column parallel for projections
- Row parallel with all-reduce for outputs
- Scales to multiple GPUs

### 6. **Batch Packing & Scheduling**
- Efficient scheduler with prefill + decode phases
- Preemption on memory pressure
- Block-based KV cache allocation

---

## 14. DATAFLOW: FROM PROMPT TO OUTPUT

```
Input: ["Hello", "World"], SamplingParams
  ↓
1. Tokenization (Hugging Face tokenizer)
  ↓
2. Add requests to Scheduler
  ["token_ids": [1, 2, 3], ...]
  ↓
3. Prefill phase:
   - Scheduler: Allocate KV cache blocks via BlockManager
   - ModelRunner: Prepare batched inputs with FlashAttn format
   - Model: Forward pass (embedding → attention → MLP → logits)
   - KV cache: Store K, V to cache
   - Sampler: Sample next tokens
  ↓
4. Decode phase (repeat):
   - Model: Single token forward
   - Attention: Use cached KV, compute new Q
   - KV cache: Append new K, V
   - Sample next token
  ↓
5. Stop when:
   - Token = EOS and not ignore_eos
   - num_completion_tokens == max_tokens
  ↓
6. Decode output tokens → text
  ↓
Output: [{"text": "...", "token_ids": [...]}]
```

---

## 15. CODE STATISTICS

- **Total Files**: 16 Python modules
- **Lines of Code**: ~1,200 (excluding model weights)
- **Main Components**:
  - Engine: ~600 LOC (llm_engine, model_runner, scheduler, sequence, block_manager)
  - Models: ~200 LOC (Qwen3 architecture)
  - Layers: ~300 LOC (attention, linear, embeddings, activations, etc.)
  - Utils: ~100 LOC (loader, context)

---

## 16. KEY DESIGN PATTERNS

### 1. **Context Pattern**
- Global mutable context for inference state
- Accessed by layers during forward pass
- Enables layer cooperation without tight coupling

### 2. **Weight Loader Pattern**
- Parameters store custom weight_loader function
- Supports tensor parallelism + weight fusion
- Generic load_model() works with any model

### 3. **Process Pool Pattern**
- Spawned workers for tensor parallelism
- SharedMemory for rank-0 to rank-n communication
- Event synchronization for method dispatch

### 4. **Block Manager Pattern**
- Abstraction for KV cache memory
- Supports prefix caching via hash table
- Efficient allocation/deallocation

### 5. **Scheduler Pattern**
- Two-phase scheduling (prefill + decode)
- Preemption on memory pressure
- Batching without padding

---

## 17. FUTURE EXTENSIBILITY

To add new models:
1. Create new file in `nanovllm/models/`
2. Implement `ModelClass(nn.Module)` with:
   - `packed_modules_mapping` (if weight fusion)
   - `forward(input_ids, positions)` method
   - `compute_logits(hidden_states)` method
3. Update ModelRunner to instantiate new model
4. Ensure all layers support tensor parallelism

To add new optimizations:
- CUDA kernels: Add to respective layer files
- Graph capture: Extend capture_cudagraph()
- Scheduling: Modify Scheduler.schedule()
- Caching: Extend BlockManager

---

## SUMMARY

NanoVLLM is a minimalist but sophisticated LLM inference engine featuring:

✅ **Architectural Clarity**: Clear separation of concerns (engine, model, layers, utils)
✅ **Production Features**: Batching, KV caching, prefix caching, CUDA graphs, tensor parallelism
✅ **Performance**: Comparable to vLLM in throughput
✅ **Educational Value**: ~1,200 LOC, easy to understand and extend
✅ **Generality**: Supports any transformer model with adapter pattern

The codebase demonstrates how to build an efficient inference engine from first principles while maintaining readability and extensibility.

