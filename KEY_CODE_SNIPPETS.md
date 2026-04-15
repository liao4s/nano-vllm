# Nano-vLLM: Key Code Snippets

## 1. Model Loading Entry Point

**File**: `nanovllm/engine/model_runner.py` (lines 15-39)

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        
        # Initialize distributed training
        dist.init_process_group("nccl", "tcp://localhost:2333", 
                               world_size=config.tensor_parallel_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # HARDCODED MODEL INSTANTIATION ← ONLY QWEN3 SUPPORTED
        self.model = Qwen3ForCausalLM(hf_config)
        
        # Load weights from safetensors
        load_model(self.model, config.model)
        
        # Initialize sampler and allocate KV cache
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        
        # Capture CUDA graphs for decode
        if not config.enforce_eager:
            self.capture_cudagraph()
```

**Key Issue**: Model hardcoded at line 31. To add new models:
```python
model_type = hf_config.model_type
if model_type == "qwen3":
    self.model = Qwen3ForCausalLM(hf_config)
elif model_type == "llama":
    self.model = LlamaForCausalLM(hf_config)
else:
    raise ValueError(f"Model {model_type} not supported")
```

---

## 2. Weight Loading Pipeline

**File**: `nanovllm/utils/loader.py` (29 lines)

```python
def load_model(model: nn.Module, path: str):
    """Load safetensors weights to model, handling packed modules."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Try to match against packed module mapping
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # Direct weight loading
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

**How it works**:
1. Reads all `*.safetensors` files in model directory
2. For each weight, checks if name matches `packed_modules_mapping` key
3. If match: extracts target module name and shard_id, calls custom `weight_loader`
4. If no match: uses default loader (direct copy)
5. Each parameter's `weight_loader` handles tensor parallel slicing

---

## 3. Packed Modules Mapping: Qwen3

**File**: `nanovllm/models/qwen3.py` (lines 186-192)

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

**Mapping Explanation**:
- HuggingFace checkpoint has separate q_proj, k_proj, v_proj
- Nano-vLLM combines them into single qkv_proj layer
- Loader unpacks HF weights and repacks into nano-vLLM structure

**Example**: For q_proj weight [vocab_size, hidden_size]:
1. Loader finds "q_proj" in weight_name
2. Maps to ("qkv_proj", "q")
3. Calls `qkv_proj.weight.weight_loader(loaded_weight, shard_id="q")`
4. QKVParallelLinear.weight_loader slices and places in correct position

---

## 4. QKVParallelLinear: Tensor Parallel Weight Loading

**File**: `nanovllm/layers/linear.py` (lines 96-128)

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, 
                 total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, 
                     loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # Narrow to TP shard for this rank
        param_data = param_data.narrow(0, shard_offset, shard_size)
        
        # Chunk loaded weight by TP size, take this rank's chunk
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        
        param_data.copy_(loaded_weight)
```

**TP Mechanism**:
- `loaded_weight.chunk(tp_size, 0)`: Split weight among TP ranks
- `[self.tp_rank]`: Get this rank's slice
- Each rank loads its portion of Q, K, V heads

---

## 5. Qwen3 Model: Architecture

**File**: `nanovllm/models/qwen3.py` (lines 161-182)

```python
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, 
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**Design Pattern**:
- Residual stream tracked separately for efficiency
- Each layer gets (positions, hidden_states, residual)
- Each layer returns (hidden_states, residual)

---

## 6. Qwen3Attention: Flash Attention + KV Cache

**File**: `nanovllm/models/qwen3.py` (lines 71-87)

```python
def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)
    
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)  # Flash attn + KV cache handling
    output = self.o_proj(o.flatten(1, -1))
    return output
```

**Key features**:
- Combined QKV projection for efficiency
- Optional Q, K normalization
- Rotary embeddings applied post-projection
- Flash attention called with context-aware mode (prefill vs decode)

---

## 7. Attention: Prefill vs Decode Mode

**File**: `nanovllm/layers/attention.py` (lines 59-75)

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    
    if context.is_prefill:
        if context.block_tables is not None:  # Prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale, causal=True, block_table=context.block_tables
        )
    else:  # Decode
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables,
            softmax_scale=self.scale, causal=True
        )
    return o
```

**Global Context Usage**:
- `get_context()` returns prefill/decode mode
- Prefill: Uses sequence lengths, supports prefix cache reuse
- Decode: Uses cached K, V from previous tokens
- Context set by `ModelRunner.run()` before forward pass

---

## 8. Main Inference Loop

**File**: `nanovllm/engine/llm_engine.py` (lines 59-93)

```python
def generate(self, prompts: list[str] | list[list[int]], 
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True) -> list[str]:
    if use_tqdm:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
    
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    
    # Add all requests
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    
    # Generate tokens until all requests finished
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        
        if use_tqdm:
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
        
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            if use_tqdm:
                pbar.update(1)
    
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), 
                "token_ids": token_ids} for token_ids in outputs]
    if use_tqdm:
        pbar.close()
    return outputs
```

**Flow**:
1. Add all prompts to scheduler.waiting
2. Loop: schedule → run → postprocess
3. Track throughput (prefill tok/s vs decode tok/s)
4. Return text and token_ids per request

---

## 9. Scheduler: Request Scheduling

**File**: `nanovllm/engine/scheduler.py` (lines 24-71)

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # Prefill phase
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
            not self.block_manager.can_allocate(seq)):
            break
        
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    
    if scheduled_seqs:
        return scheduled_seqs, True  # Prefill phase
    
    # Decode phase
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # Decode phase
```

**Key Logic**:
- Prefill: Fill batch with new sequences until tokens or seqs limit
- Decode: Add one token per sequence until batch full
- If can't append (cache full): preempt lowest priority sequence
- Returns (batch, is_prefill) for ModelRunner

---

## 10. KV Cache Block Management

**File**: `nanovllm/engine/block_manager.py` (lines 56-82)

```python
def allocate(self, seq: Sequence):
    """Allocate blocks for sequence, implementing prefix caching."""
    assert not seq.block_table
    h = -1
    cache_miss = False
    
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # Only hash full blocks
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        
        # Check if this block already exists
        block_id = self.hash_to_block_id.get(h, -1)
        
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        
        if cache_miss:
            # Allocate new block
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # Reuse cached block
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        
        seq.block_table.append(block_id)
```

**Prefix Caching**:
1. Compute xxhash of each 256-token block + prefix hash
2. Check if hash exists in `hash_to_block_id`
3. If match and tokens identical: reuse block (increase ref_count)
4. If mismatch or new: allocate new block
5. Once cache miss occurs, all subsequent blocks must be new

---

## 11. CUDA Graph Capture

**File**: `nanovllm/engine/model_runner.py` (lines 217-251)

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    
    # Allocate tensor pool for graphs
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    
    # Batch sizes to capture
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], 
                   context_lens=context_lens[:bs], 
                   block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # Warmup
        
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # Capture
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
    
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

**Graph Replay** (lines 190-206):
```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # Set input tensors
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        # Replay captured graph
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

---

## 12. Config Initialization

**File**: `nanovllm/config.py` (all 27 lines)

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, 
                                 self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

**Validation**:
- Model path must be directory
- Block size divisible by 256 (memory alignment)
- TP size 1-8
- Loads and caches HuggingFace config
- Caps max_model_len to model's limit

---

## Usage Examples

### Basic Example
```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)
params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello world"], params)
print(outputs[0]["text"])
```

### Multi-GPU
```python
llm = LLM("~/huggingface/Qwen3-0.6B/", tensor_parallel_size=2)
# Automatically spawns 2 processes via torch.multiprocessing
```

### Batch Inference
```python
prompts = ["Hello", "Hi", "Good morning"]
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
# Returns list of {"text": str, "token_ids": list[int]}
```

---

**Last Updated**: April 12, 2026
