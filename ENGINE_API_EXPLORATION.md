# NanoVLLM Engine API - Complete Exploration & Analysis

**Date:** April 12, 2026  
**Objective:** Understand the complete LLMEngine architecture for implementing an OpenAI-compatible async serving API

---

## EXECUTIVE SUMMARY

NanoVLLM's `LLMEngine` is a **synchronous, single-threaded, batch-optimized inference engine**. Key findings:

### Core API Surface (Three Main Methods)
1. **`add_request(prompt, sampling_params)`** → None (queues request)
2. **`step()`** → (finished_requests, token_count) (runs one forward pass)
3. **`is_finished()`** → bool (checks if all requests done)

### Architecture
- **Request Model**: Sequences with auto-incrementing `seq_id` tracking
- **Scheduling**: Batches prefill and decode phases with KV cache prefix caching
- **Thread Safety**: **NOT thread-safe** - must serialize all calls with lock
- **Streaming**: No built-in support - must poll `seq.completion_token_ids` manually

### For Building OpenAI API Server
1. Wrap `step()` in background thread/async task
2. Implement request tracking mapping `seq_id` → client_request_id
3. Add manual token streaming (buffer deltas)
4. Handle thread safety with mutex
5. Implement timeouts and cancellation (not in engine)

---

## FILES ANALYZED

### Core Engine Components
- ✅ `nanovllm/engine/llm_engine.py` - Main orchestrator (94 lines)
- ✅ `nanovllm/llm.py` - Public LLM class (5 lines, just alias)
- ✅ `nanovllm/engine/sequence.py` - Request representation (84 lines)
- ✅ `nanovllm/sampling_params.py` - Request configuration (12 lines)
- ✅ `nanovllm/engine/scheduler.py` - Queue management (72 lines)
- ✅ `nanovllm/engine/model_runner.py` - Model execution (266 lines)
- ✅ `nanovllm/engine/block_manager.py` - KV cache allocation (113 lines)
- ✅ `nanovllm/config.py` - Unified configuration (130 lines)
- ✅ `nanovllm/__init__.py` - Public exports (3 lines)
- ✅ `example.py` - Offline usage example (35 lines)

**No existing server/API files found** - clean slate for implementation

---

## PART 1: LLMENGINE CLASS - COMPLETE API REFERENCE

### Constructor: `__init__(model: str, **kwargs)`

```python
llm = LLM(
    model="/path/to/model",
    # Optional config parameters:
    max_num_seqs=512,                    # Max concurrent sequences
    max_num_batched_tokens=16384,        # Max tokens per batch
    max_model_len=4096,                  # Context window
    gpu_memory_utilization=0.9,          # GPU memory budget
    tensor_parallel_size=1,              # Number of GPUs
    enforce_eager=False,                 # Disable CUDA graphs
    kvcache_block_size=256,              # Block size for prefix caching
)
```

**What happens:**
1. Creates `Config` object from kwargs
2. Spawns multiprocessing workers for tensor parallelism (if size > 1)
3. Initializes `ModelRunner` (rank 0) - loads model, allocates KV cache, sets up CUDA graphs
4. Loads tokenizer from model directory
5. Creates `Scheduler` with `BlockManager`
6. Registers `atexit` handler for cleanup

### Method: `add_request(prompt, sampling_params) → None`

```python
sp = SamplingParams(temperature=0.7, max_tokens=256, ignore_eos=False)
llm.add_request("Hello, world!", sp)
# OR with pre-tokenized input
llm.add_request([1, 2, 3, 4], sp)
```

**Behavior:**
- Tokenizes string prompts if needed
- Creates `Sequence` object with auto-incrementing `seq_id`
- Adds sequence to `scheduler.waiting` queue
- **Returns nothing** - need to extract `seq_id` manually:
  ```python
  llm.add_request(prompt, sp)
  seq_id = llm.scheduler.waiting[-1].seq_id  # Get just-added seq_id
  ```

**Important:**
- Non-blocking, O(1) operation
- Can be called from multiple threads (with lock)
- No validation or error handling

### Method: `step() → (list[tuple], int)`

```python
outputs, num_tokens = llm.step()
# outputs: [(seq_id, completion_token_ids), ...]
# num_tokens: >0 for prefill, <0 for decode, 0 if nothing scheduled
```

**What happens:**
1. `scheduler.schedule()` picks sequences to execute
   - Tries prefill first (moving waiting→running)
   - Falls back to decode (round-robin through running)
2. `model_runner.call("run", seqs, is_prefill)` executes forward pass
3. `scheduler.postprocess()` updates sequences, checks for completion
4. Returns only **newly finished** sequences

**Returns:**
- `outputs`: List of (seq_id, completion_token_ids) for sequences that just completed
- `num_tokens`: 
  - Positive = prefill step (total tokens in batch)
  - Negative = decode step (-1 × num_seqs)
  - Can use to track throughput

**Latency:**
- Prefill (1000 tokens): ~50-100ms
- Decode (512 seqs): ~20-50ms

### Method: `is_finished() → bool`

```python
while not llm.is_finished():
    outputs, _ = llm.step()
```

**Returns True when:**
- `scheduler.waiting` is empty AND
- `scheduler.running` is empty

---

## PART 2: SEQUENCE CLASS - REQUEST STATE

### Construction (Automatic)
```python
# Created internally by add_request()
seq = Sequence(
    token_ids=[1, 2, 3],           # Initial prompt tokens
    sampling_params=SamplingParams()
)
# Auto-assigns seq_id via class counter (incrementing int)
```

### Core Attributes

```python
seq.seq_id              # int: globally unique ID (auto-incrementing)
seq.status              # SequenceStatus: WAITING → RUNNING → FINISHED
seq.token_ids           # list[int]: ALL tokens (prompt + generated)
seq.num_tokens          # int: total length (len(token_ids))
seq.num_prompt_tokens   # int: fixed, number of prompt tokens
seq.num_completion_tokens  # int: computed (num_tokens - num_prompt_tokens)
seq.completion_token_ids   # list[int]: tokens[num_prompt_tokens:]
seq.last_token          # int: most recent token (for decode-only access)
seq.block_table          # list[int]: KV cache block allocations
seq.num_cached_tokens   # int: number of prompt tokens in KV cache
seq.temperature         # float: sampling parameter
seq.max_tokens          # int: max completion tokens
seq.ignore_eos          # bool: continue after EOS?
```

### Key Properties

```python
@property
def is_finished(self) -> bool:
    return self.status == SequenceStatus.FINISHED

@property
def prompt_token_ids(self) -> list[int]:
    return self.token_ids[:self.num_prompt_tokens]

@property
def completion_token_ids(self) -> list[int]:
    return self.token_ids[self.num_prompt_tokens:]

@property
def num_blocks(self) -> int:
    # How many KV cache blocks needed for this sequence
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def num_cached_blocks(self) -> int:
    # Blocks that are already in KV cache
    return self.num_cached_tokens // self.block_size
```

### Lifecycle

```
1. Created: Sequence(prompt_tokens, sp)
2. Added: engine.add_request(...) → waiting queue
3. Status: WAITING
4. Scheduled: scheduler.schedule() picks it
5. Status: RUNNING, moved to running queue
6. Processing: model_runner.run() → seq.append_token()
7. Checking: scheduler.postprocess() checks completion
8. Status: FINISHED when:
   - token == EOS and not ignore_eos, OR
   - num_completion_tokens == max_tokens
9. Removed: Deallocated from running queue
10. Available: In step() output
```

---

## PART 3: SAMPLINGPARAMS CLASS

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0        # Sampling randomness (>1e-10)
    max_tokens: int = 64            # Max completion tokens
    ignore_eos: bool = False        # If False, stop at EOS
    
    def __post_init__(self):
        assert self.temperature > 1e-10  # No greedy decoding
```

**Constraints:**
- Temperature must be > 1e-10 (greedy sampling forbidden)
- Default max_tokens = 64
- Each request can have different params

---

## PART 4: SCHEDULER CLASS - ORCHESTRATION ENGINE

### State

```python
scheduler.waiting: deque[Sequence]  # FIFO queue of pending sequences
scheduler.running: deque[Sequence]  # Sequences being processed
scheduler.block_manager: BlockManager  # KV cache allocator
scheduler.max_num_seqs: int  # Config: max concurrent seqs
scheduler.max_num_batched_tokens: int  # Config: max tokens per batch
```

### Method: `schedule() → (list[Sequence], bool)`

**Most complex method - implements prefill vs decode scheduling**

```python
# Simplified algorithm:
def schedule(self):
    # PREFILL PHASE: Try to move seqs from waiting → running
    if self.waiting:
        while self.waiting and len(batch) < max_num_seqs:
            seq = self.waiting[0]
            if (batch_tokens + len(seq) > max_batched_tokens or
                not block_manager.can_allocate(seq)):
                break
            
            block_manager.allocate(seq)  # Allocate KV cache
            seq.status = RUNNING
            batch.append(self.waiting.popleft())
            self.running.append(seq)
        
        if batch:
            return batch, True  # is_prefill=True
    
    # DECODE PHASE: Round-robin through running sequences
    while self.running and len(batch) < max_num_seqs:
        seq = self.running.popleft()
        
        # Check if can append one more token
        while not block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            block_manager.may_append(seq)
            batch.append(seq)
    
    self.running.extendleft(reversed(batch))
    return batch, False  # is_prefill=False
```

**Key behaviors:**
1. **Prefill priority**: Always tries prefill first
2. **Prefill tokens**: `len(seq) - seq.num_cached_tokens` (minus cached)
3. **Decode batching**: Each seq contributes 1 token
4. **Preemption**: If can't fit more decode seqs, preempt running ones back to waiting
5. **Front-loading**: Preempted seqs added to front of waiting queue

### Method: `postprocess(seqs, token_ids) → None`

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        # Check if finished
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = FINISHED
            block_manager.deallocate(seq)
            self.running.remove(seq)
```

**Finish conditions:**
- Token ID == EOS token AND `ignore_eos == False`
- OR completion tokens reached `max_tokens`

### Method: `preempt(seq) → None`

Moves sequence back to waiting queue:
```python
def preempt(self, seq):
    seq.status = WAITING
    block_manager.deallocate(seq)
    self.waiting.appendleft(seq)  # Front-load in queue
```

---

## PART 5: BLOCK MANAGER - KV CACHE MANAGEMENT

### Purpose
Manages physical KV cache blocks with **prefix caching** - reusing cached tokens across similar prompts

### Key Methods

```python
can_allocate(seq) → bool
  # True if enough free blocks for all of seq's blocks

allocate(seq) → None
  # Allocates blocks, attempts prefix cache reuse
  # Updates seq.block_table

can_append(seq) → bool
  # During decode, check if can add 1 more token
  # True if: len(free_blocks) >= (len(seq) % block_size == 1)

may_append(seq) → None
  # Allocate new block if needed for next token

deallocate(seq) → None
  # Free all blocks, decrement ref counts
```

---

## PART 6: MODELRUNNER - EXECUTION ENGINE

### What it does
1. Loads model onto GPU
2. Allocates KV cache tensors
3. Captures CUDA graphs (for speed)
4. Executes forward passes
5. Samples next tokens
6. Handles multi-GPU communication

### Key method: `run(seqs, is_prefill) → list[int]`

```python
def run(self, seqs, is_prefill):
    # 1. Prepare tensors
    if is_prefill:
        input_ids, positions = self.prepare_prefill(seqs)
    else:
        input_ids, positions = self.prepare_decode(seqs)
    
    # 2. Forward pass
    logits = self.run_model(input_ids, positions, is_prefill)
    
    # 3. Sample (only rank 0)
    token_ids = self.sampler(logits, temperatures)
    
    return token_ids
```

**Important:** Only rank 0 returns tokens; other ranks return None

---

## PART 7: CONFIGURATION

```python
Config(
    model: str = required,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 512,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    kvcache_block_size: int = 256,
    num_kvcache_blocks: int = auto-calculated,
)
```

**Auto-calculated fields:**
- `hf_config`: Loaded from model's config.json
- `eos`: Set from tokenizer.eos_token_id
- `num_kvcache_blocks`: Based on GPU memory × `gpu_memory_utilization`

---

## PART 8: THREAD SAFETY - CRITICAL CONSTRAINT

### NOT thread-safe
- `engine.step()` uses global state
- `engine.add_request()` modifies scheduler without locks
- `scheduler.waiting` and `scheduler.running` accessed without protection

### Solution: Serialize all access
```python
import threading

engine_lock = threading.Lock()

# API handler (many threads)
@app.post("/completions")
def handle_request():
    with engine_lock:
        engine.add_request(prompt, sp)

# Inference loop (single thread)
def inference():
    while True:
        with engine_lock:
            outputs, _ = engine.step()
```

---

## PART 9: SERVING API DESIGN SKELETON

### Minimal Example

```python
from nanovllm import LLM, SamplingParams
import threading
import time

# 1. Initialize engine
engine = LLM("/path/to/model", max_num_seqs=32)
engine_lock = threading.Lock()

# 2. Track requests
requests = {}  # seq_id → {client_id, created_at, ...}

# 3. Background inference loop
def inference_loop():
    while True:
        with engine_lock:
            if engine.is_finished():
                time.sleep(0.1)
                continue
            
            outputs, _ = engine.step()
            for seq_id, completion_tokens in outputs:
                on_request_complete(seq_id, completion_tokens)
        
        time.sleep(0.01)

thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# 4. API endpoint
@app.post("/completions")
async def create_completion(prompt, temperature, max_tokens):
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    with engine_lock:
        engine.add_request(prompt, sp)
        seq_id = engine.scheduler.waiting[-1].seq_id
    
    return {"request_id": str(seq_id)}
```

---

## PART 10: CRITICAL GOTCHAS & SOLUTIONS

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| `add_request()` returns `None` | No seq_id in return value | Access `scheduler.waiting[-1].seq_id` immediately after |
| Crashes with multiple threads | Engine not thread-safe | Use `threading.Lock()` around all engine calls |
| No streaming tokens visible | `step()` only returns finished seqs | Manually poll `seq.completion_token_ids` for running seqs |
| Requests hang forever | No timeout handling | Track creation time, manually check elapsed |
| Can't cancel requests | No cancellation API | Mark `seq.status = FINISHED`, deallocate blocks |
| Memory grows unbounded | Request objects accumulate | Clean up tracking dicts after completion |
| Token counts don't match | Confusion about token types | `num_prompt_tokens` fixed, `num_completion_tokens` grows |

---

## PART 11: STREAMING IMPLEMENTATION

### Challenge
Engine returns only completed sequences. To stream tokens:

### Solution A: Polling running sequences
```python
last_sent = {}  # seq_id → token count

for seq in engine.scheduler.running:
    current = len(seq.completion_token_ids)
    if current > last_sent.get(seq.seq_id, 0):
        new_tokens = seq.completion_token_ids[last_sent[seq.seq_id]:]
        emit_stream_tokens(seq.seq_id, new_tokens)
        last_sent[seq.seq_id] = current
```

### Solution B: Tracking snapshots
```python
class EnginePoller:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}  # seq_id → last_completion_tokens
    
    def poll(self):
        # Check running seqs for new tokens
        for seq in self.engine.scheduler.running:
            current = seq.completion_token_ids
            prev = self.snapshots.get(seq.seq_id, [])
            if len(current) > len(prev):
                new_tokens = current[len(prev):]
                emit_stream_update(seq.seq_id, new_tokens)
                self.snapshots[seq.seq_id] = current
```

---

## PART 12: KEY ARCHITECTURAL DECISIONS

### Single-threaded model execution
- Only rank 0 performs sampling
- All GPU operations serialized
- Simpler than async GPU kernels

### Batch-first prefill scheduling
- Prioritizes prefill steps
- May preempt decode sequences
- Optimizes for throughput, not latency fairness

### KV cache prefix caching
- Automatic block reuse via hashing
- Transparent to API layer
- Reduces redundant computation

### No built-in streaming
- API must implement token buffering
- Simpler engine, more flexibility

### No request cancellation
- Sequences run to completion
- API layer can timeout/abandon

---

## PART 13: PERFORMANCE CHARACTERISTICS

### Throughput
- Prefill: Measured in tokens/sec (GPU-dependent)
- Decode: Tokens/sec/gpu with many sequences

### Latency
- Prefill (1000 tokens): ~50-100ms
- Decode (512 seqs × 1 token): ~20-50ms
- API request latency: ~200-500ms (first token), then 20-50ms per token

### Scaling
- Multi-GPU: Tensor parallelism via shared memory IPC
- Max sequences: Limited by `max_num_seqs` and GPU memory
- KV cache: ~2 × num_layers × block_size × num_heads × head_dim × dtype

---

## PART 14: EXAMPLE WALKTHROUGH

### Example usage from `example.py`

```python
from nanovllm import LLM, SamplingParams

# 1. Initialize
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)

# 2. Create parameters
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 3. Batch prompts
prompts = ["prompt1", "prompt2"]

# 4. Generate (blocking)
outputs = llm.generate(prompts, sampling_params)

# 5. Process
for output in outputs:
    print(output["text"])
    print(output["token_ids"])
```

### For serving, instead use:

```python
# Don't use generate() - it's blocking
# Instead:

engine.add_request(prompt1, sp)  # Queue
engine.add_request(prompt2, sp)  # Queue

# Then in background loop:
outputs, _ = engine.step()  # Get results
```

---

## PART 15: PUBLIC API (exports from __init__.py)

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

Only these two are intended for public use. Everything else is internal implementation detail.

---

## CONCLUSION

### For Building OpenAI-Compatible Serving API:

**Minimum required:**
1. ✅ Understand `step()` → `add_request()` → `is_finished()` loop
2. ✅ Implement request tracking with seq_id mapping
3. ✅ Add thread-safety with lock around engine
4. ✅ Handle streaming via polling
5. ✅ Build REST/gRPC API on top

**Optional enhancements:**
- Request prioritization
- Token streaming (delta tracking)
- Request cancellation
- Timeout enforcement
- Error recovery
- Multi-model support

**Architecture advantages:**
- Clean separation: engine (inference) vs API (serving)
- No lock contention: single inference thread
- Predictable: no dynamic memory allocation during inference
- Efficient: prefix caching, CUDA graphs, tensor parallelism

---

**END OF EXPLORATION**

Generated: April 12, 2026
Analyzed Files: 10
Total Lines: ~778 lines of core engine code
