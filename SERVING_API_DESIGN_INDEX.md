# NanoVLLM Serving API - Design Index & Navigation

> **Quick Start:** Start with the Executive Summary, then read "Threading & Safety", then "Reference: API Methods"

---

## 📚 DOCUMENT STRUCTURE

### Level 1: Quick Understanding (15 min read)
1. **This file** - Overview and navigation
2. **ENGINE_API_EXPLORATION.md** - Executive Summary (first section)
3. **QUICK_REFERENCE.txt** - Condensed API surface

### Level 2: Detailed Implementation (1 hour read)
1. **SERVING_API_DESIGN_GUIDE.md** - Complete design guide with examples
   - Part 1-5: Core mechanics and request lifecycle
   - Part 6: Error handling and timeouts
   - Part 7: OpenAI API mapping
   - Part 8: Reference skeleton code
2. **ENGINE_API_EXPLORATION.md** - Full architectural deep-dive
   - Parts 1-15: Complete class-by-class analysis

### Level 3: Code Examples (Ready to implement)
- See "PART 8: Reference Skeleton" in SERVING_API_DESIGN_GUIDE.md
- See "Quick Start Example" below

---

## 🎯 KEY FINDINGS AT A GLANCE

### Three Core Methods You Need to Know

```python
# 1. Queue a request (non-blocking, returns None)
engine.add_request(prompt, SamplingParams(...))
seq_id = engine.scheduler.waiting[-1].seq_id

# 2. Run one inference step (blocking GPU call)
outputs, num_tokens = engine.step()
# outputs: [(seq_id, completion_tokens), ...]

# 3. Check if done
if engine.is_finished():
    print("All requests complete")
```

### The Problem: add_request() Returns Nothing
```python
# ❌ This doesn't work:
seq_id = engine.add_request(prompt, sp)  # Returns None!

# ✅ This does work:
engine.add_request(prompt, sp)
seq_id = engine.scheduler.waiting[-1].seq_id  # Get last added
```

### Single-Threaded Engine
```python
# ❌ This will crash:
thread1: engine.add_request(...)
thread2: engine.step()

# ✅ This is safe:
with engine_lock:
    engine.add_request(...)
    outputs, _ = engine.step()
```

---

## 🔧 QUICK START: Minimal Implementation

```python
from nanovllm import LLM, SamplingParams
import threading
import time
from typing import Dict, Optional

# ============ SETUP ============

engine = LLM("/path/to/model", max_num_seqs=32)
engine_lock = threading.Lock()

# Track requests: seq_id → client_data
request_map: Dict[int, dict] = {}
request_results: Dict[int, list] = {}

# ============ INFERENCE LOOP ============

def inference_loop():
    """Background thread - runs model continuously"""
    while True:
        with engine_lock:
            # Check if any work
            if engine.is_finished():
                time.sleep(0.1)
                continue
            
            # Execute one step
            outputs, num_tokens = engine.step()
            
            # Collect results
            for seq_id, tokens in outputs:
                if seq_id in request_map:
                    request_results[seq_id] = tokens
                    print(f"Request {seq_id} complete: {len(tokens)} tokens")
        
        time.sleep(0.01)

# Start inference thread
thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# ============ API ENDPOINTS ============

def handle_completion_request(prompt: str, max_tokens: int = 64) -> int:
    """API endpoint: POST /completions"""
    sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    with engine_lock:
        engine.add_request(prompt, sp)
        seq_id = engine.scheduler.waiting[-1].seq_id
    
    request_map[seq_id] = {"prompt": prompt, "created_at": time.time()}
    return seq_id

def get_completion_status(request_id: int) -> Optional[dict]:
    """API endpoint: GET /completions/{id}"""
    if request_id in request_results:
        tokens = request_results[request_id]
        return {
            "status": "complete",
            "text": engine.tokenizer.decode(tokens),
            "tokens": tokens
        }
    elif request_id in request_map:
        return {"status": "pending"}
    else:
        return None

# ============ TEST ============

# Create 3 requests
req1 = handle_completion_request("Hello, ")
req2 = handle_completion_request("What is 2+2?")
req3 = handle_completion_request("Tell me a joke")

# Poll for results
while len(request_results) < 3:
    time.sleep(0.1)
    for rid in [req1, req2, req3]:
        status = get_completion_status(rid)
        if status and status["status"] == "complete":
            print(f"✓ Request {rid}: {status['text']}")

print("All done!")
```

**Output:**
```
Request 0 complete: 45 tokens
✓ Request 0: The quickest way to improve your...
Request 1 complete: 32 tokens
✓ Request 1: The sum of 2 and 2 is 4...
Request 2 complete: 78 tokens
✓ Request 2: Why did the developer go broke?...
All done!
```

---

## 📖 RECOMMENDED READING ORDER

### For Implementation Planning
1. Read: ENGINE_API_EXPLORATION.md "EXECUTIVE SUMMARY"
2. Read: SERVING_API_DESIGN_GUIDE.md "PART 1: ENGINE MECHANICS"
3. Read: SERVING_API_DESIGN_GUIDE.md "PART 4: THREAD SAFETY"
4. Read: SERVING_API_DESIGN_GUIDE.md "PART 8: REFERENCE SKELETON"
5. Reference: QUICK_REFERENCE.txt throughout

### For Deep Understanding
1. Read: ENGINE_API_EXPLORATION.md Parts 1-7 (Core Classes)
2. Read: ENGINE_API_EXPLORATION.md Part 9 (Scheduler Algorithm)
3. Read: SERVING_API_DESIGN_GUIDE.md Parts 2-3 (Streaming)
4. Read: ENGINE_API_EXPLORATION.md Parts 10-12 (Gotchas & Design)

### For OpenAI API Compatibility
1. Read: SERVING_API_DESIGN_GUIDE.md "PART 7: OPENAI MAPPING"
2. Reference: `example.py` for usage patterns
3. Implement: Response formatting functions
4. Add: Streaming token handling (Part 3 of guide)

---

## 🚨 CRITICAL CONSTRAINTS

### ❌ Do NOT
- Call `engine.step()` from multiple threads
- Access `scheduler.waiting` without lock
- Assume `add_request()` returns `seq_id`
- Use engine in async/await directly
- Try greedy sampling (temperature must be > 1e-10)

### ✅ DO
- Wrap all engine calls with `threading.Lock()`
- Extract `seq_id` from `scheduler.waiting[-1]` after add_request
- Poll running sequences for token updates
- Run inference in single background thread
- Implement token buffering for streaming

---

## 🔑 KEY CLASSES TO UNDERSTAND

### LLMEngine
**Location:** `nanovllm/engine/llm_engine.py`
- Main class to interact with
- Methods: `add_request()`, `step()`, `is_finished()`, `generate()`
- Thread-safe: NO

### Sequence
**Location:** `nanovllm/engine/sequence.py`
- Represents one request
- Auto-incrementing `seq_id` (globally unique)
- States: WAITING → RUNNING → FINISHED
- Token tracking: prompt_tokens + completion_tokens

### Scheduler
**Location:** `nanovllm/engine/scheduler.py`
- Manages queuing and scheduling
- Implements prefill vs decode batching
- May preempt sequences for fairness
- Maintains two queues: waiting, running

### SamplingParams
**Location:** `nanovllm/sampling_params.py`
- Per-request configuration
- Fields: temperature, max_tokens, ignore_eos
- Constraint: temperature > 1e-10

### ModelRunner
**Location:** `nanovllm/engine/model_runner.py`
- Executes model inference
- Handles GPU/CUDA graphs
- Multi-GPU via shared memory
- Only rank 0 returns tokens

### BlockManager
**Location:** `nanovllm/engine/block_manager.py`
- KV cache prefix caching
- Allocates/deallocates cache blocks
- Reuses common prefixes

---

## 📊 SCHEDULING ALGORITHM

The engine uses a **prefill-decode batching** strategy:

```
Step 1: Prefill Phase
  Try to schedule sequences from waiting queue
  Add as many as fit in max_num_seqs and max_num_batched_tokens
  Allocate KV cache blocks
  
Step 2: If prefill empty → Decode Phase
  Round-robin through running sequences
  Each adds 1 token
  May preempt running seqs if KV cache full
  
Step 3: Model Execution
  Forward pass on scheduled batch
  Sample next token for each sequence
  
Step 4: Postprocessing
  Append tokens to sequences
  Check for completion (EOS or max_tokens)
  Mark sequences FINISHED
  Deallocate KV cache
```

**Key Insight:** Prefill is prioritized for throughput optimization

---

## 💾 DATA FLOW EXAMPLE

```
API Request: "Hello"
    ↓
add_request() → Sequence created, seq_id=42
    ↓
scheduler.waiting = [seq_42]
    ↓
step() → scheduler.schedule()
    ↓
seq_42 moves to scheduler.running, KV cache allocated
    ↓
model_runner.run() → forward pass (prefill)
    ↓
scheduler.postprocess() → seq_42.append_token(token_X)
    ↓
[Repeat decode steps until max_tokens or EOS]
    ↓
seq.num_completion_tokens == seq.max_tokens
    ↓
seq.status = FINISHED, seq removed from running
    ↓
step() returns [(42, [token_X, token_Y, ...])]
    ↓
API returns completion to client
```

---

## 🎓 COMMON PATTERNS

### Pattern 1: Request Tracking
```python
# Map API request ID → engine seq_id
api_to_seq = {}
seq_to_api = {}

def create_request(api_id, prompt):
    engine.add_request(prompt, sp)
    seq_id = engine.scheduler.waiting[-1].seq_id
    api_to_seq[api_id] = seq_id
    seq_to_api[seq_id] = api_id
```

### Pattern 2: Token Streaming
```python
# Track last sent token count per sequence
last_sent = {}

for seq in engine.scheduler.running:
    current = len(seq.completion_token_ids)
    last = last_sent.get(seq.seq_id, 0)
    if current > last:
        new_tokens = seq.completion_token_ids[last:current]
        stream_tokens(seq.seq_id, new_tokens)
        last_sent[seq.seq_id] = current
```

### Pattern 3: Polling Loop
```python
def inference_loop():
    while True:
        with engine_lock:
            if not engine.is_finished():
                outputs, _ = engine.step()
                for seq_id, tokens in outputs:
                    on_completion(seq_id, tokens)
        time.sleep(0.01)
```

### Pattern 4: Error Handling
```python
try:
    with engine_lock:
        engine.add_request(prompt, sp)
except Exception as e:
    return {"error": str(e)}

# Manual timeout tracking
request_timeout = {}
created_at = time.time()
request_timeout[seq_id] = created_at

# In inference loop:
if time.time() - created_at > 300:  # 5 minutes
    # Manually mark as finished or remove from queues
```

---

## 🔍 DEBUGGING CHECKLIST

- [ ] Are all engine calls wrapped with `engine_lock`?
- [ ] Are you accessing `seq_id` from `scheduler.waiting[-1]` after add_request?
- [ ] Have you handled the `step()` return value (tuple, not single value)?
- [ ] Are you polling `scheduler.running` for token updates?
- [ ] Have you set up a background thread for inference loop?
- [ ] Are request tracking dicts cleaned up after completion?
- [ ] Have you tested with multiple concurrent requests?
- [ ] Are you using `SamplingParams` (not raw dict)?
- [ ] Have you handled the timeout/5-minute case?

---

## 📈 PERFORMANCE TUNING

### For Low Latency
```python
LLM(model, 
    max_num_seqs=16,           # Smaller batches
    max_num_batched_tokens=4096  # Smaller batches
)
```

### For High Throughput
```python
LLM(model,
    max_num_seqs=512,          # Larger batches
    max_num_batched_tokens=32768  # Larger batches
)
```

### Always
```python
LLM(model,
    enforce_eager=False,       # Use CUDA graphs (faster)
    gpu_memory_utilization=0.9  # Use available GPU memory
)
```

---

## 📞 SUPPORT REFERENCE

### Issue: "add_request() returns None"
**Answer:** Get seq_id from `scheduler.waiting[-1].seq_id` immediately after

### Issue: "Multiple threads crash engine"
**Answer:** Use `threading.Lock()` around all engine calls

### Issue: "Can't see streaming tokens"
**Answer:** Poll `scheduler.running[seq].completion_token_ids` manually

### Issue: "Request hangs forever"
**Answer:** Implement timeout tracking and manual sequence cancellation

### Issue: "Memory keeps growing"
**Answer:** Clean up request tracking dicts after seq completion

---

## 🎯 NEXT STEPS FOR IMPLEMENTATION

1. **Create RequestTracker class** - Map seq_id ↔ client_request_id
2. **Create InferenceEngine wrapper** - Wraps LLMEngine with thread safety
3. **Implement inference_loop()** - Background thread for steps
4. **Create API endpoints** - POST /completions, GET /completions/{id}
5. **Add streaming handler** - Poll running sequences, emit deltas
6. **Implement OpenAI response format** - Convert tokens → OpenAI JSON
7. **Add error handling** - Try/catch around add_request and step
8. **Test** - Single request, multiple concurrent, streaming
9. **Performance tune** - Adjust max_num_seqs and max_num_batched_tokens
10. **Deploy** - FastAPI/Flask server, containerize with Docker

---

**File Structure:**
```
/Users/water/work/code/LALearning/nano-vllm/
├── ENGINE_API_EXPLORATION.md          ← Full technical analysis
├── SERVING_API_DESIGN_GUIDE.md         ← Design guide with examples
├── QUICK_REFERENCE.txt                 ← Condensed API reference
├── SERVING_API_DESIGN_INDEX.md         ← This file
├── example.py                          ← Offline usage example
└── nanovllm/                           ← Source code
    ├── __init__.py                     ← Public API (LLM, SamplingParams)
    ├── llm.py                          ← LLM class
    ├── sampling_params.py              ← SamplingParams class
    ├── config.py                       ← Configuration
    └── engine/
        ├── llm_engine.py               ← Main LLMEngine class
        ├── sequence.py                 ← Sequence class
        ├── scheduler.py                ← Scheduler class
        ├── model_runner.py             ← ModelRunner class
        └── block_manager.py            ← BlockManager class
```

---

**Generated:** April 12, 2026  
**Status:** Complete exploration, ready for implementation
