# Prefix Caching Implementation Analysis - nano-vllm

## Overview
The nano-vllm codebase implements **prefix caching** using a hash-based block allocation mechanism. This allows sequences to share common prefixes in KV cache, reducing memory usage and redundant computations.

---

## 1. BLOCK MANAGER - Hash-Based Block Allocation
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/engine/block_manager.py`

### Key Data Structures

#### Block Class (Lines 8-24)
```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0           # Reference count for sharing
        self.hash = -1               # Hash of this block's tokens
        self.token_ids = []          # Actual token IDs in this block
```

#### BlockManager Class (Lines 26-113)
```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size                    # Size of each block (256 tokens)
        self.blocks: list[Block] = [...]                # All block objects
        self.hash_to_block_id: dict[int, int] = {}      # Maps hash → block_id (core cache structure)
        self.free_block_ids: deque[int] = deque(...)    # Unallocated blocks
        self.used_block_ids: set[int] = set()           # Currently allocated blocks
```

### Hash Computation (Lines 36-41)
```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))      # Include previous block's hash
    h.update(np.array(token_ids).tobytes())          # Hash current tokens
    return h.intdigest()
```

**Purpose:** 
- Uses xxhash for fast hashing
- `prefix` parameter enables **hierarchical hashing**: each block's hash includes the previous block's hash
- This creates a chain where identical prefixes have identical hashes

### Allocation Logic (Lines 59-82)
```python
def allocate(self, seq: Sequence):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # Compute hash only for full blocks
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        
        # Check if this block exists in cache
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        
        if cache_miss:
            # CACHE MISS: allocate new block
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # CACHE HIT: reuse existing block
            seq.num_cached_tokens += self.block_size  # Track cached tokens
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1                  # Increment reference count
            else:
                block = self._allocate_block(block_id)
        
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

**Key Points:**
- **Line 73:** `seq.num_cached_tokens += self.block_size` - only incremented on cache HIT
- **Lines 74-78:** Reference counting allows multiple sequences to share a block
- **Cache miss behavior (Line 68):** Once a miss is detected, all subsequent blocks are marked as misses
- **Incomplete blocks (Line 65):** Last block (if not full) gets hash=-1 and isn't cached

### Append Logic During Decode (Lines 96-112)
```python
def may_append(self, seq: Sequence):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    
    if len(seq) % self.block_size == 1:
        # New block needed, but previous block hash must be finalized
        assert last_block.hash != -1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
        
    elif len(seq) % self.block_size == 0:
        # Last block is now complete - compute and finalize its hash
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks-1)
        # KEY: Use previous block's hash as prefix for hierarchical hashing
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
    else:
        # Block still filling
        assert last_block.hash == -1
```

**Key Points:**
- **Line 107:** Uses **prefix hierarchical hashing** - the hash of block N includes hash of block N-1
- This is crucial: allows reuse detection across sequences

### Deallocation (Lines 84-91)
```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0  # Reset cached token count
    seq.block_table.clear()
```

---

## 2. SEQUENCE - Token and Cache Tracking
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/engine/sequence.py`

### Sequence Class (Lines 14-85)

#### Initialization (Lines 18-29)
```python
def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
    self.seq_id = next(Sequence.counter)
    self.status = SequenceStatus.WAITING
    self.token_ids = copy(token_ids)
    self.last_token = token_ids[-1]
    self.num_tokens = len(self.token_ids)        # Total tokens (prompt + generated)
    self.num_prompt_tokens = len(token_ids)      # Original prompt length
    self.num_cached_tokens = 0                   # KEY: initialized to 0
    self.block_table = []                        # Block allocation for this sequence
    self.temperature = sampling_params.temperature
    self.max_tokens = sampling_params.max_tokens
    self.ignore_eos = sampling_params.ignore_eos
```

#### Key Properties (Lines 54-59)
```python
@property
def num_cached_blocks(self):
    """Number of blocks with cached KV pairs"""
    return self.num_cached_tokens // self.block_size

@property
def num_blocks(self):
    """Total number of blocks needed for this sequence"""
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

**Critical:** 
- `num_cached_blocks = num_cached_tokens // 256`
- `num_cached_blocks` is the count of blocks that hit in cache

#### Usage in Serialization (Lines 74-85)
```python
def __getstate__(self):
    return (self.seq_id, self.num_tokens, self.num_prompt_tokens, 
            self.num_cached_tokens,  # Serialized
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)

def __setstate__(self, state):
    self.seq_id = state[0]
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[1:-1]
    # Restore token_ids or last_token
```

---

## 3. MODEL RUNNER - Prefix Cache Usage in prepare_prefill
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/engine/model_runner.py`

### prepare_prefill Method (Lines 278-317)

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    
    for seq in seqs:
        seqlen = len(seq)
        # KEY: Only process tokens AFTER cached tokens
        input_ids.extend(seq[seq.num_cached_tokens:])                    # Line 289
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))    # Line 290
        
        seqlen_q = seqlen - seq.num_cached_tokens  # Query sequence length (new tokens)
        seqlen_k = seqlen                           # Key sequence length (all tokens)
        
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        
        if not seq.block_table:    # warmup
            continue
        
        # Process ONLY non-cached blocks (lines 299-305)
        for i in range(seq.num_cached_blocks, seq.num_blocks):  # KEY: Start from cached blocks
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))
    
    # CRITICAL: Check if prefix cache is being used
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # Line 306: prefix cache hit detected
        block_tables = self.prepare_block_tables(seqs)
    
    # Convert to tensors and transfer to GPU...
```

**Prefix Cache Detection (Line 306-307):**
```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

**Explanation:**
- `cu_seqlens_k[-1]` = cumulative sum of ALL tokens (including cached)
- `cu_seqlens_q[-1]` = cumulative sum of NEW tokens only (excluding cached)
- When `k > q`, it means some tokens are cached (not being computed)
- In this case, need block_tables for attention to access cached KV from previous blocks

#### Impact on Computation
- **Line 289:** Only new tokens (after `num_cached_tokens`) go to input
- **Line 299:** Only non-cached blocks are written to slot_mapping (new KV cache slots)
- **Line 306-307:** Block tables passed to model when prefix cache hit occurs

### prepare_block_tables Method (Lines 272-276)
```python
def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables
```

**Purpose:** Creates padded block_table tensor for attention kernel access

---

## 4. SCHEDULER - Orchestrating Prefix Caching
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/engine/scheduler.py`

### Scheduler Class (Lines 8-72)

#### Initialization (Lines 10-16)
```python
def __init__(self, config: Config):
    self.max_num_seqs = config.max_num_seqs
    self.max_num_batched_tokens = config.max_num_batched_tokens
    self.eos = config.eos
    self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
    self.waiting: deque[Sequence] = deque()
    self.running: deque[Sequence] = deque()
```

#### Scheduling Logic - Prefill Phase (Lines 24-41)
```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # prefill
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        # Check allocation feasibility
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)  # KEY: Allocate blocks (cache hits counted here)
        # Only count NON-CACHED tokens toward batched token limit
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # Line 35: KEY
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
```

**Key Points (Line 35):**
```python
num_batched_tokens += len(seq) - seq.num_cached_tokens
```
- **After `allocate()`**, `num_cached_tokens` is updated with cache hits
- **Batched token accounting** reflects actual computation needed
- If sequence reuses 200 tokens from prefix cache, only 56 new tokens count toward batched limit

#### Scheduling Logic - Decode Phase (Lines 43-58)
```python
    # decode
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):  # Can we add one more block?
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)  # Update block hashes for partial blocks
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

#### Preemption (Lines 60-63)
```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # Deallocates blocks, resets num_cached_tokens
    self.waiting.appendleft(seq)
```

#### Postprocessing (Lines 65-71)
```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # Clean up blocks
            self.running.remove(seq)
```

---

## 5. LLM ENGINE - Integration Point
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/engine/llm_engine.py`

### Main Loop (Lines 50-60)
```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()  # LINE 51: Calls scheduler
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # LINE 52: Runs model
    self.scheduler.postprocess(seqs, token_ids)   # LINE 53: Updates sequences
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    # Free linear attention buffer slots for finished sequences
    for seq in seqs:
        if seq.is_finished:
            self.model_runner.call("free_linear_attn_slot", seq.seq_id)
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

---

## 6. Configuration Parameters
**File:** `/Users/water/work/code/LALearning/nano-vllm/nanovllm/config.py`

### Key Settings (Lines 180-200)
```python
@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: object = None
    eos: int = -1
    kvcache_block_size: int = 256          # Block size for prefix caching
    num_kvcache_blocks: int = -1           # Computed from GPU memory
```

---

## 7. COMPLETE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│ NEW REQUEST                                                      │
│ prompt_tokens = [1, 2, 3, ..., 256, 257, ..., 512]             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ add_request()
                       ▼
        ┌──────────────────────────────────┐
        │ Sequence object created          │
        │ num_cached_tokens = 0            │
        │ block_table = []                 │
        └──────────────────┬───────────────┘
                           │ scheduler.schedule()
                           ▼
        ┌──────────────────────────────────┐
        │ Prefill Phase                    │
        │ block_manager.allocate(seq)      │
        └──────────────────┬───────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │ HASH-BASED ALLOCATION LOOP          │
        │ For each block in sequence:         │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 0: tokens [0-255]                            │
        │   h = compute_hash([0-255], prefix=-1)             │
        │   block_id = hash_to_block_id.get(h, -1) → -1     │
        │   CACHE MISS → allocate new block                  │
        │   hash_to_block_id[h] = block_id                   │
        │   seq.block_table = [block_id]                     │
        │   seq.num_cached_tokens = 0                        │
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 1: tokens [256-511]                          │
        │   h = compute_hash([256-511], prefix=h_block0)     │
        │   block_id = hash_to_block_id.get(h, -1)           │
        │   CACHE MISS → allocate new block                  │
        │   hash_to_block_id[h] = block_id                   │
        │   seq.block_table = [block_id0, block_id1]         │
        │   seq.num_cached_tokens = 0                        │
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 2: tokens [512...] (incomplete)              │
        │   len([512...]) < 256 → h = -1                     │
        │   Allocate but don't hash yet                       │
        │   seq.block_table = [block_id0, block_id1, block_id2]
        │   seq.num_cached_tokens = 0                        │
        └──────────────────┬──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ PREFILL PHASE: model.run()                          │
        │ Processes tokens: seq[0:] (all tokens, no cache)    │
        │ Computes KV cache for all blocks                    │
        │ Returns next token                                  │
        └──────────────────┬──────────────────────────────────┘
                           │ scheduler.postprocess()
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ Append generated token, increment counters          │
        │ seq.token_ids.append(next_token)                    │
        │ seq.num_tokens += 1                                 │
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ DECODE LOOP:                                        │
        │ 1. block_manager.can_append() checks if need block  │
        │ 2. block_manager.may_append() finalizes block hash  │
        │ 3. model.run() with single token                    │
        │ 4. Repeat until sequence finishes                   │
        └──────────────────────────────────────────────────────┘


====== SECOND SEQUENCE WITH SAME PREFIX ======

        ┌──────────────────────────────────────────────────────┐
        │ NEW REQUEST (DIFFERENT USER)                        │
        │ prompt_tokens = [1, 2, 3, ..., 256, 257, ..., 768] │
        │ Same first 256 tokens, but continues to 768         │
        └──────────────────┬──────────────────────────────────┘
                           │ add_request()
                           │ scheduler.schedule()
                           │ block_manager.allocate(seq)
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ BLOCK 0: tokens [0-255]                            │
        │   h = compute_hash([0-255], prefix=-1)             │
        │   block_id = hash_to_block_id.get(h, -1) → FOUND! │
        │   CACHE HIT! (tokens match)                        │
        │   seq.num_cached_tokens += 256                     │
        │   block = blocks[block_id]                         │
        │   block.ref_count += 1  (now ref_count=2)          │
        │   seq.block_table = [SHARED_block_id]              │
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 1: tokens [256-511]                          │
        │   h = compute_hash([256-511], prefix=h_block0)     │
        │   block_id = hash_to_block_id.get(h, -1) → FOUND! │
        │   CACHE HIT!                                       │
        │   seq.num_cached_tokens += 256  (now 512)          │
        │   block.ref_count += 1  (now ref_count=2)          │
        │   seq.block_table = [shared_id0, shared_id1]       │
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 2: tokens [512-767]                          │
        │   h = compute_hash([512-767], prefix=h_block1)     │
        │   block_id = hash_to_block_id.get(h, -1) → NOT FOUND
        │   CACHE MISS (different continuation)              │
        │   Allocate new block                               │
        │   cache_miss_flag = True                           │
        │   seq.num_cached_tokens = 512  (no change)         │
        │   seq.block_table = [shared_id0, shared_id1, new_id]
        └──────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ BLOCK 3: tokens [768-...] (incomplete)             │
        │   h = -1 (incomplete)                              │
        │   Allocate but don't hash                          │
        │   seq.num_cached_tokens = 512  (no change)         │
        │   seq.block_table = [shared_id0, shared_id1, new_id, new_id2]
        └──────────────────┬──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ PREFILL PHASE: model.run()                          │
        │ INPUT: seq[seq.num_cached_tokens:]                  │
        │        = seq[512:]  (only NEW tokens, skip cached) │
        │ KEY/VALUE: All tokens with block_tables pointing    │
        │   to cached KV cache from shared blocks             │
        │ Returns next token                                  │
        └──────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────────────────────┐
        │ MEMORY SAVINGS:                                     │
        │ • Seq1: 3 blocks × 256 tokens/block × 16 bytes/token
        │ • Seq2: 1 block (shared) + 1 new block             │
        │ • Shared KV cache: ~6 MB saved (depends on model)  │
        │ • Computation: 512 tokens skipped for seq2          │
        └──────────────────────────────────────────────────────┘
```

---

## 8. KEY DATA FLOW SUMMARY

### Initial Allocation
```
Sequence → scheduler.schedule() 
         → block_manager.allocate(seq)
             ├─ For each block:
             │  ├─ compute_hash(tokens, prefix_hash)
             │  ├─ Check hash_to_block_id dictionary
             │  ├─ On HIT: increment ref_count, add to num_cached_tokens
             │  └─ On MISS: allocate new block, update hash_to_block_id
             └─ Returns with seq.num_cached_tokens set
         → num_batched_tokens += len(seq) - seq.num_cached_tokens  (actual work)
```

### Prefill Execution
```
model_runner.prepare_prefill(seqs)
├─ For each seq:
│  ├─ input_ids: seq[seq.num_cached_tokens:]  (skip cached tokens)
│  ├─ positions: range(num_cached_tokens, len(seq))
│  ├─ cu_seqlens_q: cumulative NEW tokens
│  ├─ cu_seqlens_k: cumulative ALL tokens
│  └─ slot_mapping: only for non-cached blocks
│
├─ if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # Prefix cache detected
│  └─ block_tables = prepare_block_tables(seqs)  # For cached block access
│
└─ Returns: input_ids, positions, block_tables (if cached)

model.forward(input_ids, positions, block_tables)
├─ Q: only new tokens (from input_ids)
├─ K/V: new tokens + cached from block_tables
└─ Attention: new*all matrix (uses cached KV)

KV Cache Write
├─ slot_mapping tells where to write in KV cache
└─ Only writes new KV pairs (cached blocks already filled)
```

### Decode Phase
```
block_manager.may_append(seq)
├─ if seq.len % 256 == 0:  # Block complete
│  ├─ Compute hash with prefix from previous block
│  └─ Update hash_to_block_id dictionary
│  
└─ Returns with block ready for append
```

---

## 9. Critical Implementation Details

### 1. Reference Counting (Line 76, 87)
```
Allocation:    block.ref_count += 1
Deallocation:  block.ref_count -= 1
               if ref_count == 0: deallocate_block()
```
Allows safe sharing of blocks between sequences.

### 2. Hash Collision Handling (Line 67)
```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```
Token-level verification prevents hash collision false positives.

### 3. Batched Token Accounting (Line 35)
```python
num_batched_tokens += len(seq) - seq.num_cached_tokens
```
Only new tokens count toward batch limits, enabling better throughput with cached prefixes.

### 4. Block Incompleteness (Line 65)
```python
h = self.compute_hash(...) if len(token_ids) == self.block_size else -1
```
Partial (last) blocks aren't hashed/cached, avoiding false positives with different continuations.

### 5. Prefix Hierarchy (Line 107)
```python
prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
h = self.compute_hash(token_ids, prefix)
```
Each block includes the previous block's hash in its hash, creating a prefix chain.

---

## 10. Code Paths Summary

| Operation | File | Lines | Function |
|-----------|------|-------|----------|
| **Hash Computation** | block_manager.py | 36-41 | `compute_hash()` |
| **Block Allocation** | block_manager.py | 59-82 | `allocate()` |
| **Cache Hit Detection** | block_manager.py | 66-68 | allocate() loop |
| **Reference Count** | block_manager.py | 75-76 | allocate() |
| **Cached Tokens Tracking** | sequence.py | 25, 73 | `__init__()`, block_manager.py |
| **Cached Blocks Property** | sequence.py | 54-55 | `num_cached_blocks` |
| **Scheduler Integration** | scheduler.py | 31-35 | `schedule()` prefill |
| **Batched Token Limit** | scheduler.py | 35 | `schedule()` |
| **Prefill Preparation** | model_runner.py | 289-290, 299 | `prepare_prefill()` |
| **Prefix Cache Detection** | model_runner.py | 306-307 | `prepare_prefill()` |
| **Block Finalization** | block_manager.py | 104-110 | `may_append()` |

---

## Summary

Prefix caching in nano-vllm uses **xxhash-based hierarchical block hashing** to:
1. **Detect** identical token prefixes across sequences
2. **Share** KV cache blocks with reference counting
3. **Skip computation** for cached tokens during prefill
4. **Account** only new tokens toward batch token limits
5. **Maintain** correct attention masking via block tables

This dramatically reduces memory and computation for workloads with common prompts or multi-turn conversations.
