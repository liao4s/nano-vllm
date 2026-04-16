# Prefix Caching - Quick Reference Guide

## File Locations
- **Block Manager**: `nanovllm/engine/block_manager.py`
- **Sequence**: `nanovllm/engine/sequence.py`
- **Model Runner**: `nanovllm/engine/model_runner.py`
- **Scheduler**: `nanovllm/engine/scheduler.py`
- **LLM Engine**: `nanovllm/engine/llm_engine.py`
- **Config**: `nanovllm/config.py`

---

## Core Data Structures

### 1. Block (block_manager.py, Lines 8-24)
```python
block.block_id      # Unique ID for this block
block.ref_count     # Number of sequences sharing this block
block.hash          # xxhash64 of token_ids (or -1 if incomplete)
block.token_ids     # Actual tokens in this block
```

### 2. BlockManager (block_manager.py, Lines 26-113)
```python
self.blocks                  # List of all Block objects
self.hash_to_block_id        # CRITICAL: Map hash → block_id for cache lookup
self.free_block_ids          # Queue of unallocated block IDs
self.used_block_ids          # Set of currently allocated block IDs
```

### 3. Sequence (sequence.py, Lines 18-29)
```python
seq.num_tokens          # Total tokens (prompt + generated)
seq.num_prompt_tokens   # Original prompt size
seq.num_cached_tokens   # Tokens from prefix cache hits (UPDATED BY BLOCK_MANAGER)
seq.block_table         # List of block IDs allocated to this sequence
```

---

## Critical Methods & Line Numbers

### Hash Computation (block_manager.py, Lines 36-41)
**Purpose**: Compute hierarchical hash for block deduplication
```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
```
**Key**: `prefix` parameter includes previous block's hash for hierarchical matching

### Block Allocation (block_manager.py, Lines 59-82)
**Purpose**: Allocate blocks for a sequence, detecting prefix cache hits
**Key Lines**:
- Line 65: `h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1`
  - Only full blocks are hashed (incomplete last block gets -1)
- Line 66-68: Cache hit detection
  - `block_id = self.hash_to_block_id.get(h, -1)` → lookup in cache
  - If found AND tokens match → CACHE HIT
- Line 73: `seq.num_cached_tokens += self.block_size` → ONLY on cache hit
- Line 75-76: Reference counting for shared blocks

### Scheduler Prefill (scheduler.py, Lines 24-41)
**Purpose**: Schedule sequence prefill, using cache hit info
**Key Lines**:
- Line 34: `self.block_manager.allocate(seq)` → updates num_cached_tokens
- Line 35: `num_batched_tokens += len(seq) - seq.num_cached_tokens`
  - CRUCIAL: Only new tokens count toward batch limits

### Model Runner Prefill (model_runner.py, Lines 278-317)
**Purpose**: Prepare inputs for prefill considering cached tokens
**Key Lines**:
- Line 289: `input_ids.extend(seq[seq.num_cached_tokens:])`
  - Skip cached tokens, only process new ones
- Line 299: `for i in range(seq.num_cached_blocks, seq.num_blocks):`
  - Process only non-cached blocks for slot_mapping
- Line 306-307: **Prefix cache detection**
  ```python
  if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # Prefix cache
      block_tables = self.prepare_block_tables(seqs)
  ```
  - When K > Q, some tokens are cached, need block_tables for KV access

---

## Data Flow: Prefix Cache Hit

### Phase 1: First Sequence (New Blocks)
```
Sequence [token_0...token_256...token_512...token_768]
           └─ Block 0      └─ Block 1      └─ Block 2

allocate(seq1):
  Block 0: hash_0 = compute_hash([0-256])
           hash_to_block_id[hash_0] = 0
           ref_count = 1
           num_cached_tokens = 0  (MISS)
  
  Block 1: hash_1 = compute_hash([256-512], prefix=hash_0)
           hash_to_block_id[hash_1] = 1
           ref_count = 1
           num_cached_tokens = 0  (MISS)
  
  Block 2: incomplete, not hashed
           num_cached_tokens = 0
```

### Phase 2: Second Sequence (Shared Prefix)
```
Sequence [token_0...token_256...token_512...token_768...token_1024]
           └─ Block 0      └─ Block 1      └─ Block 2       └─ Block 3

allocate(seq2):
  Block 0: hash_0 = compute_hash([0-256])
           Found in hash_to_block_id → block 0 EXISTS
           Tokens match → CACHE HIT
           block 0: ref_count = 2  ✓
           num_cached_tokens = 256  ✓
  
  Block 1: hash_1 = compute_hash([256-512], prefix=hash_0)
           Found in hash_to_block_id → block 1 EXISTS
           Tokens match → CACHE HIT
           block 1: ref_count = 2  ✓
           num_cached_tokens = 512  ✓
  
  Block 2: hash_2 = compute_hash([512-768], prefix=hash_1)
           NOT found in hash_to_block_id → CACHE MISS
           Allocate new block 2
           cache_miss = True
           num_cached_tokens = 512  (no increment)
  
  Block 3: incomplete, not hashed
           num_cached_tokens = 512
```

### Phase 3: Scheduler Impact
```
scheduler.schedule():
  seq1: num_batched_tokens += 768 - 0 = 768
  seq2: num_batched_tokens += 1024 - 512 = 512  ✓ REDUCED
        Total computation reduced by 512 tokens!
```

### Phase 4: Model Runner
```
prepare_prefill([seq1, seq2]):
  seq1:
    input_ids: seq1[0:768]         (all tokens)
    positions: range(0, 768)
    cu_seqlens_q: [0, 768, 768+512]
    cu_seqlens_k: [0, 768, 768+1024]
    slot_mapping: all blocks (no cache)
  
  seq2:
    input_ids: seq2[512:1024]      (skip cached!)
    positions: range(512, 1024)
    cu_seqlens_q: [0, 768, 768+512]
    cu_seqlens_k: [0, 768, 768+1024]
    slot_mapping: blocks 2, 3 only (skip cached)
  
  if cu_seqlens_k[-1] (1792) > cu_seqlens_q[-1] (1280):
    block_tables = prepare_block_tables([seq1, seq2])
    ↓
    Attention kernel uses block_tables to access cached KV from blocks 0, 1
```

---

## Key Properties to Monitor

### Before allocate()
```python
seq.num_cached_tokens = 0
seq.block_table = []
```

### After allocate()
```python
seq.num_cached_tokens > 0     # Updated by block_manager
seq.block_table = [...]       # Filled with block IDs
seq.num_cached_blocks = seq.num_cached_tokens // 256
```

### During prepare_prefill()
```python
seqlen = len(seq)                           # Total tokens
seqlen_q = seqlen - seq.num_cached_tokens   # New tokens to compute
seqlen_k = seqlen                           # All tokens for KV cache
```

---

## Reference Counting Lifecycle

```
NEW BLOCK:
  block.ref_count = 1

SEQUENCE 1 ALLOCATES:
  block.ref_count += 1  (seq1)     → ref_count = 1

SEQUENCE 2 ALLOCATES (CACHE HIT):
  block.ref_count += 1  (seq2)     → ref_count = 2

SEQUENCE 1 DEALLOCATES:
  block.ref_count -= 1             → ref_count = 1
  Block NOT freed (seq2 still using)

SEQUENCE 2 DEALLOCATES:
  block.ref_count -= 1             → ref_count = 0
  if ref_count == 0: move to free_block_ids  ✓
```

---

## Incomplete Block Handling

**Key Rule**: Last (potentially incomplete) blocks are NOT cached

```python
# In allocate() - Line 65
h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
                                      ↑
                                      Only hash full blocks
```

**Why?** Different sequences may continue differently from same partial prefix.

```
Seq A: [0-256][256-512][512-768]           ← Block 2 incomplete (256 tokens)
Seq B: [0-256][256-512][512-660][660-768] ← Block 2 incomplete (148 tokens)
                                             Different size → can't cache
```

**Resolution**: When block becomes complete (decode phase):
```python
# In may_append() - Lines 104-110
elif len(seq) % self.block_size == 0:
    # Block 2 now has 256 tokens, compute final hash
    h = self.compute_hash(token_ids, prefix)
    block.update(h, token_ids)
    hash_to_block_id[h] = block_id
```

---

## Config Parameters

```python
# config.py - Lines 190-191
kvcache_block_size: int = 256      # Tokens per block
num_kvcache_blocks: int = -1       # Total blocks (computed from GPU mem)
```

Affects:
- `num_cached_blocks = num_cached_tokens // 256`
- `num_blocks = (num_tokens + 255) // 256`

---

## Common Issues & Debugging

### Issue: num_cached_tokens not updating
**Check**: Is `allocate()` being called? (scheduler.py, line 34)
**Check**: Are blocks hitting cache? (block_manager.py, lines 66-68)
**Check**: Is `len(token_ids) == self.block_size`? (block_manager.py, line 65)

### Issue: Sequence skipping computation for wrong tokens
**Check**: `seqlen_q = seqlen - seq.num_cached_tokens` (model_runner.py, line 291)
**Check**: `input_ids.extend(seq[seq.num_cached_tokens:])` (model_runner.py, line 289)

### Issue: Incorrect attention computation
**Check**: `if cu_seqlens_k[-1] > cu_seqlens_q[-1]` (model_runner.py, line 306)
**Check**: Block tables passed to model? (model_runner.py, line 307)

### Issue: Memory leak (blocks not freed)
**Check**: `ref_count` reaching 0? (block_manager.py, line 88)
**Check**: `deallocate()` called on sequence finish? (scheduler.py, line 70)

---

## Testing Checklist

```
□ Allocate two sequences with shared 256-token prefix
  └─ Check: seq1.num_cached_tokens = 0, seq2.num_cached_tokens = 256
  └─ Check: block0.ref_count = 2, block1.ref_count = 1

□ Run prefill for seq2
  └─ Check: only 256 new tokens in input_ids (not 512)
  └─ Check: positions start at 256 (not 0)
  └─ Check: block_tables provided to model

□ Deallocate seq1, keep seq2
  └─ Check: block0.ref_count = 1 (still in use)

□ Deallocate seq2
  └─ Check: block0 moved to free_block_ids
  └─ Check: num_cached_tokens = 0

□ Same prefix, different models
  └─ Check: hash collision handling (line 67)
  └─ Check: token_ids verification
```

