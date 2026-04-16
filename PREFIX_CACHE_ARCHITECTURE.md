# Prefix Caching Architecture - Visual Guide

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM ENGINE (llm_engine.py)                     │
│  Coordinates: add_request() → schedule() → run() → postprocess()       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────────┐
   │Tokenizer│         │Scheduler│         │ModelRunner  │
   └─────────┘         └────┬────┘         └──────┬──────┘
                             │                    │
                             │ allocate()         │ prepare_prefill()
                             │                    │
                             ▼                    ▼
        ┌────────────────────────────────────────────────────┐
        │          BLOCK MANAGER (block_manager.py)          │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ hash_to_block_id: {hash → block_id}         │ │
        │  │  CORE CACHE STRUCTURE                       │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ blocks: [Block, Block, Block, ...]          │ │
        │  │  Each Block: {hash, token_ids, ref_count}   │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ free_block_ids: [deque of available IDs]    │ │
        │  │ used_block_ids: {set of allocated IDs}      │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
                             │
                             │ num_cached_tokens ← allocate()
                             ▼
        ┌────────────────────────────────────────────────────┐
        │         SEQUENCE (sequence.py)                     │
        │                                                    │
        │  num_tokens: 768          (total tokens)          │
        │  num_cached_tokens: 512   (from cache hits)       │
        │  block_table: [0, 1, 2]   (block IDs)             │
        │                                                    │
        │  Properties:                                       │
        │  • num_cached_blocks = 512 // 256 = 2            │
        │  • num_blocks = (768 + 255) // 256 = 3           │
        │                                                    │
        └────────────────────────────────────────────────────┘
```

---

## Prefix Cache Hit Detection & Data Flow

```
┌─ STEP 1: NEW REQUEST ──────────────────────────────────────────────────┐
│                                                                         │
│  Prompt: "What is machine learning? [256 tokens total]"               │
│                                                                         │
│  Scheduler.add() → waiting queue                                       │
└─ Sequence.num_cached_tokens = 0 ────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 2: SCHEDULER CALLS ALLOCATE ────────────────────────────────────┐
│                                                                         │
│  scheduler.schedule() → block_manager.allocate(seq)                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ ALLOCATION LOOP (lines 63-82):                                 │ │
│  │                                                                 │ │
│  │  for i in range(seq.num_blocks):  # 1 block (256 tokens)     │ │
│  │    token_ids = [all 256 tokens]                               │ │
│  │    h = compute_hash(token_ids, prefix=-1)                     │ │
│  │    ↓                                                           │ │
│  │    h = 0x1f2e3d4c5b6a (example xxhash64)                      │ │
│  │                                                                 │ │
│  │    block_id = hash_to_block_id.get(h, -1)                     │ │
│  │    ↓                                                           │ │
│  │    block_id = -1  (NOT FOUND - cache miss)                    │ │
│  │                                                                 │ │
│  │    Allocate: block_id = 0                                     │ │
│  │    blocks[0]: hash=h, token_ids=[...], ref_count=1            │ │
│  │    hash_to_block_id[h] = 0                                    │ │
│  │                                                                 │ │
│  │    seq.num_cached_tokens += 0  (MISS - no increment)          │ │
│  │    seq.block_table = [0]                                      │ │
│  │                                                                 │ │
│  │  Result: seq.num_cached_tokens = 0 ✓                          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Status: sequence RUNNING, ready for prefill                          │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 3: SECOND REQUEST (SAME PREFIX) ────────────────────────────────┐
│                                                                         │
│  Prompt: "What is machine learning? Give me a detailed explanation."  │
│          [same first 256 tokens] + [100 new tokens] = 356 tokens      │
│                                                                         │
│  Scheduler.add() → waiting queue                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 4: SCHEDULER CALLS ALLOCATE (2ND SEQ) ──────────────────────────┐
│                                                                         │
│  scheduler.schedule() → block_manager.allocate(seq)                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ ALLOCATION LOOP:                                               │ │
│  │                                                                 │ │
│  │  Block 0: token_ids = [same 256 tokens as first seq]           │ │
│  │    h = compute_hash(token_ids, prefix=-1)                      │ │
│  │    ↓                                                           │ │
│  │    h = 0x1f2e3d4c5b6a  (SAME HASH!)                           │ │
│  │                                                                 │ │
│  │    block_id = hash_to_block_id.get(h, -1)                     │ │
│  │    ↓                                                           │ │
│  │    block_id = 0  (FOUND!)  ✓ CACHE HIT                        │ │
│  │                                                                 │ │
│  │    Verify: blocks[0].token_ids == token_ids ✓                 │ │
│  │                                                                 │ │
│  │    ✨ CACHE HIT LOGIC (lines 73-78):                          │ │
│  │    seq.num_cached_tokens += 256  ← INCREMENTED               │ │
│  │    blocks[0].ref_count += 1  (now 2)  ← SHARED                │ │
│  │    seq.block_table = [0]                                      │ │
│  │                                                                 │ │
│  │  Block 1: token_ids = [100 new tokens] (incomplete)           │ │
│  │    len(token_ids) != 256 → h = -1                             │ │
│  │    Allocate new block 1                                       │ │
│  │    seq.num_cached_tokens = 256  (no change - incomplete)      │ │
│  │    seq.block_table = [0, 1]                                   │ │
│  │                                                                 │ │
│  │  Result: seq.num_cached_tokens = 256 ✓                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  COMPARISON:                                                            │
│  • Seq1: 256 tokens → all computed              (256 new tokens)      │
│  • Seq2: 356 tokens → only 100 computed (100 new tokens)             │
│           ↑ 156 token computations AVOIDED! ↑                         │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 5: SCHEDULER UPDATES BATCHED TOKEN COUNT ────────────────────────┐
│                                                                         │
│  scheduler.schedule() Line 35:                                         │
│  num_batched_tokens += len(seq) - seq.num_cached_tokens               │
│                                                                         │
│  Seq1: 256 - 0 = 256 new tokens                                       │
│  Seq2: 356 - 256 = 100 new tokens                                     │
│  Total: 256 + 100 = 356 new tokens (not 612)                          │
│                     ↑ MASSIVE SAVINGS                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 6: MODEL RUNNER PREPARES INPUTS ─────────────────────────────────┐
│                                                                         │
│  model_runner.prepare_prefill([seq1, seq2])                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Seq1:                                                           │ │
│  │   input_ids: extend(seq1[0:])      = all 256 tokens           │ │
│  │   positions: range(0, 256)                                     │ │
│  │   slot_mapping: all blocks (write new KV)                      │ │
│  │   cu_seqlens_q: [0, 256]  (Q sequence lengths)                │ │
│  │   cu_seqlens_k: [0, 256]  (K sequence lengths)                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Seq2:                                                           │ │
│  │   input_ids: extend(seq2[256:])   = only 100 new tokens  ✓    │ │
│  │   positions: range(256, 356)                                   │ │
│  │   slot_mapping: only block 1 (write new KV)                    │ │
│  │   cu_seqlens_q: [0, 256, 356]                                 │ │
│  │   cu_seqlens_k: [0, 256, 612]  ← Includes cached tokens       │ │
│  │                                                                 │ │
│  │   KEY DETECTION (line 306):                                    │ │
│  │   if cu_seqlens_k[-1] (612) > cu_seqlens_q[-1] (356):         │ │
│  │       block_tables = prepare_block_tables([seq1, seq2])        │ │
│  │       ↓ PROVIDES CACHED BLOCK ACCESS ↓                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  TENSORS ON GPU:                                                       │
│  • input_ids: [256 tokens from seq1] + [100 tokens from seq2]        │
│  • positions: [0-255] + [256-355]                                    │
│  • cu_seqlens_q: [0, 256, 356]                                       │
│  • cu_seqlens_k: [0, 256, 612]                                       │
│  • slot_mapping: [block0 slots] + [block1 slots]                     │
│  • block_tables: [[0, 1], [0, 1]]  ← For attention kernel            │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ STEP 7: MODEL INFERENCE ──────────────────────────────────────────────┐
│                                                                         │
│  model.forward(input_ids, positions, block_tables=block_tables)       │
│                                                                         │
│  Attention Computation:                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Q: [256 (seq1) + 100 (seq2)] = 356 new tokens                 │ │
│  │ K: [256 (seq1) + 256 (seq1) + 100 (seq2)] = 612 total tokens │ │
│  │ V: Same as K                                                   │ │
│  │                                                                 │ │
│  │ block_tables tell where to READ cached KV:                     │ │
│  │ • seq1: all from block 0 (blocks[0].k_cache, blocks[0].v_cache)
│  │ • seq2: block 0 from cache + new block 1                      │ │
│  │                                                                 │ │
│  │ Attention: newQ × (cachedKV + newKV)                          │ │
│  │                                                                 │ │
│  │ KV Cache Write (slot_mapping):                                 │ │
│  │ • block 0: already filled (from seq1 prefill)                 │ │
│  │ • block 1: write new 100 tokens                               │ │
│  │                                                                 │ │
│  │ RESULT: Computation saved on seq2 = 256 tokens               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Hash Computation Chain (Hierarchical Hashing)

```
Block 0: tokens [0-255]
  prefix = -1
  hash_0 = xxhash64(tokens[0-255])
  = 0x1f2e3d4c5b6a7f8e

Block 1: tokens [256-511]
  prefix = hash_0 = 0x1f2e3d4c5b6a7f8e
  hash_1 = xxhash64(prefix_bytes + tokens[256-511])
  = 0x9d8e7f6c5b4a3029

Block 2: tokens [512-767]
  prefix = hash_1 = 0x9d8e7f6c5b4a3029
  hash_2 = xxhash64(prefix_bytes + tokens[512-767])
  = 0x5f4e3d2c1b0a9e8d

════════════════════════════════════════════════════════════════════

NEW SEQUENCE - SAME FIRST TWO BLOCKS:

Block 0: tokens [0-255]  (same)
  hash_0 = xxhash64(tokens[0-255])
  = 0x1f2e3d4c5b6a7f8e   ← MATCHES! ✓

Block 1: tokens [256-511]  (same)
  prefix = hash_0 = 0x1f2e3d4c5b6a7f8e
  hash_1 = xxhash64(prefix_bytes + tokens[256-511])
  = 0x9d8e7f6c5b4a3029   ← MATCHES! ✓

Block 2: tokens [512-768]  (DIFFERENT - only 256 tokens vs 768)
  prefix = hash_1 = 0x9d8e7f6c5b4a3029
  hash_2 = xxhash64(prefix_bytes + tokens[512-768])
  = 0x2c3b4a5960716875   ← NEW HASH (because content different) ✗

════════════════════════════════════════════════════════════════════

KEY INSIGHT:
• Two sequences reuse first 512 tokens (blocks 0 and 1)
• Different continuation at block 2
• Hierarchical hashing detects exactly where divergence occurs
• Only compute KV cache for blocks 0, 1 once → share across sequences
```

---

## Reference Counting Lifecycle

```
Memory Allocation Timeline:
═════════════════════════════════════════════════════════════════

T=0: Seq1 allocates blocks
  ┌─────────────────────────────────────────┐
  │ Block 0: ref_count=1, hash=h0           │
  │ Block 1: ref_count=1, hash=h1           │
  │ Block 2: ref_count=1, hash=-1           │
  │ (Seq1 has 3 blocks, num_cached_tokens=0)
  └─────────────────────────────────────────┘

T=1: Seq2 allocates, cache hits on blocks 0,1
  ┌─────────────────────────────────────────┐
  │ Block 0: ref_count=2, hash=h0  ✓ SHARED │
  │ Block 1: ref_count=2, hash=h1  ✓ SHARED │
  │ Block 2: ref_count=1, hash=h2           │
  │ Block 3: ref_count=1, hash=-1           │
  │ (Seq2 has 4 blocks, num_cached_tokens=512)
  └─────────────────────────────────────────┘
  MEMORY SAVED: ~1GB (depends on model)
  COMPUTATION SAVED: 512 token computations

T=2: Seq1 finishes, deallocate
  ┌─────────────────────────────────────────┐
  │ Block 0: ref_count=1, hash=h0  ← still in use!
  │ Block 1: ref_count=1, hash=h1  ← still in use!
  │ Block 2: ref_count=0 → FREE_BLOCKS ✓
  │ Block 3: ref_count=1, hash=-1
  │ (Seq2 still running, blocks 0,1 in memory)
  └─────────────────────────────────────────┘

T=3: Seq2 finishes, deallocate
  ┌─────────────────────────────────────────┐
  │ Block 0: ref_count=0 → FREE_BLOCKS ✓
  │ Block 1: ref_count=0 → FREE_BLOCKS ✓
  │ Block 2: already freed
  │ Block 3: ref_count=0 → FREE_BLOCKS ✓
  │ (All blocks freed, ready for new sequences)
  └─────────────────────────────────────────┘
```

---

## Incomplete Block Handling

```
PROBLEM: Incomplete blocks can't be reliably cached

Scenario:
  Seq A: [block0: 256][block1: 256][block2: 128] → incomplete
  Seq B: [block0: 256][block1: 256][block2: 200] → incomplete (different size!)
  
  Can't cache block2 because:
  • Different lengths (128 vs 200)
  • Different continuation positions
  • Different hash values

SOLUTION: Lazy hashing strategy

Phase 1 - ALLOCATION (lines 63-82):
  for i in range(num_blocks):
    if len(block_tokens) == 256:
      h = compute_hash(tokens, prefix)
      hash_to_block_id[h] = block_id  ← cache full blocks
    else:
      h = -1  ← incomplete block, NOT cached
      
Phase 2 - DECODE, WHEN BLOCK COMPLETES (lines 104-110):
  if len(seq) % 256 == 0:  ← Block now has 256 tokens
    token_ids = seq.block(seq.num_blocks - 1)
    prefix = blocks[block_table[-2]].hash  ← previous block's hash
    h = compute_hash(token_ids, prefix)
    block.update(h, token_ids)
    hash_to_block_id[h] = block_id  ← now cache it
    
RESULT:
  Block only cached AFTER completion
  Other sequences can reuse it if they have same continuation
  False positives avoided
```

---

## Batch Token Accounting Impact

```
Without Prefix Caching:
═════════════════════════════════════════════════════════════════

Batched Tokens Budget: 16,384 tokens/batch

Request 1: 512 tokens
  Batched tokens = 512

Request 2: 512 tokens (same prefix)
  Batched tokens = 512 + 512 = 1,024
  
Request 3: 512 tokens (different)
  Batched tokens = 1,024 + 512 = 1,536
  ...can fit 10 such requests per batch

═════════════════════════════════════════════════════════════════

With Prefix Caching:
═════════════════════════════════════════════════════════════════

Request 1: 512 tokens (new)
  num_cached_tokens = 0
  Batched tokens = 512 - 0 = 512

Request 2: 512 tokens (shares 256-token prefix with Req1)
  num_cached_tokens = 256  ← UPDATED BY ALLOCATE
  Batched tokens = 512 - 256 = 256  ← ONLY COUNT NEW
  Total = 512 + 256 = 768

Request 3: 512 tokens (shares 256-token prefix with Req1,2)
  num_cached_tokens = 256
  Batched tokens = 512 - 256 = 256
  Total = 768 + 256 = 1,024

Request 4: 512 tokens (same)
  num_cached_tokens = 256
  Batched tokens = 512 - 256 = 256
  Total = 1,024 + 256 = 1,280

...can fit 12-13 such requests per batch (vs 10)!

THROUGHPUT INCREASE: 20-30% for workloads with common prefixes
```

---

## Model Runner Context Passing

```
prepare_prefill() context passes to attention kernel:
═════════════════════════════════════════════════════════════════

set_context(
    is_prefill=True,
    cu_seqlens_q=cu_seqlens_q,        ← Q sequence lengths (new tokens)
    cu_seqlens_k=cu_seqlens_k,        ← K sequence lengths (all tokens)
    max_seqlen_q=max_seqlen_q,        ← Max query length
    max_seqlen_k=max_seqlen_k,        ← Max key length
    slot_mapping=slot_mapping,        ← KV cache write locations
    block_tables=block_tables,        ← ONLY if prefix cache hit!
)

ATTENTION KERNEL USAGE:
═════════════════════════════════════════════════════════════════

if block_tables is not None:
    # Prefix cache is being used
    for seq_idx in range(num_seqs):
        q_start, q_end = cu_seqlens_q[seq_idx:seq_idx+2]
        k_start, k_end = cu_seqlens_k[seq_idx:seq_idx+2]
        
        # Query: only new tokens
        Q = input_ids[q_start:q_end]
        
        # Key/Value: all tokens (via block tables)
        block_table = block_tables[seq_idx]
        K = gather_from_blocks(block_table, blocks)
        V = gather_from_blocks(block_table, blocks)
        
        # Attention: new Q × all KV
        output = attention(Q, K, V)
```

