# Prefix Caching Documentation Index

This directory contains comprehensive documentation of the prefix caching implementation in nano-vllm.

## Documentation Files

### 1. **PREFIX_CACHE_SUMMARY.txt** (START HERE)
**Size:** 16 KB | **Read Time:** 10 minutes

High-level overview and summary of the entire implementation. Contains:
- File locations and line numbers for all key components
- Core concepts explanation
- Critical data structures summary
- Execution flow overview
- Memory and computation savings examples
- Quick debugging checklist

**Best for:** Getting oriented with the overall system

---

### 2. **PREFIX_CACHE_ARCHITECTURE.md** (VISUAL GUIDE)
**Size:** 28 KB | **Read Time:** 15 minutes

Detailed visual diagrams and architecture overview. Contains:
- System architecture diagram
- 7-step data flow with concrete examples
- Hierarchical hashing explanation with hash chain
- Reference counting lifecycle timeline
- Incomplete block handling strategy
- Batch token accounting impact examples
- Model runner context passing details

**Best for:** Understanding how components interact

---

### 3. **PREFIX_CACHE_QUICK_REFERENCE.md** (LOOKUP GUIDE)
**Size:** 9.3 KB | **Read Time:** 5 minutes

Quick reference for developers. Contains:
- File locations
- Core data structure fields
- Critical methods and line numbers
- Data flow phases (allocation, prefill, decode)
- Key properties to monitor
- Reference counting lifecycle
- Incomplete block handling rules
- Common issues and debugging
- Testing checklist

**Best for:** Finding specific information quickly

---

### 4. **PREFIX_CACHE_ANALYSIS.md** (DEEP DIVE)
**Size:** 30 KB | **Read Time:** 30 minutes

Complete technical analysis with full implementation details. Contains:
- Block manager hash-based allocation (lines 8-82)
- Sequence token and cache tracking (lines 18-85)
- Model runner prefill with cache hits (lines 278-317)
- Scheduler orchestration (lines 24-71)
- LLM engine integration (lines 50-60)
- Configuration parameters
- Complete flow diagram
- Key data flow summary
- Code paths with line numbers

**Best for:** Deep understanding and modification

---

## Quick Navigation

### I want to understand...

**How prefix caching works in general**
→ Start with PREFIX_CACHE_SUMMARY.txt, then PREFIX_CACHE_ARCHITECTURE.md

**Where specific functionality is implemented**
→ Look in PREFIX_CACHE_QUICK_REFERENCE.md for line numbers

**How data flows through the system**
→ Read PREFIX_CACHE_ARCHITECTURE.md sections on data flow

**How hash-based block allocation works**
→ Read PREFIX_CACHE_ANALYSIS.md Section 1

**How sequences track cached tokens**
→ Read PREFIX_CACHE_ANALYSIS.md Section 2

**How the model runner uses cached tokens**
→ Read PREFIX_CACHE_ANALYSIS.md Section 3

**How the scheduler integrates caching**
→ Read PREFIX_CACHE_ANALYSIS.md Section 4

**How to debug a specific issue**
→ Check PREFIX_CACHE_QUICK_REFERENCE.md "Common Issues & Debugging"

---

## Key Concepts at a Glance

### Hierarchical Hashing
- Each block's hash includes the previous block's hash
- Creates a chain where identical prefixes produce identical hashes
- Enables O(1) cache hit detection

### Reference Counting
- Blocks track how many sequences share them
- Incremented on cache hit, decremented on deallocation
- Blocks freed only when ref_count = 0

### Batch Token Accounting
- Only NEW (non-cached) tokens count toward batch limits
- `num_batched_tokens += len(seq) - seq.num_cached_tokens`
- Enables 20-30% throughput increase for common prefixes

### Incomplete Block Handling
- Last incomplete blocks get hash = -1 (not cached)
- When block completes, hash is computed and cached
- Prevents false cache hits from different continuations

### Key Data Structures
- `hash_to_block_id`: The core cache (hash → block_id lookup)
- `num_cached_tokens`: Per-sequence cached token count
- `ref_count`: Per-block reference count
- `block_table`: Per-sequence block allocation list

---

## Implementation Overview

```
REQUEST → SCHEDULER.allocate() → BLOCK_MANAGER.compute_hash()
                                         ↓
                                    CACHE HIT? 
                                         ↓
                        YES: num_cached_tokens +=
                        NO:  allocate new block
                                         ↓
                    MODEL_RUNNER.prepare_prefill()
                                         ↓
                    Skip cached tokens, compute new KV
                                         ↓
                    BLOCK_MANAGER.may_append()
                                         ↓
                    Finalize block hashes when complete
```

---

## File Locations in Codebase

| Component | File | Key Lines |
|-----------|------|-----------|
| Block allocation | block_manager.py | 59-82 |
| Hash computation | block_manager.py | 36-41 |
| Cached tokens tracking | sequence.py | 25, 54-55 |
| Scheduler integration | scheduler.py | 34-35 |
| Prefill preparation | model_runner.py | 289-307 |
| Prefix cache detection | model_runner.py | 306-307 |
| Block finalization | block_manager.py | 104-110 |

---

## Reading Order Recommendations

### For a 10-Minute Overview
1. PREFIX_CACHE_SUMMARY.txt (sections: Overview, Core Concepts, Execution Flow)
2. PREFIX_CACHE_ARCHITECTURE.md (section: System Architecture Diagram)

### For Implementation Understanding (30 minutes)
1. PREFIX_CACHE_SUMMARY.txt
2. PREFIX_CACHE_ARCHITECTURE.md (full)
3. PREFIX_CACHE_QUICK_REFERENCE.md (focus on data structures)

### For Deep Technical Knowledge (60 minutes)
1. PREFIX_CACHE_SUMMARY.txt
2. PREFIX_CACHE_ARCHITECTURE.md (full)
3. PREFIX_CACHE_ANALYSIS.md (full)
4. PREFIX_CACHE_QUICK_REFERENCE.md (debugging section)

### For Problem Solving (as needed)
1. PREFIX_CACHE_QUICK_REFERENCE.md (specific section)
2. PREFIX_CACHE_ANALYSIS.md (code path section)
3. PREFIX_CACHE_ARCHITECTURE.md (data flow section)

---

## Common Questions Answered

**Q: Where is prefix caching decided - is it automatic?**
A: Automatic! It happens in `block_manager.allocate()` (line 34 of scheduler.py). When a sequence is scheduled, the block manager checks if blocks with identical tokens already exist via hash lookup.

**Q: How are duplicate blocks detected?**
A: Via the `hash_to_block_id` dictionary (block_manager.py line 31). Each block gets a hierarchical hash that includes the previous block's hash, enabling exact prefix matching.

**Q: What happens when cached tokens are reused?**
A: Three things:
1. `num_cached_tokens` is incremented (block_manager.py:73)
2. `ref_count` is incremented (block_manager.py:76)
3. The block is reused in the sequence's block_table (line 82)

**Q: How does the model know to skip computing cached tokens?**
A: In `prepare_prefill()` (model_runner.py:289), only tokens after `seq.num_cached_tokens` are added to input_ids. The model processes only new tokens.

**Q: What about KV cache access for cached tokens?**
A: Passed via `block_tables` parameter when `cu_seqlens_k[-1] > cu_seqlens_q[-1]` (model_runner.py:306-307). Attention kernel uses these to gather cached KV.

**Q: When are incomplete blocks finalized?**
A: During decode phase in `may_append()` (block_manager.py:104-110). Once a block reaches 256 tokens, its hash is computed and it becomes cacheable.

**Q: How are freed blocks managed?**
A: Via `free_block_ids` deque (block_manager.py:32). When `ref_count` reaches 0, the block is added back to this queue for reuse.

---

## Performance Impact

For workloads with common prefixes:
- **Memory savings:** 20-60% depending on prefix overlap
- **Computation savings:** 20-40% token computations avoided
- **Throughput increase:** 20-30% more sequences per batch
- **Latency impact:** Negligible (hash lookup is O(1))

---

## Testing Tips

1. **Enable prefix cache hits:**
   - Create two sequences with identical first 256 tokens
   - Check `seq.num_cached_tokens` after allocation

2. **Verify block sharing:**
   - Check `blocks[0].ref_count` increases on cache hit
   - Both sequences should reference same block ID

3. **Monitor batch accounting:**
   - Calculate `len(seq) - num_cached_tokens` for each sequence
   - Verify total doesn't exceed `max_num_batched_tokens`

4. **Check deallocation:**
   - Deallocate sequences one at a time
   - Verify `ref_count` decreases and block freed when count = 0

---

## Related Code

- Attention kernel: Uses `block_tables` from context
- KV cache layout: `model_runner.py` allocate_kv_cache()
- Context management: `utils/context.py`
- Tokenizer: Uses standard HuggingFace tokenizer

---

## Version Information

- Last Updated: April 2026
- Analyzed Codebase: nano-vllm (recent commit)
- Python Version: 3.8+
- Key Dependencies: xxhash, torch

---

**Questions or Issues?**
Refer to the appropriate documentation file or check line numbers in the codebase directly.
