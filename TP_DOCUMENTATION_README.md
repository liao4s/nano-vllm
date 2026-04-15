# Tensor Parallelism Analysis Documentation

## 🎯 What This Is

A comprehensive analysis of tensor parallelism (TP) issues in nanovllm, specifically:
- How TP works in the **working Qwen3 model**
- Why **Qwen3.5 model breaks** with TP size > 2
- Detailed code comparison and fix recommendations

## 📚 Documentation Files

### 1. **TENSOR_PARALLELISM_INDEX.md** ⭐ START HERE
Your guide to navigating all the documentation.
- Overview of all documents
- What each one covers
- How to use them based on your needs
- Critical line numbers to remember
- Validation checklist

**Best for:** Understanding what exists and picking the right doc to read

---

### 2. **TENSOR_PARALLELISM_QUICK_SUMMARY.md** ⏱️ 5-MINUTE READ
One-page executive summary of the issue.
- Issue at a glance
- Key files and line numbers  
- Max TP sizes table
- Quick fixes overview

**Best for:** Getting the gist quickly before diving deep

---

### 3. **TENSOR_PARALLELISM_ANALYSIS.md** 🔍 DETAILED ANALYSIS
Complete technical breakdown of TP implementation.
- How heads are divided by TP size (Section 1)
- ColumnParallel/RowParallel/QKVParallel layers (Section 2)
- QKV weight organization (Section 3)
- Configuration values for both models (Section 4)
- How tp_size is obtained (Section 5)
- Root cause analysis (Section 6)

**Best for:** Deep technical understanding of every component

---

### 4. **TENSOR_PARALLELISM_CODE_COMPARISON.md** 🔀 SIDE-BY-SIDE
Code comparison and error tracing.
- Qwen3Attention vs Qwen3_5FullAttention (init)
- Forward pass comparison (exact failure point)
- Weight projection differences
- Configuration math for both models
- Error stack trace explanation
- Two fix options with code

**Best for:** Code review, implementing fixes, understanding exact errors

---

### 5. **TENSOR_PARALLELISM_VISUAL.md** 📊 DIAGRAMS
Visual flowcharts and ASCII diagrams.
- Head division diagrams
- Data flow through layers
- Weight sharding patterns
- Attention head assignment
- AllReduce message passing
- Config math matrices
- Error propagation paths
- Code path flowcharts
- Divisibility matrices
- Solution comparison

**Best for:** Visual learners, presentations, quick reference

---

## 🚀 Quick Navigation

### "I need to fix this now"
1. Read: QUICK_SUMMARY.md
2. Read: CODE_COMPARISON.md (Fix Options section)
3. Implement one of the two fixes
4. Add validation tests

**Time: ~15 minutes**

---

### "I need to understand it deeply"
1. Read: QUICK_SUMMARY.md (overview)
2. Read: ANALYSIS.md (sections 1-3)
3. Reference: CODE_COMPARISON.md
4. Look at: VISUAL.md (diagrams)

**Time: ~45 minutes**

---

### "I'm doing code review"
1. Reference: CODE_COMPARISON.md
2. Check: ANALYSIS.md (Root Cause section)
3. Verify: Both Q and KV head divisibility
4. Compare: Against "Fix Options" in CODE_COMPARISON.md

**Time: ~20 minutes**

---

### "I want to learn TP in general"
1. Start: VISUAL.md (diagrams 1-5)
2. Read: ANALYSIS.md (sections 1-2)
3. Code walk: CODE_COMPARISON.md
4. Deep dive: ANALYSIS.md (sections 3-6)

**Time: ~1 hour**

---

## 🎓 The Issue (30-Second Summary)

**Problem:**
- Qwen3 has 8 KV heads → divides nicely (8÷4=2, 8÷8=1)
- Qwen3.5 has 2 KV heads → divides poorly (2÷4=0 ❌)

**Impact:**
- Qwen3: Works with TP size 1, 2, 4, 8
- Qwen3.5: Only works with TP size 1, 2 (max)

**Technical Cause:**
```python
self.num_kv_heads = 2 // 4  # = 0 ❌
k = proj(x).view(-1, 0, 256)  # RuntimeError!
```

**Solutions:**
1. **Quick fix**: Add divisibility check
2. **Full fix**: Replicate KV heads instead of sharding

---

## 📋 Key Concepts

### Tensor Parallelism (TP)
Splitting model computation across multiple GPUs/ranks.
- Each rank handles subset of parameters
- Results synchronized with AllReduce

### Head Sharding
Attention heads divided across TP ranks.
- Query (Q): 32 heads → 4 GPUs = 8 heads/GPU ✓
- Key/Value (K,V): 8 heads → 4 GPUs = 2 heads/GPU ✓

### ColumnParallelLinear
Outputs sharded across TP ranks.
- Input size: shared (full)
- Output size: divided by TP size

### RowParallelLinear  
Inputs sharded across TP ranks.
- Input size: divided by TP size
- Output size: full (with AllReduce to sync)

---

## 📍 Key Line Numbers

### Qwen3 (Working)
| File | Lines | What |
|------|-------|------|
| qwen3.py | 14-88 | Qwen3Attention class |
| qwen3.py | 29 | `tp_size = dist.get_world_size()` |
| qwen3.py | 32-35 | Head division with assertions ✓ |
| qwen3.py | 42-48 | QKVParallelLinear usage |
| qwen3.py | 76-77 | Forward: QKV split |

### Qwen3.5 (Broken) 
| File | Lines | What |
|------|-------|------|
| qwen3_5.py | 457-577 | Qwen3_5FullAttention class |
| qwen3_5.py | 488 | `tp_size = dist.get_world_size()` |
| qwen3_5.py | 490 | Head division: Q ✓ |
| qwen3_5.py | 492 | Head division: KV ❌ (can be 0!) |
| qwen3_5.py | 557 | Forward: K reshape crash ❌ |

### Config
| File | Lines | What |
|------|-------|------|
| qwen3.5-config | 71 | `num_attention_heads: 16` |
| qwen3.5-config | 75 | `num_key_value_heads: 2` ⚠️ |

### Linear Layers
| File | Lines | What |
|------|-------|------|
| linear.py | 54-74 | ColumnParallelLinear |
| linear.py | 131-154 | RowParallelLinear |
| linear.py | 96-129 | QKVParallelLinear |
| linear.py | 37-52 | ReplicatedLinear |

---

## ✅ Before You Implement

Verify these facts:

- [ ] Qwen3 has 8 KV heads, Qwen3.5 has 2
- [ ] 2 KV heads // 4 TP ranks = 0 heads/rank ❌
- [ ] 8 KV heads // 4 TP ranks = 2 heads/rank ✓
- [ ] Error occurs in `.view(-1, 0, 256)` reshape
- [ ] Error is at line 557 in qwen3_5.py forward pass
- [ ] ColumnParallelLinear divides by tp_size
- [ ] ReplicatedLinear doesn't divide (full output)
- [ ] Qwen3 uses QKVParallelLinear (unified)
- [ ] Qwen3.5 uses separate linear layers (K, V separate from Q)

---

## 🛠️ Implementation Checklist

### Option 1: Add Assertion (Minimal Fix)
- [ ] Add check in `Qwen3_5FullAttention.__init__`
- [ ] Check: `num_kv_heads % tp_size == 0`
- [ ] Raise clear error if check fails
- [ ] Test with TP=2 (should pass)
- [ ] Test with TP=4 (should raise error)

### Option 2: Head Replication (Full Fix)
- [ ] Import `ReplicatedLinear` in qwen3_5.py
- [ ] Check if `num_kv_heads < tp_size`
- [ ] If yes: use `ReplicatedLinear` for k_proj, v_proj
- [ ] If no: use `ColumnParallelLinear` (current behavior)
- [ ] Set `self.num_kv_heads = num_kv_heads` (don't divide if replicated)
- [ ] Test with TP=1, 2, 4, 8
- [ ] Verify all tests pass

---

## 📞 Questions to Ask Yourself

1. **Why does Qwen3 work?**
   - Because 8 KV heads divides evenly by any TP size up to 8

2. **Why does Qwen3.5 break with TP=4?**
   - Because 2 KV heads // 4 = 0, can't reshape to 0 dimensions

3. **Where exactly does it crash?**
   - Line 557: `k.view(-1, self.num_kv_heads, self.head_dim)`
   - Becomes: `k.view(-1, 0, 256)` → RuntimeError

4. **What's the simplest fix?**
   - Add divisibility assertion, limit TP size to ≤ 2

5. **What's the better fix?**
   - Replicate K/V heads instead of sharding when num_kv_heads < tp_size

6. **Why replicate instead of shard?**
   - Some operations need K,V to be complete (not sharded)
   - Replication means each GPU computes K,V independently
   - Still shard Q, use AllReduce to sum Q contributions

---

## 🎯 Success Criteria

After fix is complete:

- [ ] Qwen3.5 initializes without error with TP=1
- [ ] Qwen3.5 initializes without error with TP=2
- [ ] Qwen3.5 forward pass completes without error (TP=1 or 2)
- [ ] If using Option 1: TP=4 raises clear error message
- [ ] If using Option 2: Qwen3.5 works with TP=4, 8, etc.
- [ ] Tests pass for different TP sizes
- [ ] Code follows nanovllm patterns (matches Qwen3 style)

---

## 📚 Additional References

### PyTorch Distributed
- `dist.get_world_size()`: Number of TP ranks
- `dist.get_rank()`: Current rank index
- `dist.all_reduce()`: Sum values across all ranks

### Attention Mechanics
- Q (Query): one head per position, used to "look" at K,V
- K,V: fewer heads (KV heads), can be replicated or sharded
- Scaling: Important for numerical stability

### Model Architecture
- Linear Attention: GatedDeltaNet (3 layers per 4)
- Full Attention: With output gating and partial RoPE (1 layer per 4)
- MoE: Top-K router with shared expert

---

## 📝 Notes

- All analysis is based on code inspection
- Line numbers may shift with future edits
- Always verify critical sections before implementing
- Test thoroughly with different TP sizes
- Consider memory implications of head replication

---

**Created:** 2026-04-15  
**Status:** Ready for Implementation  
**Files:** 5 comprehensive documents + this README

**Total Documentation:** 1,936 lines of detailed analysis
