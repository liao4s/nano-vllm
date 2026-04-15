# Tensor Parallelism Documentation Index

This directory contains comprehensive analysis of tensor parallelism (TP) implementation in nanovllm, specifically comparing the working Qwen3 model with the broken Qwen3.5 model.

## 📄 Documents Overview

### 1. **TENSOR_PARALLELISM_ANALYSIS.md** (Start Here)
**Comprehensive deep-dive into TP implementation**
- Full explanation of how TP division works
- Line-by-line code breakdown for all 5 key areas
- Configuration values for both models
- Root cause analysis with detailed explanations
- Recommended fixes with code examples

**Read this if you want:** Complete understanding of the issue and how TP works

---

### 2. **TENSOR_PARALLELISM_QUICK_SUMMARY.md** (Quick Reference)
**One-page summary for busy developers**
- Issue at a glance
- Key files and line numbers
- How TP division works (with examples)
- Maximum supported TP sizes
- Quick fixes

**Read this if you want:** Fast understanding of what's wrong and where

---

### 3. **TENSOR_PARALLELISM_CODE_COMPARISON.md** (For Code Review)
**Side-by-side comparison of working vs broken code**
- Init comparison (Qwen3 ✓ vs Qwen3.5 ❌)
- Forward pass comparison
- Weight projection differences
- Configuration math breakdowns
- Error stack trace explanation
- Specific fix options

**Read this if you want:** To understand exactly what went wrong and how to fix it

---

## 🎯 The Issue (TL;DR)

```
Qwen3.5 has 2 KV heads but TP tries to divide them by 4:
  2 // 4 = 0 ❌ ZERO KV heads per rank!

This causes:
  .view(-1, 0, 256) → RuntimeError: invalid reshape
  
Qwen3 works because:
  8 // 4 = 2 ✓ Each rank gets 2 KV heads
```

---

## 📍 Key File Locations

| Concept | File | Lines |
|---------|------|-------|
| **Qwen3 Attention (working)** | `nanovllm/models/qwen3.py` | 14-88 |
| **Qwen3.5 Full Attention (broken)** | `nanovllm/models/qwen3_5.py` | 457-577 |
| **ColumnParallelLinear** | `nanovllm/layers/linear.py` | 54-74 |
| **RowParallelLinear** | `nanovllm/layers/linear.py` | 131-154 |
| **QKVParallelLinear** | `nanovllm/layers/linear.py` | 96-129 |
| **Qwen3.5 Config** | `qwen3.5/qwen3.5-35B-A3B-config` | 71, 75 |
| **Context setup** | `nanovllm/utils/context.py` | 1-28 |

---

## 🔍 What Each Document Covers

### TENSOR_PARALLELISM_ANALYSIS.md

**Section 1: Head Division**
- How Qwen3 divides heads correctly
- How Qwen3.5 fails with 0 heads per rank
- The specific math behind the problem

**Section 2: Linear Layers**
- ColumnParallelLinear (output sharding)
- RowParallelLinear (input sharding + all-reduce)
- QKVParallelLinear (combined Q,K,V handling)

**Section 3: QKVParallelLinear Details**
- Weight organization across ranks
- Weight loading strategy
- Per-rank output layout

**Section 4: Config Values**
- Qwen3 config: 32 Q heads, 8 KV heads
- Qwen3.5 config: 16 Q heads, 2 KV heads ⚠️
- Config math for different TP sizes

**Section 5: Context Setup**
- How tp_size is obtained: `dist.get_world_size()`
- How tp_rank is obtained: `dist.get_rank()`
- Usage patterns across the codebase

**Section 6: Root Cause Analysis**
- Why Qwen3 works (8 KV heads divide evenly)
- Why Qwen3.5 fails (2 KV heads don't divide by 4)
- Recommended fixes and validation

---

### TENSOR_PARALLELISM_QUICK_SUMMARY.md

**Issue Overview**
- Problem statement
- Working vs broken patterns
- Max TP sizes table

**Code Locations**
- Key files with line numbers
- What each file does

**TP Division Examples**
- ColumnParallelLinear math
- QKVParallelLinear math
- Forward pass examples

**Attention Forward Pass**
- Qwen3 (working) forward code
- Qwen3.5 (broken) forward code
- Where exactly it crashes

**Fixes**
- Immediate constraint check
- Better head replication solution
- Documentation guidelines

**Testing Strategy**
- Test TP=4 with Qwen3 (should work)
- Test TP=4 with Qwen3.5 (should fail)
- Test TP=2 with Qwen3.5 (should work with fix)

---

### TENSOR_PARALLELISM_CODE_COMPARISON.md

**Init Comparison**
- Qwen3Attention.__init__ ✓ (lines 14-40)
- Qwen3_5FullAttention.__init__ ❌ (lines 476-496)
- Why Qwen3 works and Qwen3.5 doesn't

**Forward Pass Comparison**
- Qwen3Attention.forward ✓ (lines 71-87)
- Qwen3_5FullAttention.forward ❌ (lines 542-576)
- Exact point of failure

**Weight Projection Usage**
- Qwen3: QKVParallelLinear (unified)
- Qwen3.5: Separate ColumnParallelLinear layers
- Why separate projections cause issues

**Configuration Math**
- Qwen3 divisibility math
- Qwen3.5 divisibility failures
- Per-rank calculations

**Error Stack Trace**
- Exact error that occurs
- Line-by-line path to failure
- How RuntimeError is triggered

**Fix Options**
- Option 1: Add assertion (minimal)
- Option 2: Replicate KV heads (better)
- Code for each option

---

## 🛠️ How to Use These Docs

### Scenario 1: "I need to understand the issue"
1. Start with **QUICK_SUMMARY.md** (5 min read)
2. Read the "Issue at a Glance" section
3. Look at "Key Files and Line Numbers"

### Scenario 2: "I need to fix the code"
1. Read **CODE_COMPARISON.md** (10 min read)
2. Focus on "Init Comparison" section
3. Skip to "Fix Options" at the end
4. Implement one of the proposed fixes

### Scenario 3: "I need deep understanding"
1. Start with **QUICK_SUMMARY.md** to get overview
2. Read **ANALYSIS.md** for detailed explanations
3. Use **CODE_COMPARISON.md** as reference for specific code

### Scenario 4: "I'm doing code review"
1. Open **CODE_COMPARISON.md**
2. Use as checklist against proposed changes
3. Verify fixes against "Fix Options" section

---

## 📊 Problem Summary Table

| Aspect | Qwen3 | Qwen3.5 | Issue |
|--------|-------|---------|-------|
| `num_attention_heads` | 32 | 16 | Lower in Qwen3.5 |
| `num_key_value_heads` | 8 | 2 | **Too low for TP!** |
| `head_dim` | 128 | 256 | Larger in Qwen3.5 |
| With TP=4: Q heads/rank | 8 | 4 | Both OK |
| With TP=4: KV heads/rank | 2 | 0 | **Qwen3.5 FAILS** |
| Max TP size | 8 | **2** | Limited for Qwen3.5 |

---

## ✅ Validation Checklist

Before implementing fixes, verify:

- [ ] You understand why 2 KV heads can't be divided by 4
- [ ] You found lines 489-492 in qwen3_5.py where division happens
- [ ] You can see the exact error: `view(-1, 0, 256)`
- [ ] You understand ColumnParallelLinear divides output by tp_size
- [ ] You know Qwen3 works because 8 % 4 == 0
- [ ] You can trace the forward pass from projection to reshape
- [ ] You understand the two fix options (assertion vs replication)

---

## 🔗 Related Code Patterns

### Pattern 1: Head Division in Attention
```python
# This is used in BOTH Qwen3 and Qwen3.5:
tp_size = dist.get_world_size()
self.num_heads = total_num_heads // tp_size
self.num_kv_heads = total_num_kv_heads // tp_size
```

**Works if:** `total_num_kv_heads % tp_size == 0`
**Fails if:** `total_num_kv_heads < tp_size`

### Pattern 2: ColumnParallelLinear
```python
# Used for Q, K, V projections
self.q_proj = ColumnParallelLinear(
    input_size,
    output_size,  # Will be divided by tp_size
)
```

**Key:** Output is always divided by tp_size

### Pattern 3: Forward reshape
```python
# This is where failures show up:
k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
# If num_kv_heads is 0, this fails!
```

---

## 📝 Key Equations

### Head Count per Rank
```
heads_per_rank = total_heads // tp_size

Valid when: total_heads % tp_size == 0
```

### ColumnParallel Output Size
```
output_per_rank = total_output_size // tp_size
```

### QKV Total Output
```
total_output = (num_heads + 2*num_kv_heads) * head_dim

Example Qwen3:
= (32 + 2*8) * 128 = 5120
Per rank: 5120 // 4 = 1280

Example Qwen3.5 (broken):
Q output: 16*256*2 // 4 = 2048  ✓
K output: 2*256 // 4 = 128      (should contain 0 heads!) ✗
V output: 2*256 // 4 = 128      (should contain 0 heads!) ✗
```

---

## 🚀 Next Steps

1. **Read the appropriate document** based on your needs
2. **Implement a fix** using CODE_COMPARISON.md as guide
3. **Add validation** with assertion for num_kv_heads divisibility
4. **Test with TP=1,2,4** to verify the fix works
5. **Consider head replication** if broader TP support is needed

---

## 📞 Cross-References

When reading these documents, you'll encounter references to:

- **Tensor Parallelism**: The distribution of model computation across multiple GPUs
- **TP rank**: The index of the current GPU (0, 1, 2, ...)
- **TP size**: Number of GPUs in the TP group (4, 8, etc.)
- **Head sharding**: Splitting attention heads across GPUs
- **All-reduce**: Distributed synchronization to sum contributions
- **ColumnParallel**: Sharding output dimension
- **RowParallel**: Sharding input dimension

---

## 🎓 Learning Path

**Beginner:** QUICK_SUMMARY.md → Issue section
**Intermediate:** QUICK_SUMMARY.md → CODE_COMPARISON.md → Fix Options
**Advanced:** ANALYSIS.md (full detailed sections) → CODE_COMPARISON.md → Implement

---

## 📌 Critical Lines to Remember

- **Line 29 (qwen3.py):** `tp_size = dist.get_world_size()`
- **Line 32 (qwen3.py):** `self.num_heads = self.total_num_heads // tp_size` ✓
- **Line 35 (qwen3.py):** `self.num_kv_heads = self.total_num_kv_heads // tp_size` ✓
- **Line 488 (qwen3_5.py):** `tp_size = dist.get_world_size()`
- **Line 490 (qwen3_5.py):** `self.num_heads = num_heads // tp_size` ✓
- **Line 492 (qwen3_5.py):** `self.num_kv_heads = num_kv_heads // tp_size` ❌ **PROBLEM**
- **Line 557 (qwen3_5.py):** `k = self.k_proj(...).view(-1, self.num_kv_heads, ...)` ❌ **CRASH HERE**

---

**Last Updated:** 2026-04-15
**Status:** Analysis Complete, Ready for Implementation
