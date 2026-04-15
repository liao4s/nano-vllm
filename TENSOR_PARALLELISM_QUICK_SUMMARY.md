# Tensor Parallelism (TP) Issue in Qwen3.5 - Quick Summary

## The Issue at a Glance

**Qwen3.5 has 2 KV heads, but TP tries to divide them across 4+ ranks:**

```
num_key_value_heads = 2
tp_size = 4
Result: 2 // 4 = 0 KV heads per rank ❌ CRASH
```

## The Working Pattern (Qwen3)

```
num_key_value_heads = 8
tp_size = 4
Result: 8 // 4 = 2 KV heads per rank ✓ WORKS
```

---

## Key Files and Line Numbers

### 1. **Qwen3 (Working) Attention Module**
- **File:** `nanovllm/models/qwen3.py`
- **Lines:** 14-88 (Qwen3Attention class)
- **Key Lines:**
  - Line 29: `tp_size = dist.get_world_size()`
  - Line 32: `self.num_heads = self.total_num_heads // tp_size`
  - Line 35: `self.num_kv_heads = self.total_num_kv_heads // tp_size`

### 2. **Qwen3.5 (Broken) Full Attention Module**
- **File:** `nanovllm/models/qwen3_5.py`
- **Lines:** 457-577 (Qwen3_5FullAttention class)
- **Problem Lines:**
  - Line 488: `tp_size = dist.get_world_size()`
  - Line 490: `self.num_heads = num_heads // tp_size`
  - Line 492: `self.num_kv_heads = num_kv_heads // tp_size` ❌ Can be 0!

### 3. **ColumnParallelLinear (Shards output)**
- **File:** `nanovllm/layers/linear.py`
- **Lines:** 54-74
- **Key:** Divides output dimension by `tp_size`

### 4. **RowParallelLinear (Shards input)**
- **File:** `nanovllm/layers/linear.py`
- **Lines:** 131-154
- **Key:** Divides input dimension by `tp_size` with all_reduce

### 5. **QKVParallelLinear (Handles Q, K, V sharding)**
- **File:** `nanovllm/layers/linear.py`
- **Lines:** 96-129
- **Key:** Splits Q, K, V outputs, each sharded separately

### 6. **Qwen3.5 Configuration**
- **File:** `qwen3.5/qwen3.5-35B-A3B-config`
- **Key Lines:**
  - Line 71: `"num_attention_heads": 16`
  - Line 75: `"num_key_value_heads": 2` ⚠️ THE PROBLEM

---

## How TP Division Works

### ColumnParallelLinear Example
```
Input: hidden_states [N, 2048]
Full output would be: [N, 8192]

With tp_size = 4:
Each rank computes: [N, 8192 // 4] = [N, 2048]
```

### QKVParallelLinear Example
```
Total output needed: [Q | K | V]
= [(16 heads * 256 dim) + (2 kv_heads * 256 dim) * 2]
= [4096 + 1024] = 5120

With tp_size = 4:
Each rank computes per-head splits:
  Q per rank: (16 // 4) * 256 = 4 * 256 = 1024
  K per rank: (2 // 4) * 256 = 0 * 256 = 0 ❌
  V per rank: (2 // 4) * 256 = 0 * 256 = 0 ❌
```

---

## Attention Forward Pass

### Qwen3Attention (Line 76-87)
```python
qkv = self.qkv_proj(hidden_states)  # Each rank gets its shard
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

q = q.view(-1, self.num_heads, self.head_dim)           # [N, 8, 128]
k = k.view(-1, self.num_kv_heads, self.head_dim)        # [N, 2, 128] ✓
v = v.view(-1, self.num_kv_heads, self.head_dim)        # [N, 2, 128] ✓
```

### Qwen3_5FullAttention (Line 547-575)
```python
q_out = self.q_proj(hidden_states)  # Each rank gets its shard
q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)

k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
# Line 557: view(-1, 0, 256) ❌ FAILS! Can't reshape to 0 dimensions
```

---

## Root Causes

### Issue 1: Head Count Mismatch
- Qwen3: 8 KV heads → divides evenly by any tp_size ≤ 8
- Qwen3.5: 2 KV heads → only works with tp_size ≤ 2

### Issue 2: No Validation
- Neither model checks if `num_kv_heads % tp_size == 0`
- Silent failure leads to 0-dimensional heads

### Issue 3: No Head Replication Support
- Some frameworks replicate small head counts instead of sharding
- nanovllm assumes pure sharding is always possible

---

## Maximum Supported TP Sizes

| Model | Q Heads | KV Heads | Max TP Size |
|-------|---------|----------|------------|
| Qwen3 | 32 | 8 | 8 |
| Qwen3.5 | 16 | 2 | **2** ⚠️ |

---

## Fixes Required

### Immediate (Constraint Check)
```python
# In Qwen3_5FullAttention.__init__
assert num_kv_heads % tp_size == 0, (
    f"Qwen3.5 has {num_kv_heads} KV heads, incompatible with tp_size={tp_size}. "
    f"Maximum supported tp_size: {num_kv_heads}"
)
```

### Better (Head Replication)
```python
if num_kv_heads < tp_size:
    # Don't shard, replicate instead
    self.num_kv_heads = num_kv_heads
    # Use ReplicatedLinear for k_proj, v_proj
else:
    self.num_kv_heads = num_kv_heads // tp_size
    # Use ColumnParallelLinear
```

### Documentation
```
Qwen3.5-35B:
- Requires: tp_size <= 2
- Optimal: tp_size = 1 (single GPU) or 2 (2 GPUs)
- Single GPU: Load on one device
- Two GPUs: Split model across 2 devices
```

---

## Context Setup

### How tp_size is obtained
- **File:** `nanovllm/utils/context.py`
- **Method:** `dist.get_world_size()` from PyTorch Distributed
- **Usage:** All linear layers and attention modules query this

### The get_context() function
- Provides: `is_prefill`, `cu_seqlens_q`, `slot_mapping`, etc.
- Does NOT provide: `tp_size` or `tp_rank`
- Those come directly from `torch.distributed`

---

## Testing Strategy

### Test 1: Qwen3 with TP=4
✓ Should work (8 KV heads → 2 heads/rank)

### Test 2: Qwen3.5 with TP=4
❌ Should fail (2 KV heads → 0 heads/rank)

### Test 3: Qwen3.5 with TP=2
✓ Should work with fix (2 KV heads → 1 head/rank)

### Test 4: Qwen3.5 with TP=1
✓ Should work (no sharding)

---

## References in Code

- **TP initialization:** `dist.get_world_size()`, `dist.get_rank()`
- **Head division:** Lines 29-35 (Qwen3), Lines 488-492 (Qwen3.5)
- **Weight loading:** `QKVParallelLinear.weight_loader()`
- **Forward pass:** Lines 76-87 (Qwen3), Lines 547-575 (Qwen3.5)
- **All-reduce sync:** Line 152 in `RowParallelLinear.forward()`
