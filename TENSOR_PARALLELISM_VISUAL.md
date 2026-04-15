# Tensor Parallelism: Visual Diagrams & Flowcharts

## 1. Head Division Overview

### ✓ Qwen3 (Working)
```
Total Config:
  Q Heads: 32
  KV Heads: 8
  Head Dim: 128

With TP Size = 4:

GPU 0: Q[0-7]    KV[0-1]     Head Dim: 128
GPU 1: Q[8-15]   KV[2-3]     Head Dim: 128  
GPU 2: Q[16-23]  KV[4-5]     Head Dim: 128
GPU 3: Q[24-31]  KV[6-7]     Head Dim: 128

Per-rank output: [8*128 + 2*128 + 2*128] = 1280 features
Total: 1280 * 4 = 5120 ✓
```

### ❌ Qwen3.5 (Broken)
```
Total Config:
  Q Heads: 16
  KV Heads: 2
  Head Dim: 256

With TP Size = 4:

GPU 0: Q[0-3]    KV[???]        ❌ 2//4 = 0 heads!
GPU 1: Q[4-7]    KV[???]        ❌ 2//4 = 0 heads!
GPU 2: Q[8-11]   KV[???]        ❌ 2//4 = 0 heads!
GPU 3: Q[12-15]  KV[???]        ❌ 2//4 = 0 heads!

Attempted output per rank: [4*256 + 0*256 + 0*256] = 1024 features ❌
Problem: Can't reshape 128 features into (num_heads=0, head_dim=256)
```

---

## 2. Data Flow Through Layers

### Qwen3 Forward Pass ✓
```
INPUT: hidden_states [N, 4096]
           |
           v
    QKVParallelLinear
    - Unified Q,K,V projection
    - Outputs: [Q_shard | K_shard | V_shard]
           |
           v
    Each GPU gets [N, 1280]:
    - [N, 1024] Q features → [N, 8 heads, 128 dim]  ✓
    - [N, 128] K features  → [N, 2 heads, 128 dim]  ✓
    - [N, 128] V features  → [N, 2 heads, 128 dim]  ✓
           |
           v
    Attention Computation
    - Q: [N, 8, 128]
    - K: [N, 2, 128]
    - V: [N, 2, 128]
    - Output: [N, 8, 128]
           |
           v
    RowParallelLinear + AllReduce
    - Each GPU computes partial output
    - AllReduce sums across GPUs
           |
           v
    OUTPUT: [N, 4096] ✓
```

### Qwen3.5 Forward Pass ❌
```
INPUT: hidden_states [N, 2048]
           |
           v
    ColumnParallelLinear (Q)
    - Outputs: [N, 2048] (per GPU)
           |
           v
    Reshape Q: [N, 4, 512] ✓  (Q has 4 heads per GPU)
           |
           v
    Split gate and query: [N, 4, 256] each ✓
           |
           v
    ColumnParallelLinear (K)
    - Outputs: [N, 128] (per GPU)
           |
           v
    Reshape K: [N, ???, 256]
    where ??? = num_kv_heads = 2 // 4 = 0  ❌
           |
           v
    RuntimeError: Cannot reshape 128 into shape [N, 0, 256]
    CRASH! 💥
```

---

## 3. Weight Sharding Pattern

### ColumnParallelLinear (Used for Q, K, V)
```
Full Checkpoint:
┌─────────────────────────────────────┐
│      QKV Output (5120 features)     │
│ [Q_full|K_full|V_full]              │
│ [1024 + 512 + 512]                  │
└─────────────────────────────────────┘
              |
              | ColumnParallel Shard by TP Size
              |
    ┌─────────┴─────────┬──────────┬──────────┐
    |                   |          |          |
    v                   v          v          v
┌─────────┐         ┌─────────┐┌─────────┐┌─────────┐
│ GPU 0   │         │ GPU 1   ││ GPU 2   ││ GPU 3   │
│ 1280    │         │ 1280    ││ 1280    ││ 1280    │
│ features│         │ features││ features││ features│
└─────────┘         └─────────┘└─────────┘└─────────┘
  Output from each GPU is concatenated
  Final output: [N, 5120] total ✓

Per-GPU weight shape: [1280, hidden_size]
Full weight shape: [5120, hidden_size]
```

---

## 4. Attention Head Assignment

### Qwen3 with TP=4
```
Logical attention heads:   [Q0][Q1][Q2][Q3]...[Q31]
                           [K0-1][K2-3]...[K6-7]

Distribution across 4 GPUs:
GPU 0: [Q0-7] [K0-1] → handles attention heads 0-7 for Q, 0-1 for K
GPU 1: [Q8-15] [K2-3] → handles attention heads 8-15 for Q, 2-3 for K
GPU 2: [Q16-23] [K4-5] → handles attention heads 16-23 for Q, 4-5 for K
GPU 3: [Q24-31] [K6-7] → handles attention heads 24-31 for Q, 6-7 for K

All computations stay local within each GPU ✓
```

### Qwen3.5 with TP=4 (Broken)
```
Logical attention heads:   [Q0][Q1]...[Q15]
                           [K0-1]

Distribution across 4 GPUs:
GPU 0: [Q0-3] [K???] → No heads to assign to K!
GPU 1: [Q4-7] [K???] → No heads to assign to K!
GPU 2: [Q8-11] [K???] → No heads to assign to K!
GPU 3: [Q12-15] [K???] → No heads to assign to K!

Reshaping [128 features] to [num_heads=0, 256] → Invalid! ❌
```

---

## 5. Message Passing: AllReduce in RowParallel

### RowParallelLinear Output Projection
```
Input to RowParallel:
[N, num_heads*head_dim] = [N, 1024] per GPU

GPU 0 computes: [N, 4096] * (1024 weights) = [N, 4096] partial output
GPU 1 computes: [N, 4096] * (1024 weights) = [N, 4096] partial output
GPU 2 computes: [N, 4096] * (1024 weights) = [N, 4096] partial output
GPU 3 computes: [N, 4096] * (1024 weights) = [N, 4096] partial output

AllReduce (sum all contributions):
   GPU0_out
      +
   GPU1_out
      +
   GPU2_out
      +
   GPU3_out
      ↓
Final output = sum of all GPUs = [N, 4096] ✓
```

---

## 6. Config Math Comparison

### Qwen3 Config Math
```
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128

For TP_SIZE = 1 (Single GPU):
  heads_per_rank = 32 // 1 = 32 ✓
  kv_heads_per_rank = 8 // 1 = 8 ✓

For TP_SIZE = 2:
  heads_per_rank = 32 // 2 = 16 ✓
  kv_heads_per_rank = 8 // 2 = 4 ✓

For TP_SIZE = 4:
  heads_per_rank = 32 // 4 = 8 ✓
  kv_heads_per_rank = 8 // 4 = 2 ✓

For TP_SIZE = 8:
  heads_per_rank = 32 // 8 = 4 ✓
  kv_heads_per_rank = 8 // 8 = 1 ✓

For TP_SIZE = 16:
  heads_per_rank = 32 // 16 = 2 ✓
  kv_heads_per_rank = 8 // 16 = 0.5 ❌ Not integer!
```

### Qwen3.5 Config Math
```
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256

For TP_SIZE = 1 (Single GPU):
  heads_per_rank = 16 // 1 = 16 ✓
  kv_heads_per_rank = 2 // 1 = 2 ✓

For TP_SIZE = 2:
  heads_per_rank = 16 // 2 = 8 ✓
  kv_heads_per_rank = 2 // 2 = 1 ✓

For TP_SIZE = 4:
  heads_per_rank = 16 // 4 = 4 ✓
  kv_heads_per_rank = 2 // 4 = 0 ❌ ZERO HEADS!

For TP_SIZE = 8:
  heads_per_rank = 16 // 8 = 2 ✓
  kv_heads_per_rank = 2 // 8 = 0 ❌ ZERO HEADS!

Maximum usable TP_SIZE = 2 (due to 2 KV heads)
```

---

## 7. Error Propagation Path

### Where Qwen3.5 Breaks with TP=4
```
Step 1: Initialization (qwen3_5.py:492)
  tp_size = dist.get_world_size()        # 4
  num_kv_heads = 2 // 4                 # 0 ❌
  kv_size = 0 * 256                     # 0 ❌

Step 2: Forward pass (qwen3_5.py:557)
  k_proj_out = k_proj(hidden_states)    # [N, 128]
  k = k_proj_out.view(-1, 0, 256)       # Invalid shape! ❌

Step 3: Error
  RuntimeError: shape '[N, 0, 256]' is invalid for input of size [N*128]
  
Step 4: Failure
  Model initialization completes silently
  Error only appears at inference/training
```

---

## 8. Parameter Count Comparison

### Qwen3: QKV Projection
```
Total output features: (32 + 8 + 8) * 128 = 5760
Per GPU (TP=4): 5760 / 4 = 1440
Weight shape: [1440, 4096] = 5,898,240 params per GPU

Total across 4 GPUs: 5,898,240 * 4 / 4 = 5,898,240 (unique params)
```

### Qwen3.5: Q, K, V Separate Projections (Broken)
```
Q projection output: 16 * 256 * 2 = 8192
  Per GPU: 8192 / 4 = 2048
  Weight: [2048, 2048] = 4,194,304 per GPU

K projection output: 2 * 256 = 512
  Per GPU: 512 / 4 = 128 ❌ (Should be divisible into heads!)
  Weight: [128, 2048] = 262,144 per GPU
  Cannot reshape to [batch, 0, 256] ❌

V projection output: 2 * 256 = 512
  Per GPU: 512 / 4 = 128 ❌ (Should be divisible into heads!)
  Weight: [128, 2048] = 262,144 per GPU
  Cannot reshape to [batch, 0, 256] ❌
```

---

## 9. The Fix: Head Replication Option

### Before (Broken)
```
With TP=4:
  GPU0: Q_heads[0-3]   KV_heads[]      ← No KV heads!
  GPU1: Q_heads[4-7]   KV_heads[]      ← No KV heads!
  GPU2: Q_heads[8-11]  KV_heads[]      ← No KV heads!
  GPU3: Q_heads[12-15] KV_heads[]      ← No KV heads!
```

### After (With Replication)
```
With TP=4 and head replication:
  GPU0: Q_heads[0-3]   KV_heads[0,1] ← Replicated!
  GPU1: Q_heads[4-7]   KV_heads[0,1] ← Replicated!
  GPU2: Q_heads[8-11]  KV_heads[0,1] ← Replicated!
  GPU3: Q_heads[12-15] KV_heads[0,1] ← Replicated!

Each GPU computes full attention with replicated K,V
Then AllReduce sums Q contributions
```

---

## 10. Code Path Flowchart

### Initialization Path (Qwen3.5FullAttention)
```
Qwen3_5DecoderLayer.__init__
    |
    └─→ Qwen3_5FullAttention.__init__
            |
            ├─→ get_world_size()            # Line 488
            |   (tp_size = 4)
            |
            ├─→ self.total_num_heads = 16   # Line 489
            |
            ├─→ self.num_heads = 16 // 4 = 4 # Line 490 ✓
            |
            ├─→ self.total_num_kv_heads = 2 # Line 491
            |
            ├─→ self.num_kv_heads = 2 // 4  # Line 492 ❌ PROBLEM HERE
            |   (num_kv_heads = 0)
            |
            ├─→ self.kv_size = 0 * 256      # Line 495
            |
            ├─→ Create ColumnParallelLinear layers
            |   q_proj, k_proj, v_proj
            |
            └─→ Initialization completes silently ✓
                (Error happens at forward!)
```

### Forward Pass Path (Qwen3_5FullAttention)
```
forward(positions, hidden_states)
    |
    ├─→ q_out = q_proj(hidden_states)      # Line 548
    |   Output: [N, 2048] ✓
    |
    ├─→ Reshape & split: [N, 4, 256] ✓    # Line 551-554
    |
    ├─→ k = k_proj(hidden_states)           # Line 557
    |   Output: [N, 128]
    |
    ├─→ .view(-1, self.num_kv_heads, ...)  # Line 557
    |       |
    |       └─→ .view(-1, 0, 256)          # ❌ INVALID!
    |
    └─→ RuntimeError: Cannot reshape!
        💥 MODEL CRASHES
```

---

## 11. Divisibility Matrix

```
        num_kv_heads=2
            |
            v
    ┌───────┬───────┬───────┬───────┐
    │  TP=1 │  TP=2 │  TP=4 │  TP=8 │
    ├───────┼───────┼───────┼───────┤
    │  2÷1  │  2÷2  │  2÷4  │  2÷8  │
    │  = 2  │  = 1  │  = 0  │  = 0  │
    │  ✓    │  ✓    │  ❌   │  ❌   │
    └───────┴───────┴───────┴───────┘
         ||
         vv
    Max TP size for Qwen3.5: 2
    
    vs
    
        num_kv_heads=8 (Qwen3)
            |
            v
    ┌───────┬───────┬───────┬───────┐
    │  TP=1 │  TP=2 │  TP=4 │  TP=8 │
    ├───────┼───────┼───────┼───────┤
    │  8÷1  │  8÷2  │  8÷4  │  8÷8  │
    │  = 8  │  = 4  │  = 2  │  = 1  │
    │  ✓    │  ✓    │  ✓    │  ✓    │
    └───────┴───────┴───────┴───────┘
         ||
         vv
    Max TP size for Qwen3: 8
```

---

## 12. Solution Comparison

### Option 1: Add Assertion (Quick Fix)
```
Qwen3_5FullAttention.__init__():
  if num_kv_heads % tp_size != 0:
    raise ValueError(
      f"num_kv_heads ({num_kv_heads}) not divisible by tp_size ({tp_size})"
    )
  
  Pro: ✓ Quick, clear error message
  Con: ❌ Limits TP to max size of KV heads
  Result: Users can only use TP size ≤ 2
```

### Option 2: Head Replication (Full Fix)
```
Qwen3_5FullAttention.__init__():
  if num_kv_heads < tp_size:
    # Replicate KV heads instead of sharding
    self.k_proj = ReplicatedLinear(...)  # Not ColumnParallel
    self.v_proj = ReplicatedLinear(...)
    self.num_kv_heads = num_kv_heads     # Don't divide!
  else:
    # Normal sharding
    self.k_proj = ColumnParallelLinear(...)
    self.v_proj = ColumnParallelLinear(...)
    self.num_kv_heads = num_kv_heads // tp_size
  
  Pro: ✓ Allows any TP size, K/V computed redundantly
  Con: ❌ More memory for K/V, slight redundant computation
  Result: Users can use any TP size (1, 4, 8, etc.)
```

---

## Summary Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   TENSOR PARALLELISM ISSUE               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Qwen3:  32 Q heads + 8 KV heads → TP friendly          │
│  ✓ Divides evenly with TP=1,2,4,8                       │
│                                                          │
│  Qwen3.5: 16 Q heads + 2 KV heads → TP hostile          │
│  ❌ Only works with TP=1,2                              │
│                                                          │
│  Root cause:                                            │
│  2 KV heads // 4 TP ranks = 0 heads/rank               │
│                                                          │
│  Manifestation:                                         │
│  .view(-1, 0, 256) → RuntimeError                       │
│                                                          │
│  Solutions:                                             │
│  1. Add divisibility check (quick)                      │
│  2. Replicate K/V heads (full)                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

