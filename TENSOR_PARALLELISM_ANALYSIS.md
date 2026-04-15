# Tensor Parallelism (TP) Implementation Analysis: Qwen3 vs Qwen3.5

## Summary of Findings

This document provides a detailed analysis of how tensor parallelism is implemented in the Qwen3 (working) model, with identification of the issues in Qwen3.5 that need fixing.

---

## 1. How `num_kv_heads` and `num_heads` are divided by `tp_size`

### Qwen3 Model (Working) - `/repo/nanovllm/models/qwen3.py`

**File:** `nanovllm/models/qwen3.py`  
**Class:** `Qwen3Attention`  
**Lines:** 14-88

```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        # LINE 29: Get TP size from distributed context
        tp_size = dist.get_world_size()
        
        # LINES 30-35: Total vs per-rank heads
        self.total_num_heads = num_heads                    # e.g., 32 (total)
        assert self.total_num_heads % tp_size == 0         # Must be divisible
        self.num_heads = self.total_num_heads // tp_size    # e.g., 32 // 4 = 8
        
        self.total_num_kv_heads = num_kv_heads            # e.g., 8 (total)
        assert self.total_num_kv_heads % tp_size == 0     # Must be divisible
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # e.g., 8 // 4 = 2
        
        # LINE 36: head_dim stays the same across all ranks
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        
        # LINES 37-38: Compute query and KV buffer sizes for THIS rank
        self.q_size = self.num_heads * self.head_dim       # [8 * 128] per rank
        self.kv_size = self.num_kv_heads * self.head_dim   # [2 * 128] per rank
```

**Key Insight:**
- **Total heads** (from config) are divided equally across TP ranks
- **Per-rank heads** = `total_heads // tp_size`
- **head_dim** uses the total heads count (constant across all ranks)
- Each rank processes only its fraction of the heads

**Usage in QKV Projection (Lines 42-48):**
```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,              # Per-rank head dimension
    self.total_num_heads,       # Total Q heads (pre-division)
    self.total_num_kv_heads,    # Total KV heads (pre-division)
    bias=qkv_bias,
)
```

---

### Qwen3.5 Model (Full Attention) - `/repo/nanovllm/models/qwen3_5.py`

**File:** `nanovllm/models/qwen3_5.py`  
**Class:** `Qwen3_5FullAttention`  
**Lines:** 457-577

```python
class Qwen3_5FullAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        # LINE 488: Get TP size from distributed context
        tp_size = dist.get_world_size()
        
        # LINES 489-492: Head divisions (SAME PATTERN AS QWEN3)
        self.total_num_heads = num_heads                    # e.g., 16
        self.num_heads = num_heads // tp_size              # e.g., 16 // 4 = 4
        
        self.total_num_kv_heads = num_kv_heads             # e.g., 2
        self.num_kv_heads = num_kv_heads // tp_size        # e.g., 2 // 4 = 0.5 ❌ ISSUE!
        
        # LINE 493: head_dim is passed as a parameter (not computed)
        self.head_dim = head_dim                            # e.g., 256
```

**The Problem:**
```
num_kv_heads = 2
tp_size = 4
num_kv_heads // tp_size = 2 // 4 = 0  ❌ ZERO HEADS PER RANK!
```

When `num_kv_heads < tp_size`, you get 0 heads per rank, which causes:
- Division by zero in reshape operations
- Incorrect attention computation
- Model failure

---

## 2. ColumnParallelLinear and RowParallelLinear Implementation

### File: `/repo/nanovllm/layers/linear.py`

#### ColumnParallelLinear (Lines 54-74)

```python
class ColumnParallelLinear(LinearBase):
    """
    Shards output dimension across TP ranks.
    Output is split column-wise (along output features).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # LINE 63: Initialize with sharded output size
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)
        
        # Weight shape: [output_size // tp_size, input_size]
        # tp_dim = 0 (shard along output dimension)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load checkpoint weights for this rank's shard."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # size(0) = output_size // tp_size
        start_idx = self.tp_rank * shard_size      # Start offset for this rank
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each rank computes its output shard independently
        # Output from all ranks are concatenated
        return F.linear(x, self.weight, self.bias)
```

**Key Points:**
- **Weight shape:** `[output_size // tp_size, input_size]`
- **tp_dim:** 0 (output dimension is sharded)
- **Load strategy:** Each rank loads only its shard of the output dimension
- **Forward:** Each rank produces `output_size // tp_size` outputs independently

**Usage in Qwen3.5 FullAttention (Line 499-503):**
```python
self.q_proj = ColumnParallelLinear(
    hidden_size,
    num_heads * head_dim * 2,  # Query + gate output
    bias=False,
)
# Weight shape: [16*256*2 // tp_size, hidden_size] = [2048 // tp_size, 2048]
# On tp_rank=0: [512, 2048]
# On tp_rank=1: [512, 2048]
# ...
```

---

#### RowParallelLinear (Lines 131-154)

```python
class RowParallelLinear(LinearBase):
    """
    Shards input dimension across TP ranks.
    Input is split row-wise (along sequence/batch items receive different features).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # LINE 140: Initialize with sharded input size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)
        
        # Weight shape: [output_size, input_size // tp_size]
        # tp_dim = 1 (shard along input dimension)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Load checkpoint weights for this rank's shard."""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)  # size(1) = input_size // tp_size
        start_idx = self.tp_rank * shard_size      # Start offset for this rank
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LINE 150: Each rank computes partial output
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        # LINE 152-153: All-reduce to sum contributions from all ranks
        if self.tp_size > 1:
            dist.all_reduce(y)  # y += y_from_all_other_ranks
        return y
```

**Key Points:**
- **Weight shape:** `[output_size, input_size // tp_size]`
- **tp_dim:** 1 (input dimension is sharded)
- **Load strategy:** Each rank loads its shard of the input dimension
- **Forward:** 
  - Each rank computes partial output: `y = x_shard @ W_shard.T`
  - All-reduce sums contributions: `final_y = sum(y_all_ranks)`
  - Only rank 0 adds bias

**Usage in Qwen3 (Line 49-53):**
```python
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,  # Total input from all Q heads
    hidden_size,
    bias=False,
)
# Weight shape: [hidden_size, (total_num_heads * head_dim) // tp_size]
# Example: [4096, (32 * 128) // 4] = [4096, 1024]
```

---

## 3. QKVParallelLinear Implementation

### File: `/repo/nanovllm/layers/linear.py`
**Lines:** 96-129

```python
class QKVParallelLinear(ColumnParallelLinear):
    """
    Specialized column-parallel linear that handles Q, K, V head sharding.
    
    Output layout per rank:
    [q_shard | k_shard | v_shard]
    
    Each shard contains:
    - q_shard: num_heads // tp_size * head_dim
    - k_shard: num_kv_heads // tp_size * head_dim
    - v_shard: num_kv_heads // tp_size * head_dim
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        # LINES 108-110: Pre-compute per-rank head counts
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)           # Q heads per rank
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)     # KV heads per rank
        
        # LINE 111: Total output size (before sharding)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        
        # LINE 112: Initialize as ColumnParallel with sharded output
        super().__init__(hidden_size, output_size, bias)
        
        # Final weight shape: [output_size // tp_size, input_size]
        # Example (tp_size=4, total_num_heads=32, total_num_kv_heads=8, head_dim=128):
        # output_size = (32 + 2*8) * 128 = 5120
        # weight shape: [5120 // 4, hidden_size] = [1280, hidden_size]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        Load weights for Q, K, or V separately.
        
        Layout in loaded_weight (full checkpoint):
        [Q_full | K_full | V_full]
        where:
          Q_full: total_num_heads * head_dim
          K_full: total_num_kv_heads * head_dim
          V_full: total_num_kv_heads * head_dim
        
        Layout in param_data (this rank's weight):
        [Q_shard | K_shard | V_shard]
        where:
          Q_shard: (total_num_heads // tp_size) * head_dim
          K_shard: (total_num_kv_heads // tp_size) * head_dim
          V_shard: (total_num_kv_heads // tp_size) * head_dim
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # LINES 117-125: Determine offset and size for this shard
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # LINE 126: Extract this rank's shard from full weight
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # LINE 127: Split loaded weight across TP ranks, select this rank's partition
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        # LINE 128: Copy into parameters
        param_data.copy_(loaded_weight)
```

**Weight Organization:**

```
Loaded checkpoint weight shape: [total_output_size, hidden_size]
  = [(total_num_heads + 2*total_num_kv_heads)*head_dim, hidden_size]

Per-rank weight shape: [total_output_size // tp_size, hidden_size]

Layout per rank (tp_rank=0, tp_size=4):
[Q_heads_0:8 | K_heads_0:2 | V_heads_0:2] = [8*128 | 2*128 | 2*128] = [1280 features]

Example with Qwen3:
- total_num_heads = 32
- total_num_kv_heads = 8
- head_dim = 128
- tp_size = 4

Per rank:
- num_heads = 8
- num_kv_heads = 2
- Output per rank: [8*128 + 2*128 + 2*128] = [1280]
- Total across all ranks: [1280 * 4] = [5120] ✓
```

---

## 4. Qwen3.5 Config Values

### File: `/repo/qwen3.5/qwen3.5-35B-A3B-config`

**Text Config Section:**

```json
{
  "text_config": {
    "num_attention_heads": 16,      // LINE 71
    "num_key_value_heads": 2,       // LINE 75
    "head_dim": 256,                // LINE 14
    "hidden_size": 2048,            // LINE 16
    ...
    "full_attention_interval": 4,   // Every 4th layer is full attention
  }
}
```

**Critical Numbers:**
- `num_attention_heads` = 16 (Q heads)
- `num_key_value_heads` = 2 (KV heads)
- `head_dim` = 256
- `hidden_size` = 2048

**The Problem (with TP):**
```
If tp_size = 4:
  num_heads_per_rank = 16 // 4 = 4  ✓ OK
  num_kv_heads_per_rank = 2 // 4 = 0  ❌ ZERO!

If tp_size = 2:
  num_heads_per_rank = 16 // 2 = 8  ✓ OK
  num_kv_heads_per_rank = 2 // 2 = 1  ✓ OK (minimum viable)
  
If tp_size = 1 (no TP):
  num_heads_per_rank = 16  ✓ OK
  num_kv_heads_per_rank = 2  ✓ OK
```

---

## 5. How `get_context()` Provides `tp_size` and `tp_rank`

### File: `/repo/nanovllm/utils/context.py`

```python
import torch.distributed as dist

# tp_size and tp_rank are obtained directly from PyTorch Distributed
tp_size = dist.get_world_size()   # Number of TP ranks
tp_rank = dist.get_rank()         # This rank's index (0 to tp_size-1)
```

**How it's used:**

1. **Linear layers (via dist):**
   ```python
   # In LinearBase.__init__
   self.tp_rank = dist.get_rank()
   self.tp_size = dist.get_world_size()
   ```

2. **Attention module (in qwen3.py):**
   ```python
   # In Qwen3Attention.__init__
   tp_size = dist.get_world_size()
   self.num_heads = self.total_num_heads // tp_size
   ```

3. **Full attention module (in qwen3_5.py):**
   ```python
   # In Qwen3_5FullAttention.__init__
   tp_size = dist.get_world_size()
   self.num_heads = num_heads // tp_size
   self.num_kv_heads = num_kv_heads // tp_size  # ❌ CAN BE ZERO!
   ```

---

## Root Cause Analysis: Qwen3.5 TP Issue

### The Problem:

Qwen3.5 has only **2 key-value heads** but attempts to use TP with potentially 4+ ranks:

```
num_key_value_heads = 2
tp_size = 4
num_kv_heads_per_rank = 2 // 4 = 0 ❌
```

This causes:
1. **Shape errors in reshape operations:** `view(-1, 0, 256)` is invalid
2. **Attention computation failures:** Can't reshape attention output
3. **Weight loading issues:** Trying to split 2 heads across 4 ranks

### Why Qwen3 Works:

Qwen3 has **8 key-value heads**, which divides evenly across TP ranks:

```
num_key_value_heads = 8
tp_size = 4
num_kv_heads_per_rank = 8 // 4 = 2 ✓
```

---

## Recommended Fixes

### 1. **Constraint Check**
Add validation in `Qwen3_5FullAttention.__init__`:

```python
assert num_kv_heads % tp_size == 0, (
    f"num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({tp_size}). "
    f"For Qwen3.5 with {num_kv_heads} KV heads, use tp_size <= {num_kv_heads}."
)
```

### 2. **Support Head Replication** (if needed)
For cases where `num_kv_heads < tp_size`, replicate K/V heads across ranks:

```python
if num_kv_heads < tp_size:
    # Replicate instead of shard
    self.num_kv_heads = num_kv_heads
    # Use ReplicatedLinear for K/V instead of ColumnParallelLinear
else:
    self.num_kv_heads = num_kv_heads // tp_size
```

### 3. **Documentation**
Update model documentation to specify maximum supported `tp_size`:

```
Qwen3.5-35B Configuration:
- Maximum tp_size: 2 (due to 2 KV heads)
- Recommended tp_size: 1 or 2
- TP with size > 2 will fail during initialization
```

---

## Summary Table

| Aspect | Qwen3 | Qwen3.5 | Status |
|--------|-------|---------|--------|
| `num_attention_heads` | 32 | 16 | Qwen3.5 lower |
| `num_key_value_heads` | 8 | 2 | ❌ Qwen3.5 problematic |
| `head_dim` | 128 | 256 | Qwen3.5 larger |
| Max TP rank (divisible) | 8 | 2 | Qwen3.5 limited |
| Qwen3 TP=4 works | ✓ | N/A | Reference works |
| Qwen3.5 TP=4 works | N/A | ❌ | Issue: 2 KV heads → 0 heads/rank |

