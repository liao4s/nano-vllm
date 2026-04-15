# Side-by-Side Code Comparison: Qwen3 vs Qwen3.5 Attention

## Initialization Comparison

### Qwen3Attention ✓ WORKS
**File:** `nanovllm/models/qwen3.py:14-40`

```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,               # e.g., 32
        num_kv_heads: int,            # e.g., 8
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()                          # Get TP size
        
        # ✓ Store total heads before division
        self.total_num_heads = num_heads                         # 32
        assert self.total_num_heads % tp_size == 0              # ✓ CHECK!
        self.num_heads = self.total_num_heads // tp_size         # 32 // 4 = 8
        
        # ✓ Same pattern for KV heads
        self.total_num_kv_heads = num_kv_heads                  # 8
        assert self.total_num_kv_heads % tp_size == 0           # ✓ CHECK!
        self.num_kv_heads = self.total_num_kv_heads // tp_size   # 8 // 4 = 2
        
        # ✓ head_dim computed from TOTAL heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 4096 // 32 = 128
        
        # ✓ Sizes for THIS rank only
        self.q_size = self.num_heads * self.head_dim            # 8 * 128 = 1024
        self.kv_size = self.num_kv_heads * self.head_dim        # 2 * 128 = 256
```

**Summary:**
- ✅ Has assertions to validate divisibility
- ✅ Stores both total and per-rank head counts
- ✅ Computes head_dim correctly using total heads
- ✅ Works because 8 KV heads % 4 == 0

---

### Qwen3_5FullAttention ❌ BROKEN
**File:** `nanovllm/models/qwen3_5.py:476-496`

```python
class Qwen3_5FullAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,               # e.g., 16
        num_kv_heads: int,            # e.g., 2
        head_dim: int,                # e.g., 256 (PASSED IN)
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        tp_size = dist.get_world_size()                         # Get TP size
        
        # ✓ Store total heads
        self.total_num_heads = num_heads                        # 16
        self.num_heads = num_heads // tp_size                   # 16 // 4 = 4
        
        # ❌ Same division pattern BUT...
        self.total_num_kv_heads = num_kv_heads                  # 2
        self.num_kv_heads = num_kv_heads // tp_size             # 2 // 4 = 0 ❌ PROBLEM!
        
        # ❌ head_dim is a parameter, not computed
        self.head_dim = head_dim                                # 256 (not validated)
        
        # Computed sizes
        self.q_size = self.num_heads * head_dim                # 4 * 256 = 1024
        self.kv_size = self.num_kv_heads * head_dim             # 0 * 256 = 0 ❌
        self.scaling = head_dim ** -0.5
```

**Issues:**
- ❌ NO assertions - silently creates 0 heads per rank
- ❌ head_dim passed as parameter (not validated)
- ❌ Works only if num_kv_heads % tp_size == 0
- ❌ Fails at runtime in forward pass when reshaping

---

## Forward Pass Comparison

### Qwen3Attention Forward ✓ WORKS
**File:** `nanovllm/models/qwen3.py:71-87`

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # Get QKV projection (each rank gets its shard)
    qkv = self.qkv_proj(hidden_states)  # [N, q_size + kv_size + kv_size]
    
    # Split into Q, K, V
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # q: [N, 1024], k: [N, 256], v: [N, 256]
    
    # Reshape to heads
    q = q.view(-1, self.num_heads, self.head_dim)           # [N, 8, 128] ✓
    k = k.view(-1, self.num_kv_heads, self.head_dim)        # [N, 2, 128] ✓
    v = v.view(-1, self.num_kv_heads, self.head_dim)        # [N, 2, 128] ✓
    
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)
    
    output = self.o_proj(o.flatten(1, -1))  # Flatten and reduce
    return output
```

**Success because:**
- `self.kv_size = 2 * 128 = 256` ✓
- `v.view(-1, 2, 128)` ✓ Valid reshape

---

### Qwen3_5FullAttention Forward ❌ BROKEN
**File:** `nanovllm/models/qwen3_5.py:542-576`

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # Q projection: outputs [N, num_heads * head_dim * 2 / tp_size]
    q_out = self.q_proj(hidden_states)  # [N, 1024]
    
    # Reshape to [N, num_heads, head_dim * 2], then split query and gate
    q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
    # [N, 4, 512] ✓ Still works because Q heads = 4
    
    query, gate = q_out.chunk(2, dim=-1)  # each [N, 4, 256]
    gate = gate.reshape(-1, self.num_heads * self.head_dim)  # [N, 1024]
    
    # K, V projections
    k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
    #                                    [-, 0, 256]
    # ❌ CRASH! Can't reshape to 0 dimensions!
    
    v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
    #                                    [-, 0, 256]
    # ❌ CRASH! Can't reshape to 0 dimensions!
    
    # ... rest unreachable
```

**Failure because:**
- `self.num_kv_heads = 2 // 4 = 0` ❌
- `.view(-1, 0, 256)` is invalid PyTorch reshape

---

## Weight Projection Comparison

### Qwen3: QKVParallelLinear Usage ✓
**File:** `nanovllm/models/qwen3.py:42-48`

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,              # 128 (per-rank dimension)
    self.total_num_heads,       # 32 (total, BEFORE division)
    self.total_num_kv_heads,    # 8 (total, BEFORE division)
    bias=qkv_bias,
)

# QKVParallelLinear will:
# - Compute num_heads = 32 // 4 = 8 per rank
# - Compute num_kv_heads = 8 // 4 = 2 per rank
# - Create output: [8*128 + 2*128 + 2*128] = [1280] per rank
# - Total: 1280 * 4 = 5120 ✓
```

---

### Qwen3.5: ColumnParallelLinear Usage (Q) ✓
**File:** `nanovllm/models/qwen3_5.py:499-503`

```python
self.q_proj = ColumnParallelLinear(
    hidden_size,
    num_heads * head_dim * 2,    # 16 * 256 * 2 = 8192
    bias=False,
)

# ColumnParallelLinear will:
# - Divide output: 8192 // 4 = 2048 per rank ✓
```

---

### Qwen3.5: ColumnParallelLinear Usage (K,V) ❌
**File:** `nanovllm/models/qwen3_5.py:504-512`

```python
self.k_proj = ColumnParallelLinear(
    hidden_size,
    num_kv_heads * head_dim,     # 2 * 256 = 512
    bias=False,
)

# ColumnParallelLinear will:
# - Divide output: 512 // 4 = 128 per rank
# - Each rank gets 128 output features
# - But forward expects it to be divisible into 0 heads! ❌

self.v_proj = ColumnParallelLinear(
    hidden_size,
    num_kv_heads * head_dim,     # 2 * 256 = 512
    bias=False,
)

# Same problem: 128 features can't be reshaped to (0, 256) ❌
```

---

## QKVParallelLinear Details

### File: `nanovllm/layers/linear.py:96-129`

```python
class QKVParallelLinear(ColumnParallelLinear):
    """Specializes ColumnParallelLinear to handle Q, K, V sharding."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,      # 32
        total_num_kv_heads: int,   # 8
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        # Pre-compute per-rank divisions
        self.head_size = head_size                              # 128
        self.num_heads = divide(total_num_heads, tp_size)      # 32 // 4 = 8
        self.num_kv_heads = divide(total_num_kv_heads, tp_size) # 8 // 4 = 2
        
        # Total output size (before sharding)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        # = (32 + 2*8) * 128 = 5120
        
        # Initialize parent ColumnParallel (which divides by tp_size)
        super().__init__(hidden_size, output_size, bias)
        # Weight shape becomes: [5120 // 4, hidden_size] = [1280, hidden_size]
        
        # This is the KEY: QKV is handled as ONE output dimension
        # All sharding happens at the END (after concatenation)
```

---

## Configuration Comparison

### Qwen3 Config Values

```json
{
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "hidden_size": 4096
}
```

**Math with tp_size=4:**
```
Q heads per rank: 32 // 4 = 8  ✓
KV heads per rank: 8 // 4 = 2  ✓

Divisibility check:
  32 % 4 = 0 ✓
  8 % 4 = 0 ✓
```

---

### Qwen3.5 Config Values

**File:** `qwen3.5/qwen3.5-35B-A3B-config:71,75`

```json
{
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "hidden_size": 2048
}
```

**Math with tp_size=4:**
```
Q heads per rank: 16 // 4 = 4   ✓
KV heads per rank: 2 // 4 = 0   ❌

Divisibility check:
  16 % 4 = 0  ✓
  2 % 4 = 2   ❌ FAILS!
```

**Math with tp_size=2:**
```
Q heads per rank: 16 // 2 = 8   ✓
KV heads per rank: 2 // 2 = 1   ✓

Divisibility check:
  16 % 2 = 0  ✓
  2 % 2 = 0   ✓ WORKS!
```

---

## Key Differences Summary

| Aspect | Qwen3 | Qwen3.5 |
|--------|-------|---------|
| Q heads | 32 (divisible by many) | 16 (less flexible) |
| KV heads | 8 (divisible by many) | 2 (**very limited**) |
| head_dim computation | In Attention class | Passed as parameter |
| Assertions | ✓ Present | ❌ Missing |
| QKV handling | Unified QKVParallelLinear | Separate projections |
| head_dim usage | Computed from total heads | Parameter (not validated) |
| TP compatibility | tp_size ≤ 8 | **tp_size ≤ 2** |

---

## Error Stack Trace (What Actually Happens)

### Qwen3.5 with tp_size=4:

```
File nanovllm/models/qwen3_5.py, line 492
  self.num_kv_heads = num_kv_heads // tp_size
  # num_kv_heads = 2 // 4 = 0

File nanovllm/models/qwen3_5.py, line 495
  self.kv_size = self.num_kv_heads * head_dim
  # kv_size = 0 * 256 = 0

File nanovllm/models/qwen3_5.py, line 557
  k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
  # k = self.k_proj(hidden_states).view(-1, 0, 256)
  
RuntimeError: shape '[N, 0, 256]' is invalid for input of size [N*128]
```

The forward pass fails when trying to reshape the KV projection output to 0 heads.

---

## The Fix

### Option 1: Add Assertion (Minimal)
**In Qwen3_5FullAttention.__init__ after line 492:**

```python
self.num_kv_heads = num_kv_heads // tp_size

# ADD THIS:
if self.num_kv_heads == 0:
    raise ValueError(
        f"num_kv_heads ({num_kv_heads}) < tp_size ({tp_size}) not supported. "
        f"For Qwen3.5 with {num_kv_heads} KV heads, maximum tp_size is {num_kv_heads}."
    )
```

### Option 2: Replicate KV Heads (Better)
**In Qwen3_5FullAttention.__init__:**

```python
if num_kv_heads < tp_size:
    # Don't shard KV heads, replicate them
    self.num_kv_heads = num_kv_heads
    # Use ReplicatedLinear for k_proj, v_proj
    self.k_proj = ReplicatedLinear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.v_proj = ReplicatedLinear(hidden_size, num_kv_heads * head_dim, bias=False)
else:
    # Normal sharding
    self.num_kv_heads = num_kv_heads // tp_size
    self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, bias=False)
```

