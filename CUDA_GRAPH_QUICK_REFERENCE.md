# CUDA Graph & GatedDeltaNet: Quick Reference

## The Problem in 30 Seconds

```python
# GatedDeltaNet uses Python dict to store per-sequence state:
self._recurrent_states[seq_id] = state_tensor  # PYTHON-SIDE STATE
self._conv_states[seq_id] = conv_tensor

# CUDA graphs cannot re-execute Python code during replay()
# → State lookups fail
# → New tensors created at different GPU addresses
# → Graph replay crashes or produces wrong results
```

**Current workaround**: Force `enforce_eager=True` (disables graphs, ~2-3× slower decode)

---

## Config Values (Qwen3.5)

```
linear_num_key_heads: 16          # K heads
linear_num_value_heads: 32        # V heads
linear_key_head_dim: 128          # dim per K head
linear_value_head_dim: 128        # dim per V head
linear_conv_kernel_dim: 4         # Conv kernel size

Derived:
- key_dim = 16 × 128 = 2048
- value_dim = 32 × 128 = 4096
- conv_dim = 2×2048 + 4096 = 8192
```

---

## Solution: Pre-Allocated Buffers

Replace dict with GPU tensors at **fixed addresses**:

```python
# BEFORE (dict-based, incompatible with graph)
self._recurrent_states: dict[int, Tensor] = {}

# AFTER (buffer-based, compatible with graph)
self.recurrent_state_buffer: torch.Tensor  # [max_bs, num_v_heads, head_k_dim, head_v_dim]
self.conv_state_buffer: torch.Tensor       # [max_bs, conv_dim, conv_kernel_size - 1]
```

### Memory Requirements (Qwen3.5, max_bs=512)

```
Per linear attention layer:
- Recurrent buffer: 512 × 32 × 128 × 128 × 4 bytes = 1.0 GB (float32)
- Conv buffer:      512 × 8192 × 3 × 4 bytes = 50 MB
- Total per layer:  ~1.05 GB

For 10 linear layers: ~10.5 GB (dominant cost)

Optimization: Use bfloat16 → 5 GB (50% reduction)
```

---

## State Flow

### Current (Eager Mode)

```
Prefill:
  Input → GatedDeltaNet → Output
            ↓
         Save to dict[seq_id]

Decode step 1:
  Input + dict[seq_id] → GatedDeltaNet → Output
            ↓ (read from dict)       ↓ (update dict)
         Update dict[seq_id]

Decode step 2: (Same pattern)
```

### Proposed (Graph Mode)

```
Pre-Allocate:
  buffer[0..511] = all sequence states (GPU memory)

Prefill:
  Input → GatedDeltaNet → Output
            ↓
         Write to buffer[seq_id]

Decode step 1 (BEFORE graph.replay()):
  Copy buffer[seq_id] into buffer[batch_idx]  ← CPU-side update
  
  Input + buffer[batch_idx] → [GRAPH CAPTURE: kernels operate on fixed buffer addresses]
                  ↓
             Output
  
  Copy buffer[batch_idx] back to buffer[seq_id]  ← CPU-side update

Decode step 2: (Same pattern, reuses graph)
```

---

## Exact Tensor Operations

### Recurrent State Update (Decode)

```python
# Input: query [1,1,32,128], key [1,1,32,128], value [1,1,32,128], 
#        recurrent_state [1,32,128,128]

# For each token i:
g_t = g[..., i].exp()                                    # Decay factor
q_t = query[..., i]                                      # [1,32,128]
k_t = key[..., i]                                        # [1,32,128]
v_t = value[..., i]                                      # [1,32,128]

# Operation 1: Decay
recurrent_state *= g_t.unsqueeze(-1).unsqueeze(-1)      # [1,32,128,128] *= [1,32,1,1]

# Operation 2: Fetch from memory
kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(-2)  # [1,32,128]

# Operation 3: Update
delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)           # [1,32,128]
recurrent_state += k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # [1,32,128,128]

# Operation 4: Output
output = (recurrent_state * q_t.unsqueeze(-1)).sum(-2)  # [1,32,128]
```

### Conv State Update (Decode)

```python
# Input: mixed_qkv [1,8192,1], conv_state [1,8192,3] (from previous step)

# Concatenate: [1, 8192, 3] + [1, 8192, 1] = [1, 8192, 4]
conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)

# Apply depthwise conv (kernel_size=4, requires full window)
conv_output = F.conv1d(conv_input, weight, groups=8192)  # [1,8192,1]

# Save last 3 elements for next step
conv_state = conv_input[:, :, -3:]                       # [1,8192,3]
```

---

## Implementation Checklist

### Phase 1: Setup
- [ ] Add `allocate_linear_attn_states()` in ModelRunner
- [ ] Pre-allocate buffers as layer attributes
- [ ] Extend `Context` with `seq_id_to_buffer_idx`
- [ ] Add `_seq_id_to_state` dict to ModelRunner

### Phase 2: Graph Capture
- [ ] Zero-init buffers before capture
- [ ] Create seq_id→buffer_idx mapping for test batch

### Phase 3: Graph Replay
- [ ] **Before replay**: Copy saved state from dict into buffer[batch_idx]
- [ ] Call `graph.replay()`
- [ ] **After replay**: Copy buffer[batch_idx] back to dict

### Phase 4: GatedDeltaNet
- [ ] Add `recurrent_state_buffer` tensor
- [ ] Add `conv_state_buffer` tensor
- [ ] Modify `_forward_decode_one(buffer_idx)` to use buffers
- [ ] Modify `_forward_prefill()` to zero-init buffer
- [ ] Remove dict state management

### Phase 5: Testing
- [ ] Verify graph capture succeeds
- [ ] Benchmark graph vs eager
- [ ] Profile memory usage
- [ ] Test state correctness (prefill→decode continuity)

---

## Comparison: Dict vs Pre-Allocated

| Aspect | Dict | Pre-Allocated |
|---|---|---|
| GPU Addresses | Dynamic | Fixed ✓ |
| CUDA Graph Compatible | ❌ | ✅ |
| Memory Overhead | ~1-10 MB active | 1-10 GB pre-allocated |
| Code Complexity | Simple | Medium |
| State Lifetime | Sparse dict | Dense buffer |
| Cleanup | Per-sequence | Zero buffer slot |
| Performance | ~100 tok/s (eager) | ~300 tok/s (graph) |

---

## Code Locations

### Key Files
- **model_runner.py**: Lines 45-53 (workaround), 223-239 (graph replay), 264-299 (graph capture)
- **qwen3_5.py**: Lines 228-273 (init), 285-347 (prefill), 349-415 (decode), 417-452 (forward)
- **config**: qwen3.5/qwen3.5-35B-A3B-config (lines 60-64)

### Related Files
- **attention.py**: How normal attention works with fixed k_cache/v_cache
- **context.py**: Where context variables are passed (needs seq_id→buffer_idx mapping)

---

## Why This Works

1. **Fixed GPU addresses**: Pre-allocated tensors stay at same GPU memory throughout inference
2. **CUDA graph constraint**: Graphs only record GPU operations, cannot re-execute Python
3. **Pre-update pattern**: CPU-side code updates buffer values BEFORE replay()
4. **Post-update pattern**: CPU-side code reads buffer values AFTER replay()

The graph only sees: "Read from GPU address X, write to GPU address Y" — doesn't care when/how values were set.

---

## Potential Issues

### Issue 1: Memory Overhead
- **Solution**: Use bfloat16 instead of float32 (50% reduction)
- **Alternative**: Reduce max_batch_size for graph capture

### Issue 2: State Synchronization
- **Solution**: Careful copy semantics between dict and buffer
- **Test**: Verify prefill→decode state continuity

### Issue 3: Multi-Sequence Batches
- **Solution**: Create seq_id→buffer_idx mapping before graph.replay()
- **Ensure**: Each sequence gets its own buffer slot

---

## Benchmarking Expected Improvements

```
Qwen3.5-35B on single GPU (assume 512 batch size for graphs):

Decode (current - eager):
- Per-token latency: ~10-15ms
- Throughput: ~100 tokens/s

Decode (proposed - with graph):
- Per-token latency: ~3-5ms (skip kernel launch overhead)
- Throughput: ~300 tokens/s

Expected speedup: 3× for large decode batches
```

---

## Safety Checklist

Before enabling graphs:
- [ ] State is correctly initialized to zero for first token
- [ ] Prefill→decode state transfer works correctly
- [ ] Multiple sequences in batch don't interfere
- [ ] State cleanup doesn't leak GPU memory
- [ ] Graph replay produces identical results to eager mode

