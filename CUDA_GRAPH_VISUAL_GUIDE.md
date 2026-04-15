# CUDA Graph & GatedDeltaNet: Visual Guide

## 1. Current Architecture (Eager Mode)

### Model Runner Setup
```
┌─────────────────────────────────────────────────────────┐
│                    ModelRunner Init                      │
├─────────────────────────────────────────────────────────┤
│ 1. Load Qwen3.5 model                                   │
│ 2. Detect model_type == 'qwen3_5_moe'                   │
│ 3. Check: enforce_eager enabled? YES → Skip graph setup │
│    [Lines 45-53: Force eager mode]                      │
│ 4. Call run() → always uses run_model eager path        │
└─────────────────────────────────────────────────────────┘
```

### Inference Flow (Eager)
```
Input tokens
    ↓
[Prefill or Decode?]
    ├─ PREFILL: process full sequence
    │   ├─ GatedDeltaNet._forward_prefill()
    │   │   ├─ Compute query/key/value
    │   │   ├─ Apply conv1d
    │   │   ├─ Run gated delta rule
    │   │   └─ [SAVE STATE] → dict[seq_id]  ← PYTHON-SIDE STATE
    │   └─ Output
    │
    └─ DECODE: process one token at a time
        ├─ GatedDeltaNet._forward_decode_one()
        │   ├─ [READ STATE] ← dict[seq_id]  ← PYTHON-SIDE LOOKUP (breaks graphs!)
        │   ├─ Compute query/key/value
        │   ├─ Apply conv1d with cached state
        │   ├─ Run gated delta rule (recurrent)
        │   └─ [UPDATE STATE] → dict[seq_id]  ← PYTHON-SIDE MUTATION
        └─ Output

ISSUE: dict[seq_id] lookups and mutations can't be recorded in CUDA graphs!
```

### Dict-Based State Storage
```
Qwen3_5GatedDeltaNet instance (in each layer):

self._recurrent_states = {          self._conv_states = {
    0: Tensor[1,32,128,128],            0: Tensor[1,8192,3],
    1: Tensor[1,32,128,128],            1: Tensor[1,8192,3],
    2: Tensor[1,32,128,128],            2: Tensor[1,8192,3],
    ...                                 ...
}                                   }

Problem: Each tensor created with NEW GPU address
         CUDA graphs can't reference dynamic addresses
```

---

## 2. CUDA Graph Internals (How They Work)

### Graph Capture Process
```
PHASE 1: WARMUP (not recorded)
    Input tensors → Forward pass → Kernels execute
    ↓
    All GPU kernel code paths are compiled/loaded

PHASE 2: CAPTURE (recorded to graph)
    Input tensors → Forward pass → Kernels execute
    ↑                                      ↓
    Graph records:                    Graph records:
    - Kernel launch addresses         - Memory read/write addresses
    - Thread block config             - Data transfer patterns
    - Parameter buffers              - Synchronization points
    ↓
    Result: Graph = "cached kernel sequence"

PHASE 3: REPLAY (fast execution)
    graph.replay()
    ↓
    GPU executes: "do all recorded kernel launches in sequence"
    NO CPU involvement → ~100-200× kernel launch overhead reduction
```

### Key Constraint: Fixed GPU Addresses

```
Capture Phase:
    Tensor A at GPU address 0x123ABC
    ├─ Kernel 1: read from 0x123ABC
    ├─ Kernel 2: write to 0x123ABC + offset
    └─ Graph records: "kernel 1 reads from 0x123ABC"

Replay Phase:
    Tensor A still at GPU address 0x123ABC  ✓ SAME ADDRESS
    ├─ Kernel 1: read from 0x123ABC         ✓ Works!
    └─ Output matches

BUT IF:
    Tensor A reallocated to 0x456DEF        ✗ DIFFERENT ADDRESS
    └─ Kernel 1 reads from stale 0x123ABC  ✗ Wrong memory!
```

---

## 3. How Attention Handles This (Reference Implementation)

### Fixed Cache Pattern (Works with Graphs!)

```
ModelRunner.__init__:
    ↓
    allocate_kv_cache()
    ├─ Create: self.kv_cache = torch.empty(...)
    │          GPU address: 0xAAAA (fixed!)
    ├─ Assign to each Attention layer:
    │   └─ module.k_cache = self.kv_cache[0, layer_id]
    │   └─ module.v_cache = self.kv_cache[1, layer_id]
    │      Both point to FIXED GPU addresses
    └─ Status: Tensors allocated ONCE, never reallocated

Graph Capture:
    ├─ set_context(slot_mapping, context_lens, block_tables)
    ├─ model(input_ids[:bs], positions[:bs])
    │   └─ Attention.forward()
    │       ├─ store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    │       │  └─ Kernel reads from FIXED k_cache/v_cache addresses
    │       └─ flash_attn_with_kvcache(..., cache_seqlens=context_lens, ...)
    │          └─ Kernel reads from FIXED cache, uses dynamic context_lens
    └─ Graph records: kernel ops on 0xAAAA (k_cache) and 0xBBBB (v_cache)

Graph Replay (Per Token):
    ├─ Update context_lens: 5 → 6  (CPU-side, before replay)
    ├─ Update slot_mapping: [...] (CPU-side, before replay)
    ├─ graph.replay()
    │   └─ Kernels read from SAME 0xAAAA/0xBBBB addresses
    │   └─ Use UPDATED context values
    └─ Result: Correct!

KEY INSIGHT: GPU addresses are CONSTANT, CPU values (context) UPDATED before replay()
```

### Attention Module Structure
```
class Attention(nn.Module):
    def __init__(self):
        self.k_cache = torch.tensor([])   # Pre-allocated by ModelRunner
        self.v_cache = torch.tensor([])   # Pre-allocated by ModelRunner
        # GPU addresses: FIXED for life of engine

    def forward(self, q, k, v):
        context = get_context()
        
        # Store k,v to fixed cache using slot_mapping (which slots to write to)
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        #              ↓    ↓   ↓          ↓          ↓
        #              From attention     To fixed   Where to store
        #                                 cache      (dynamic)
        
        # Use cached k,v with context information
        o = flash_attn_with_kvcache(..., 
                                    cache_seqlens=context.context_lens,
                                    block_table=context.block_tables)
        #                                         ↑ Dynamic values
        #                                         ↓ Updated before graph.replay()
        return o
```

---

## 4. Why GatedDeltaNet Breaks (Current Implementation)

### Dict-Based State Problem

```
Decode Step 1 (Captured):
    ┌─────────────────────────────────┐
    │ GatedDeltaNet._forward_decode_one
    ├─────────────────────────────────┤
    │ recurrent_state = self._recurrent_states.get(0)
    │                   GPU address: 0xCCCC (created on-demand)
    │ 
    │ ... compute ...
    │ 
    │ self._recurrent_states[0] = last_recurrent_state
    │                             GPU address: 0xDDDD (NEW!)
    │
    │ Graph records: "operations on 0xCCCC and 0xDDDD"
    └─────────────────────────────────┘

Decode Step 2 (Replay):
    ┌─────────────────────────────────┐
    │ graph.replay()
    ├─────────────────────────────────┤
    │ Tries to read from 0xCCCC       ✗ Wrong! 
    │                   (old address)
    │ Tries to write to 0xDDDD        ✗ Wrong!
    │                   (old address)
    │
    │ But tensors are NOW at 0xEEEE and 0xFFFF!
    │ Kernels access garbage memory
    └─────────────────────────────────┘

Result: Crash or silent data corruption
```

### Why Dict Breaks Graphs

```
CUDA Graph Constraint: "All GPU memory addresses must be fixed at capture time"

Dict-based state violates this:

    Capture Phase:
    ├─ seq_id=0: create tensor at 0xAAAA, graph records "read 0xAAAA"
    ├─ seq_id=1: create tensor at 0xBBBB, graph records "read 0xBBBB"
    └─ etc.

    Replay Phase 1 (seq_id=0 in batch):
    ├─ Need to read seq_id=0 state
    ├─ But dict[0] was deleted after seq_id=0 finished!
    ├─ OR dict[0] now points to NEW tensor at 0xCCCC (not 0xAAAA)
    └─ Graph tries to read 0xAAAA → garbage

    Replay Phase 2 (seq_id=0 again, different batch):
    ├─ Create new tensor for seq_id=0 at 0xDDDD
    ├─ Graph still tries to read 0xAAAA
    └─ Mismatch!
```

---

## 5. Proposed Solution: Pre-Allocated Buffers

### Architecture Change

```
BEFORE (Dict-based):
┌──────────────────────┐
│ GatedDeltaNet Layer  │
├──────────────────────┤
│ self._recurrent      │ (Python dict, sparse)
│ _states: {           │
│   0: Tensor@0xCCCC,  │ ← Created on-demand
│   1: Tensor@0xDDDD,  │ ← Different addresses each time
│   ...                │ ✗ Not compatible with CUDA graphs
│ }                    │
│                      │
│ self._conv_states {  │
│   ...                │
│ }                    │
└──────────────────────┘

AFTER (Pre-allocated):
┌──────────────────────────────────┐
│ GatedDeltaNet Layer              │
├──────────────────────────────────┤
│ self.recurrent_state_buffer:     │
│   Tensor[512, 32, 128, 128]      │
│   GPU address: 0xAAAA (FIXED!)   │
│   ├─ buffer[0] → seq_id 0        │
│   ├─ buffer[1] → seq_id 1        │
│   └─ ...                         │
│                                  │
│ self.conv_state_buffer:          │
│   Tensor[512, 8192, 3]           │
│   GPU address: 0xBBBB (FIXED!)   │
│   ├─ buffer[0] → seq_id 0        │
│   ├─ buffer[1] → seq_id 1        │
│   └─ ...                         │
│                                  │
│ ✓ Compatible with CUDA graphs    │
└──────────────────────────────────┘
```

### Pre-Allocation Flow

```
ModelRunner Initialization:
    ├─ allocate_kv_cache()           (existing, for Attention)
    │
    └─ allocate_linear_attn_states() (NEW, for GatedDeltaNet)
        │
        ├─ For each GatedDeltaNet layer:
        │   ├─ Create recurrent_state_buffer[512, 32, 128, 128]
        │   │  GPU: 0xAAAA (ONE allocation, stays forever)
        │   │
        │   └─ Create conv_state_buffer[512, 8192, 3]
        │      GPU: 0xBBBB (ONE allocation, stays forever)
        │
        └─ Status: All buffers allocated at FIXED addresses
```

---

## 6. State Flow: Current vs. Proposed

### Current Flow (Per Decode Token)

```
Token from seq_id=0:
    ↓
    ┌─────────────────────────────────────┐
    │ run_model()                         │
    ├─────────────────────────────────────┤
    │ is_prefill=False, enforce_eager=True│
    │ → return self.model(...)             │
    │   (Never uses CUDA graph)           │
    └─────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────┐
    │ model.forward()                     │
    ├─────────────────────────────────────┤
    │ └─ GatedDeltaNet._forward_decode_one
    │    ├─ recurrent = dict[0]  (lookup)│
    │    ├─ conv = dict_conv[0]  (lookup)│
    │    ├─ ... compute ...               │
    │    ├─ dict[0] = new_recurrent      │
    │    ├─ dict_conv[0] = new_conv      │
    │    └─ return output                 │
    └─────────────────────────────────────┘
    ↓
    Output

Speed: ~100 tokens/s (kernel launch overhead)
```

### Proposed Flow (Per Decode Token with Graphs)

```
Token from seq_id=0:
    ↓
    ┌──────────────────────────────────────────┐
    │ run_model() → select graph for batch_size│
    ├──────────────────────────────────────────┤
    │ bs=1, context.seq_ids=[0]                │
    │                                          │
    │ [PRE-REPLAY UPDATE]  ← CPU-side         │
    │  Copy dict[0] → buffer[0] at 0xAAAA    │
    │  Copy dict_conv[0] → buffer[0] at 0xBB │
    │                                          │
    │ graph.replay()  ← Fast GPU execution    │
    │   ├─ Kernels read state from 0xAAAA    │
    │   ├─ Kernels write state to 0xAAAA     │
    │   ├─ All operations on fixed addresses  │
    │   └─ ~100-200× faster (no CPU launches) │
    │                                          │
    │ [POST-REPLAY UPDATE]  ← CPU-side        │
    │  Copy buffer[0] → dict[0]              │
    │  Copy buffer[0] → dict_conv[0]         │
    └──────────────────────────────────────────┘
    ↓
    Output

Speed: ~300 tokens/s (3× speedup!)
```

---

## 7. Memory Layout Diagram

### Pre-Allocated Buffers (Per Layer)

```
Recurrent State Buffer:
┌─────────────────────────────────────────────────┐
│ Shape: [512, 32, 128, 128]                      │
│ Size: 1.0 GB (float32) / 0.5 GB (bfloat16)     │
│                                                 │
│ GPU Memory: 0xAAAA (START)                      │
│                                                 │
│ ┌──────────────────────┐                        │
│ │ seq_id=0 state [1,1] │ ← 32 × 128 × 128 els │
│ ├──────────────────────┤                        │
│ │ seq_id=1 state [1,1] │                        │
│ ├──────────────────────┤                        │
│ │ seq_id=2 state [1,1] │                        │
│ ├──────────────────────┤                        │
│ │ ...                  │                        │
│ ├──────────────────────┤                        │
│ │ seq_id=511 state     │                        │
│ └──────────────────────┘                        │
│                                                 │
│ GPU Memory: 0xAAAA + 4MB×512 (END)              │
└─────────────────────────────────────────────────┘

Conv State Buffer:
┌─────────────────────────────────────────────────┐
│ Shape: [512, 8192, 3]                           │
│ Size: 50 MB (float32 / bfloat16)                │
│                                                 │
│ GPU Memory: 0xBBBB (START)                      │
│                                                 │
│ ┌──────────────────────┐                        │
│ │ seq_id=0 state  [1]  │ ← 8192 × 3 elements  │
│ ├──────────────────────┤                        │
│ │ seq_id=1 state  [1]  │                        │
│ ├──────────────────────┤                        │
│ │ ...                  │                        │
│ └──────────────────────┘                        │
│                                                 │
│ GPU Memory: 0xBBBB + 100KB×512 (END)            │
└─────────────────────────────────────────────────┘

Total per layer: ~1.05 GB
For 10 linear layers: ~10.5 GB (or 5 GB with bfloat16)
```

---

## 8. Sequence Lifecycle

### Prefill Phase

```
new_sequence: "What is AI?"
    ↓
    prepare_prefill()
    ├─ input_ids = [tok1, tok2, tok3]  (entire sequence at once)
    ├─ context.seq_ids = [0]  (seq_id assigned)
    └─ set_context(is_prefill=True, cu_seqlens=..., seq_ids=[0])

    ↓ (CPU)
    model.forward(input_ids, positions)
    │
    ├─ GatedDeltaNet._forward_prefill(hidden_states, seq_id=0)
    │   ├─ buffer[0].zero_()  ← Initialize recurrent state
    │   ├─ buffer_conv[0].zero_()  ← Initialize conv state
    │   ├─ ... process sequence ...
    │   ├─ buffer[0].copy_(final_recurrent_state)  ← Save
    │   ├─ buffer_conv[0].copy_(final_conv_state)  ← Save
    │   └─ return output
    │
    └─ Continue through rest of model

    ↓
    Save to dict: _seq_id_to_state[0] = (rec_state, conv_state)
    (Buffers still hold the state too)

    ↓
    Output tokens, add to sequence
```

### Decode Phase (Batched)

```
Batch 1:
  Sequences: [0, 1, 2]  (3 sequences in decode batch)

    prepare_decode()
    ├─ context.seq_ids = [0, 1, 2]
    ├─ context.seq_id_to_buffer_idx = {0: 0, 1: 1, 2: 2}
    └─ set_context(is_prefill=False, seq_ids=[0,1,2], ...)

    ↓ (CPU-side pre-update)
    For i=0: dict[0] → buffer[0]  ← Restore seq_0 state
    For i=1: dict[1] → buffer[1]  ← Restore seq_1 state
    For i=2: dict[2] → buffer[2]  ← Restore seq_2 state

    ↓ (GPU-side, GRAPH REPLAY)
    graph.replay()
    ├─ Token 0: read buffer[0], compute, write buffer[0]
    ├─ Token 1: read buffer[1], compute, write buffer[1]
    ├─ Token 2: read buffer[2], compute, write buffer[2]
    └─ All use FIXED GPU addresses

    ↓ (CPU-side post-update)
    For i=0: buffer[0] → dict[0]  ← Save seq_0 state
    For i=1: buffer[1] → dict[1]  ← Save seq_1 state
    For i=2: buffer[2] → dict[2]  ← Save seq_2 state

    ↓
    Output 3 tokens, add to sequences
```

### Sequence Finish

```
sequence_id=0 completes (max_len or stop token)

    ↓
    ModelRunner.run() calls _clear_linear_attn_states([0])

    ├─ For each GatedDeltaNet layer:
    │   ├─ buffer[0].zero_()  ← Clear slot
    │   └─ buffer_conv[0].zero_()
    │
    └─ _seq_id_to_state.pop(0)  ← Remove from dict

    Status: Slot 0 is now available for new sequences
```

---

## 9. Decision Tree: When to Use Each Approach

```
Does my layer have state that persists across tokens?
│
├─ NO → Use standard Attention approach
│   └─ Pre-allocate k_cache/v_cache (fixed addresses)
│   └─ ✓ Works with CUDA graphs by default
│
└─ YES → Need to choose:
    │
    ├─ Option A: Use dict-based state
    │   ├─ Pros: Simple, dynamic, minimal memory
    │   └─ Cons: ✗ Incompatible with CUDA graphs
    │           ✗ Requires enforce_eager=True
    │           ✗ 3× slower decode (~100 tok/s)
    │
    └─ Option B: Use pre-allocated buffers
        ├─ Pros: ✓ Compatible with CUDA graphs
        │        ✓ 3× faster decode (~300 tok/s)
        │        ✓ Fixed GPU addresses
        │
        └─ Cons: 1-10 GB memory per layer
                Complex seq_id → buffer_idx mapping
                Requires careful state management
```

