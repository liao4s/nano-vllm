# Nano-vLLM Architecture Diagrams & Flows

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Application                              │
│                    llm.generate(prompts, params)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LLMEngine (Main)      │
                    │  ┌──────────────────┐   │
                    │  │ Tokenizer        │   │
                    │  │ Scheduler        │   │
                    │  │ Config           │   │
                    │  └──────────────────┘   │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼──────────┐    ┌────────▼─────────┐     ┌───────▼────────┐
   │ Scheduler     │    │ ModelRunner      │     │  BlockManager  │
   │               │    │ (Rank 0 - Main)  │     │ (KV Cache)     │
   │ • waiting Q   │    │ • Model          │     │ • Block table  │
   │ • running Q   │    │ • Sampler        │     │ • Prefix cache │
   │ • preempt     │    │ • CUDA graphs    │     │ • Allocation   │
   └───────────────┘    │ • GPU exec       │     └────────────────┘
                        │ • NCCL Dist.     │
                        └────────┬─────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼────────┐ ┌─────▼──────┐ ┌──────▼──────┐
         │ ModelRunner   │ │ ModelRunner│ │ModelRunner  │
         │ (Rank 1)      │ │ (Rank 2)   │ │ (Rank 3)    │
         │ • Watches SHM │ │• Watches SHM│ │• Watches SHM│
         └──────┬────────┘ └─────┬──────┘ └──────┬──────┘
                │                │                │
                └────────────────┼────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   GPU Devices   │
                        │                 │
                        │ NCCL Collective │
                        │ Operations      │
                        └─────────────────┘
```

---

## 2. Request Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION                                                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Tokenize prompt → token_ids
        ├─ Create Sequence object
        │  └─ seq_id, status=WAITING, num_prompt_tokens, max_tokens
        │
        └─ Scheduler.add(seq) → waiting queue

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: PREFILL (Process new requests)                                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Scheduler.schedule()
        │  └─ While waiting_seqs available:
        │     ├─ Check memory (BlockManager.can_allocate)
        │     ├─ Allocate KV cache with prefix caching
        │     │  └─ Hash-based deduplication
        │     ├─ seq.status = RUNNING
        │     └─ Add to running queue
        │
        ├─ ModelRunner.run(seqs, is_prefill=True)
        │  ├─ prepare_prefill(): Format batch
        │  │  └─ cu_seqlens_q/k, slot_mapping, block_tables
        │  ├─ run_model(): Forward pass
        │  │  └─ Hidden states → Logits
        │  └─ Sampler(): Sample tokens
        │
        ├─ Scheduler.postprocess(): Update sequences
        │  └─ seq.append_token(new_token)
        │     Check: num_completion_tokens == max_tokens?
        │            token_id == EOS?
        │     → If yes, seq.status = FINISHED
        │
        └─ (Return to scheduling if more work)

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: DECODE (Auto-regressive generation)                            │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Scheduler.schedule() [Decode Phase]
        │  └─ For each running_seq:
        │     ├─ Check if room for 1 more token
        │     ├─ If full: Preempt (move to waiting)
        │     ├─ If ok: BlockManager.may_append(seq)
        │     │  └─ Allocate new block if crossing 256-token boundary
        │     └─ Add to scheduled batch
        │
        ├─ ModelRunner.run(seqs, is_prefill=False)
        │  ├─ prepare_decode(): Format batch (1 token per seq)
        │  │  └─ input_ids, positions, slot_mapping, block_tables
        │  ├─ run_model(): Forward pass
        │  │  ├─ Check if bs <= max_bs in CUDA graphs
        │  │  ├─ If yes: Replay graph (GPU-only, ~10x faster)
        │  │  └─ If no: Run eagerly
        │  ├─ Sampler(): Sample token
        │  └─ Store KV to cache (Triton kernel)
        │
        ├─ Scheduler.postprocess(): Update sequences
        │  └─ seq.append_token(new_token)
        │
        └─ Repeat until all sequences finished

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: COMPLETION                                                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ Collect all finished sequences
        ├─ Decode token_ids → text
        ├─ Return [{"text": ..., "token_ids": ...}, ...]
        │
        └─ (Optional) Deallocate KV cache on exit
```

---

## 3. Scheduling Algorithm (Detailed)

```
┌──────────────────────────────────────────────────────────┐
│  scheduler.schedule() → (seqs, is_prefill)               │
└──────────────────────────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │ Any       │
                    │ running?  │
                    └────┬─────┘
                    ┌────┴────┐
                   NO        YES
                    │         │
    ┌───────────────┘         └──────────────────────┐
    │                                                │
    ▼ PREFILL PHASE                    DECODE PHASE ▼
    │                                                │
    ├─ Check waiting_seqs                ├─ Check running_seqs
    │  for i in waiting while:           │  for i in running:
    │    ├─ num_seqs < max              │    ├─ if can_append?
    │    ├─ num_batched_tokens < max    │    │  ├─ YES: Schedule
    │    ├─ can_allocate(seq)?          │    │  │  └─ Add to batch
    │    │  ├─ YES:                      │    │  └─ NO: Preempt
    │    │  │  ├─ Allocate KV cache     │    │     ├─ deallocate()
    │    │  │  ├─ status = RUNNING      │    │     ├─ status = WAITING
    │    │  │  ├─ add to running        │    │     └─ append to waiting
    │    │  │  └─ add to scheduled      │    │
    │    │  └─ NO: Break                │    ├─ may_append(seq)
    │    │                               │    │  ├─ If at boundary:
    │    │                               │    │  │  └─ allocate new block
    │    │                               │    │  └─ Update hash table
    │    │                               │    │
    │    └─ Return (scheduled, True)     │    └─ Return (scheduled, False)
    │                                    │
    └────────────────────┬───────────────┘
                         │
                    ┌────▼─────────────┐
                    │ No sequences?    │
                    │ assert False     │
                    └──────────────────┘
```

---

## 4. KV Cache Memory Layout

```
┌────────────────────────────────────────────────────────────────┐
│               GPU Memory (90% allocated to KV cache)           │
└────────────────────────────────────────────────────────────────┘

Shape: [2, num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim]
       [K/V, Layers,   Blocks,         256 tokens, Heads,        Dim]

Example: [2, 24, 2048, 256, 8, 64] for model with:
  - 24 layers
  - 8 KV heads (with TP=1)
  - 64-dim heads
  - ~2048 blocks total
  - Qwen3 config

┌──────────────────────────────────────────────────────────────┐
│ ALLOCATION BY SEQUENCE (Prefix Caching)                      │
└──────────────────────────────────────────────────────────────┘

Sequence A: [token_0, ..., token_255, token_256, ..., token_511]
            └─────── Block 0 ────────────────────────────────┘
                                      └─ Block 1 ───────┘
            block_table: [0, 1]
            num_cached_blocks: 1  (after prefill, Block 0 is "cached")

Sequence B: [token_0, ..., token_255, token_256, ...]
            └────── Block 0 (REUSED!) ────────────┘
                                    └─ Block 2 (new)
            block_table: [0, 2]
            num_cached_blocks: 1  (prefix hits Block 0!)

Hash Table:
  hash_of_[token_0...token_255] → Block 0 (ref_count = 2)

┌──────────────────────────────────────────────────────────────┐
│ SLOT MAPPING (Where to write KV in cache)                    │
└──────────────────────────────────────────────────────────────┘

Flat indexing: slot = block_id * 256 + token_offset

Seq A, token 100: slot_mapping[seq_a_pos] = 0 * 256 + 100 = 100
Seq B, token 100: slot_mapping[seq_b_pos] = 0 * 256 + 100 = 100  (same!)
                  → Both write to same KV cache location
                  → Multi-request sharing via reference counting

Triton Kernel (store_kvcache):
  for idx in parallel_for(num_tokens):
    slot = slot_mapping[idx]
    if slot != -1:
      k_cache[slot] = key[idx]
      v_cache[slot] = value[idx]
```

---

## 5. Tensor Parallelism Execution

```
┌────────────────────────────────────────────────────────────────┐
│ MULTI-PROCESS EXECUTION (tensor_parallel_size=4)               │
└────────────────────────────────────────────────────────────────┘

Main Process (Rank 0)          Worker Processes (Rank 1-3)
│                              │          │          │
├─ Prepare batch               │          │          │
│  (input_ids, positions)      │          │          │
│                              │          │          │
├─ write_shm(method, *args)    │          │          │
│  ┌─ Pickle data              │          │          │
│  ├─ Write to SHM buffer      │          │          │
│  └─ Set event for each rank  │          │          │
│                              │          │          │
│                              ├─────────►│          │
│                              │          ├─────────►│
│                              │          │          │
│                    ┌─────────▼─────┬────▼──────┬──▼──────┐
│                    │ read_shm()    │read_shm() │read_shm()│
│                    │ Unpickle      │Unpickle   │ Unpickle │
│                    └────┬──────────┴────┬──────┴─────┬────┘
│                         │               │           │
│                    ┌────▼──────────────▼───────────▼───┐
│                    │ call(method, *args)               │
│                    │                                   │
│                    │ model.run(seqs, is_prefill)      │
│                    │ ├─ Local forward pass            │
│                    │ ├─ nccl.all_reduce() [sync]      │
│                    │ ├─ Local backward (if needed)    │
│                    │ └─ Return logits                 │
│                    │                                   │
│                    │ NCCL Collective Operations:      │
│                    │ • all_reduce() - aggregate       │
│                    │ • barrier() - synchronize        │
│                    │ • gather() - collect to rank 0   │
│                    └────┬──────────────┬────────────┬──┘
│                         │               │            │
│                    ┌────▼──────────────▼──────────▼──┐
│                    │ Synchronize at barrier           │
│                    └────┬──────────────┬────────────┬──┘
│                         │               │            │
│ ◄────────────────────────┴───────────────┴────────────┘
│
├─ Sampler() [rank 0 only]
├─ Postprocess tokens
│
└─ Continue to next step
```

**Tensor Parallelism Patterns:**

```
Model Layer Structure (tp_size=2):

┌─────────────────────────────────────────────────┐
│ Input: (batch, hidden_size)                     │
│ Example: (32, 768)                              │
└────┬────────────────────────────────────────────┘
     │
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
     ▼ Rank 0              ▼ Rank 1              │ (local compute)
┌──────────────┐      ┌──────────────┐          │
│ QKVProj      │      │ QKVProj      │          │
│ input: 768   │      │ input: 768   │          │
│ out: 96      │      │ out: 96      │          │
└──────┬───────┘      └──────┬───────┘          │
       │ Q: (32,32)         │ Q: (32,32)        │
       │ K: (32,16)         │ K: (32,16)        │
       │ V: (32,16)         │ V: (32,16)        │
       ▼                     ▼                   │
  ┌─────────┐          ┌─────────┐              │ (no sync needed)
  │ RoPE    │          │ RoPE    │              │
  │ Attn    │          │ Attn    │              │
  └────┬────┘          └────┬────┘              │
       │ (32, 32)           │ (32, 32)         │
       ▼                     ▼                   │
  ┌──────────────┐      ┌──────────────┐       │ (all_reduce here!)
  │ Output Proj  │      │ Output Proj  │       │
  │ in: 64       │      │ in: 64       │       │
  │ out: 384     │      │ out: 384     │       │
  └──────┬───────┘      └──────┬───────┘       │
         │ (32, 384)           │ (32, 384)     │
         └────────────┬────────┘               │
                      ▼                         │
            ┌──────────────────┐               │
            │ all_reduce()     │ ← Aggregate  │
            │ Result: (32, 768)│   across TP  │
            └────────┬─────────┘               │
                     ▼                         │
              Output: (32, 768)
```

---

## 6. Attention with Prefix Caching & Block Tables

```
┌────────────────────────────────────────────────────────────┐
│ PREFILL: Variable-length sequences with prefix cache       │
└────────────────────────────────────────────────────────────┘

Batch Input:
  Seq0: [cached_part] [new_part]  (100 cached + 100 new)
  Seq1: [new_part]                (200 new)
  Seq2: [cached_part] [new_part]  (50 cached + 150 new)

cu_seqlens_q: [0, 100, 300, 450]      (query lengths: new tokens only)
cu_seqlens_k: [0, 100, 300, 450]      (key lengths: total tokens)
              └─ Seq0 ─┘└── Seq1 ──┘└──── Seq2 ────┘

Query positions:
  Seq0: [0, 1, ..., 99]       (new tokens start at pos 0)
  Seq1: [0, 1, ..., 199]
  Seq2: [0, 1, ..., 149]

KV positions:
  Seq0: [100, 101, ..., 199]  (cached at 0-99, new at 100-199)
  Seq1: [0, 1, ..., 199]
  Seq2: [50, 51, ..., 199]    (cached at 0-49, new at 50-199)

Block Tables:
  Seq0: [[0, 1], [0, 1], ...]  (maps seq indices to block IDs)
  Seq1: [[2, 3], [2, 3], ...]
  Seq2: [[0, 4], [0, 4], ...]

Flash Attention varlen:
  ├─ Take queries at cu_seqlens_q
  ├─ Use sparse K/V via block_tables (don't compute for cached part!)
  └─ Return attention output for new tokens


┌────────────────────────────────────────────────────────────┐
│ DECODE: Single token, variable context lengths             │
└────────────────────────────────────────────────────────────┘

Batch Input (one token per sequence):
  Seq0: [last_token]   (context_len = 200)
  Seq1: [last_token]   (context_len = 200)
  Seq2: [last_token]   (context_len = 200)

input_ids: [token_a, token_b, token_c]
positions:  [199, 199, 199]       (last token position)
context_lens: [200, 200, 200]      (full context length for each)

slot_mapping: [block_id*256 + offset for each seq]
  [0 * 256 + 199,  (Seq0: last token slot in block 0)
   2 * 256 + 199,  (Seq1: last token slot in block 2)
   4 * 256 + 199]  (Seq2: last token slot in block 4)

Block Tables:
  [[0, 1], [2, 3], [0, 4]]  (multi-block context for each seq)

Flash Attention with KV cache:
  ├─ Store new (K, V) to cache at slot_mapping
  ├─ Lookup existing K/V via block_tables (efficient!)
  └─ Single-token attention only
```

---

## 7. CUDA Graph Capture Flow

```
┌──────────────────────────────────────────────────────────┐
│ INITIALIZATION: Capture graphs for common batch sizes    │
└──────────────────────────────────────────────────────────┘

for bs in [1, 2, 4, 8, 16, 32, 48, ..., max_bs]:
    │
    ├─ Allocate dummy tensors
    │  ├─ input_ids: (bs, )
    │  ├─ positions: (bs, )
    │  ├─ slot_mapping: (bs, )
    │  └─ block_tables: (bs, num_blocks)
    │
    ├─ Create CUDA graph
    │  ├─ graph = torch.cuda.CUDAGraph()
    │  │
    │  ├─ Forward pass (warmup)
    │  │  model(input_ids, positions)
    │  │
    │  ├─ torch.cuda.graph(graph) context:
    │  │  ├─ Reset device
    │  │  └─ Forward pass (capture)
    │  │     • All kernels recorded
    │  │     • No CPU overhead
    │  │
    │  └─ graph.pool() ← Memory pool for all graphs
    │
    └─ Store graph[bs]

┌──────────────────────────────────────────────────────────┐
│ RUNTIME: Replay graph for decode step                    │
└──────────────────────────────────────────────────────────┘

decode step:
    │
    ├─ Determine batch size (e.g., 24)
    │  └─ Find closest captured size: max(bs for bs in [1,2,4,...] if bs >= 24)
    │  └─ graph_bs = 32
    │
    ├─ Update graph variables (in-place)
    │  ├─ graph_vars["input_ids"][:24] = new_token_ids
    │  ├─ graph_vars["positions"][:24] = current_positions
    │  ├─ graph_vars["slot_mapping"][:24] = cache_slots
    │  └─ graph_vars["context_lens"][:24] = context_lengths
    │
    ├─ graph.replay()  ← Execute pre-recorded kernels
    │  • Pure GPU execution
    │  • No CPU <-> GPU communication
    │  • ~10x faster than eager
    │
    └─ Extract outputs[:24]

┌──────────────────────────────────────────────────────────┐
│ MEMORY EFFICIENCY                                        │
└──────────────────────────────────────────────────────────┘

Single pool for all graphs:
    [1]:   ◄─ Smallest graph
    [2]:   ◄─ Uses pool
    [4]:   ◄─ Uses pool
    ...
    [512]: ◄─ Largest graph

Advantages:
  • Shared memory pool
  • Faster graph execution
  • Reduced memory fragmentation
```

---

## 8. Model Forward Pass (Qwen3)

```
INPUT: (batch_size, seq_length) int64 tokens
POSITIONS: (batch_size, seq_length) int64 positions

↓
┌────────────────────────────────────┐
│ VocabParallelEmbedding             │
│  • Split vocab across TP ranks    │
│  • Embed tokens → hidden_size      │
└────────────┬───────────────────────┘
             ↓
       hidden_states: (B*L, hidden_size)

↓ [24 Layers × Qwen3DecoderLayer]
┌────────────────────────────────────┐
│ 1. RMSNorm(input_layernorm)         │
│    hidden → normalized_hidden       │
├────────────────────────────────────┤
│ 2. Qwen3Attention                  │
│    ├─ QKVParallelLinear (TP)       │
│    │  hidden → [Q, K, V]           │
│    ├─ RotaryEmbedding               │
│    │  [Q, K] + positions → RoPE    │
│    ├─ Attention (FlashAttn)        │
│    │  Q, K, V → output             │
│    │  └─ Store K/V to cache        │
│    │  └─ Use block_tables for sparse│
│    └─ RowParallelLinear (TP)       │
│       output → hidden_size          │
├────────────────────────────────────┤
│ 3. Residual add (fused in RMSNorm)  │
│    hidden + attn_output            │
├────────────────────────────────────┤
│ 4. RMSNorm(post_attention_layernorm)│
│    hidden → normalized_hidden       │
├────────────────────────────────────┤
│ 5. Qwen3MLP                        │
│    ├─ MergedColumnParallelLinear    │
│    │  hidden → [gate, value]       │
│    ├─ SiluAndMul                    │
│    │  [gate, value] → gated_value  │
│    └─ RowParallelLinear             │
│       gated_value → hidden_size     │
├────────────────────────────────────┤
│ 6. Residual add (fused in next)     │
│    hidden + mlp_output              │
└────────────┬───────────────────────┘
             ↓ (repeat x24)
             ...

↓
┌────────────────────────────────────┐
│ RMSNorm(final)                      │
└────────────┬───────────────────────┘
             ↓
       hidden_states: (batch, seq_len, hidden_size)

↓
┌────────────────────────────────────┐
│ ParallelLMHead                      │
│  • Extract last token per sequence  │
│  • Project to vocab size (TP)      │
│  • Gather logits to rank 0          │
└────────────┬───────────────────────┘
             ↓
       logits: (batch, vocab_size)

↓
┌────────────────────────────────────┐
│ Sampler (rank 0 only)               │
│  • Temperature scaling               │
│  • Softmax                          │
│  • Gumbel-max sampling              │
└────────────┬───────────────────────┘
             ↓
       token_ids: (batch,) int64
```

---

## 9. Memory Allocation & Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│ INITIALIZATION                                           │
└──────────────────────────────────────────────────────────┘

GPU Memory:
  Total: 8GB (RTX 4070)
  │
  ├─ Model weights: ~1-2 GB
  │
  ├─ KV Cache (90% of remaining):
  │  ├─ Shape: [2, 24 layers, blocks, 256, heads, head_dim]
  │  ├─ Example: ~5 GB for 2048 blocks
  │  │
  │  └─ Pre-allocated at startup
  │     └─ Never reallocated (efficient)
  │
  ├─ Activations: ~1-2 GB (varies with batch size)
  │
  └─ Workspace (CUDA, cuBLAS, etc.): ~0.5 GB

┌──────────────────────────────────────────────────────────┐
│ RUNTIME: Request Scheduling                             │
└──────────────────────────────────────────────────────────┘

Prefill Phase:
  ┌─ New request arrives
  ├─ BlockManager.can_allocate(seq)?
  │  └─ Check: free_blocks >= num_blocks_needed
  ├─ If YES:
  │  ├─ Allocate blocks with prefix caching
  │  ├─ seq.block_table ← [block_id, block_id, ...]
  │  └─ seq.status = RUNNING
  └─ If NO:
     └─ seq stays in waiting queue (preempt if needed)

Decode Phase:
  ┌─ For each running request:
  ├─ BlockManager.can_append(seq)?
  │  └─ Check: Is there a free block for next token?
  ├─ If YES:
  │  └─ BlockManager.may_append(seq)
  │     ├─ Allocate new block if crossing 256-token boundary
  │     └─ Update hash table for prefix cache
  └─ If NO:
     └─ Preempt (deallocate blocks, back to waiting)

Deallocation:
  ├─ When seq.status = FINISHED:
  │  └─ BlockManager.deallocate(seq)
  │     ├─ Decrement ref_count for each block
  │     ├─ If ref_count == 0:
  │     │  ├─ Remove from hash table
  │     │  ├─ Mark block as free
  │     │  └─ Add to free_block_ids
  │     └─ seq.block_table.clear()
  │
  └─ Exit: Deallocate all blocks, free GPU memory
```

---

## 10. Data Pipeline (End-to-End)

```
┌───────────────────────────────────────────────────────────┐
│ USER INPUT                                                │
│ prompts = ["Hello world", "Tell me a joke"]               │
│ sampling_params = SamplingParams(max_tokens=256)          │
└────────────┬──────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ TOKENIZATION (LLMEngine.__init__)                         │
│ • Load HF tokenizer                                       │
│ • Encode prompts → token_ids                              │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ prompt_0: [5, 10, 23, ...]
             ├─ prompt_1: [67, 2, 45, ...]
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ REQUEST QUEUEING (LLMEngine.generate)                     │
│ • Create Sequence objects                                 │
│ • Add to scheduler.waiting queue                          │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ Sequence(token_ids=[5,10,23,...], sp=params_0)
             ├─ Sequence(token_ids=[67,2,45,...], sp=params_1)
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 1: PREFILL (LLMEngine.step)                          │
│ • Scheduler allocates KV cache (with prefix caching)      │
│ • ModelRunner processes full prompts in batch             │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ Batch: [token_0...100 token_ids]
             ├─ Forward: Model([5,10,23,...,67,2,45,...])
             ├─ Output: logits (2 sequences × vocab_size)
             ├─ Sample: [next_token_0, next_token_1]
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 2-257: DECODE (Repeated)                             │
│ • Single token generation per request                     │
│ • Uses CUDA graphs for speed                              │
│ • Stores K/V to cache (Triton kernel)                     │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ Batch: [last_token_0, last_token_1]
             ├─ Forward: Model([token_a, token_b])
             ├─ Output: logits (2 sequences × vocab_size)
             ├─ Sample: [next_token_a, next_token_b]
             │
             ├─ Check: seq_0.num_completion_tokens == 256?
             │ └─ YES: seq_0.status = FINISHED
             │ └─ NO: continue
             │
             └─ Check: seq_1.num_completion_tokens == 256?
                └─ YES: seq_1.status = FINISHED
                └─ NO: continue

             ... repeat until both finished ...

             ▼
┌───────────────────────────────────────────────────────────┐
│ FINALIZATION (LLMEngine.generate)                         │
│ • Collect completed sequences                             │
│ • Decode token_ids → text strings                         │
│ • Return results                                          │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ outputs = {
             │   0: [5, 10, 23, ..., 256 completion tokens],
             │   1: [67, 2, 45, ..., 256 completion tokens]
             │ }
             │
             ├─ decoded = tokenizer.decode(token_ids)
             │   0: "Hello world... [256 tokens generated]"
             │   1: "Tell me a joke... [256 tokens generated]"
             │
             └─ return [
                   {"text": "Hello world...", "token_ids": [...]},
                   {"text": "Tell me a joke...", "token_ids": [...]}
                ]
```

