# Nano-vLLM Architecture Visual Guide

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        User Code                        │
│  llm = LLM(model_path)                                  │
│  outputs = llm.generate(prompts, sampling_params)      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              LLMEngine (llm_engine.py)                  │
│  ┌────────────────────────────────────────────────┐   │
│  │ add_request()    │ step()    │ generate()     │   │
│  │ ────────────────────────────────────────────  │   │
│  │ - Tokenize      │ - Schedule│ - Loop until   │   │
│  │ - Create Seq    │ - Execute │   finished     │   │
│  │ - Add to queue  │ - Process │ - Return texts │   │
│  └────────────────────────────────────────────────┘   │
└──────────┬─────────────────────────────────┬───────────┘
           │                                 │
           ▼                                 ▼
    ┌────────────────┐            ┌──────────────────────┐
    │  Scheduler     │            │  ModelRunner (GPU)   │
    │ (scheduler.py) │            │(model_runner.py)     │
    ├────────────────┤            ├──────────────────────┤
    │ ┌─Waiting──┐   │            │ - Model forward pass │
    │ │         │   │            │ - CUDA graphs        │
    │ ├─Running──┤   │            │ - Sampling           │
    │ │         │   │            │ - KV cache mgmt      │
    │ └─────────┘   │            │                      │
    │               │            │  Per-rank process:   │
    │ Two-phase:    │            │  rank 0: main        │
    │ - Prefill     │            │  rank 1+: loop       │
    │ - Decode      │            └──────────────────────┘
    └────────┬───────┘                     │
             │                             │
             ▼                             ▼
    ┌──────────────────┐      ┌────────────────────────┐
    │  BlockManager    │      │  Qwen3ForCausalLM     │
    │(block_manager.py)│      │  (models/qwen3.py)    │
    ├──────────────────┤      ├────────────────────────┤
    │ - Allocate       │      │ ┌──────────────────┐   │
    │ - Deallocate     │      │ │ Embedding        │   │
    │ - Prefix cache   │      │ ├──────────────────┤   │
    │ - Hash-based     │      │ │ [32 × Decoder]   │   │
    │   dedup          │      │ ├──────────────────┤   │
    │                  │      │ │ RMSNorm          │   │
    │ Manages:         │      │ ├──────────────────┤   │
    │ - Block[]        │      │ │ LM Head (vocab)  │   │
    │ - block_table    │      │ └──────────────────┘   │
    │ - free_blocks    │      │                        │
    │ - hash_to_block  │      │ ┌──────────────────┐   │
    └──────────────────┘      │ │ QKV Proj (fused) │   │
                              │ │ Gate-Up (fused)  │   │
                              │ │ Flash-Attention  │   │
                              │ │ RoPE             │   │
                              │ └──────────────────┘   │
                              └────────────────────────┘
                                      │
                                      ▼
                              ┌──────────────────┐
                              │  GPU Memory      │
                              ├──────────────────┤
                              │ Model weights    │
                              │ KV cache blocks  │
                              │ (prefix cached)  │
                              └──────────────────┘
```

---

## Request Processing Pipeline

```
Request 1: "Hello"     Request 2: "How are"   Request 3: "I am"
    │                      │                       │
    ▼                      ▼                       ▼
[ Tokenize ]          [ Tokenize ]           [ Tokenize ]
    │                      │                       │
    ▼                      ▼                       ▼
Seq 1: [7, 34, ...]   Seq 2: [42, 19, ...]  Seq 3: [9, 23, ...]
    │                      │                       │
    └──────────────────────┼───────────────────────┘
                           ▼
                   [ Scheduler waiting queue ]
                   [Seq1, Seq2, Seq3]
                           │
                           ▼
    ┌──────────────[ PREFILL PHASE ]──────────────┐
    │                                             │
    │  If batch has space:                       │
    │  1. Allocate KV cache blocks               │
    │  2. Concatenate all tokens                 │
    │  3. Run model on full prompt               │
    │  4. Move to running queue                  │
    │                                             │
    │  Input:  [7, 34, ... | 42, 19, ... | ...]  │
    │  Output: [logits1, logits2, logits3]      │
    │          ▼          ▼          ▼           │
    │  Sample: [tok1]    [tok1]    [tok1]        │
    │                                             │
    └──────────────────────┬──────────────────────┘
                           ▼
              [ Scheduler running queue ]
              [Seq1, Seq2, Seq3]
              num_tokens: [5, 4, 3]
                           │
                           ▼
    ┌──────────────[ DECODE PHASE ]───────────────┐
    │                                             │
    │  Per step (1 token at a time):             │
    │  1. Get last token from each seq           │
    │  2. Run model (fast, uses CUDA graph)      │
    │  3. Sample next token                      │
    │  4. Append to seq, update block_table      │
    │                                             │
    │  Input:  [tok1, tok1, tok1]  (last tokens) │
    │  Output: [logits1, logits2, logits3]      │
    │          ▼          ▼          ▼           │
    │  Sample: [tok2]    [tok2]    [tok2]        │
    │                                             │
    │  Check: EOS or max_tokens?                 │
    │  If yes: move to outputs                   │
    │  If no:  stay in running queue             │
    │                                             │
    │  Repeat until all sequences finish         │
    └──────────────────────┬──────────────────────┘
                           ▼
                     [ Return outputs ]
                Seq1: text + token_ids
                Seq2: text + token_ids
                Seq3: text + token_ids
```

---

## Qwen3 Transformer Layer

```
┌──────────────────────────────────────────────────────────────┐
│              Input: [batch, seq_len, hidden_size]           │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
                 [ Input LayerNorm ]
                   RMSNorm(x, eps)
                         ▼
    ┌────────────────────────────────────────────────┐
    │         MULTI-HEAD SELF-ATTENTION              │
    │                                                │
    │  Hidden ──┐                                   │
    │           ├─→ [QKVParallelLinear]            │
    │           │   (fused Q, K, V projection)      │
    │           └─→ Split into Q, K, V              │
    │                                                │
    │  Q ──→ [RMSNorm] ──→ [RoPE] ──┐              │
    │        (if no bias)            │              │
    │                                ├─→ [Attention]│
    │  K ──→ [RMSNorm] ──→ [RoPE] ──│              │
    │        (if no bias)            │              │
    │                                │              │
    │  V ──────────────────────────→ │              │
    │                                │              │
    │  Attention:                    │              │
    │  - Flash-Attention v2          │              │
    │  - Store K, V in cache         │              │
    │  - Attention weights           │              │
    │  - Output aggregation          │              │
    │        ↓                       │              │
    │  Output ←─────────────────────┘              │
    │        ↓                                      │
    │  [RowParallelLinear] (O_proj)                │
    │  (distributed across TP ranks)               │
    │        ↓                                      │
    │  Output: [batch, seq_len, hidden_size]      │
    └────────────────────────┬──────────────────────┘
                             ▼
                  [ Add Residual Connection ]
                         ▼
                 [ Post-Attention LayerNorm ]
                   RMSNorm(x, eps)
                         ▼
    ┌────────────────────────────────────────────────┐
    │         FEED-FORWARD NETWORK (MLP)             │
    │                                                │
    │  Hidden ──→ [MergedColumnParallel]           │
    │             Linear (fused Gate+Up)            │
    │             Output: [batch, seq, 2×ffn]      │
    │                         ▼                     │
    │                  [Split in half]              │
    │                  Gate | Up                    │
    │                    ▼   ▼                      │
    │              [SiLU ⊙ multiply]               │
    │              (compiled op)                    │
    │                    ▼                          │
    │            [RowParallel Down]                │
    │            Linear back to hidden_size         │
    │                    ▼                          │
    │  Output: [batch, seq_len, hidden_size]      │
    └────────────────────────┬──────────────────────┘
                             ▼
                  [ Add Residual Connection ]
                             ▼
        [ Output: [batch, seq_len, hidden_size] ]
        (passed to next layer or to RMSNorm+LMHead)
```

---

## KV Cache Block Management

### Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache Memory                          │
│  Shape: [2, num_layers, num_kvcache_blocks, block_size,    │
│          num_kv_heads, head_dim]                            │
└─────────────────────────────────────────────────────────────┘

Example with block_size=256:
┌─────────────────────────────────────────────────────────────┐
│     Block 0        Block 1        Block 2        ...        │
│  [256 tokens]  [256 tokens]   [256 tokens]                 │
│  K[layer0]     K[layer0]      K[layer0]                    │
│  V[layer0]     V[layer0]      V[layer0]                    │
│  K[layer1]     K[layer1]      K[layer1]                    │
│  V[layer1]     V[layer1]      V[layer1]                    │
│  ...           ...            ...                           │
└─────────────────────────────────────────────────────────────┘

Per sequence: block_table = [block_id1, block_id2, ...]
Example: Seq1 with 600 tokens → 3 blocks [0, 5, 12]
```

### Prefix Caching Example

```
Scenario: Two sequences with shared prefix

Seq 1: [token_A, token_B, token_C, token_D, token_E]
Seq 2: [token_A, token_B, token_F, token_G]

Naive approach:
  Block 0: [A, B, C]  → Seq1, Block 1: [D, E]    → Seq1
  Block 2: [A, B, F]  → Seq2, Block 3: [G]       → Seq2
  Memory: 4 blocks

With prefix caching:
  Step 1: Process Seq1 first
    Hash(A) = h0
    Hash(B|h0) = h1
    Hash(C|h1) = h2
    Hash(D|h2) = h3
    Hash(E|h3) = h4
    
    Block allocation:
    Block 0: [A, B, C] → hash h2 → hash_to_block_id[h2] = 0
    Block 1: [D, E]    → hash h4 → hash_to_block_id[h4] = 1
    
  Step 2: Process Seq2
    Hash(A) = h0 (same as before)
    Hash(B|h0) = h1 (same as before)
    
    Block lookup:
    Block 0: [A, B, ?] found in cache! hash h1 → block 0
    num_cached_tokens += 256 (reuse the whole block!)
    
    Block allocation for [F, G]:
    Hash(F|h1) = h_new (different from h2 because F ≠ C)
    Block 2: [F, G] → hash h_new → new allocation
    
    Result: Seq1.block_table = [0, 1]
            Seq2.block_table = [0, 2]  ← Reused block 0!
    
    Memory saved: 1 block (25% reduction)
    
    ┌─────────────────────────────────────┐
    │       Block 0  Block 1  Block 2    │
    │   ┌─[A,B,C]─┬─[D,E]─┬─[F,G]─┐    │
    │   │        │       │        │    │
    │ Seq1────────────────────────────│────
    │   │        │       │              │
    │ Seq2───────────────────┐         │
    │   └────────────────────┘         │
    │                                   │
    │ ref_count: [2, 1, 1]  ← block 0  │
    │                        shared!   │
    └─────────────────────────────────┘
```

---

## Tensor Parallelism

### Weight Sharding

```
Original (TP=1):
  Q_weight: [hidden, hidden]
  K_weight: [hidden, hidden]  
  V_weight: [hidden, hidden]
  ───────────────────────────
  Total: 3 × hidden²

TP=2 (Split across 2 GPUs):
  GPU 0:
    Q_weight: [hidden, hidden/2]
    K_weight: [hidden, hidden/2]
    V_weight: [hidden, hidden/2]
    
  GPU 1:
    Q_weight: [hidden, hidden/2]
    K_weight: [hidden, hidden/2]
    V_weight: [hidden, hidden/2]
    
  Computation:
    GPU 0: q0 = input @ Q0.T  (shape: [batch, seq, hidden/2])
    GPU 1: q1 = input @ Q1.T  (shape: [batch, seq, hidden/2])
    Concat: q = [q0 || q1]    (shape: [batch, seq, hidden])
```

### All-Reduce for Output Projection

```
After attention output (per-GPU computation):
  GPU 0: o0 = attention0_output @ O0.T  ([batch, seq, hidden/2])
  GPU 1: o1 = attention1_output @ O1.T  ([batch, seq, hidden/2])

All-reduce to combine:
  o = o0 + o1  (element-wise add across GPUs)
  Result: [batch, seq, hidden]
```

---

## CUDA Graph Capture & Replay

### What Gets Captured

```
┌──────────────────────────────────────────────────────────┐
│           CUDA Graph for Batch Size 8                    │
│                                                          │
│  ┌─ Kernel 0: Embedding lookup                          │
│  ├─ Kernel 1: Attention QKV projection                  │
│  ├─ Kernel 2: RoPE application                          │
│  ├─ Kernel 3: Flash-Attention                           │
│  ├─ Kernel 4: Output projection                         │
│  ├─ Kernel 5: MLP gate projection                       │
│  ├─ Kernel 6: SiLU + multiply                           │
│  ├─ Kernel 7: MLP down projection                       │
│  └─ Kernel 8: Sampling (on CPU)                         │
│                                                          │
│  Total GPU time: ~2-3ms (vs 5-10ms with Python overhead)│
└──────────────────────────────────────────────────────────┘

Captured for batch sizes:
  [1, 2, 4, 8, 16, 32, 48, 64, ...] up to max_num_seqs
  
  Benefit: Uses largest graph that fits batch size
  Example: batch_size=20 → use graph_bs[16]
```

### Replay Process

```
┌─────────────────────────┐
│  Input Tensors:         │
│  - input_ids: [bs, 1]   │
│  - positions: [bs, 1]   │
│  - slot_mapping: [bs]   │
│  - block_tables: [bs, K]│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Update Graph Variables                         │
│  graph_vars["input_ids"][:bs] = input_ids       │
│  graph_vars["positions"][:bs] = positions       │
│  graph_vars["slot_mapping"][:bs] = slot_mapping │
│  graph_vars["block_tables"][:bs] = block_tables │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  graph.replay()                                 │
│  - Pre-recorded GPU kernels execute             │
│  - No Python interpreter overhead               │
│  - Deterministic execution                      │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Extract Output                                 │
│  logits = graph_vars["outputs"][:bs]            │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Sampling (CPU)                                 │
│  token_ids = sampler(logits, temperatures)      │
└─────────────────────────────────────────────────┘
```

---

## Attention Mechanism Detail

### Prefill Phase (Variable Length Sequences)

```
Requests:
  Seq 1: [T1, T2, T3, T4, T5]         (5 tokens)
  Seq 2: [T6, T7, T8]                 (3 tokens)
  Seq 3: [T9, T10]                    (2 tokens)

Flatten for batch processing:
  input_ids = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]  (10 tokens)
  
CU_SEQLENS (cumulative sequence lengths):
  cu_seqlens_q = [0, 5, 8, 10]   ← Query endpoints
  cu_seqlens_k = [0, 5, 8, 10]   ← Key/Value endpoints
  
  Seq 1 queries: indices [0:5]
  Seq 2 queries: indices [5:8]
  Seq 3 queries: indices [8:10]

Flash Attention (variable length):
  flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=[0, 5, 8, 10],
    cu_seqlens_k=[0, 5, 8, 10],
    max_seqlen_q=5,
    max_seqlen_k=5,
    causal=True
  )
  
  Result: Attention output (10 tokens, reshaped to [3 seqs])
```

### Decode Phase (Fixed Single Token)

```
Running sequences after prefill:
  Seq 1: [T1, T2, T3, T4, T5, token_new1]     (6 tokens total)
  Seq 2: [T6, T7, T8, token_new1]             (4 tokens total)
  Seq 3: [T9, T10, token_new1]                (3 tokens total)

Extract last token of each:
  input_ids = [token_new1, token_new1, token_new1]  (batch=3)
  
Cache info:
  context_lens = [6, 4, 3]  ← How many tokens to attend to
  block_tables = [[0, 1], [0, 2], [0]]  ← Where cached KVs are
  
Flash Attention with KV cache:
  flash_attn_with_kvcache(
    q.unsqueeze(1),          # [batch, 1, hidden] (single token)
    k_cache, v_cache,        # Pre-computed and stored
    cache_seqlens=[6, 4, 3], # Attention lengths per seq
    block_table=[[0,1],[0,2],[0]],  # Physical block IDs
    causal=True
  )
  
  Result: Attention output (batch=3, 1 token each)
          Values used from cache (KV), only Q computed
```

---

## Memory Consumption Breakdown

```
Model: Qwen3-0.6B
Hardware: RTX 4070 (8 GB)

Memory breakdown:
┌─────────────────────────────────────────┐
│ Model Weights:            ~2.5 GB       │ (0.6B params × 4 bytes)
├─────────────────────────────────────────┤
│ Intermediate Activations: ~1.2 GB       │ (prefill batch)
├─────────────────────────────────────────┤
│ KV Cache:                 ~3.0 GB       │ (90% GPU memory util)
│  - num_kvcache_blocks: ~1500 blocks     │ (auto-computed)
│  - block_size: 256 tokens               │
│  - Can hold ~384k tokens total          │ (256 × 1500)
├─────────────────────────────────────────┤
│ Other (CUDA overhead):    ~0.3 GB       │
├─────────────────────────────────────────┤
│ Total:                    ~7.0 GB ✓     │ (90% utilization)
└─────────────────────────────────────────┘

Scaling with TP=2:
  Memory per GPU ≈ Model size / 2 + KV cache (mostly unchanged)
  Model weights per GPU: ~1.25 GB
  KV cache per GPU: ~3.0 GB
  Total per GPU: ~4.5 GB
```

---

## Data Structures

```python
# Main data structure relationships

Sequence (sequence.py)
  ├─ seq_id: int (unique ID)
  ├─ token_ids: list[int] (all tokens: prompt + generated)
  ├─ block_table: list[int] (physical block IDs for KV cache)
  ├─ num_cached_tokens: int (how many tokens cached)
  └─ status: SequenceStatus (WAITING | RUNNING | FINISHED)

BlockManager (block_manager.py)
  ├─ blocks: list[Block] (physical cache blocks)
  ├─ free_block_ids: deque[int] (available blocks)
  ├─ used_block_ids: set[int] (allocated blocks)
  └─ hash_to_block_id: dict[int, int] (prefix cache)

Block (block_manager.py)
  ├─ block_id: int
  ├─ ref_count: int (how many sequences use it)
  ├─ hash: int (for prefix caching)
  └─ token_ids: list[int] (tokens in this block)

Config (config.py)
  ├─ model: str (model directory path)
  ├─ max_num_batched_tokens: int
  ├─ max_num_seqs: int
  ├─ gpu_memory_utilization: float
  ├─ num_kvcache_blocks: int (auto-computed)
  └─ ... (9 more parameters)
```

---

## Execution Timeline

```
Timeline for 3 requests over multiple steps:

Step 0 (Prefill):
  ├─ Time: 0ms
  ├─ Input: [Seq1 full, Seq2 full, Seq3 full]
  ├─ GPU kernels: Embedding, Attention×32, MLP×32, LMHead
  ├─ Output: [token1, token2, token3]
  └─ Duration: ~50ms (all in parallel)

Step 1-5 (Decode):
  ├─ Time: 50ms
  ├─ Input: [last_token_seq1, last_token_seq2, last_token_seq3]
  ├─ GPU kernels: Attention×32, MLP×32, LMHead (cached embeddings)
  ├─ Output: [token4, token5, token6]
  ├─ Duration: ~6ms (CUDA graph replay)
  └─ Repeat...

Total Time for 256 total tokens:
  Prefill: 50ms (process 10 input tokens + first 3 output)
  Decode: ~80ms (process remaining ~250 tokens × 3 seqs ÷ throughput)
  ═══════════════════════════════════════════
  Total: ~130ms for ~250 output tokens
  
  Throughput: 250 tokens / 0.13s ≈ 1923 tok/s
```

---

**These diagrams show how Nano-vLLM orchestrates requests, manages memory, and executes computations efficiently!** 🚀
