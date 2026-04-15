# NanoVLLM - Detailed Dataflow & Architecture

## 1. COMPLETE REQUEST-TO-RESPONSE FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER CODE                                                                   │
│ llm.generate(["Hello", "World"], SamplingParams(max_tokens=10))            │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLMENGINE.GENERATE()                                                        │
│ ├─ Tokenize each prompt                                                    │
│ │  "Hello" → [1, 2, 3]                                                     │
│ │  "World" → [4, 5]                                                        │
│ │                                                                           │
│ ├─ Create Sequence objects                                                 │
│ │  Sequence(token_ids=[1,2,3], sampling_params=...)                        │
│ │  Sequence(token_ids=[4,5], sampling_params=...)                          │
│ │                                                                           │
│ └─ Add to Scheduler (waiting queue)                                        │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MAIN GENERATION LOOP: while not is_finished()                              │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────────────────────────────────────┐
              │                                                      │
              ▼                                                      │
      ┌─────────────────────────────────────────────────────────┐   │
      │ PREFILL PHASE (first time sequences enter)             │   │
      │                                                          │   │
      │ 1. SCHEDULER.SCHEDULE() → prefill=True                  │   │
      │    ├─ Load sequences from waiting queue                 │   │
      │    ├─ Check memory: can_allocate()?                     │   │
      │    └─ Return: [Seq1, Seq2], is_prefill=True             │   │
      │                                                          │   │
      │ 2. BLOCKMANAGER.ALLOCATE(seq)                           │   │
      │    ├─ For each block in sequence:                       │   │
      │    │  ├─ Hash block tokens                              │   │
      │    │  ├─ Check prefix cache (hash_to_block_id)          │   │
      │    │  ├─ If cache hit:                                  │   │
      │    │  │  └─ Reuse block (increment ref count)           │   │
      │    │  │     → seq.num_cached_tokens += block_size       │   │
      │    │  └─ If cache miss:                                 │   │
      │    │     └─ Allocate new block from free pool           │   │
      │    └─ Store block IDs in seq.block_table                │   │
      │                                                          │   │
      │ 3. MODELRUNNER.PREPARE_PREFILL(seqs)                    │   │
      │    ├─ Pack inputs for Flash Attention (varlen format)   │   │
      │    │                                                    │   │
      │    │  Sequence 1: [1, 2, 3] (3 tokens)                  │   │
      │    │  Sequence 2: [4, 5]    (2 tokens)                  │   │
      │    │                                                    │   │
      │    │  input_ids:   [1, 2, 3, 4, 5]                      │   │
      │    │  positions:   [0, 1, 2, 0, 1]                      │   │
      │    │  cu_seqlens_q: [0, 3, 5]                           │   │
      │    │  cu_seqlens_k: [0, 3, 5]                           │   │
      │    │  slot_mapping: [block_ids with offset]             │   │
      │    │                                                    │   │
      │    └─ set_context(is_prefill=True, ...)                 │   │
      │                                                          │   │
      │ 4. MODEL.FORWARD(input_ids, positions)                  │   │
      │    ├─ Embed: [1,2,3,4,5] → embeddings                   │   │
      │    └─ For each decoder layer:                           │   │
      │       ├─ QKVParallelLinear: embed → Q, K, V             │   │
      │       │  K, V shapes: [5 tokens, num_kv_heads, head_dim]│   │
      │       ├─ RoPE: Apply position embeddings to Q, K        │   │
      │       ├─ Attention (Flash):                             │   │
      │       │  ├─ store_kvcache: K, V → cache                 │   │
      │       │  │  (Write K,V to GPU cache at slot_mapping)    │   │
      │       │  └─ flash_attn_varlen: Q,K,V → output           │   │
      │       ├─ Output proj: reshape & all-reduce (if TP)      │   │
      │       ├─ MLP (gate*up through fused layers)             │   │
      │       └─ Residual connections + norm                    │   │
      │    └─ Output: [5, hidden_size]                          │   │
      │                                                          │   │
      │ 5. COMPUTE_LOGITS(hidden_states) [using ParallelLMHead] │   │
      │    ├─ Extract last token per sequence (via cu_seqlens_q)│   │
      │    │  Prefill: Take indices [2, 4] (last per seq)       │   │
      │    ├─ Linear projection → logits                        │   │
      │    └─ Gather from all TP ranks (if TP)                  │   │
      │    → [2, vocab_size]                                     │   │
      │                                                          │   │
      │ 6. SAMPLER(logits, temperatures)                        │   │
      │    ├─ Divide logits by temperature                      │   │
      │    ├─ Softmax → probabilities                           │   │
      │    ├─ Gumbel-max trick: sample                          │   │
      │    └─ token_ids = [100, 234]  (sampled next tokens)     │   │
      │                                                          │   │
      │ 7. SCHEDULER.POSTPROCESS(seqs, token_ids)               │   │
      │    └─ Append sampled tokens to each sequence            │   │
      │       Seq1: [1,2,3,100]  (num_tokens=4)                │   │
      │       Seq2: [4,5,234]    (num_tokens=3)                │   │
      │                                                          │   │
      │ 8. Check completion:                                    │   │
      │    ├─ num_completion_tokens == max_tokens?              │   │
      │    └─ sampled_token == eos_token?                       │   │
      │    → Not finished, move to decode                       │   │
      └─────────────┬───────────────────────────────────────────┘   │
                    │                                               │
                    ▼                                               │
      ┌─────────────────────────────────────────────────────────┐   │
      │ DECODE PHASE (subsequent steps)                        │   │
      │                                                          │   │
      │ 1. SCHEDULER.SCHEDULE() → prefill=False                 │   │
      │    ├─ Move sequences from waiting → running             │   │
      │    ├─ Check memory: can_append()?                       │   │
      │    │  (Do we have room for one more block?)             │   │
      │    ├─ If no: preempt low-priority seq                   │   │
      │    └─ Return running sequences                          │   │
      │                                                          │   │
      │ 2. BLOCKMANAGER.MAY_APPEND(seq)                         │   │
      │    ├─ If len(seq) % block_size == 1:                    │   │
      │    │  └─ Allocate new block (need space)                │   │
      │    └─ If len(seq) % block_size == 0:                    │   │
      │       └─ Finalize block with hash (for future cache)    │   │
      │                                                          │   │
      │ 3. MODELRUNNER.PREPARE_DECODE(seqs)                     │   │
      │    ├─ Extract last token per sequence                   │   │
      │    │  Seq1 last: 100                                    │   │
      │    │  Seq2 last: 234                                    │   │
      │    ├─ Get context length per sequence                   │   │
      │    │  Seq1: 4 (1+2+3+100)                              │   │
      │    │  Seq2: 3 (4+5+234)                                │   │
      │    ├─ Map to KV cache slots                             │   │
      │    │  (Where to write new K,V)                          │   │
      │    │                                                    │   │
      │    └─ set_context(is_prefill=False, ...)                │   │
      │                                                          │   │
      │ 4. MODEL.FORWARD([100, 234], [3, 2])                    │   │
      │    ├─ Embed: [100, 234] → embeddings [2, hidden]        │   │
      │    └─ For each decoder layer:                           │   │
      │       ├─ QKVParallelLinear: embed → Q, K, V             │   │
      │       │  Shapes: [2 tokens, num_kv_heads, head_dim]     │   │
      │       ├─ RoPE: Apply position embeddings                │   │
      │       ├─ Attention (Flash with KV cache):               │   │
      │       │  ├─ store_kvcache: New K,V → cache              │   │
      │       │  │  (Append at existing block locations)        │   │
      │       │  ├─ flash_attn_with_kvcache:                    │   │
      │       │  │  (Use cached K,V + new Q for attention)      │   │
      │       │  └─ Output: attended values                     │   │
      │       ├─ Output proj + MLP + residual                   │   │
      │       └─ Output: [2, hidden_size]                       │   │
      │                                                          │   │
      │ 5. COMPUTE_LOGITS: [2, hidden_size] → [2, vocab]        │   │
      │    └─ token_ids = [50, 60]  (next tokens)               │   │
      │                                                          │   │
      │ 6. SCHEDULER.POSTPROCESS()                              │   │
      │    └─ Append tokens                                     │   │
      │       Seq1: [1,2,3,100,50]  (5 tokens)                 │   │
      │       Seq2: [4,5,234,60]    (4 tokens)                 │   │
      │                                                          │   │
      │ 7. Check completion → repeat if not done                │   │
      │                                                          │   │
      │ *** SECOND DECODE STEP ***                              │   │
      │ (Process only sequences still running)                  │   │
      │                                                          │   │
      │ Seq1: last=50, len=5                                    │   │
      │ Seq2: last=60, len=4                                    │   │
      │ → Process with single forward, sample, append, check    │   │
      │                                                          │   │
      │ *** THIRD DECODE STEP ***                               │   │
      │ ...repeat until max_tokens or EOS                       │   │
      └─────────────┬───────────────────────────────────────────┘   │
                    │                                               │
                    └──────────────────────────────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ RETURN RESULTS                                             │
              │ ├─ Decode token IDs → text                                │
              │ ├─ Seq1: [1,2,3,100,50,...] → "Hello there..."            │
              │ └─ Return [{"text": "...", "token_ids": [...]}]            │
              └─────────────────────────────────────────────────────────────┘
```

## 2. QWEN3 MODEL LAYER ARCHITECTURE (Detailed)

```
INPUT: input_ids [batch, seq_len], positions [batch, seq_len]
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ QWEN3MODEL                                                      │
│                                                                 │
│ Step 1: EMBED_TOKENS (VocabParallelEmbedding)                  │
│  ├─ Input: [B, S] token IDs                                    │
│  ├─ Embedding table: [vocab_size, hidden_size]                 │
│  │  (or [vocab_size/TP, hidden_size] per GPU)                  │
│  └─ Output: [B*S, hidden_size] (flattened)                     │
│     [batch*seq embedding vectors]                              │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
        For each layer i=0..num_layers-1:
        ┌─────────────────────────────────────────────────────────┐
        │ QWEN3DECODERLAYER                                       │
        │                                                         │
        │ Step A: INPUT_LAYERNORM (RMSNorm)                       │
        │  ├─ Input: hidden [B*S, H]                              │
        │  ├─ RMS normalize each token embedding                  │
        │  └─ Output: normed [B*S, H]                             │
        │                                                         │
        │ Step B: SELF_ATTENTION (Qwen3Attention)                 │
        │  ├─────────────────────────────────────────────────────┤
        │  │ B1. QKVParallelLinear                                │
        │  │  ├─ Input: [B*S, H]                                  │
        │  │  ├─ Weights: [H, (Q_dim+2*KV_dim)] split across TP   │
        │  │  └─ Output: qkv [B*S, Q_dim+2*KV_dim]                │
        │  │                                                     │
        │  │ B2. Split & Reshape                                  │
        │  │  ├─ q [B*S, Q_dim] → [B*S, num_heads, head_dim]      │
        │  │  ├─ k [B*S, KV_dim] → [B*S, num_kv_heads, head_dim]  │
        │  │  └─ v [B*S, KV_dim] → [B*S, num_kv_heads, head_dim]  │
        │  │                                                     │
        │  │ B3. Optional Norm (if no qkv bias)                   │
        │  │  ├─ q_norm(q) → normalized queries                  │
        │  │  └─ k_norm(k) → normalized keys                     │
        │  │                                                     │
        │  │ B4. RoPE (Rotary Position Embeddings)                │
        │  │  ├─ Input: positions, q, k                           │
        │  │  ├─ Lookup cos/sin from pre-computed cache           │
        │  │  ├─ Apply rotation: (q,k) = rotate(q,k, cos, sin)    │
        │  │  └─ Output: rotated q, k                             │
        │  │                                                     │
        │  │ B5. Attention (using Flash Attention)                │
        │  │  ├─ store_kvcache():                                 │
        │  │  │  ├─ Take k [B*S, num_kv_heads, head_dim]          │
        │  │  │  ├─ Take v [B*S, num_kv_heads, head_dim]          │
        │  │  │  ├─ Get slot_mapping from context                 │
        │  │  │  └─ Write to k_cache & v_cache at slots           │
        │  │  │     (Triton kernel: concurrent writes)            │
        │  │  │                                                  │
        │  │  ├─ Flash Attention:                                 │
        │  │  │  ├─ Prefill mode:                                │
        │  │  │  │  ├─ If prefix cache: use cached k,v            │
        │  │  │  │  └─ flash_attn_varlen_func(q, k, v, ...)       │
        │  │  │  │     (Efficient attention over variable lengths) │
        │  │  │  └─ Decode mode:                                 │
        │  │  │     └─ flash_attn_with_kvcache(q, k_cache, ...)   │
        │  │  │        (Attend over cached KV + new Q)            │
        │  │  │                                                  │
        │  │  └─ Output: attn_out [B*S, Q_dim] or [B*S, 1, H]     │
        │  │                                                     │
        │  │ B6. Output Projection (RowParallelLinear)            │
        │  │  ├─ Input: attn_out [B*S, num_heads*head_dim]        │
        │  │  ├─ Each GPU handles output_size/TP columns          │
        │  │  ├─ Compute local linear: [B*S, H/TP]                │
        │  │  ├─ All-reduce across TP ranks to sum                │
        │  │  └─ Output: attn_output [B*S, H]                     │
        │  └─────────────────────────────────────────────────────┤
        │  Output: attention_out [B*S, H]                         │
        │                                                         │
        │ Step C: POST_ATTENTION_LAYERNORM (RMSNorm)              │
        │  ├─ Input: residual [B*S, H]                            │
        │  ├─ Add & Norm: (attn_out + hidden) → norm              │
        │  └─ Output: normed [B*S, H], residual updated           │
        │                                                         │
        │ Step D: MLP (Qwen3MLP)                                  │
        │  ├─────────────────────────────────────────────────────┤
        │  │ D1. MergedColumnParallelLinear (gate_up_proj)        │
        │  │  ├─ Input: [B*S, H]                                  │
        │  │  ├─ Two outputs: gate and up                         │
        │  │  ├─ Fused layer: combines gate & up projections      │
        │  │  └─ Output: [B*S, 2*intermediate_size]               │
        │  │                                                     │
        │  │ D2. SiluAndMul (Gating)                              │
        │  │  ├─ Split: gate [B*S, intermediate] & up [...]       │
        │  │  ├─ gate = SiLU(gate)                                │
        │  │  └─ Output: gate * up [B*S, intermediate]            │
        │  │                                                     │
        │  │ D3. RowParallelLinear (down_proj)                    │
        │  │  ├─ Input: [B*S, intermediate_size]                  │
        │  │  ├─ Project back to hidden_size                      │
        │  │  ├─ All-reduce to combine TP results                 │
        │  │  └─ Output: [B*S, H]                                 │
        │  └─────────────────────────────────────────────────────┤
        │  Output: mlp_out [B*S, H]                               │
        │                                                         │
        │ Step E: Residual Connection                             │
        │  ├─ hidden = mlp_out + attn_residual                    │
        │  └─ Return to next layer or output                      │
        └─────────────────────────────────────────────────────────┘
              │
              ▼
        ┌─────────────────────────────────────────────────────────┐
        │ FINAL_LAYERNORM (RMSNorm)                               │
        │ ├─ Input: [B*S, H]                                      │
        │ └─ Output: [B*S, H] (final hidden states)               │
        └─────────────┬───────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │ LM_HEAD (ParallelLMHead)                                │
        │                                                         │
        │ Step 1: Select tokens for logit computation             │
        │  ├─ Prefill: Take last token per sequence               │
        │  │  (via cu_seqlens_q indices)                          │
        │  │  e.g., [5, 2] if batch 1 has 5 tokens, batch 2 has 2 │
        │  └─ Decode: Take all tokens                             │
        │     (all B tokens need logits)                          │
        │                                                         │
        │ Step 2: Linear projection                               │
        │  ├─ Input: [num_tokens, H]                              │
        │  ├─ Weight: [vocab_size/TP, H] (per GPU)                │
        │  └─ Output: [num_tokens, vocab_size/TP]                 │
        │                                                         │
        │ Step 3: Gather from TP ranks                            │
        │  ├─ Each rank has partial vocab logits                  │
        │  ├─ Gather to rank 0: [num_tokens, vocab_size/TP] →    │
        │  │ [num_tokens, vocab_size]                             │
        │  └─ Rank 0 proceeds to sampling                         │
        └─────────────────────────────────────────────────────────┘
```

## 3. KV CACHE MEMORY LAYOUT

```
┌─ KV Cache Tensor Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
│                          │  │           │         │          │             │
│                          │  │           │         │          │             └─ Head embedding dimension (e.g., 128)
│                          │  │           │         │          └─ Number of KV heads (e.g., 8)
│                          │  │           │         └─ Tokens per block (e.g., 256)
│                          │  │           └─ Total blocks allocated (e.g., 1000)
│                          │  └─ Number of transformer layers (e.g., 24)
│                          └─ 0=K cache, 1=V cache
│
├─ Example dimensions:
│  ├─ num_layers = 24
│  ├─ num_blocks = 1000
│  ├─ block_size = 256
│  ├─ num_kv_heads = 8
│  ├─ head_dim = 128
│  └─ Total memory = 2 * 24 * 1000 * 256 * 8 * 128 * 2 bytes = ~39 GB (float32)
│                                                              = ~20 GB (float16)
│
└─ Access during prefill:
   For token position i in block b:
   ├─ k_cache[b, block_size+i] = new_k_value  (write via slot_mapping)
   └─ v_cache[b, block_size+i] = new_v_value

   During decode:
   ├─ Read: k_cache[block_ids, :] (all tokens for this sequence)
   ├─ Read: v_cache[block_ids, :] (all tokens for this sequence)
   └─ Attention: Q @ K^T → softmax → V (using only cached KV)
```

## 4. SCHEDULER STATE MACHINE

```
Request arrives
    │
    ▼
Add to Scheduler.waiting queue
    │
    ▼
     ┌──────────────────────────────────────────┐
     │ PREFILL PHASE (first time)               │
     │                                          │
     │ Schedule.schedule() called                │
     │ ├─ Waiting queue has sequence            │
     │ ├─ BlockManager.can_allocate()? → YES    │
     │ ├─ Allocate KV cache blocks              │
     │ ├─ seq.status = RUNNING                  │
     │ └─ Move to running queue                 │
     │                                          │
     │ ModelRunner executes                     │
     │ ├─ Forward pass (all prompt tokens)      │
     │ ├─ Sample first completion token         │
     │ └─ Append to token_ids                   │
     └──────────────┬──────────────────────────┘
                    │
                    ▼
             ┌──────────────────────┐
             │ Is finished?         │
             │ ├─ token == EOS?     │
             │ └─ max_tokens hit?   │
             └──────┬──────┬────────┘
                    │ NO   │ YES
                    │      │
                    ▼      ▼
              ┌──────────┐ ┌──────────────────┐
              │ DECODE   │ │ FINISHED         │
              │ PHASE    │ │ ├─ Remove from   │
              │ ├─ seq in│ │ │ running queue  │
              │ │ running│ │ ├─ Deallocate    │
              │ │ queue  │ │ │ KV cache       │
              │ ├─ Sched.│ │ └─ Return output │
              │ │ returns│ │                  │
              │ │ for    │ └──────────────────┘
              │ │ decode │
              │ ├─ One   │
              │ │ forward│
              │ │ pass   │
              │ ├─ Sample│
              │ │ next   │
              │ │ token  │
              │ └─ Repeat│
              └──────┬───┘
                     │
                     └─────────┐
                               │
                     ┌─────────▼──────┐
                     │ Check finished │
                     └─────────┬──────┘
                               │
                               ├──→ (back to top of loop)
```

## 5. TENSOR PARALLELISM WEIGHT DISTRIBUTION

```
For TP_SIZE = 2 (2 GPUs):

┌─────────────────────────────────────────────────────────────┐
│ QWEN3ATTENTION (per layer)                                  │
│                                                             │
│ Rank 0 (GPU 0):                                            │
│ ├─ QKVParallelLinear:                                      │
│ │  ├─ Weight shape: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim / 2]
│ │  │  (50% of output queries, 50% of keys, 50% of values) │
│ │  └─ Output: [seq_len, output_size/2]                    │
│ │     {Q:50%, K:50%, V:50%}                               │
│ │                                                         │
│ ├─ After attention:                                        │
│ │  └─ All-reduce needed (sum across ranks)                │
│ │                                                         │
│ ├─ RowParallelLinear (output proj):                        │
│ │  ├─ Input split: [seq_len, hidden_size/2]              │
│ │  ├─ Weight shape: [hidden_size/2, hidden_size]          │
│ │  └─ Local output: [seq_len, hidden_size]                │
│ │     (only handles its part of output_size)              │
│ │                                                         │
│ └─ All-reduce to combine rank outputs                     │
│                                                             │
│ Rank 1 (GPU 1):                                            │
│ ├─ QKVParallelLinear:                                      │
│ │  ├─ Weight shape: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim / 2]
│ │  │  (50% of output queries, 50% of keys, 50% of values) │
│ │  └─ Output: [seq_len, output_size/2]                    │
│ │     {Q:50%, K:50%, V:50%}                               │
│ │                                                         │
│ └─ Same communication pattern                              │
│                                                             │
│ *** ATTENTION OUTPUT PHASE ***                             │
│ All ranks have full Q,K,V after all-reduce                │
│ ├─ Attention: Q @ K^T (independent, no communication)     │
│ └─ All ranks produce output (which is then all-reduced)   │
│                                                             │
│ NCCL OPERATIONS PER LAYER:                                │
│ ├─ After QKV projection: all-reduce (combine K,V slices) │
│ ├─ After attention: all-reduce (combine output slices)   │
│ ├─ During output proj: all-reduce (combine results)      │
│ └─ After MLP: all-reduce (combine outputs)               │
└─────────────────────────────────────────────────────────────┘
```

## 6. CUDA GRAPH CAPTURE & REPLAY

```
┌─────────────────────────────────────────────────────────────┐
│ CUDA GRAPH CAPTURE (initialization)                         │
│                                                             │
│ For each batch size in [1, 2, 4, 8, 16, 32, ...]:          │
│                                                             │
│ 1. Create empty CUDA graph                                  │
│    graph = torch.cuda.CUDAGraph()                           │
│                                                             │
│ 2. Warmup run (outside graph):                              │
│    outputs[:bs] = model(input_ids[:bs], positions[:bs])     │
│    └─ Populate GPU caches, warmup kernels                   │
│                                                             │
│ 3. Begin graph capture:                                     │
│    with torch.cuda.graph(graph, pool):                      │
│        outputs[:bs] = model(input_ids[:bs], positions[:bs]) │
│        └─ Record all GPU operations to graph                │
│                                                             │
│ 4. Store graph and create shared memory pool:               │
│    self.graphs[bs] = graph                                  │
│    self.graph_pool = graph.pool()                           │
│    └─ Reuse pool for subsequent captures                    │
│                                                             │
│ Repeat for all batch sizes...                               │
│                                                             │
│ Store input tensors for graph replay:                       │
│ graph_vars = {                                              │
│     "input_ids": torch.zeros(max_bs, ...),                 │
│     "positions": torch.zeros(max_bs, ...),                 │
│     "slot_mapping": torch.zeros(max_bs, ...),              │
│     "context_lens": torch.zeros(max_bs, ...),              │
│     "block_tables": torch.zeros(max_bs, max_blocks, ...),   │
│     "outputs": torch.zeros(max_bs, hidden_size, ...),      │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CUDA GRAPH REPLAY (during inference)                        │
│                                                             │
│ Each decode step:                                           │
│                                                             │
│ 1. Get batch size (num_sequences)                           │
│    bs = len(seqs)                                           │
│                                                             │
│ 2. Find appropriate graph:                                  │
│    graph = graphs[min(bs in [1,2,4,8,16,...] >= bs)]       │
│    ├─ If bs=5: use graphs[8]                              │
│    ├─ If bs=100: use graphs[512]                           │
│    └─ Batch gets padded to graph size                      │
│                                                             │
│ 3. Prepare inputs:                                          │
│    graph_vars["input_ids"][:bs] = input_ids                │
│    graph_vars["positions"][:bs] = positions                │
│    graph_vars["slot_mapping"][:bs] = slot_mapping          │
│    graph_vars["context_lens"][:bs] = context_lens          │
│    graph_vars["block_tables"][:bs, :] = block_tables       │
│                                                             │
│ 4. Replay graph:                                            │
│    graph.replay()                                           │
│    └─ All GPU kernels replay (no CPU overhead!)            │
│                                                             │
│ 5. Extract outputs:                                         │
│    logits = graph_vars["outputs"][:bs]                     │
│    └─ Use only first bs rows                               │
│                                                             │
│ *** SPEEDUP MECHANISM ***                                  │
│ Normal (no graph):                                          │
│ ├─ Copy input to GPU (CPU↔GPU)                             │
│ ├─ Launch kernels (kernel overhead)                        │
│ ├─ CPU waits for GPU                                       │
│ └─ Copy output from GPU                                    │
│                                                             │
│ With graph replay:                                          │
│ ├─ Inputs already in GPU memory                            │
│ ├─ All kernels replay instantly (pre-optimized)            │
│ └─ No CPU-GPU synchronization                              │
└─────────────────────────────────────────────────────────────┘
```

## 7. PREFIX CACHING EXAMPLE

```
┌─────────────────────────────────────────────────────────────┐
│ SCENARIO: Two requests with shared prefix                   │
│                                                             │
│ Request 1: "What is machine learning? Explain in detail."   │
│ Request 2: "What is deep learning? Explain in detail."      │
│                                                             │
│ Tokens after encoding:                                      │
│ Req1: [101, 2054, 2003, 3698, 4083, 1029, 18443, ...]      │
│ Req2: [101, 2054, 2003, 6576, 4083, 1029, 18443, ...]      │
│        └─ Same start: "What is" + "Explain"                 │
│            But different middle: "machine learning" vs      │
│            "deep learning"                                  │
│                                                             │
│ Block size: 256 tokens                                      │
│                                                             │
│ *** PREFILL PHASE ***                                       │
│                                                             │
│ Process Req1:                                               │
│ ├─ Block 0 tokens: [101, 2054, 2003, ..., 3698] (first 256) │
│ │  ├─ Hash: hash(block0_tokens) = 0x123456                 │
│ │  ├─ BlockManager: No match in hash_to_block_id            │
│ │  └─ Allocate new block: block_id=0                        │
│ │     └─ hash_to_block_id[0x123456] = 0                    │
│ │                                                           │
│ ├─ Block 1 tokens: [4083, 1029, 18443, ..., ...] (next 256) │
│ │  ├─ Hash: hash(block1_tokens, prefix=0x123456) = 0xABC   │
│ │  ├─ BlockManager: No match                                │
│ │  └─ Allocate new block: block_id=1                        │
│ │     └─ hash_to_block_id[0xABC] = 1                       │
│ │                                                           │
│ └─ Req1.block_table = [0, 1]  (2 blocks)                    │
│                                                             │
│ *** PREFILL PHASE (Req2) ***                                │
│                                                             │
│ Process Req2:                                               │
│ ├─ Block 0 tokens: [101, 2054, 2003, ..., 6576] (first 256) │
│ │  ├─ Hash: hash(block0_tokens) = 0xABC123  (Different!)    │
│ │  │  (Different because 3698 ≠ 6576)                      │
│ │  ├─ BlockManager: No match in hash_to_block_id            │
│ │  └─ Allocate new block: block_id=2                        │
│ │     └─ hash_to_block_id[0xABC123] = 2                    │
│ │                                                           │
│ ├─ Block 1 tokens: [4083, 1029, 18443, ..., ...] (next 256) │
│ │  ├─ Hash: hash(block1_tokens, prefix=0xABC123) = 0xDEF   │
│ │  │  (Different prefix!)                                  │
│ │  ├─ BlockManager: No match                                │
│ │  └─ Allocate new block: block_id=3                        │
│ │                                                           │
│ └─ Req2.block_table = [2, 3]  (2 blocks, no reuse)         │
│                                                             │
│ *** BETTER SCENARIO: Shared prefix after tokenization ***   │
│                                                             │
│ Request 1: "What is the capital of France?"                 │
│ Request 2: "What is the capital of Germany?"                │
│                                                             │
│ Req1: [101, 2054, 2003, 1996, 3007, 1997, 2289]            │
│ Req2: [101, 2054, 2003, 1996, 3007, 1997, 2088]            │
│        └─ Shared: "What is the capital of"                  │
│ ├─ Block 0: [101, 2054, 2003, 1996, 3007, 1997, ...]       │
│ │  ├─ Req1 prefill: hash=0x111, allocate block_id=0        │
│ │  │  hash_to_block_id[0x111] = 0                          │
│ │  │  Req1.num_cached_tokens += 256                         │
│ │  │                                                        │
│ │  ├─ Req2 prefill: hash=0x111 (MATCH!)                    │
│ │  │  ├─ Hash matches, token values match                  │
│ │  │  ├─ REUSE block_id=0                                  │
│ │  │  ├─ blocks[0].ref_count += 1 (now = 2)                │
│ │  │  └─ Req2.num_cached_tokens += 256                     │
│ │  │     (No new KV computation for this block!)            │
│ │  │                                                        │
│ │  └─ *** SAVINGS: Half the KV cache computation ***        │
│ │                                                           │
│ └─ Block 1: [2289] vs [2088] (only last token differs)      │
│    └─ Different hash → allocate separately                  │
│                                                             │
│ *** MEMORY SAVINGS ***                                      │
│                                                             │
│ Without prefix caching:                                     │
│ ├─ Req1: 2 blocks × 256 tokens/block = 512 tokens cached    │
│ ├─ Req2: 2 blocks × 256 tokens/block = 512 tokens cached    │
│ └─ Total cached: 1024 tokens                                │
│                                                             │
│ With prefix caching:                                        │
│ ├─ Req1: 2 blocks = 512 tokens cached                       │
│ ├─ Req2: 1 block (reused) + 1 block (new) = 256 new cached  │
│ └─ Total cached: 768 tokens (25% savings!)                  │
│                                                             │
│ *** COMPUTE SAVINGS ***                                     │
│ Prefill involves:                                           │
│ ├─ Embedding lookup                                         │
│ ├─ Attention computation                                    │
│ └─ MLP computation                                          │
│                                                             │
│ With prefix cache, Req2's first block:                      │
│ ├─ Skip embedding (already have embeddings)                 │
│ ├─ Skip attention (already have KV cache values)            │
│ ├─ Skip MLP (already have outputs)                          │
│ └─ *** Prefill speedup: ~2x for cached portion ***          │
└─────────────────────────────────────────────────────────────┘
```

## 8. MULTI-GPU COMMUNICATION FLOW

```
┌─────────────────────────────────────────────────────────────┐
│ TENSOR PARALLELISM: 4 GPUs (Rank 0, 1, 2, 3)                │
│                                                             │
│ INITIALIZATION:                                             │
│                                                             │
│ Main Process                                                │
│ ├─ Create Config(tensor_parallel_size=4)                   │
│ ├─ Spawn 3 worker processes (ranks 1, 2, 3)                │
│ ├─ Initialize ModelRunner rank 0                           │
│ │  └─ Calls: dist.init_process_group("nccl", ...)          │
│ │     (Blocks until all ranks join)                         │
│ └─ Workers also call: dist.init_process_group(...)         │
│    (Now all ranks are connected via NCCL)                   │
│                                                             │
│ *** INFERENCE STEP (each rank executes independently) ***   │
│                                                             │
│ Input: input_ids, positions                                 │
│                                                             │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│ │ Rank 0   │  │ Rank 1   │  │ Rank 2   │  │ Rank 3   │    │
│ │ (GPU 0)  │  │ (GPU 1)  │  │ (GPU 2)  │  │ (GPU 3)  │    │
│ └──────┬───┘  └──────┬───┘  └──────┬───┘  └──────┬───┘    │
│        │             │             │             │         │
│        └─────────────┴─────────────┴─────────────┘         │
│               All wait at barrier                           │
│                                                             │
│ ├─ Embedding (each rank handles vocab_size/4):             │
│ │  ├─ Rank 0: embed tokens if token_id in [0, vocab/4)    │
│ │  ├─ Rank 1: embed tokens if token_id in [vocab/4, vocab/2)
│ │  ├─ Rank 2: embed tokens if token_id in [vocab/2, 3*vocab/4)
│ │  ├─ Rank 3: embed tokens if token_id in [3*vocab/4, vocab)
│ │  └─ All-reduce: each rank gets full embeddings          │
│ │     (all non-matching ranks return 0, sum brings in valid)
│ │                                                         │
│ ├─ For each layer:                                         │
│ │  ├─ QKVParallelLinear (ColumnParallel):                 │
│ │  │  ├─ Each rank computes Q,K,V for its output slice    │
│ │  │  └─ NO communication (independent computation)        │
│ │  │                                                      │
│ │  ├─ RoPE (Rotary embeddings):                           │
│ │  │  └─ Local (no communication)                         │
│ │  │                                                      │
│ │  ├─ Attention (local):                                  │
│ │  │  ├─ Rank 0: Q @ K^T (local Q and full K/V)           │
│ │  │  │  └─ Wait, K is split across ranks!                │
│ │  │  │     Need all-gather to get full K/V first         │
│ │  │  └─ All-gather K,V from all ranks:                   │
│ │  │     ├─ Rank 0: gather K slices from all ranks        │
│ │  │     ├─ Combine into full K tensor                    │
│ │  │     ├─ same for V                                    │
│ │  │     └─ Now can compute attention (full Q @ full K)   │
│ │  │                                                      │
│ │  ├─ RowParallelLinear (output projection):              │
│ │  │  ├─ Input split across ranks (each rank has input/4) │
│ │  │  ├─ Each rank: local linear [B*S, H/4] → [B*S, H]   │
│ │  │  ├─ All-reduce to sum contributions:                 │
│ │  │  │  ├─ Rank 0: Y0 (partial output)                   │
│ │  │  │  ├─ Rank 1: Y1 (partial output)                   │
│ │  │  │  ├─ All-reduce: Y = Y0 + Y1 + Y2 + Y3            │
│ │  │  │  └─ All ranks now have full output                │
│ │  │  └─ NO communication (all-reduce handles it)         │
│ │  │                                                      │
│ │  ├─ MLP:                                                │
│ │  │  ├─ gate_up (ColumnParallel): local, no comm         │
│ │  │  ├─ down (RowParallel): requires all-reduce          │
│ │  │  └─ (same pattern as output projection)              │
│ │  │                                                      │
│ │  └─ Barrier: all ranks sync                             │
│ │     (wait for slowest to finish)                        │
│ │                                                         │
│ ├─ LM Head (ParallelLMHead):                              │
│ │  ├─ Each rank: compute partial logits [num_tokens, vocab/4]
│ │  ├─ Only rank 0 gathers:                               │
│ │  │  ├─ gather(rank1_logits, rank2_logits, rank3_logits) │
│ │  │  ├─ Concatenate: [num_tokens, vocab]                │
│ │  │  └─ Rank 0 now has full logits                      │
│ │  └─ Other ranks discard logits (None)                   │
│ │                                                         │
│ ├─ Sampling (only rank 0):                                │
│ │  └─ Rank 0 samples tokens, all ranks receive via        │
│ │     dist.broadcast()                                    │
│ │                                                         │
│ └─ Barrier at end of iteration                            │
│    (sync before next step)                                │
│                                                             │
│ *** COMMUNICATION PATTERN ***                              │
│ Per layer:                                                  │
│ ├─ 0 collectives (ColumnParallel layers)                   │
│ ├─ 1 all-reduce (RowParallel output)                       │
│ ├─ 1 all-reduce (RowParallel MLP)                          │
│ ├─ ... other all-reduces ...                               │
│ └─ Total: ~4-6 all-reduce operations per layer             │
│                                                             │
│ *** LATENCY BREAKDOWN (example) ***                         │
│ ├─ Computation: 100ms                                       │
│ ├─ NCCL communication: 20ms per all-reduce × 4 = 80ms       │
│ ├─ Synchronization barriers: 5ms                           │
│ └─ Total: ~185ms (5-10x slower than single GPU)            │
│                                                             │
│ *** THROUGHPUT (tokens/sec) ***                            │
│ ├─ Single GPU: 1000 tok/s                                  │
│ ├─ 4 GPUs (TP): 3000-3500 tok/s (not 4000 due to comm)     │
│ └─ Efficiency: ~80-90% (good for TP)                       │
└─────────────────────────────────────────────────────────────┘
```

