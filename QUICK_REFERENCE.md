# Nano-vLLM Quick Reference Guide

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER CODE                                │
│                    llm.generate(prompts)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLMEngine (llm_engine.py)                  │
│  • Tokenization   • Request Management  • Main Loop             │
│  • Coordinates TP ranks via IPC         • Returns Results       │
└────────────────┬──────────────────┬──────────────────┬──────────┘
                 │                  │                  │
        ┌────────▼─────────┐  ┌────▼──────────┐  ┌────▼──────────┐
        │  Tokenizer       │  │   Scheduler   │  │  ModelRunner  │
        │  (AutoTokenizer) │  │ (scheduler.py)│  │  (rank 0)     │
        └──────────────────┘  └──┬───────┬────┘  └────┬──────────┘
                                 │       │           │
                    ┌────────────┘       │           │
                    │                    │           │
        ┌───────────▼──────────────┐ ┌──▼─────┐ ┌──▼──────────┐
        │  Scheduler internals:    │ │Sequence│ │ModelRunner  │
        │  • prefill/decode phase  │ │States  │ │ • Init Model│
        │  • preemption logic      │ │        │ │ • CUDA Graphs
        │  • BlockManager (KV)     │ └────────┘ │ • Batch prep│
        │  • Linear Attn slots     │            │ • FORWARD   │
        └─────────────────────────┘            └─────────────┘
```

## Request Lifecycle

```
                    ┌─────────────────────────────┐
                    │  User calls add_request()   │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Tokenize + Create Seq      │
                    │  allocate_linear_attn_slot()│ ← IMPORTANT!
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Scheduler: WAITING queue   │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                                                     │
    ┌───▼───────────────────┐                   ┌────────────▼────────┐
    │  step() → PREFILL     │                   │  step() → DECODE    │
    │  (First iteration)    │                   │  (Loop until end)   │
    │                       │                   │                     │
    │  prepare_prefill()    │                   │  prepare_decode()   │
    │  ├─ input_ids         │                   │  ├─ input_ids       │
    │  ├─ cu_seqlens_q/k    │                   │  ├─ context_lens    │
    │  ├─ slot_mapping      │                   │  ├─ slot_mapping    │
    │  └─ block_tables      │                   │  └─ block_tables    │
    │                       │                   │                     │
    │  run_model(eager)     │                   │  run_model(graph?)  │
    │  │ (no CUDA graphs)   │                   │  │ (CUDA graph if   │
    │  │                    │                   │  │  batch_size ok)  │
    │  └─ model.forward()   │                   │  └─ graph.replay()  │
    │     ├─ Embedding      │                   │                     │
    │     ├─ Layers         │                   │  sampler()          │
    │     │  ├─ Attention   │                   │                     │
    │     │  ├─ Attn KV→cache                   │  scheduler.        │
    │     │  ├─ LinearAttn→slot               │  postprocess()      │
    │     │  └─ MoE         │                   │  ├─ append_token()  │
    │     └─ LMHead         │                   │  ├─ check EOS/len   │
    │                       │                   │  ├─ free_slot()  ◄──┼─ Finished!
    │  sampler()            │                   │  └─ deallocate()    │
    │                       │                   │                     │
    │  scheduler.postprocess│                   │  Back to WAITING    │
    │  └─ append_token()    │                   │  (preemption)       │
    │     RUNNING           │                   │                     │
    └───────────────────────┘                   └─────────────────────┘
```

## Memory Management

```
Total GPU Memory = total_memory * gpu_memory_utilization

├─ Linear Attention Budget (pre-computed)
│  └─ 32 slots × (num_layers × state_size) ≈ 1GB (Qwen3.5-35B)
│
├─ CUDA Graph Reserve (pre-computed)
│  └─ Max batch × peak_activation + 2MB/layer
│
├─ KV Cache (main consumer)
│  └─ [2, num_layers, num_blocks, 256, kv_heads, head_dim]
│     num_blocks = available_budget // block_size_bytes
│
└─ Model Weights (fixed)
```

## CUDA Graph Capture & Replay

### Capture (During Init)
```
for bs in [1, 2, 4, 8, 16, 32, 48, ...]:
    Create graph
    Set context (slots, block_tables)
    Warmup: model(input[:bs])  # Outside graph
    Capture: with torch.cuda.graph():
                 model(input[:bs])  # Inside graph, recorded
    Store: graphs[bs] = graph
```

### Replay (During Decode)
```
batch_size = actual_batch
graph = graphs[next(x for x in graph_bs if x >= batch_size)]
graph_vars["input_ids"][:batch_size] = actual_input_ids
graph_vars["linear_attn_slot_indices"][:batch_size] = slot_indices
graph.replay()  # Entire forward pass replayed deterministically
output = graph_vars["outputs"][:batch_size]
```

## Model Type Selection

```
config.model_type (from config.json)
         │
    ┌────┴────┐
    │          │
"qwen3_5_moe"  "qwen3" (default)
    │          │
    ▼          ▼
Qwen3_5    Qwen3
ForCausalLM ForCausalLM

Key Differences:
┌────────────────────────────────────────────────────────┐
│              │ Qwen3        │ Qwen3.5-MoE               │
├──────────────┼──────────────┼──────────────────────────┤
│ Attention    │ Standard     │ Hybrid (linear + full)   │
│              │ (every layer)│ (3 linear + 1 full)      │
├──────────────┼──────────────┼──────────────────────────┤
│ MLP          │ Standard     │ MoE (top-2) + shared     │
├──────────────┼──────────────┼──────────────────────────┤
│ RMSNorm      │ Standard (1*w)  │ (1+w) style (w init 0) │
│              │ (w ≈ 1.0)       │                        │
├──────────────┼──────────────┼──────────────────────────┤
│ State cache  │ KV only      │ KV + recurrent + conv    │
├──────────────┼──────────────┼──────────────────────────┤
│ CUDA Graphs  │ Works easily │ Slot indices as variable │
└────────────────────────────────────────────────────────┘
```

## Key Files Reference

### Core Engine
- **llm_engine.py**: Main loop, request management
- **model_runner.py**: Model initialization, forward pass, CUDA graphs
- **scheduler.py**: Prefill/decode scheduling, preemption
- **block_manager.py**: KV cache allocation, prefix caching
- **sequence.py**: Per-sequence state

### Layers
- **attention.py**: Flash-attention wrapper, KV cache storage
- **linear.py**: TP-aware linear layers (ColumnParallel, RowParallel, etc.)
- **rotary_embedding.py**: RoPE with partial rotation support
- **layers/sampler.py**: Gumbel-max sampling

### Models
- **models/qwen3.py**: Standard Qwen3 (transformer)
- **models/qwen3_5.py**: Qwen3.5-MoE (hybrid attention + MoE)

### Utilities
- **context.py**: ThreadLocal batch context
- **loader.py**: SafeTensors weight loading

## Critical Sequences

### Adding a Request (With Linear Attention)
```python
llm.add_request(prompt, sampling_params)
  ├─ tokenize(prompt) → token_ids
  ├─ Sequence(token_ids) → seq
  ├─ model_runner.call("allocate_linear_attn_slot", seq.seq_id)  ◄─ KEY!
  │  └─ Pops slot from free_slots, zeroes state buffers
  └─ scheduler.add(seq)  # WAITING
```

### Finishing a Sequence (With Linear Attention)
```python
scheduler.postprocess(seqs, token_ids)
  for seq in seqs:
    if seq.is_finished:
      ├─ model_runner.call("free_linear_attn_slot", seq.seq_id)  ◄─ KEY!
      │  └─ Returns slot to free_slots
      ├─ block_manager.deallocate(seq)  # KV cache
      └─ Remove from running
```

### Prefill Forward
```
input_ids: [token_ids_1, token_ids_2, ...]  (flattened)
positions: [absolute positions]
cu_seqlens_q: [0, len1, len1+len2, ...]  (only new tokens)
cu_seqlens_k: [0, len1_total, len1_total+len2_total, ...]

Flash-attn receives:
  q, k, v (for NEW tokens)
  cu_seqlens_q/k (variable lengths)
  block_tables (if prefix cached)
  → Efficient ragged batching

Model updates:
  kv_cache[block_id][slot] ← k,v for new tokens
  linear_attn_state[slot_idx] ← recurrent state at end of seq
```

### Decode Forward
```
input_ids: [last_token_1, last_token_2, ..., last_token_N]
positions: [seq_len_1, seq_len_2, ..., seq_len_N]
context_lens: [seq_len_1, seq_len_2, ..., seq_len_N]
slot_mapping: [kv_cache_slot_1, kv_cache_slot_2, ...]
block_tables: [block_ids for each seq]
linear_attn_slots: [state_buffer_slot_1, slot_2, ..., slot_N]

Via CUDA Graph (if captured):
  graph_vars["input_ids"][:N] = input_ids
  graph_vars["linear_attn_slot_indices"][:N] = linear_attn_slots
  graph.replay()
    └─ Model processes all N tokens in single captured kernel
       Each seq reads/writes its slot in state buffer

Via Eager (if graph not captured):
  model.forward(input_ids, positions)
    └─ Same computation, individual kernel launches
```

## Performance Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gpu_memory_utilization` | 0.9 | % of GPU RAM to allocate to KV cache |
| `max_num_seqs` | 512 | Max sequences in batch (limits max_linear_attn_slots) |
| `max_num_batched_tokens` | 16384 | Max total tokens in prefill batch |
| `kvcache_block_size` | 256 | Must be multiple of 256; larger = fewer blocks but less fragmentation |
| `tensor_parallel_size` | 1 | TP degree (1 = single GPU) |
| `enforce_eager` | False | Disable CUDA graphs if True (debug mode) |

## Common Issues & Debug

### "Not enough KV cache blocks"
- Reduce `max_num_seqs`
- Increase `gpu_memory_utilization`
- Reduce `max_model_len`

### Linear attention slot exhaustion
- Reduce `max_num_seqs`
- Reduce concurrent sequences via batch size limiting

### CUDA graph capture fails
- Set `enforce_eager=True` to skip graphs
- Check batch size < 512
- Ensure decode phase (not prefill)

### Weight loading warnings
- Expected: "skipped weight 'mtp.'" etc.
- These are intentional (skip vision modules)
