# Nano-vLLM Complete Architecture Documentation

This directory contains comprehensive documentation of the Nano-vLLM architecture. Choose the document that best fits your needs:

## 📋 Documentation Files

### 1. **ARCHITECTURE.md** (26 KB) - Complete Technical Reference
**Use this for:** Deep dives into implementation details

**Contents:**
- Project summary & key features
- High-level architecture overview
- Detailed component documentation (12 sections):
  - LLMEngine & configuration
  - Request/Sequence management
  - Scheduling & block management
  - Model execution & context
  - Qwen3 model implementation
  - Layer implementations (attention, linear, normalization, etc.)
- Data flow example (end-to-end execution)
- Optimization techniques (7 categories)
- Key design decisions
- File structure summary
- Performance characteristics
- Usage examples
- Limitations & future work

**Good for:** Building systems, understanding trade-offs, reviewing code

---

### 2. **ARCHITECTURE_SUMMARY.md** (6 KB) - Quick Reference Guide
**Use this for:** Getting oriented quickly, refresher lookups

**Contents:**
- Component overview table (11 rows)
- High-level execution flow
- Key data structures (Sequence, Block, Context)
- Scheduling algorithms (pseudocode)
- Prefix caching algorithm
- Tensor parallelism patterns
- Optimization techniques (6 main techniques)
- Memory layout examples
- Qwen3 model architecture
- Configuration parameters
- Files by category
- Performance insights vs vLLM
- Common workflows
- Key insights (6 takeaways)

**Good for:** Team communication, design discussions, onboarding

---

### 3. **ARCHITECTURE_DIAGRAMS.md** (38 KB) - Visual Architecture Guide
**Use this for:** Understanding system components & data flow

**Contents:**
- 10 detailed ASCII diagrams:
  1. System architecture overview
  2. Request lifecycle flow (4 phases)
  3. Scheduling algorithm (detailed)
  4. KV cache memory layout
  5. Tensor parallelism execution
  6. Attention with prefix caching & block tables
  7. CUDA graph capture flow
  8. Model forward pass (Qwen3)
  9. Memory allocation & lifecycle
  10. Data pipeline (end-to-end)

**Good for:** Visual learners, presentations, documentation

---

## 🎯 Quick Start by Use Case

### "I need to understand the entire system"
1. Start with **ARCHITECTURE_SUMMARY.md** (5 min overview)
2. Review **ARCHITECTURE_DIAGRAMS.md** sections 1-2 (system structure)
3. Read **ARCHITECTURE.md** sections 1-3 (core components)

### "I want to modify the scheduler"
1. **ARCHITECTURE_SUMMARY.md** - Scheduling algorithms section
2. **ARCHITECTURE_DIAGRAMS.md** - Diagram 3 (Scheduling algorithm)
3. **ARCHITECTURE.md** - Section 2.2 (Scheduler detailed)
4. Review actual code: `nanovllm/engine/scheduler.py`

### "I need to optimize KV cache"
1. **ARCHITECTURE_SUMMARY.md** - Prefix caching algorithm
2. **ARCHITECTURE_DIAGRAMS.md** - Diagram 4 (Memory layout)
3. **ARCHITECTURE.md** - Section 2.3 (Block Manager)
4. Review actual code: `nanovllm/engine/block_manager.py`

### "I want to add tensor parallelism"
1. **ARCHITECTURE_SUMMARY.md** - Tensor parallelism patterns
2. **ARCHITECTURE_DIAGRAMS.md** - Diagram 5 (TP execution)
3. **ARCHITECTURE.md** - Section 3.1 (ModelRunner)
4. Review actual code: `nanovllm/engine/model_runner.py`, `nanovllm/layers/linear.py`

### "I need to support a new model"
1. **ARCHITECTURE.md** - Section 4 (Qwen3 Model)
2. **ARCHITECTURE_DIAGRAMS.md** - Diagram 8 (Forward pass)
3. Review actual code: `nanovllm/models/qwen3.py`
4. Create similar structure for your model

### "I'm doing code review"
1. Skim **ARCHITECTURE_SUMMARY.md** (component table)
2. Reference **ARCHITECTURE.md** sections for specific components
3. Use **ARCHITECTURE_DIAGRAMS.md** for data flow verification

---

## 📊 Architecture at a Glance

```
CORE FLOW:
  User Code → LLMEngine.generate()
           → Scheduler.schedule()
           → ModelRunner.run()
           → Sampler()
           → repeat until finished
           → Return results

KEY COMPONENTS:
  • LLMEngine: Main orchestrator
  • Scheduler: Request scheduling with preemption
  • ModelRunner: GPU execution + tensor parallelism
  • BlockManager: KV cache management + prefix caching
  • Sequence: Request state tracking
  • Qwen3Model: Transformer architecture

OPTIMIZATIONS:
  ✓ Prefix caching (hash-based token sharing)
  ✓ Block-based KV cache (flexible, efficient)
  ✓ CUDA graphs (10x faster decode)
  ✓ Flash attention (IO-optimized)
  ✓ Tensor parallelism (multi-GPU)
  ✓ Preemption (handle overflow)

PERFORMANCE:
  • 1,434 tokens/s (vs vLLM's 1,362 tok/s)
  • 1.053x speedup on RTX 4070
  • ~1,200 lines of code
```

---

## 🔍 File Cross-Reference

### By Component

**Engine Layer:**
- `engine/llm_engine.py` - Main orchestrator → See ARCHITECTURE.md 1.1
- `engine/model_runner.py` - GPU execution → See ARCHITECTURE.md 3.1
- `engine/scheduler.py` - Scheduling → See ARCHITECTURE.md 2.2
- `engine/sequence.py` - Request state → See ARCHITECTURE.md 2.1
- `engine/block_manager.py` - KV cache → See ARCHITECTURE.md 2.3

**Layers:**
- `layers/attention.py` - Attention with KV → See ARCHITECTURE.md 5.1
- `layers/linear.py` - Tensor parallel → See ARCHITECTURE.md 5.2
- `layers/layernorm.py` - RMSNorm → See ARCHITECTURE.md 5.3
- `layers/rotary_embedding.py` - RoPE → See ARCHITECTURE.md 5.4
- `layers/activation.py` - SiLU/gating → See ARCHITECTURE.md 5.5
- `layers/embed_head.py` - Embeddings → See ARCHITECTURE.md 5.6
- `layers/sampler.py` - Sampling → See ARCHITECTURE.md 5.7

**Model:**
- `models/qwen3.py` - Qwen3 architecture → See ARCHITECTURE.md 4.0

**Utilities:**
- `utils/context.py` - Thread-local state → See ARCHITECTURE.md 3.2
- `utils/loader.py` - Weight loading → See ARCHITECTURE.md 3.3

**Configuration:**
- `config.py` - System config → See ARCHITECTURE.md 1.2
- `sampling_params.py` - Generation params → See ARCHITECTURE.md 1.3

---

## 📝 Key Concepts Explained

| Concept | Location | Key Points |
|---------|----------|-----------|
| **Prefix Caching** | SUMMARY, DIAGRAMS-4, ARCH 2.3 | Hash-based block sharing for identical prefixes |
| **Block-Based KV** | SUMMARY, DIAGRAMS-4, ARCH 2.3 | 256 tokens per block, slot mapping for access |
| **CUDA Graphs** | SUMMARY, DIAGRAMS-7, ARCH 3.1 | Pre-record decode forward passes, ~10x speedup |
| **Tensor Parallelism** | SUMMARY, DIAGRAMS-5, ARCH 5.2 | Column/row parallel patterns, NCCL sync |
| **Scheduling** | SUMMARY, DIAGRAMS-2,3, ARCH 2.2 | Prefill-first, preemption for overflow |
| **Attention** | SUMMARY, DIAGRAMS-6, ARCH 5.1 | Flash attention varlen + KV cache storage |

---

## 🚀 Performance Insights

**Why Nano-vLLM is fast (1.053x vs vLLM):**

1. **Efficient KV Cache** - Prefix caching reduces memory/compute
2. **CUDA Graphs** - No CPU overhead for decode steps
3. **Simple Scheduling** - Less overhead than complex schedulers
4. **Flash Attention** - IO-optimized kernels
5. **Lean Codebase** - Less Python overhead

**Trade-offs:**
- No multi-node distribution
- Qwen3-only (currently)
- No speculative decoding
- No quantization

---

## 📚 External References

These docs reference key concepts from:
- **Flash Attention**: `flash_attn_varlen_func`, `flash_attn_with_kvcache`
- **CUDA Graphs**: `torch.cuda.CUDAGraph()`
- **Tensor Parallelism**: `torch.distributed` (NCCL backend)
- **RoPE**: Rotary Position Embeddings (Su et al., 2021)
- **xxHash**: `xxhash.xxh64()` for fast hashing

---

## 💡 How to Use These Docs

### For Reading Code
1. Find the component in your code editor
2. Look it up in the component overview table
3. Jump to the corresponding section in ARCHITECTURE.md
4. Review the diagram in ARCHITECTURE_DIAGRAMS.md

### For Design Discussions
1. Use ARCHITECTURE_SUMMARY.md for quick facts
2. Show ARCHITECTURE_DIAGRAMS.md for visual explanation
3. Reference ARCHITECTURE.md for detailed trade-offs

### For Debugging
1. Use ARCHITECTURE_DIAGRAMS.md Section 10 (data pipeline)
2. Trace your issue through the flow
3. Reference ARCHITECTURE.md for component details

### For Extensions
1. Review "Limitations & Future Work" in ARCHITECTURE.md
2. Study relevant component in ARCHITECTURE.md (section 4 for new model)
3. Check ARCHITECTURE_DIAGRAMS.md for integration points

---

## ✅ Verification Checklist

After reading these docs, you should understand:

- [ ] What LLMEngine does and how it orchestrates requests
- [ ] How Scheduler decides which sequences to process
- [ ] How prefix caching works and why it matters
- [ ] What CUDA graphs optimize and when they apply
- [ ] How tensor parallelism splits computation
- [ ] How KV cache is laid out in memory
- [ ] What happens during prefill vs decode phases
- [ ] How attention works with sparse block tables
- [ ] What each layer does in Qwen3
- [ ] Why Nano-vLLM is faster than vLLM

---

## 📞 Document Version

- **Created:** April 2026
- **Nano-vLLM Version:** 0.2.0
- **Model:** Qwen3-0.6B
- **Hardware Reference:** RTX 4070 (8GB)

---

**Happy learning! 🚀**

For questions or clarifications, reference the specific section and revisit the code.

