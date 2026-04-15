# =====================================================# ...省略其他代码... 


# ==== 修改后代码 ====
# Qwen3.5 Linear Attention Roofline 分析脚本
# =====================================================# ...省略其他代码... 


# ==== 修改后代码 ====

import math

# === GPU 参数 ===
# 可选：A100, H100, H800 等
gpu_configs = {
    "A100_SXM": {"bf16_tflops": 312, "mem_bw_gbps": 2039},
    "H100_SXM": {"bf16_tflops": 989, "mem_bw_gbps": 3350},
    "H800":     {"bf16_tflops": 989, "mem_bw_gbps": 3350},
}

# === 模型参数 ===
hidden_size = 2048
linear_num_key_heads = 16
linear_num_value_heads = 32
linear_key_head_dim = 128
linear_value_head_dim = 128
conv_kernel_dim = 4

key_dim = linear_num_key_heads * linear_key_head_dim    # 2048
value_dim = linear_num_value_heads * linear_value_head_dim  # 4096

# === 分析函数 ===
def analyze_linear_attention_layer(batch_tokens, gpu_name="H100_SXM", tp_size=1):
    """
    分析单个 Linear Attention 层的 Compute vs Memory bound 情况
    batch_tokens: 当前 batch 中的 token 数
                  - Decode: 通常 = batch_size (每个请求1个token)
                  - Prefill: = sequence_length (单个请求) 或总 token 数
    """
    gpu = gpu_configs[gpu_name]
    ridge_point = gpu["bf16_tflops"] * 1e12 / (gpu["mem_bw_gbps"] * 1e9)
    
    B = batch_tokens
    bytes_per_param = 2  # BF16
    
    print(f"\n{'='*60}")
    print(f"GPU: {gpu_name}, TP={tp_size}, Batch Tokens={B}")
    print(f"Ridge Point: {ridge_point:.1f} FLOPs/Byte")
    print(f"{'='*60}")
    
    # --- 1. in_proj_qkvz: [B, 2048] × [2048, (key_dim*2+value_dim*2)/tp] ---
    out_dim_qkvz = (key_dim * 2 + value_dim * 2) // tp_size  # 12288/tp
    flops_qkvz = 2 * B * hidden_size * out_dim_qkvz
    # 权重 + 输入 + 输出
    bytes_qkvz = (hidden_size * out_dim_qkvz + B * hidden_size + B * out_dim_qkvz) * bytes_per_param
    ai_qkvz = flops_qkvz / bytes_qkvz
    bound_qkvz = "Compute" if ai_qkvz > ridge_point else "Memory"
    print(f"\n[in_proj_qkvz] FLOPs={flops_qkvz:.2e}, Bytes={bytes_qkvz:.2e}, AI={ai_qkvz:.1f} → {bound_qkvz}-bound")
    
    # --- 2. in_proj_ba: [B, 2048] × [2048, num_v_heads*2/tp] ---
    out_dim_ba = (linear_num_value_heads * 2) // tp_size  # 64/tp
    flops_ba = 2 * B * hidden_size * out_dim_ba
    bytes_ba = (hidden_size * out_dim_ba + B * hidden_size + B * out_dim_ba) * bytes_per_param
    ai_ba = flops_ba / bytes_ba
    bound_ba = "Compute" if ai_ba > ridge_point else "Memory"
    print(f"[in_proj_ba]   FLOPs={flops_ba:.2e}, Bytes={bytes_ba:.2e}, AI={ai_ba:.1f} → {bound_ba}-bound")
    
    # --- 3. Core GDN Attention ---
    # Prefill (chunk-based): O(seq_len * dk * dv * num_heads)
    # Decode (recurrent): O(num_k_heads * dk * dv) per token
    num_k_h = linear_num_key_heads // tp_size
    num_v_h = linear_num_value_heads // tp_size
    dk = linear_key_head_dim
    dv = linear_value_head_dim
    
    # Chunk-based prefill: 假设 chunk_size=64
    chunk_size = 64
    if B > chunk_size:
        # 粗略估计: 每个 chunk 内 O(C^2 * dk) + chunk 间 O(C * dk * dv)
        n_chunks = math.ceil(B / chunk_size)
        # intra-chunk: QK^T 类似操作
        flops_core = n_chunks * (2 * chunk_size * chunk_size * dk * num_k_h + 
                                  2 * chunk_size * dk * dv * num_k_h)
        # 状态读写
        state_size_bytes = num_k_h * dk * dv * 4  # FP32 状态
        bytes_core = (B * (dk * 2 + dv * 2) * bytes_per_param * num_k_h // linear_num_key_heads +
                      n_chunks * state_size_bytes * 2 + 
                      B * dv * num_v_h * bytes_per_param)
    else:
        # Decode: recurrent per token
        flops_core = B * (2 * num_k_h * dk * dv +  # state update
                          2 * num_k_h * dk * dv)     # query @ state
        state_size_bytes = num_k_h * dk * dv * 4
        bytes_core = (state_size_bytes * 2 +  # 读+写状态
                      B * (dk * 2 + dv * 2) * bytes_per_param +  # 输入
                      B * dv * num_v_h * bytes_per_param)  # 输出
    
    ai_core = flops_core / bytes_core if bytes_core > 0 else 0
    bound_core = "Compute" if ai_core > ridge_point else "Memory"
    print(f"[GDN Core]     FLOPs={flops_core:.2e}, Bytes={bytes_core:.2e}, AI={ai_core:.1f} → {bound_core}-bound")
    
    # --- 4. out_proj: [B, value_dim/tp] × [value_dim/tp, hidden_size] ---
    in_dim_out = value_dim // tp_size
    flops_out = 2 * B * in_dim_out * hidden_size
    bytes_out = (in_dim_out * hidden_size + B * in_dim_out + B * hidden_size) * bytes_per_param
    ai_out = flops_out / bytes_out
    bound_out = "Compute" if ai_out > ridge_point else "Memory"
    print(f"[out_proj]     FLOPs={flops_out:.2e}, Bytes={bytes_out:.2e}, AI={ai_out:.1f} → {bound_out}-bound")
    
    # --- 总结 ---
    total_flops = flops_qkvz + flops_ba + flops_core + flops_out
    total_bytes = bytes_qkvz + bytes_ba + bytes_core + bytes_out
    total_ai = total_flops / total_bytes
    total_bound = "Compute" if total_ai > ridge_point else "Memory"
    print(f"\n[总计]         FLOPs={total_flops:.2e}, Bytes={total_bytes:.2e}, AI={total_ai:.1f} → {total_bound}-bound")
    
    # 需要多少 batch_tokens 才能达到 compute-bound?
    # 对于最大的 GEMM (in_proj_qkvz):
    # AI = 2*B*H*O / ((H*O + B*H + B*O)*2)  ≈ B*H*O / (H*O) = B  当 B << H, O
    # 需要 AI > ridge → B > ridge (粗略)
    print(f"\n[提示] 对于 GEMM 操作，大约需要 batch_tokens ≈ {int(ridge_point)} "
          f"才能使 GEMM 操作接近 Compute-bound")


# === 运行分析 ===
print("=" * 60)
print("Qwen3.5-35B-A3B Linear Attention Roofline Analysis")
print("=" * 60)

# Decode 场景
for bs in [1, 8, 32, 128, 256]:
    analyze_linear_attention_layer(bs, "H100_SXM", tp_size=1)

print("\n\n" + "=" * 60)
print("Prefill 场景")
print("=" * 60)
for seq_len in [512, 1024, 4096, 8192, 32768]:
    analyze_linear_attention_layer(seq_len, "H100_SXM", tp_size=1)