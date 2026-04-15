#!/usr/bin/env python3
"""
Test script: Verify Qwen3.5 model can be successfully constructed and weights loaded.

Usage:
    python example/test_qwen3_5_load.py [--model-path /path/to/qwen3.5]

Requirements:
    - Python 3.10+ (for union type syntax used in nanovllm)
    - PyTorch
    - safetensors

This script:
1. Loads the Qwen3.5 config from the local config file
2. Constructs the model architecture on CPU
3. If safetensors weight files exist, loads weights and verifies them
4. Checks weight name matching between checkpoint index and model params
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)


def test_config_loading(model_path: str):
    """Test 1: Load and validate config."""
    print("=" * 60)
    print("[Test 1] Loading Qwen3.5 config...")
    print("=" * 60)

    # Import config loading directly (avoid triggering nanovllm.__init__ which
    # chains into engine imports requiring CUDA)
    import importlib.util
    config_spec = importlib.util.spec_from_file_location(
        "nanovllm_config",
        os.path.join(PROJECT_ROOT, "nanovllm", "config.py"),
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    load_hf_config = config_module.load_hf_config
    config = load_hf_config(model_path)

    print(f"  Model type:         {config.model_type}")
    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Num layers:         {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Num KV heads:       {config.num_key_value_heads}")
    print(f"  Head dim:           {config.head_dim}")
    print(f"  Vocab size:         {config.vocab_size}")
    print(f"  Max position:       {config.max_position_embeddings}")
    print(f"  Torch dtype:        {config.torch_dtype}")

    text_config = config.text_config
    print(f"\n  --- Text Config (MoE) ---")
    print(f"  Num experts:        {text_config.num_experts}")
    print(f"  Experts per token:  {text_config.num_experts_per_tok}")
    print(f"  MoE intermediate:   {text_config.moe_intermediate_size}")
    print(f"  Shared expert int:  {text_config.shared_expert_intermediate_size}")

    print(f"\n  --- Linear Attention ---")
    print(f"  Num key heads:      {text_config.linear_num_key_heads}")
    print(f"  Num value heads:    {text_config.linear_num_value_heads}")
    print(f"  Key head dim:       {text_config.linear_key_head_dim}")
    print(f"  Value head dim:     {text_config.linear_value_head_dim}")
    print(f"  Conv kernel dim:    {text_config.linear_conv_kernel_dim}")

    # Count layer types
    layer_types = text_config.layer_types
    type_counts = defaultdict(int)
    for lt in layer_types:
        type_counts[lt] += 1
    print(f"\n  --- Layer Types ---")
    for lt, count in type_counts.items():
        print(f"  {lt}: {count} layers")

    print(f"\n  [PASS] Config loaded successfully!")
    return config


def test_model_construction(config):
    """Test 2: Construct model and print structure."""
    print("\n" + "=" * 60)
    print("[Test 2] Constructing Qwen3.5 model...")
    print("=" * 60)

    import torch
    import torch.distributed as dist

    # Initialize a minimal distributed environment for model construction
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")

    from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM

    print("  Building model on CPU (this may take a moment)...")
    with torch.device("cpu"):
        model = Qwen3_5ForCausalLM(config)

    # Count parameters
    total_params = 0
    param_groups: dict[str, int] = defaultdict(int)
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        parts = name.split(".")
        if len(parts) >= 3:
            group = f"{parts[0]}.{parts[1]}.{parts[2]}"
        else:
            group = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        param_groups[group] += num_params

    print(f"\n  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"\n  Top parameter groups:")
    for group, count in sorted(param_groups.items(), key=lambda x: -x[1])[:15]:
        print(f"    {group}: {count:,} ({count / 1e6:.1f}M)")

    # Print model structure
    print(f"\n  Model structure (top level):")
    for name, module in model.named_children():
        print(f"    {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for subname, submodule in module.named_children():
                if subname == 'layers':
                    num_layers = len(list(submodule.children()))
                    print(f"      {subname}: ModuleList ({num_layers} layers)")
                    for i, (lname, layer) in enumerate(submodule.named_children()):
                        if i < 4 or i >= num_layers - 2:
                            layer_type = getattr(layer, 'layer_type', 'unknown')
                            print(f"        [{lname}] {type(layer).__name__} ({layer_type})")
                        elif i == 4:
                            print(f"        ... ({num_layers - 6} more layers)")
                else:
                    print(f"      {subname}: {type(submodule).__name__}")

    # Count attention types
    num_full_attn = 0
    num_linear_attn = 0
    for m in model.modules():
        if type(m).__name__ == 'Qwen3_5FullAttention':
            num_full_attn += 1
        elif type(m).__name__ == 'Qwen3_5GatedDeltaNet':
            num_linear_attn += 1

    print(f"\n  Full attention layers (with KV cache): {num_full_attn}")
    print(f"  Linear attention layers (GatedDeltaNet): {num_linear_attn}")

    print(f"\n  [PASS] Model constructed successfully!")
    return model


def test_weight_loading(model, model_path: str):
    """Test 3: Load weights from safetensors files."""
    import torch
    from glob import glob

    safetensor_files = glob(os.path.join(model_path, "*.safetensors"))

    if not safetensor_files:
        print("\n" + "=" * 60)
        print("[Test 3] Weight loading - SKIPPED (no .safetensors files found)")
        print("=" * 60)
        print(f"  No safetensors files in: {model_path}")
        print(f"  To test weight loading, download model weights to this directory.")
        return False

    print("\n" + "=" * 60)
    print(f"[Test 3] Loading weights from {len(safetensor_files)} safetensors files...")
    print("=" * 60)

    from nanovllm.utils.loader import load_model

    try:
        load_model(model, model_path)
        print(f"  [PASS] All weights loaded successfully!")
    except Exception as e:
        print(f"  [FAIL] Weight loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify weights are non-zero
    print(f"\n  Verifying loaded weights...")
    zero_params = []
    nonzero_params = 0
    for name, param in model.named_parameters():
        if param.abs().sum().item() == 0:
            zero_params.append(name)
        else:
            nonzero_params += 1

    print(f"  Non-zero parameters: {nonzero_params}")
    if zero_params:
        print(f"  Zero parameters (might be unloaded): {len(zero_params)}")
        for name in zero_params[:10]:
            print(f"    - {name}")
        if len(zero_params) > 10:
            print(f"    ... and {len(zero_params) - 10} more")

    return True


def test_weight_name_matching(model, model_path: str):
    """Test 4: Verify checkpoint weight names match model parameters."""
    print("\n" + "=" * 60)
    print("[Test 4] Checking weight name matching...")
    print("=" * 60)

    # Load safetensors index if available
    index_files = [
        os.path.join(model_path, "model.safetensors.index.json"),
        os.path.join(model_path, "qwen3.5-35B-A3B_model.safetensors.index.json"),
    ]

    index_file = None
    for f in index_files:
        if os.path.exists(f):
            index_file = f
            break

    if index_file is None:
        print("  No safetensors index file found, skipping.")
        return

    with open(index_file, 'r') as f:
        index = json.load(f)

    weight_names = set(index.get('weight_map', {}).keys())
    weight_prefix = getattr(model, 'weight_prefix', '')
    skip_prefixes = getattr(model, 'skip_prefixes', ())
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', {})

    # Get model parameter names
    model_params = set(name for name, _ in model.named_parameters())

    # Map checkpoint names to expected model param names
    matched = []
    skipped = []
    unmatched = []

    for wn in sorted(weight_names):
        # Check skip
        skip = False
        for sp in skip_prefixes:
            if wn.startswith(sp):
                skip = True
                break
        if skip:
            skipped.append(wn)
            continue

        # Strip prefix
        param_name = wn
        if weight_prefix and param_name.startswith(weight_prefix):
            param_name = param_name[len(weight_prefix):]

        # Check packed mapping
        for k in packed_modules_mapping:
            if k in param_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = param_name.replace(k, v)
                break

        if param_name in model_params:
            matched.append((wn, param_name))
        else:
            unmatched.append((wn, param_name))

    print(f"  Checkpoint weights: {len(weight_names)}")
    print(f"  Model parameters:   {len(model_params)}")
    print(f"  Matched:           {len(matched)}")
    print(f"  Skipped:           {len(skipped)}")
    print(f"  Unmatched:         {len(unmatched)}")

    if unmatched:
        print(f"\n  Unmatched checkpoint weights (first 30):")
        for wn, pn in unmatched[:30]:
            print(f"    {wn} -> {pn}")
        if len(unmatched) > 30:
            print(f"    ... and {len(unmatched) - 30} more")

    # Check for model params not in checkpoint
    checkpoint_mapped = set(pn for _, pn in matched)
    unmapped_model_params = model_params - checkpoint_mapped
    if unmapped_model_params:
        print(f"\n  Model params not in checkpoint ({len(unmapped_model_params)}):")
        for p in sorted(unmapped_model_params)[:20]:
            print(f"    {p}")
        if len(unmapped_model_params) > 20:
            print(f"    ... and {len(unmapped_model_params) - 20} more")

    if not unmatched and not unmapped_model_params:
        print(f"\n  [PASS] Perfect match between checkpoint and model!")
    elif not unmatched:
        print(f"\n  [PASS] All checkpoint weights map to model parameters!")
    else:
        print(f"\n  [WARN] {len(unmatched)} checkpoint weights don't match model parameters")


def test_weight_name_analysis_standalone(config, model_path: str):
    """Analyze weight names without constructing the model.

    Uses the config + index file to predict what model parameter names
    should be after prefix stripping, and verifies the naming scheme.
    """
    print("\n" + "=" * 60)
    print("[Test 2b] Standalone weight name analysis...")
    print("=" * 60)

    index_files = [
        os.path.join(model_path, "model.safetensors.index.json"),
        os.path.join(model_path, "qwen3.5-35B-A3B_model.safetensors.index.json"),
    ]

    index_file = None
    for f in index_files:
        if os.path.exists(f):
            index_file = f
            break

    if index_file is None:
        print("  No safetensors index file found, skipping.")
        return

    with open(index_file, 'r') as f:
        index = json.load(f)

    weight_names = sorted(index.get('weight_map', {}).keys())
    weight_prefix = "model."
    skip_prefixes = ("mtp.", "model.visual.", "model.merger.")

    text_config = config.text_config

    # Build expected parameter name patterns
    expected_patterns = set()
    for layer_idx in range(text_config.num_hidden_layers):
        layer_type = text_config.layer_types[layer_idx]
        prefix = f"language_model.layers.{layer_idx}"

        # Common
        expected_patterns.add(f"{prefix}.input_layernorm.weight")
        expected_patterns.add(f"{prefix}.post_attention_layernorm.weight")

        # MoE MLP
        expected_patterns.add(f"{prefix}.mlp.gate.weight")
        expected_patterns.add(f"{prefix}.mlp.experts.gate_up_proj")
        expected_patterns.add(f"{prefix}.mlp.experts.down_proj")
        expected_patterns.add(f"{prefix}.mlp.shared_expert.gate_proj.weight")
        expected_patterns.add(f"{prefix}.mlp.shared_expert.up_proj.weight")
        expected_patterns.add(f"{prefix}.mlp.shared_expert.down_proj.weight")
        expected_patterns.add(f"{prefix}.mlp.shared_expert_gate.weight")

        if layer_type == "linear_attention":
            expected_patterns.add(f"{prefix}.linear_attn.in_proj_qkv.weight")
            expected_patterns.add(f"{prefix}.linear_attn.in_proj_z.weight")
            expected_patterns.add(f"{prefix}.linear_attn.in_proj_a.weight")
            expected_patterns.add(f"{prefix}.linear_attn.in_proj_b.weight")
            expected_patterns.add(f"{prefix}.linear_attn.conv1d.weight")
            expected_patterns.add(f"{prefix}.linear_attn.A_log")
            expected_patterns.add(f"{prefix}.linear_attn.dt_bias")
            expected_patterns.add(f"{prefix}.linear_attn.norm.weight")
            expected_patterns.add(f"{prefix}.linear_attn.out_proj.weight")
        elif layer_type == "full_attention":
            expected_patterns.add(f"{prefix}.self_attn.q_proj.weight")
            expected_patterns.add(f"{prefix}.self_attn.k_proj.weight")
            expected_patterns.add(f"{prefix}.self_attn.v_proj.weight")
            expected_patterns.add(f"{prefix}.self_attn.o_proj.weight")
            expected_patterns.add(f"{prefix}.self_attn.q_norm.weight")
            expected_patterns.add(f"{prefix}.self_attn.k_norm.weight")

    expected_patterns.add("language_model.embed_tokens.weight")
    expected_patterns.add("language_model.norm.weight")
    expected_patterns.add("lm_head.weight")

    # Process checkpoint weights
    matched = []
    skipped = []
    unmatched = []

    for wn in weight_names:
        skip = False
        for sp in skip_prefixes:
            if wn.startswith(sp):
                skip = True
                break
        if skip:
            skipped.append(wn)
            continue

        param_name = wn
        if weight_prefix and param_name.startswith(weight_prefix):
            param_name = param_name[len(weight_prefix):]

        if param_name in expected_patterns:
            matched.append((wn, param_name))
        else:
            unmatched.append((wn, param_name))

    # Check reverse: expected patterns not in checkpoint
    checkpoint_mapped = set(pn for _, pn in matched)
    missing_from_checkpoint = expected_patterns - checkpoint_mapped

    print(f"  Checkpoint weights total:  {len(weight_names)}")
    print(f"  Expected model params:     {len(expected_patterns)}")
    print(f"  Matched:                   {len(matched)}")
    print(f"  Skipped (visual/mtp):      {len(skipped)}")
    print(f"  Unmatched from checkpoint: {len(unmatched)}")
    print(f"  Missing from checkpoint:   {len(missing_from_checkpoint)}")

    if unmatched:
        print(f"\n  Unmatched checkpoint weights (first 15):")
        for wn, pn in unmatched[:15]:
            print(f"    {wn} -> {pn}")

    if missing_from_checkpoint:
        print(f"\n  Expected but missing from checkpoint (first 15):")
        for p in sorted(missing_from_checkpoint)[:15]:
            print(f"    {p}")

    if not unmatched and not missing_from_checkpoint:
        print(f"\n  [PASS] Perfect match between checkpoint and expected model params!")
    elif not unmatched:
        print(f"\n  [PASS] All language model checkpoint weights matched!")
    else:
        print(f"\n  [INFO] Some checkpoint weights don't match expected patterns")
        print(f"         This may be normal if there are additional weights in the checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3.5 model loading")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "qwen3.5"),
        help="Path to Qwen3.5 model directory",
    )
    args = parser.parse_args()
    model_path = os.path.abspath(args.model_path)

    print(f"Model path: {model_path}")
    print(f"Python version: {sys.version}")
    print()

    # Test 1: Config loading (works on any Python 3.7+)
    config = test_config_loading(model_path)

    # Test 2-4: Model construction and weight loading (requires Python 3.10+)
    if sys.version_info < (3, 10):
        print(f"\n  [NOTE] Python {sys.version_info.major}.{sys.version_info.minor} detected.")
        print(f"  The nanovllm codebase uses Python 3.10+ syntax (union types).")
        print(f"  Tests 2-4 (model construction, weight loading) require Python 3.10+.")
        print(f"  Config loading (Test 1) passed successfully.")
        print(f"\n  Running standalone weight name analysis instead...")
        test_weight_name_analysis_standalone(config, model_path)
        print(f"\n  To run full tests, use Python 3.10+:")
        print(f"    python3.10 example/test_qwen3_5_load.py --model-path {model_path}")
    else:
        # Test 2: Model construction
        model = test_model_construction(config)

        # Test 3: Weight loading (if files exist)
        test_weight_loading(model, model_path)

        # Test 4: Weight name matching (using index file)
        test_weight_name_matching(model, model_path)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
