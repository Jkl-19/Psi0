#!/usr/bin/env python3
"""
Build TensorRT engines for Psi0 action-head ONNX files.

GR00T equivalent:
  scripts/deployment/build_tensorrt_engine.py
"""

from __future__ import annotations

import argparse
import os

import onnx
import tensorrt as trt


def derive_shapes_from_onnx(
    onnx_path: str,
    max_batch: int = 1,
    opt_seq_lens: dict[str, int] | None = None,
):
    model = onnx.load(onnx_path, load_external_data=False)
    opt_seq_lens = opt_seq_lens or {}

    min_shapes = {}
    opt_shapes = {}
    max_shapes = {}

    for inp in model.graph.input:
        name = inp.name
        dims = inp.type.tensor_type.shape.dim

        min_shape = []
        opt_shape = []
        max_shape = []

        for i, d in enumerate(dims):
            if d.dim_value > 0:
                value = int(d.dim_value)
                min_shape.append(value)
                opt_shape.append(value)
                max_shape.append(value)
            else:
                dim_name = d.dim_param if d.dim_param else f"dim_{i}"

                if dim_name == "batch_size" or i == 0:
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(max_batch)
                elif dim_name in opt_seq_lens:
                    opt = int(opt_seq_lens[dim_name])
                    min_shape.append(1)
                    opt_shape.append(opt)
                    max_shape.append(max(opt * 2, opt + 64))
                else:
                    min_shape.append(1)
                    opt_shape.append(256)
                    max_shape.append(512)

        min_shapes[name] = tuple(min_shape)
        opt_shapes[name] = tuple(opt_shape)
        max_shapes[name] = tuple(max_shape)

    return min_shapes, opt_shapes, max_shapes


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    max_batch: int = 1,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flags = 0
    if "EXPLICIT_BATCH" in getattr(trt.NetworkDefinitionCreationFlag, "__members__", {}):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mb * 1024 * 1024,
    )

    strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")

    if not strongly_typed:
        if precision == "bf16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp32":
            pass
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    min_shapes, opt_shapes, max_shapes = derive_shapes_from_onnx(
        onnx_path=onnx_path,
        max_batch=max_batch,
    )

    profile = builder.create_optimization_profile()

    has_dynamic_input = False

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = tuple(inp.shape)

        if any(dim < 0 for dim in shape):
            has_dynamic_input = True
            profile.set_shape(name, min_shapes[name], opt_shapes[name], max_shapes[name])
            print(f"{name}: min={min_shapes[name]} opt={opt_shapes[name]} max={max_shapes[name]}")

    if has_dynamic_input:
        config.add_optimization_profile(profile)

    print(f"Building TensorRT engine: {engine_path}")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Saved engine: {engine_path}")


def build_action_head_engines(
    onnx_dir: str,
    engine_dir: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    max_batch: int = 1,
):
    os.makedirs(engine_dir, exist_ok=True)

    onnx_files = {
        "obs_encoder": "obs_encoder.onnx",
        "action_encoder": "action_encoder.onnx",
        "dit": f"dit_{precision}.onnx",
        "action_decoder": "action_decoder.onnx",
    }

    engine_files = {
        "obs_encoder": "obs_encoder.engine",
        "action_encoder": "action_encoder.engine",
        "dit": f"dit_{precision}.engine",
        "action_decoder": "action_decoder.engine",
    }

    for name, onnx_name in onnx_files.items():
        onnx_path = os.path.join(onnx_dir, onnx_name)
        engine_path = os.path.join(engine_dir, engine_files[name])

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Missing ONNX for {name}: {onnx_path}")

        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=precision,
            workspace_mb=workspace_mb,
            max_batch=max_batch,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="psi0_trt_deployment/onnx")
    parser.add_argument("--engine-dir", default="psi0_trt_deployment/engines")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--max-batch", type=int, default=1)
    args = parser.parse_args()

    build_action_head_engines(
        onnx_dir=args.onnx_dir,
        engine_dir=args.engine_dir,
        precision=args.precision,
        workspace_mb=args.workspace_mb,
        max_batch=args.max_batch,
    )


if __name__ == "__main__":
    main()
