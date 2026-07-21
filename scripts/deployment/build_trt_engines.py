#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build TensorRT engines from exported ONNX pipeline components."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import onnx
import tensorrt as trt


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Component:
    """Describe one ONNX input and its TensorRT engine output."""

    name: str
    onnx_name: str
    engine_name: str
    precision: str


def load_export_metadata(onnx_dir: str | Path) -> dict:
    """Load captured shape hints and the fixed vision grid."""
    path = Path(onnx_dir) / "full_pipeline_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported export metadata schema: {metadata.get('schema_version')}")
    return metadata


def _shape_hints(
    metadata: dict,
    max_batch: int,
    max_seq_len: int,
) -> dict[str, tuple[int, int, int]]:
    """Convert captured dimensions into TensorRT min/opt/max values."""
    llm_seq_len = int(metadata.get("llm_seq_len", 256))
    vlm_seq_len = int(metadata.get("vlm_seq_len", llm_seq_len))
    obs_seq_len = int(metadata.get("obs_seq_len", vlm_seq_len + 1))
    obs_extra_tokens = obs_seq_len - vlm_seq_len

    return {
        "batch": (1, 1, max_batch),
        "batch_size": (1, 1, max_batch),
        "seq_len": (1, llm_seq_len, max(max_seq_len, llm_seq_len)),
        "vlm_seq_len": (1, vlm_seq_len, max(max_seq_len, vlm_seq_len)),
        "obs_seq_len": (
            1 + obs_extra_tokens,
            obs_seq_len,
            max(max_seq_len + obs_extra_tokens, obs_seq_len),
        ),
    }


def derive_shapes_from_onnx(
    onnx_path: str | Path,
    hints: dict[str, tuple[int, int, int]],
) -> tuple[dict, dict, dict]:
    """Build input shape profiles from ONNX dimensions and named hints."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    minimum: dict[str, tuple[int, ...]] = {}
    optimum: dict[str, tuple[int, ...]] = {}
    maximum: dict[str, tuple[int, ...]] = {}

    for graph_input in model.graph.input:
        min_shape = []
        opt_shape = []
        max_shape = []
        for index, dimension in enumerate(graph_input.type.tensor_type.shape.dim):
            if dimension.dim_value > 0:
                values = (int(dimension.dim_value),) * 3
            else:
                dimension_name = dimension.dim_param or ("batch" if index == 0 else "")
                if dimension_name not in hints:
                    raise ValueError(
                        f"No shape hint for dynamic dimension {dimension_name!r} "
                        f"of input {graph_input.name!r} in {onnx_path}."
                    )
                values = hints[dimension_name]
            min_shape.append(values[0])
            opt_shape.append(values[1])
            max_shape.append(values[2])

        minimum[graph_input.name] = tuple(min_shape)
        optimum[graph_input.name] = tuple(opt_shape)
        maximum[graph_input.name] = tuple(max_shape)

    return minimum, optimum, maximum


def _validate_strong_precision(network, precision: str) -> None:
    """Ensure a strongly typed graph contains the requested float dtype."""
    expected = {
        "bf16": "BF16",
        "fp16": "HALF",
        "fp32": "FLOAT",
    }[precision]
    names = {
        network.get_input(index).dtype.name for index in range(network.num_inputs)
    } | {
        network.get_output(index).dtype.name for index in range(network.num_outputs)
    }
    if expected not in names:
        raise ValueError(
            f"Requested {precision}, but the strongly typed ONNX graph exposes {sorted(names)}. "
            "Re-export the component at the requested precision."
        )


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str,
    workspace_mb: int,
    minimum: dict,
    optimum: dict,
    maximum: dict,
) -> Path:
    """Parse one ONNX graph, configure its profiles, and serialize an engine."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    if strongly_typed:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing %s", onnx_path)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024**2)
    if strongly_typed:
        _validate_strong_precision(network, precision)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")

    dynamic_inputs = []
    for index in range(network.num_inputs):
        network_input = network.get_input(index)
        if any(dimension < 0 for dimension in network_input.shape):
            dynamic_inputs.append(network_input.name)

    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for name in dynamic_inputs:
            accepted = profile.set_shape(name, minimum[name], optimum[name], maximum[name])
            if accepted is False:
                raise ValueError(
                    f"TensorRT rejected profile for {name}: "
                    f"min={minimum[name]}, opt={optimum[name]}, max={maximum[name]}"
                )
            logger.info(
                "%s profile: min=%s opt=%s max=%s",
                name,
                minimum[name],
                optimum[name],
                maximum[name],
            )
        config.add_optimization_profile(profile)

    logger.info("Building %s", engine_path)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_path.open("wb") as file:
        file.write(serialized)
    logger.info("Saved %s", engine_path)
    return engine_path


def pipeline_components(
    mode: str,
    precision: str,
    vision_precision: str,
) -> list[Component]:
    """Return the engine components selected by the requested build mode."""
    backbone = [
        Component("vit", f"vit_{vision_precision}.onnx", "vit.engine", vision_precision),
        Component("llm", f"llm_{precision}.onnx", f"llm_{precision}.engine", precision),
    ]
    action_head = [
        Component("obs_encoder", "obs_encoder.onnx", "obs_encoder.engine", precision),
        Component("action_encoder", "action_encoder.onnx", "action_encoder.engine", precision),
        Component("dit", f"dit_{precision}.onnx", f"dit_{precision}.engine", precision),
        Component("action_decoder", "action_decoder.onnx", "action_decoder.engine", precision),
    ]
    if mode == "backbone":
        return backbone
    if mode == "action_head":
        return action_head
    if mode == "full_pipeline":
        return backbone + action_head
    raise ValueError(f"Unsupported build mode: {mode}")


def build_pipeline_engines(
    onnx_dir: str | Path,
    engine_dir: str | Path,
    mode: str = "full_pipeline",
    precision: str = "bf16",
    vision_precision: str = "fp32",
    workspace_mb: int = 8192,
    max_batch: int = 1,
    max_seq_len: int = 512,
) -> dict[str, Path]:
    """Build the selected backbone and action-head engine bundle."""
    onnx_dir = Path(onnx_dir)
    engine_dir = Path(engine_dir)
    metadata = load_export_metadata(onnx_dir)
    if mode != "action_head" and not metadata:
        raise FileNotFoundError(
            f"Missing {onnx_dir / 'full_pipeline_metadata.json'}. "
            "Run the full-pipeline exporter before building backbone engines."
        )

    if metadata:
        if metadata.get("precision") != precision:
            raise ValueError(
                f"Export precision is {metadata.get('precision')}, "
                f"but build precision is {precision}."
            )
        if metadata.get("vision_precision") != vision_precision:
            raise ValueError(
                "Vision export/build precision mismatch: "
                f"{metadata.get('vision_precision')} vs {vision_precision}."
            )

    hints = _shape_hints(metadata, max_batch=max_batch, max_seq_len=max_seq_len)
    results = {}
    for component in pipeline_components(mode, precision, vision_precision):
        onnx_path = onnx_dir / component.onnx_name
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX for {component.name}: {onnx_path}")
        minimum, optimum, maximum = derive_shapes_from_onnx(onnx_path, hints)
        results[component.name] = build_engine(
            onnx_path=onnx_path,
            engine_path=engine_dir / component.engine_name,
            precision=component.precision,
            workspace_mb=workspace_mb,
            minimum=minimum,
            optimum=optimum,
            maximum=maximum,
        )

    if metadata:
        engine_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            onnx_dir / "full_pipeline_metadata.json",
            engine_dir / "full_pipeline_metadata.json",
        )
    return results


def main() -> None:
    """Parse CLI options and build the selected engine bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["action_head", "backbone", "full_pipeline"],
        default="full_pipeline",
    )
    parser.add_argument("--onnx-dir", default="psi0_trt_deployment/onnx")
    parser.add_argument("--engine-dir", default="psi0_trt_deployment/engines")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--vision-precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    results = build_pipeline_engines(
        onnx_dir=args.onnx_dir,
        engine_dir=args.engine_dir,
        mode=args.mode,
        precision=args.precision,
        vision_precision=args.vision_precision,
        workspace_mb=args.workspace_mb,
        max_batch=args.max_batch,
        max_seq_len=args.max_seq_len,
    )
    for name, path in results.items():
        print(f"[build] {name}: {path}")


if __name__ == "__main__":
    main()
