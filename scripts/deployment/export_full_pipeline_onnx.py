#!/usr/bin/env python3

"""Export the vision-language backbone and action head to ONNX."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

from scripts.deployment.export_psi0_action_head_onnx import (
    ActionHeadInputCapture,
    export_action_head_to_onnx,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _precision_to_dtype(precision: str) -> torch.dtype:
    """Map an export precision name to its PyTorch dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision: {precision}") from exc


def verify_onnx_export(onnx_path: str | Path) -> None:
    """Check that ONNX can parse and validate an exported graph."""
    import onnx

    logger.info("Verifying ONNX export: %s", onnx_path)
    onnx.checker.check_model(str(onnx_path))
    logger.info("ONNX model verified successfully.")


def _consolidate_external_data(
    onnx_path: Path,
    files_before_export: set[str],
) -> None:
    """Store newly created external weights in one data file."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    created = set(os.listdir(onnx_path.parent)) - files_before_export
    scattered = [
        name
        for name in created
        if name != onnx_path.name and not name.endswith((".onnx", ".json", ".data"))
    ]
    if not scattered:
        return

    data_name = f"{onnx_path.name}.data"
    logger.info("Consolidating %d external weight files into %s", len(scattered), data_name)
    model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )
    onnx.save(model, str(onnx_path))

    for name in scattered:
        path = onnx_path.parent / name
        if path.is_file():
            path.unlink()


class VisionInputCapture:
    """Capture the real vision encoder shapes and fixed image grid."""

    def __init__(self):
        """Initialize empty capture fields for one vision forward call."""
        self.captured = False
        self.pixel_values_shape: tuple[int, ...] | None = None
        self.grid_thw: torch.Tensor | None = None
        self.output_shape: tuple[int, ...] | None = None
        self.deepstack_shapes: list[tuple[int, ...]] = []

    def hook_fn(self, module, args, kwargs, output) -> None:
        """Save the first vision forward call observed by the hook."""
        if self.captured:
            return

        pixel_values = args[0] if args else kwargs["hidden_states"]
        grid_thw = args[1] if len(args) > 1 else kwargs.get("grid_thw")
        if grid_thw is None:
            raise RuntimeError("Vision input capture did not receive grid_thw.")

        image_embeds, deepstack = output
        self.pixel_values_shape = tuple(pixel_values.shape)
        self.grid_thw = grid_thw.detach().cpu().clone()
        self.output_shape = tuple(image_embeds.shape)
        self.deepstack_shapes = [tuple(feature.shape) for feature in deepstack]
        self.captured = True

        logger.info("Captured vision inputs:")
        logger.info("  pixel_values: %s", self.pixel_values_shape)
        logger.info("  grid_thw: %s", self.grid_thw.tolist())
        logger.info("  image_embeds: %s", self.output_shape)
        logger.info("  deepstack features: %s", self.deepstack_shapes)


class LanguageInputCapture:
    """Capture the tensors passed into the language transformer."""

    def __init__(self):
        """Initialize empty capture fields for one language forward call."""
        self.captured = False
        self.inputs_embeds: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.position_ids: torch.Tensor | None = None
        self.visual_pos_masks: torch.Tensor | None = None
        self.deepstack_visual_embeds: list[torch.Tensor] = []

    def hook_fn(self, module, args, kwargs) -> None:
        """Save the first language-model forward call observed by the hook."""
        if self.captured:
            return

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and args:
            inputs_embeds = args[0]
        if inputs_embeds is None:
            raise RuntimeError("Language input capture did not receive inputs_embeds.")

        self.inputs_embeds = inputs_embeds.detach().cpu().clone()

        for name in ("attention_mask", "position_ids", "visual_pos_masks"):
            value = kwargs.get(name)
            if value is not None:
                setattr(self, name, value.detach().cpu().clone())

        deepstack = kwargs.get("deepstack_visual_embeds")
        if deepstack is not None:
            self.deepstack_visual_embeds = [value.detach().cpu().clone() for value in deepstack]

        self.captured = True
        logger.info("Captured language-model inputs:")
        logger.info("  inputs_embeds: %s", tuple(self.inputs_embeds.shape))
        if self.attention_mask is not None:
            logger.info("  attention_mask: %s", tuple(self.attention_mask.shape))
        if self.position_ids is not None:
            logger.info("  position_ids: %s", tuple(self.position_ids.shape))
        if self.visual_pos_masks is not None:
            logger.info("  visual_pos_masks: %s", tuple(self.visual_pos_masks.shape))
        logger.info(
            "  deepstack features: %s",
            [tuple(value.shape) for value in self.deepstack_visual_embeds],
        )


@torch.inference_mode()
def capture_full_pipeline_inputs(
    model,
    observations,
    states,
    instructions,
    num_inference_steps: int,
    traj2ds=None,
):
    """Run one real inference and capture backbone plus action-head inputs."""
    model.eval()

    vision_capture = VisionInputCapture()
    language_capture = LanguageInputCapture()
    action_capture = ActionHeadInputCapture()

    inner_model = model.vlm_model.model
    handles = [
        inner_model.visual.register_forward_hook(vision_capture.hook_fn, with_kwargs=True),
        inner_model.language_model.register_forward_pre_hook(
            language_capture.hook_fn,
            with_kwargs=True,
        ),
        model.action_header.register_forward_pre_hook(
            action_capture.hook_fn,
            with_kwargs=True,
        ),
    ]

    try:
        model.predict_action(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=num_inference_steps,
            traj2ds=traj2ds,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = []
    if not vision_capture.captured:
        missing.append("vision encoder")
    if not language_capture.captured:
        missing.append("language model")
    if not action_capture.captured:
        missing.append("action head")
    if missing:
        raise RuntimeError(f"Failed to capture inputs for: {', '.join(missing)}")

    return vision_capture, language_capture, action_capture


def _apply_rotary_real(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings with ONNX-compatible real-valued operations."""
    original_dtype = tensor.dtype
    tensor = tensor.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
    return (tensor * cos + rotated * sin).to(original_dtype)


def _make_vision_attention_forward(attention, chunk_sizes: list[int]):
    """Create an ONNX-friendly attention forward for one vision block."""

    def forward(
        hidden_states,
        cu_seqlens=None,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        """Run chunked vision attention with traceable tensor operations."""
        del cu_seqlens, rotary_pos_emb, kwargs
        sequence_length = hidden_states.shape[0]
        qkv = attention.qkv(hidden_states)
        qkv = qkv.reshape(sequence_length, 3, attention.num_heads, -1)
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)

        cos, sin = position_embeddings
        query = _apply_rotary_real(query, cos, sin)
        key = _apply_rotary_real(key, cos, sin)

        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            torch.split(query, chunk_sizes, dim=0),
            torch.split(key, chunk_sizes, dim=0),
            torch.split(value, chunk_sizes, dim=0),
        ):
            query_chunk = query_chunk.transpose(0, 1)
            key_chunk = key_chunk.transpose(0, 1)
            value_chunk = value_chunk.transpose(0, 1)
            weights = torch.matmul(query_chunk, key_chunk.transpose(-2, -1))
            weights = F.softmax(weights.float() * attention.scaling, dim=-1)
            output = torch.matmul(weights.to(value_chunk.dtype), value_chunk)
            outputs.append(output.transpose(0, 1))

        output = torch.cat(outputs, dim=0).reshape(sequence_length, -1).contiguous()
        return attention.proj(output)

    return forward


def _patch_vision_attention(vision_model, chunk_sizes: list[int]):
    """Replace vision attention forwards temporarily for ONNX tracing."""
    originals = []
    for block in vision_model.blocks:
        originals.append(block.attn.forward)
        block.attn.forward = _make_vision_attention_forward(block.attn, chunk_sizes)
    return originals


def _restore_vision_attention(vision_model, originals) -> None:
    """Restore the original vision attention implementations."""
    for block, original in zip(vision_model.blocks, originals):
        block.attn.forward = original


def _restore_vision_rotary_frequency(vision_model) -> None:
    """Rebuild the non-persistent vision RoPE frequency buffer in FP32."""
    rotary = vision_model.rotary_pos_emb
    head_dim = vision_model.config.hidden_size // vision_model.config.num_heads
    with torch.device(rotary.inv_freq.device):
        fresh_rotary = type(rotary)(head_dim // 2)
    value = fresh_rotary.inv_freq.detach().to(
        device=rotary.inv_freq.device,
        dtype=torch.float32,
    )
    rotary.register_buffer("inv_freq", value, persistent=False)


class VisionEncoderForExport(nn.Module):
    """Traceable vision encoder with a fixed image grid baked into buffers."""

    def __init__(self, vision_model, grid_thw: torch.Tensor):
        """Store vision modules and precompute grid-dependent position buffers."""
        super().__init__()
        self.patch_embed = vision_model.patch_embed
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list

        with torch.no_grad():
            position = vision_model.fast_pos_embed_interpolate(grid_thw)
            rotary = vision_model.rot_pos_emb(grid_thw)
            rotary = torch.cat((rotary, rotary), dim=-1)

        self.register_buffer("position", position.detach().contiguous())
        self.register_buffer("rotary_cos", rotary.cos().detach().contiguous())
        self.register_buffer("rotary_sin", rotary.sin().detach().contiguous())

    def forward(self, pixel_values):
        """Encode image patches and return final plus deepstack embeddings."""
        hidden_states = self.patch_embed(pixel_values) + self.position
        position_embeddings = (self.rotary_cos, self.rotary_sin)

        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=None,
                position_embeddings=position_embeddings,
            )
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )

        image_embeds = self.merger(hidden_states)
        if deepstack_features:
            deepstack = torch.stack(deepstack_features)
        else:
            deepstack = image_embeds.new_zeros((1, 1, 1))
        return image_embeds, deepstack


def export_vision_encoder_to_onnx(
    model,
    captured: VisionInputCapture,
    output_dir: str | Path,
    precision: str = "fp32",
) -> Path:
    """Export the vision encoder using its captured static image grid."""
    if captured.grid_thw is None or captured.pixel_values_shape is None:
        raise ValueError("Vision inputs were not captured.")

    dtype = _precision_to_dtype(precision)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_model = model.vlm_model.model.visual
    if precision == "fp32":
        _restore_vision_rotary_frequency(vision_model)
    vision_model = vision_model.to(device="cuda", dtype=dtype).eval()

    grid_thw = captured.grid_thw.to(device="cuda")
    chunk_sizes = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).tolist()
    originals = _patch_vision_attention(vision_model, chunk_sizes)
    wrapper = VisionEncoderForExport(vision_model, grid_thw).to(dtype=dtype).eval().cuda()
    pixel_values = torch.randn(captured.pixel_values_shape, device="cuda", dtype=dtype)

    output_path = output_dir / f"vit_{precision}.onnx"
    files_before_export = set(os.listdir(output_dir))
    logger.info("Exporting vision encoder: %s", output_path)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (pixel_values,),
                str(output_path),
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_features"],
                opset_version=19,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
        _consolidate_external_data(output_path, files_before_export)
        verify_onnx_export(output_path)
    finally:
        _restore_vision_attention(vision_model, originals)

    return output_path


class LanguageModelForExport(nn.Module):
    """Traceable language transformer with explicit multimodal inputs."""

    def __init__(self, text_model, number_of_layers: int, number_of_deepstack: int):
        """Create eager-attention layers and copy the trained language weights."""
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextRotaryEmbedding,
        )

        config = copy.deepcopy(text_model.config)
        config._attn_implementation = "eager"
        config.num_hidden_layers = number_of_layers

        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, index) for index in range(number_of_layers)]
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.number_of_deepstack = number_of_deepstack

        source_state = text_model.state_dict()
        incompatible = self.load_state_dict(source_state, strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith(("embed_tokens.", "norm."))
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "Language export wrapper weight mismatch: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}"
            )

    @staticmethod
    def _causal_mask(inputs_embeds, attention_mask):
        """Build a traceable causal mask with optional padding."""
        batch_size, sequence_length = inputs_embeds.shape[:2]
        mask_value = torch.finfo(inputs_embeds.dtype).min * 0.5
        causal = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                mask_value,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ),
            diagonal=1,
        )
        causal = causal[None, None].expand(batch_size, 1, -1, -1)
        padding = (1.0 - attention_mask[:, None, None, :].to(inputs_embeds.dtype))
        return causal + padding * mask_value

    @staticmethod
    def _add_deepstack(hidden_states, visual_pos_masks, visual_embeds):
        """Scatter one deepstack tensor into its visual token positions."""
        mask = visual_pos_masks.unsqueeze(-1).expand_as(hidden_states)
        delta = torch.zeros_like(hidden_states).masked_scatter(mask, visual_embeds)
        return hidden_states + delta

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        visual_pos_masks=None,
        deepstack_0=None,
        deepstack_1=None,
        deepstack_2=None,
    ):
        """Run the exported language layers and return pre-norm features."""
        sequence_length = inputs_embeds.shape[1]
        attention = self._causal_mask(inputs_embeds, attention_mask)
        text_position_ids = position_ids[0]
        cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        deepstack = [deepstack_0, deepstack_1, deepstack_2]
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if layer_index < self.number_of_deepstack:
                hidden_states = self._add_deepstack(
                    hidden_states,
                    visual_pos_masks,
                    deepstack[layer_index],
                )

        return hidden_states


def export_language_model_to_onnx(
    model,
    captured: LanguageInputCapture,
    output_dir: str | Path,
    precision: str = "bf16",
) -> Path:
    """Export the language transformer with dynamic text sequence length."""
    required = (captured.inputs_embeds, captured.attention_mask, captured.position_ids)
    if any(value is None for value in required):
        raise ValueError("Language inputs were not fully captured.")
    if len(captured.deepstack_visual_embeds) > 3:
        raise ValueError("The exporter currently supports at most three deepstack tensors.")

    dtype = _precision_to_dtype(precision)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_model = model.vlm_model.model.language_model
    number_of_layers = len(text_model.layers)
    number_of_deepstack = len(captured.deepstack_visual_embeds)

    wrapper = LanguageModelForExport(
        text_model=text_model,
        number_of_layers=number_of_layers,
        number_of_deepstack=number_of_deepstack,
    ).to(device="cuda", dtype=dtype).eval()

    inputs_embeds = captured.inputs_embeds.to(device="cuda", dtype=dtype)
    attention_mask = captured.attention_mask.to(device="cuda", dtype=torch.int64)
    position_ids = captured.position_ids.to(device="cuda", dtype=torch.int64)
    export_inputs = [inputs_embeds, attention_mask, position_ids]
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "embeddings": {1: "seq_len"},
    }

    if number_of_deepstack:
        if captured.visual_pos_masks is None:
            raise ValueError("Deepstack inputs require visual_pos_masks.")
        visual_pos_masks = captured.visual_pos_masks.to(device="cuda", dtype=torch.bool)
        export_inputs.append(visual_pos_masks)
        input_names.append("visual_pos_masks")
        dynamic_axes["visual_pos_masks"] = {1: "seq_len"}

        for index, value in enumerate(captured.deepstack_visual_embeds):
            export_inputs.append(value.to(device="cuda", dtype=dtype))
            input_names.append(f"deepstack_{index}")

    output_path = output_dir / f"llm_{precision}.onnx"
    files_before_export = set(os.listdir(output_dir))
    logger.info("Exporting language model with %d layers: %s", number_of_layers, output_path)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(export_inputs),
            str(output_path),
            input_names=input_names,
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
            dynamo=False,
        )

    _consolidate_external_data(output_path, files_before_export)
    verify_onnx_export(output_path)
    return output_path


def _write_export_metadata(
    model,
    vision_capture: VisionInputCapture,
    language_capture: LanguageInputCapture,
    action_capture: ActionHeadInputCapture,
    output_dir: str | Path,
    precision: str,
    vision_precision: str,
) -> Path:
    """Record the static grid and dynamic-shape hints used by the builder."""
    if (
        vision_capture.grid_thw is None
        or vision_capture.pixel_values_shape is None
        or vision_capture.output_shape is None
    ):
        raise ValueError("Vision metadata is unavailable.")
    if language_capture.inputs_embeds is None:
        raise ValueError("Language metadata is unavailable.")

    vlm_seq_len = int(action_capture.vlm_hidden_states.shape[2])
    state_seq_len = int(action_capture.states.shape[1])
    metadata = {
        "schema_version": 1,
        "precision": precision,
        "vision_precision": vision_precision,
        "batch_size": int(language_capture.inputs_embeds.shape[0]),
        "vit_grid_thw": vision_capture.grid_thw.tolist(),
        "pixel_values_shape": list(vision_capture.pixel_values_shape),
        "llm_seq_len": int(language_capture.inputs_embeds.shape[1]),
        "vlm_seq_len": vlm_seq_len,
        "obs_seq_len": vlm_seq_len + state_seq_len,
        "num_visual_tokens": int(vision_capture.output_shape[0]),
        "num_deepstack": len(language_capture.deepstack_visual_embeds),
        "language_layers": len(model.vlm_model.model.language_model.layers),
    }

    output_path = Path(output_dir) / "full_pipeline_metadata.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    logger.info("Saved export metadata: %s", output_path)
    return output_path


def export_full_pipeline_to_onnx(
    model,
    vision_capture: VisionInputCapture,
    language_capture: LanguageInputCapture,
    action_capture: ActionHeadInputCapture,
    output_dir: str | Path,
    precision: str = "bf16",
    vision_precision: str = "fp32",
    include_action_head: bool = True,
) -> dict[str, Path | str]:
    """Export backbone engines and optionally the four action-head engines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path | str] = {}
    paths["vit"] = export_vision_encoder_to_onnx(
        model=model,
        captured=vision_capture,
        output_dir=output_dir,
        precision=vision_precision,
    )
    paths["llm"] = export_language_model_to_onnx(
        model=model,
        captured=language_capture,
        output_dir=output_dir,
        precision=precision,
    )

    if include_action_head:
        paths.update(
            export_action_head_to_onnx(
                model=model,
                captured=action_capture,
                output_dir=str(output_dir),
                precision=precision,
                batch_size=1,
            )
        )

    paths["metadata"] = _write_export_metadata(
        model=model,
        vision_capture=vision_capture,
        language_capture=language_capture,
        action_capture=action_capture,
        output_dir=output_dir,
        precision=precision,
        vision_precision=vision_precision,
    )
    return paths


def export_from_loaded_model_and_batch(
    model,
    observations,
    states,
    instructions,
    traj2ds,
    output_dir: str | Path,
    num_inference_steps: int = 4,
    precision: str = "bf16",
    vision_precision: str = "fp32",
    include_action_head: bool = True,
) -> dict[str, Path | str]:
    """Capture one real batch and export the requested full-pipeline graphs."""
    vision_capture, language_capture, action_capture = capture_full_pipeline_inputs(
        model=model,
        observations=observations,
        states=states,
        instructions=instructions,
        num_inference_steps=num_inference_steps,
        traj2ds=traj2ds,
    )
    return export_full_pipeline_to_onnx(
        model=model,
        vision_capture=vision_capture,
        language_capture=language_capture,
        action_capture=action_capture,
        output_dir=output_dir,
        precision=precision,
        vision_precision=vision_precision,
        include_action_head=include_action_head,
    )
