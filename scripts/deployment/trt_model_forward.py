from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from scripts.deployment.trt_torch import Engine


class TensorRTBackbone:
    """Run the vision encoder and language transformer with TensorRT."""

    def __init__(
        self,
        vlm_model,
        engine_dir: str | Path,
        precision: str = "bf16",
    ):
        """Load backbone engines and retain the lightweight PyTorch glue modules."""
        self.vlm_model = vlm_model
        self.inner_model = vlm_model.model
        self.engine_dir = Path(engine_dir)
        self.precision = precision

        metadata_path = self.engine_dir / "full_pipeline_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"TensorRT backbone metadata not found: {metadata_path}. "
                "Build the backbone engines with build_trt_engines.py."
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.vision_encoder = Engine(self.engine_dir / "vit.engine")
        self.language_model = Engine(self.engine_dir / f"llm_{precision}.engine")
        self.embedding_layer = self.inner_model.language_model.get_input_embeddings()

        if self.metadata.get("precision") != precision:
            raise ValueError(
                f"Backbone engine precision is {self.metadata.get('precision')}, "
                f"but runtime precision is {precision}."
            )

        print("[TRT] TensorRT backbone initialized.")
        print(f"[TRT] precision={precision}")
        print(f"[TRT] engine_dir={self.engine_dir}")

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        """Reject batches or image grids that do not match the export contract."""
        expected_batch = int(self.metadata["batch_size"])
        if input_ids.shape[0] != expected_batch:
            raise ValueError(
                f"Backbone engines were exported for batch {expected_batch}, "
                f"but runtime batch is {input_ids.shape[0]}."
            )

        expected_grid = torch.tensor(
            self.metadata["vit_grid_thw"],
            dtype=image_grid_thw.dtype,
            device=image_grid_thw.device,
        )
        if image_grid_thw.shape != expected_grid.shape or not torch.equal(
            image_grid_thw,
            expected_grid,
        ):
            raise ValueError(
                "Runtime image_grid_thw does not match the grid baked into vit.engine: "
                f"runtime={image_grid_thw.tolist()}, expected={expected_grid.tolist()}"
            )

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final backbone features consumed by the action head."""
        self._validate_inputs(input_ids, image_grid_thw)

        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(pixel_values, dim=0)
        vision_dtype = self.vision_encoder.dtype_of("pixel_values")
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values.to(dtype=vision_dtype),
        )
        image_embeds = vision_outputs["image_embeds"]
        deepstack_tensor = vision_outputs.get("deepstack_features")
        deepstack = []
        if deepstack_tensor is not None and deepstack_tensor.numel() > 1:
            deepstack = list(deepstack_tensor.unbind(0))

        inputs_embeds = self.embedding_layer(input_ids)
        image_embeds = image_embeds.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        image_mask, _ = self.inner_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        position_ids, rope_deltas = self.inner_model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        self.inner_model.rope_deltas = rope_deltas

        if attention_mask.shape[0] > 1 and not torch.equal(
            attention_mask,
            attention_mask[0:1].expand_as(attention_mask),
        ):
            raise ValueError(
                "TensorRT backbone requires batch 1 or identical padding masks across the batch."
            )

        valid_mask = attention_mask[0] == 1
        if not bool(valid_mask.all()):
            inputs_embeds = inputs_embeds[:, valid_mask]
            attention_mask = attention_mask[:, valid_mask]
            position_ids = position_ids[:, :, valid_mask]
            visual_pos_masks = visual_pos_masks[:, valid_mask]

        language_dtype = self.language_model.dtype_of("inputs_embeds")
        payload = {
            "inputs_embeds": inputs_embeds.to(language_dtype),
            "attention_mask": attention_mask.to(torch.int64),
            "position_ids": position_ids.to(torch.int64),
        }

        if "visual_pos_masks" in self.language_model.input_names:
            payload["visual_pos_masks"] = visual_pos_masks
            for index, feature in enumerate(deepstack):
                name = f"deepstack_{index}"
                if name in self.language_model.input_names:
                    payload[name] = feature.to(language_dtype)

        missing = [name for name in self.language_model.input_names if name not in payload]
        if missing:
            raise RuntimeError(
                f"Language engine expects uncaptured runtime inputs: {missing}. "
                "Re-export with the same image configuration."
            )

        outputs = self.language_model(**payload)
        return outputs["embeddings"].to(torch.bfloat16)


class TensorRTActionHead:
    """TensorRT replacement for the diffusion action head.

    This mirrors the split:
      obs_encoder.engine
      action_encoder.engine
      dit_bf16.engine
      action_decoder.engine

    Lightweight glue, especially time_ins_embed, stays in PyTorch.
    """

    def __init__(
        self,
        action_header,
        engine_dir: str | Path,
        precision: str = "bf16",
    ):
        """Load the four action-head engines and retain timestep embedding."""
        self.action_header = action_header
        self.engine_dir = Path(engine_dir)
        self.precision = precision

        self.obs_encoder = Engine(self.engine_dir / "obs_encoder.engine")
        self.action_encoder = Engine(self.engine_dir / "action_encoder.engine")
        self.dit = Engine(self.engine_dir / f"dit_{precision}.engine")
        self.action_decoder = Engine(self.engine_dir / "action_decoder.engine")

        # Keep timestep embedding in PyTorch.
        self.time_ins_embed = action_header.time_ins_embed

        if precision == "bf16":
            self.engine_dtype = torch.bfloat16
        elif precision == "fp16":
            self.engine_dtype = torch.float16
        elif precision == "fp32":
            self.engine_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported TRT precision: {precision}")

        print("[TRT] TensorRT action head initialized.")
        print(f"[TRT] precision={precision}")
        print(f"[TRT] engine_dir={self.engine_dir}")

    def _maybe_add(
        self,
        payload: dict,
        engine: Engine,
        name: str,
        value: Optional[torch.Tensor],
    ) -> None:
        """Only add optional input if the engine actually expects it."""
        if value is not None and name in engine.input_names:
            payload[name] = value

    @torch.inference_mode()
    def encode_obs(
        self,
        vlm_hidden_states: torch.Tensor,
        states: torch.Tensor,
        traj2ds: Optional[torch.Tensor] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run obs_encoder.engine once before the denoising loop."""

        payload = {}

        if "vlm_hidden_states" in self.obs_encoder.input_names:
            payload["vlm_hidden_states"] = vlm_hidden_states.to(self.engine_dtype)
        elif "views" in self.obs_encoder.input_names:
            payload["views"] = vlm_hidden_states.to(self.engine_dtype)
        else:
            raise KeyError(
                f"obs_encoder input names do not include vlm_hidden_states/views: "
                f"{self.obs_encoder.input_names}"
            )

        if "states" in self.obs_encoder.input_names:
            payload["states"] = states.to(self.engine_dtype)
        elif "obs" in self.obs_encoder.input_names:
            payload["obs"] = states.to(self.engine_dtype)
        else:
            raise KeyError(
                f"obs_encoder input names do not include states/obs: "
                f"{self.obs_encoder.input_names}"
            )

        self._maybe_add(payload, self.obs_encoder, "traj2ds", traj2ds)
        self._maybe_add(payload, self.obs_encoder, "vlm_attn_mask", vlm_attn_mask)

        outputs = self.obs_encoder(**payload)

        if "obs_hidden_states" in outputs:
            obs_hidden_states = outputs["obs_hidden_states"]
        else:
            obs_hidden_states = outputs[self.obs_encoder.output_names[0]]

        obs_token_mask = outputs.get("obs_token_mask", None)

        return obs_hidden_states, obs_token_mask

    @torch.inference_mode()
    def step(
        self,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one denoising step through TRT action encoder + DiT + decoder."""

        # 1. action encoder
        action_payload = {}

        if "action_samples" in self.action_encoder.input_names:
            action_payload["action_samples"] = action_samples.to(self.engine_dtype)
        elif "noisy_actions" in self.action_encoder.input_names:
            action_payload["noisy_actions"] = action_samples.to(self.engine_dtype)
        else:
            # Fall back to the only input.
            action_payload[self.action_encoder.input_names[0]] = action_samples.to(
                self.engine_dtype
            )

        action_outputs = self.action_encoder(**action_payload)

        if "action_hidden_states" in action_outputs:
            action_hidden_states = action_outputs["action_hidden_states"]
        else:
            action_hidden_states = action_outputs[self.action_encoder.output_names[0]]

        # 2. timestep embedding stays in PyTorch glue.
        # Keep timestep dtype close to original PyTorch behavior.
        temb = self.time_ins_embed(timestep)
        temb = temb.to(self.engine_dtype)

        # 3. DiT / transformer blocks
        dit_payload = {}

        if "action_hidden_states" in self.dit.input_names:
            dit_payload["action_hidden_states"] = action_hidden_states
        else:
            dit_payload[self.dit.input_names[0]] = action_hidden_states

        if "obs_hidden_states" in self.dit.input_names:
            dit_payload["obs_hidden_states"] = obs_hidden_states.to(self.engine_dtype)

        if "temb" in self.dit.input_names:
            dit_payload["temb"] = temb

        self._maybe_add(dit_payload, self.dit, "obs_token_mask", obs_token_mask)

        dit_outputs = self.dit(**dit_payload)

        if "action_hidden_states_out" in dit_outputs:
            action_hidden_states_out = dit_outputs["action_hidden_states_out"]
        else:
            action_hidden_states_out = dit_outputs[self.dit.output_names[0]]

        # 4. action decoder
        decoder_payload = {}

        if "action_hidden_states" in self.action_decoder.input_names:
            decoder_payload["action_hidden_states"] = action_hidden_states_out
        elif "x" in self.action_decoder.input_names:
            decoder_payload["x"] = action_hidden_states_out
        else:
            decoder_payload[self.action_decoder.input_names[0]] = action_hidden_states_out

        if "temb" in self.action_decoder.input_names:
            decoder_payload["temb"] = temb
        elif "t" in self.action_decoder.input_names:
            decoder_payload["t"] = temb

        decoder_outputs = self.action_decoder(**decoder_payload)

        if "model_pred" in decoder_outputs:
            model_pred = decoder_outputs["model_pred"]
        elif "action" in decoder_outputs:
            model_pred = decoder_outputs["action"]
        else:
            model_pred = decoder_outputs[self.action_decoder.output_names[0]]

        return model_pred


def setup_action_head_tensorrt(
    model,
    engine_dir: str | Path,
    precision: str = "bf16",
    free_pytorch_modules: bool = False,
):
    """Attach the TensorRT action-head runtime to a loaded model."""

    model.trt_action_head = TensorRTActionHead(
        action_header=model.action_header,
        engine_dir=engine_dir,
        precision=precision,
    )
    model.use_trt_action_head = True

    if free_pytorch_modules:
        # Keep time_ins_embed because TRT runtime still uses it.
        if hasattr(model.action_header, "obs_proj"):
            del model.action_header.obs_proj
        if hasattr(model.action_header, "action_proj_in"):
            del model.action_header.action_proj_in
        if hasattr(model.action_header, "transformer_blocks"):
            del model.action_header.transformer_blocks
        if hasattr(model.action_header, "action_proj_out"):
            del model.action_header.action_proj_out

        torch.cuda.empty_cache()

    print("[TRT] model.use_trt_action_head=True")
    return model


def setup_backbone_tensorrt(
    model,
    engine_dir: str | Path,
    precision: str = "bf16",
    free_pytorch_modules: bool = False,
):
    """Attach the TensorRT backbone runtime to a loaded model."""
    model.trt_backbone = TensorRTBackbone(
        vlm_model=model.vlm_model,
        engine_dir=engine_dir,
        precision=precision,
    )
    model.use_trt_backbone = True

    if free_pytorch_modules:
        inner_model = model.vlm_model.model
        if hasattr(inner_model, "visual"):
            del inner_model.visual
        if hasattr(inner_model.language_model, "layers"):
            del inner_model.language_model.layers
        if hasattr(inner_model.language_model, "norm"):
            del inner_model.language_model.norm
        torch.cuda.empty_cache()

    print("[TRT] model.use_trt_backbone=True")
    return model


def setup_full_pipeline_tensorrt(
    model,
    engine_dir: str | Path,
    precision: str = "bf16",
    free_pytorch_modules: bool = False,
):
    """Attach both TensorRT backbone and action-head runtimes."""
    setup_backbone_tensorrt(
        model=model,
        engine_dir=engine_dir,
        precision=precision,
        free_pytorch_modules=free_pytorch_modules,
    )
    setup_action_head_tensorrt(
        model=model,
        engine_dir=engine_dir,
        precision=precision,
        free_pytorch_modules=free_pytorch_modules,
    )
    print("[TRT] Full pipeline initialized.")
    return model
