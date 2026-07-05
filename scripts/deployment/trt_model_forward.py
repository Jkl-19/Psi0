from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from scripts.deployment.trt_torch import Engine


class Psi0TRTActionHead:
    """TensorRT replacement for Psi0 ActionTransformerModel action head.

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

        print("[TEMP TRT] Psi0 TensorRT action head initialized.")
        print(f"[TEMP TRT] precision={precision}")
        print(f"[TEMP TRT] engine_dir={self.engine_dir}")

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


def setup_psi0_action_head_trt(
    model,
    engine_dir: str | Path,
    precision: str = "bf16",
    free_pytorch_modules: bool = False,
):
    """Attach TensorRT action-head runtime to an existing Psi0Model."""

    model.trt_action_head = Psi0TRTActionHead(
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

    print("[TEMP TRT] model.use_trt_action_head=True")
    return model
