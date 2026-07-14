#!/usr/bin/env python3
"""
Export action-head components to ONNX for TensorRT.
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.onnx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def verify_onnx_export(onnx_path: str) -> None:
    import onnx

    logger.info(f"Verifying ONNX export: {onnx_path}")
    onnx.checker.check_model(onnx_path)
    logger.info("ONNX model verified successfully.")

def verify_onnx_with_ort(
    onnx_path: str,
    pytorch_module: torch.nn.Module,
    sample_inputs: dict[str, torch.Tensor],
    output_names: list[str],
    label: str = "model",
) -> dict[str, float]:
    """Compare ONNX Runtime outputs against PyTorch using cosine similarity."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping ORT verification")
        return {}

    logger.info(f"ORT verification for {label}...")

    # Run PyTorch.
    with torch.inference_mode():
        pt_inputs = tuple(sample_inputs.values())
        pt_outputs = pytorch_module(*pt_inputs)

    if not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)

    # Run ONNX Runtime.
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ort_inputs = {
        name: tensor.cpu().numpy()
        for name, tensor in sample_inputs.items()
    }
    ort_outputs = session.run(output_names, ort_inputs)

    # Compare outputs.
    results = {}
    for index, name in enumerate(output_names):
        pt_flat = pt_outputs[index].float().flatten().cpu()
        ort_flat = torch.tensor(ort_outputs[index]).float().flatten()

        cosine = torch.nn.functional.cosine_similarity(
            pt_flat.unsqueeze(0),
            ort_flat.unsqueeze(0),
        ).item()

        results[name] = cosine
        logger.info(f"{name}: ORT vs PyTorch cosine = {cosine:.6f}")

    return results
    
class ActionHeadInputCapture:
    """Capture the real inputs entering action_header during predict_action."""

    def __init__(self):
        self.captured = False

        self.action_samples = None
        self.timestep = None
        self.vlm_hidden_states = None
        self.states = None
        self.traj2ds = None
        self.vlm_attn_mask = None

    def hook_fn(self, module, args, kwargs):
        if self.captured:
            return

        joint_kwargs = kwargs["joint_attention_kwargs"]

        self.action_samples = (
            joint_kwargs["action_hidden_embeds"].detach().cpu().clone()
        )
        self.vlm_hidden_states = joint_kwargs["views"].detach().cpu().clone()
        self.states = joint_kwargs["obs"].detach().cpu().clone()
        self.timestep = kwargs["timestep"].detach().cpu().clone()

        traj2ds = joint_kwargs.get("traj2ds")
        if traj2ds is not None:
            self.traj2ds = traj2ds.detach().cpu().clone()

        vlm_attn_mask = kwargs.get("vlm_attn_mask")
        if vlm_attn_mask is not None:
            self.vlm_attn_mask = vlm_attn_mask.detach().cpu().clone()

        self.captured = True

        logger.info("Captured action_head inputs:")
        logger.info(f"  action_samples:    {self.action_samples.shape} {self.action_samples.dtype}")
        logger.info(f"  vlm_hidden_states: {self.vlm_hidden_states.shape} {self.vlm_hidden_states.dtype}")
        logger.info(f"  states:            {self.states.shape} {self.states.dtype}")
        logger.info(f"  timestep:          {self.timestep.shape} {self.timestep.dtype}")

        if self.traj2ds is not None:
            logger.info(f"  traj2ds:           {self.traj2ds.shape} {self.traj2ds.dtype}")

        if self.vlm_attn_mask is not None:
            logger.info(f"  vlm_attn_mask:     {self.vlm_attn_mask.shape} {self.vlm_attn_mask.dtype}")


@torch.inference_mode()
def capture_action_head_inputs(
    model,
    observations,
    states,
    instructions,
    num_inference_steps: int,
    traj2ds=None,
):
    """Run one normal inference and capture action_header inputs."""
    model.eval()

    capture = ActionHeadInputCapture()

    hook_handle = model.action_header.register_forward_pre_hook(
        capture.hook_fn,
        with_kwargs=True,
    )

    try:
        _ = model.predict_action(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=num_inference_steps,
            traj2ds=traj2ds,
        )
    finally:
        hook_handle.remove()

    if not capture.captured:
        raise RuntimeError(
            "Failed to capture action_head inputs. "
            "Make sure model.predict_action(...) calls model.action_header."
        )

    return capture


def _make_export_tensors(captured, dtype: torch.dtype, batch_size: int):
    """Create ONNX tracing tensors using captured real shapes."""
    action_shape = (batch_size,) + captured.action_samples.shape[1:]
    vlm_shape = (batch_size,) + captured.vlm_hidden_states.shape[1:]
    state_shape = (batch_size,) + captured.states.shape[1:]
    timestep_shape = (batch_size,) + captured.timestep.shape[1:]

    action_samples = torch.randn(action_shape, dtype=dtype, device="cuda")
    vlm_hidden_states = torch.randn(vlm_shape, dtype=dtype, device="cuda")
    states = torch.randn(state_shape, dtype=dtype, device="cuda")

    if captured.timestep.dtype.is_floating_point:
        timestep = torch.ones(timestep_shape, dtype=dtype, device="cuda")
    else:
        timestep = torch.ones(timestep_shape, dtype=torch.int64, device="cuda")

    traj2ds = None
    if captured.traj2ds is not None:
        traj2ds_shape = (batch_size,) + captured.traj2ds.shape[1:]
        traj2ds = torch.randn(traj2ds_shape, dtype=dtype, device="cuda")

    vlm_attn_mask = None
    if captured.vlm_attn_mask is not None:
        mask_shape = (batch_size,) + captured.vlm_attn_mask.shape[1:]
        vlm_attn_mask = torch.ones(mask_shape, dtype=captured.vlm_attn_mask.dtype, device="cuda")

    return action_samples, vlm_hidden_states, states, timestep, traj2ds, vlm_attn_mask


def export_obs_encoder_to_onnx(
    action_header,
    captured,
    output_dir: str,
    precision: str = "bf16",
    batch_size: int = 1,
):
    logger.info("\n" + "=" * 80)
    logger.info("Exporting obs_encoder to ONNX")
    logger.info("=" * 80)

    dtype = _precision_to_dtype(precision)
    obs_proj = action_header.obs_proj.to(dtype=dtype, device="cuda").eval()

    _, vlm_hidden_states, states, _, traj2ds, vlm_attn_mask = _make_export_tensors(
        captured, dtype=dtype, batch_size=batch_size
    )

    has_traj2ds = traj2ds is not None
    has_vlm_attn_mask = vlm_attn_mask is not None

    class ObsEncoderWrapper(torch.nn.Module):
        def __init__(self, obs_proj, use_traj2ds: bool, use_vlm_attn_mask: bool):
            super().__init__()
            self.obs_proj = obs_proj
            self.use_traj2ds = use_traj2ds
            self.use_vlm_attn_mask = use_vlm_attn_mask

        def forward(self, vlm_hidden_states, states, traj2ds=None, vlm_attn_mask=None):
            obs_hidden_states, obs_token_mask = self.obs_proj(
                views=vlm_hidden_states,
                obs=states,
                traj2ds=traj2ds if self.use_traj2ds else None,
                text_embeddings=None,
                vlm_attn_mask=vlm_attn_mask if self.use_vlm_attn_mask else None,
            )

            if self.use_vlm_attn_mask:
                return obs_hidden_states, obs_token_mask

            return obs_hidden_states

    export_inputs = [vlm_hidden_states, states]
    input_names = ["vlm_hidden_states", "states"]
    output_names = ["obs_hidden_states"]

    if has_traj2ds:
        export_inputs.append(traj2ds)
        input_names.append("traj2ds")

    if has_vlm_attn_mask:
        export_inputs.append(vlm_attn_mask)
        input_names.append("vlm_attn_mask")
        output_names.append("obs_token_mask")

    output_path = os.path.join(output_dir, "obs_encoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    wrapper = ObsEncoderWrapper(obs_proj, has_traj2ds, has_vlm_attn_mask).eval().cuda()

    logger.info("Obs encoder export inputs:")
    for name, tensor in zip(input_names, export_inputs):
        logger.info(f"  {name}: {tensor.shape} {tensor.dtype}")

    # Keep only the sequence dimensions that can vary at runtime dynamic.
    # The server produced vlm_hidden_states with a different VLM token length
    # than the sample used for export, so dim 2 must not be frozen.
    dynamic_axes = {
        "vlm_hidden_states": {
            0: "batch",
            2: "vlm_seq_len",
        },
        "states": {
            0: "batch",
        },
        "obs_hidden_states": {
            0: "batch",
            1: "obs_seq_len",
        },
    }

    if has_traj2ds:
        dynamic_axes["traj2ds"] = {
            0: "batch",
        }

    if has_vlm_attn_mask:
        dynamic_axes["vlm_attn_mask"] = {
            0: "batch",
            1: "vlm_seq_len",
        }
        dynamic_axes["obs_token_mask"] = {
            0: "batch",
            1: "obs_seq_len",
        }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    verify_onnx_export(output_path)
    return output_path


def export_action_encoder_to_onnx(
    action_header,
    captured,
    output_dir: str,
    precision: str = "bf16",
    batch_size: int = 1,
):
    """Export action_proj_in as action_encoder.onnx"""
    logger.info("\n" + "=" * 80)
    logger.info("Exporting action_encoder to ONNX")
    logger.info("=" * 80)

    dtype = _precision_to_dtype(precision)
    action_proj_in = action_header.action_proj_in.to(dtype=dtype, device="cuda").eval()

    action_samples, _, _, _, _, _ = _make_export_tensors(
        captured, dtype=dtype, batch_size=batch_size
    )

    output_path = os.path.join(output_dir, "action_encoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  action_samples: {action_samples.shape} {action_samples.dtype}")

    with torch.inference_mode():
        torch.onnx.export(
            action_proj_in,
            (action_samples,),
            output_path,
            input_names=["action_samples"],
            output_names=["action_hidden_states"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    verify_onnx_export(output_path)
    return output_path


def _get_intermediate_shapes(action_header, captured, dtype: torch.dtype, batch_size: int):
    """Run PyTorch submodules once to derive DiT and decoder shapes."""
    action_samples, vlm_hidden_states, states, timestep, traj2ds, vlm_attn_mask = (
        _make_export_tensors(captured, dtype=dtype, batch_size=batch_size)
    )

    action_header = action_header.to(dtype=dtype, device="cuda").eval()

    with torch.inference_mode():
        action_hidden_states = action_header.action_proj_in(action_samples)

        obs_hidden_states, obs_token_mask = action_header.obs_proj(
            views=vlm_hidden_states,
            obs=states,
            traj2ds=traj2ds,
            text_embeddings=None,
            vlm_attn_mask=vlm_attn_mask,
        )

        temb = action_header.time_ins_embed(timestep)

    return action_hidden_states, obs_hidden_states, obs_token_mask, temb


def export_dit_to_onnx(
    action_header,
    captured,
    output_dir: str,
    precision: str = "bf16",
    batch_size: int = 1,
):
    logger.info("\n" + "=" * 80)
    logger.info("Exporting DiT core to ONNX")
    logger.info("=" * 80)

    dtype = _precision_to_dtype(precision)
    action_header = action_header.to(dtype=dtype, device="cuda").eval()

    action_hidden_states, obs_hidden_states, obs_token_mask, temb = _get_intermediate_shapes(
        action_header, captured, dtype=dtype, batch_size=batch_size
    )

    has_obs_token_mask = obs_token_mask is not None

    class DiTWrapper(torch.nn.Module):
        def __init__(self, transformer_blocks, use_obs_token_mask: bool):
            super().__init__()
            self.transformer_blocks = transformer_blocks
            self.use_obs_token_mask = use_obs_token_mask

        def forward(self, action_hidden_states, obs_hidden_states, temb, obs_token_mask=None):
            mask = obs_token_mask if self.use_obs_token_mask else None

            for block in self.transformer_blocks:
                action_hidden_states, obs_hidden_states = block(
                    action_hidden_states=action_hidden_states,
                    obs_hidden_states=obs_hidden_states,
                    temb=temb,
                    obs_token_mask=mask,
                )

            return action_hidden_states

    wrapper = DiTWrapper(
        transformer_blocks=action_header.transformer_blocks,
        use_obs_token_mask=has_obs_token_mask,
    ).eval().cuda()

    export_inputs = [action_hidden_states, obs_hidden_states, temb]
    input_names = ["action_hidden_states", "obs_hidden_states", "temb"]

    if has_obs_token_mask:
        export_inputs.append(obs_token_mask)
        input_names.append("obs_token_mask")

    precision_tag = precision
    output_path = os.path.join(output_dir, f"dit_{precision_tag}.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("DiT export inputs:")
    for name, tensor in zip(input_names, export_inputs):
        logger.info(f"  {name}: {tensor.shape} {tensor.dtype}")

    # obs_hidden_states comes from obs_encoder, so its token sequence length
    # must stay dynamic whenever vlm_seq_len is dynamic.
    dynamic_axes = {
        "action_hidden_states": {
            0: "batch",
        },
        "obs_hidden_states": {
            0: "batch",
            1: "obs_seq_len",
        },
        "temb": {
            0: "batch",
        },
        "action_hidden_states_out": {
            0: "batch",
        },
    }

    if has_obs_token_mask:
        dynamic_axes["obs_token_mask"] = {
            0: "batch",
            1: "obs_seq_len",
        }

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            tuple(export_inputs),
            output_path,
            input_names=input_names,
            output_names=["action_hidden_states_out"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    verify_onnx_export(output_path)
    return output_path


def export_action_decoder_to_onnx(
    action_header,
    captured,
    output_dir: str,
    precision: str = "bf16",
    batch_size: int = 1,
):
    """Export action_proj_out as action_decoder.onnx."""
    logger.info("\n" + "=" * 80)
    logger.info("Exporting action_decoder to ONNX")
    logger.info("=" * 80)

    dtype = _precision_to_dtype(precision)
    action_header = action_header.to(dtype=dtype, device="cuda").eval()

    action_hidden_states, _, _, temb = _get_intermediate_shapes(
        action_header, captured, dtype=dtype, batch_size=batch_size
    )

    decoder = action_header.action_proj_out.to(dtype=dtype, device="cuda").eval()

    class ActionDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, action_hidden_states, temb):
            return self.decoder(x=action_hidden_states, t=temb)

    wrapper = ActionDecoderWrapper(decoder).eval().cuda()

    output_path = os.path.join(output_dir, "action_decoder.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"  action_hidden_states: {action_hidden_states.shape} {action_hidden_states.dtype}")
    logger.info(f"  temb:                 {temb.shape} {temb.dtype}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (action_hidden_states, temb),
            output_path,
            input_names=["action_hidden_states", "temb"],
            output_names=["model_pred"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    verify_onnx_export(output_path)
    return output_path


def export_action_head_to_onnx(
    model,
    captured,
    output_dir: str,
    precision: str = "bf16",
    batch_size: int = 1,
):
    """Export all action-head components to ONNX."""
    action_header = model.action_header

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "obs_encoder": export_obs_encoder_to_onnx(
            action_header, captured, output_dir, precision, batch_size
        ),
        "action_encoder": export_action_encoder_to_onnx(
            action_header, captured, output_dir, precision, batch_size
        ),
        "dit": export_dit_to_onnx(
            action_header, captured, output_dir, precision, batch_size
        ),
        "action_decoder": export_action_decoder_to_onnx(
            action_header, captured, output_dir, precision, batch_size
        ),
    }

    logger.info("Exported action-head ONNX files:")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    return paths


def export_from_loaded_model_and_batch(
    model,
    observations,
    states,
    instructions,
    traj2ds,
    output_dir: str,
    num_inference_steps: int = 4,
    precision: str = "bf16",
    batch_size: int = 1,
):
    """Capture inputs from one real batch, then export all action-head ONNX files."""
    captured = capture_action_head_inputs(
        model=model,
        observations=observations,
        states=states,
        instructions=instructions,
        num_inference_steps=num_inference_steps,
        traj2ds=traj2ds,
    )

    return export_action_head_to_onnx(
        model=model,
        captured=captured,
        output_dir=output_dir,
        precision=precision,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="psi0_trt_deployment/onnx")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    raise RuntimeError(
        "Call export_from_loaded_model_and_batch(...) from mock_client.py first, "
        "because mock_client.py already constructs model, observations, states, "
        "instructions, and traj2ds.\n\n"
        "Example:\n"
        "from scripts.deployment.export_psi0_action_head_onnx import "
        "export_from_loaded_model_and_batch\n\n"
        "export_from_loaded_model_and_batch(\n"
        "    model=model,\n"
        "    observations=observations,\n"
        "    states=states,\n"
        "    instructions=instructions,\n"
        "    traj2ds=traj2ds,\n"
        f"    output_dir='{args.output_dir}',\n"
        f"    num_inference_steps={args.num_inference_steps},\n"
        f"    precision='{args.precision}',\n"
        f"    batch_size={args.batch_size},\n"
        ")\n"
    )


if __name__ == "__main__":
    main()
