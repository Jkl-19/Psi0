#!/usr/bin/env python3
"""Load one dataset frame and export every TensorRT pipeline component."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from psi.config.config import LaunchConfig
from psi.models.psi0 import Psi0Model
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything
from scripts.deployment.export_full_pipeline_onnx import (
    export_from_loaded_model_and_batch,
)


DEFAULT_TASK = "Remove the cap, turn on the faucet, and fill the bottle with water."


def frame_to_pil(frame) -> Image.Image:
    """Convert a dataset frame in CHW or HWC layout to a PIL image."""
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def load_export_batch(args, launch_config, device):
    """Build the same image and state tensors used by deployment inference."""
    dataset_root = Path(args.dataset_root) / args.dataset_repo_id
    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=dataset_root)
    batch = dataset[args.frame_idx]

    from torchvision.transforms import v2

    model_transform = launch_config.data.transform.model
    transform = v2.Compose(
        [
            model_transform.resize(),
            model_transform.center_crop(),
        ]
    )
    observations = [[transform(frame_to_pil(batch["observation.images.egocentric"]))]]

    raw_states = batch["states"]
    if hasattr(raw_states, "detach"):
        raw_states = raw_states.detach().cpu().numpy()
    else:
        raw_states = np.asarray(raw_states)

    hand_joints = raw_states[:14].astype(np.float32)
    arm_joints = raw_states[14:28].astype(np.float32)
    torso_rpy = np.zeros(3, dtype=np.float32)
    torso_height = np.array([0.75], dtype=np.float32)
    state = np.concatenate([hand_joints, arm_joints, torso_rpy, torso_height])

    state_transform = launch_config.data.transform.field
    if state_transform.pad_state_dim != len(state):
        state = pad_to_len(state, state_transform.pad_state_dim, dim=0)[0]
    state = state_transform.normalize_state_func(state)
    states = torch.from_numpy(state[None, None]).to(device)
    return observations, states


def main() -> None:
    """Parse CLI arguments, load a checkpoint, and export ONNX files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt-step", required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default="Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water",
    )
    parser.add_argument("--dataset-root", default="./data/real")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-dir", default="psi0_trt_deployment/onnx")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--vision-precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Export only ViT and LLM graphs and reuse existing action-head ONNX files.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ONNX export.")

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    config_template: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    launch_config = config_template.model_validate_json(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    seed_everything(launch_config.seed or 42)

    print("[export] Loading model checkpoint...")
    model = Psi0Model.from_pretrained(
        run_dir=run_dir,
        ckpt_step=args.ckpt_step,
        launch_config=launch_config,
        device=args.device,
    )
    model.to(device).eval()

    print("[export] Preparing one real dataset frame...")
    observations, states = load_export_batch(args, launch_config, device)

    print("[export] Exporting full pipeline...")
    paths = export_from_loaded_model_and_batch(
        model=model,
        observations=observations,
        states=states,
        instructions=[args.task.lower()],
        traj2ds=None,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        precision=args.precision,
        vision_precision=args.vision_precision,
        include_action_head=not args.backbone_only,
    )
    for name, path in paths.items():
        print(f"[export] {name}: {path}")


if __name__ == "__main__":
    main()
