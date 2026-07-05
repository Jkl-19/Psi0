import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from psi.config.config import LaunchConfig
from psi.models.psi0 import Psi0Model
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything

from scripts.deployment.export_psi0_action_head_onnx import (
    export_from_loaded_model_and_batch,
)


QWEN_TASK = "Spray the bowl and wipe it and stack it up."


def frame_to_pil(frame) -> Image.Image:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.array(frame)

    # LeRobot commonly gives C,H,W. Server/client payload uses H,W,C.
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return Image.fromarray(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt-step", type=str, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water",
    )
    parser.add_argument("--dataset-root", type=str, default="./data/real")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--task", type=str, default=QWEN_TASK)
    parser.add_argument("--output-dir", type=str, default="psi0_trt_deployment/onnx")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16"])
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device)

    # Load the same launch config style as the server.
    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = (run_dir / "run_config.json").open("r").read()
    launch_config = config_.model_validate_json(conf)
    seed_everything(launch_config.seed or 42)

    print("[export] Loading Psi0 model...")
    model = Psi0Model.from_pretrained(
        run_dir=run_dir,
        ckpt_step=args.ckpt_step,
        launch_config=launch_config,
        device=args.device,
    )
    model.to(device)
    model.eval()

    # Load same transform objects as server.
    model_transform = launch_config.data.transform.model
    maxmin = launch_config.data.transform.field

    dataset_root = Path(args.dataset_root) / args.dataset_repo_id
    print(f"[export] Loading dataset from {dataset_root}")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
    )

    batch = dataset[args.frame_idx]

    # Match mock_client/server image source.
    frame = batch["observation.images.egocentric"]
    img = frame_to_pil(frame)

    from torchvision.transforms import v2

    t = v2.Compose([
        model_transform.resize(),
        model_transform.center_crop(),
    ])

    # Match server shape: observations = [[processed_cam0_image]]
    observations = [[t(img)]]

    # Match mock_client state split and server normalization.
    raw_states = batch["states"]
    if hasattr(raw_states, "detach"):
        raw_states = raw_states.detach().cpu().numpy()
    else:
        raw_states = np.array(raw_states)

    hand_joints = raw_states[:14].astype(np.float32)
    arm_joints = raw_states[14:28].astype(np.float32)
    tmp_torso_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tmp_torso_height = np.array([0.75], dtype=np.float32)

    obs = np.concatenate(
        [hand_joints, arm_joints, tmp_torso_rpy, tmp_torso_height],
        axis=-1,
    )

    if maxmin.pad_state_dim != len(obs):
        obs = pad_to_len(obs, maxmin.pad_state_dim, dim=0)[0]

    obs = maxmin.normalize_state_func(obs)
    obs = obs[np.newaxis, np.newaxis, :]  # (1, 1, state_dim)

    states = torch.from_numpy(obs).to(device)

    print("[export] Exporting Psi0 action-head ONNX files...")
    export_from_loaded_model_and_batch(
        model=model,
        observations=observations,
        states=states,
        instructions=[args.task.lower()],
        traj2ds=None,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        precision=args.precision,
        batch_size=1,
    )

    print(f"[export] Done. ONNX files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
