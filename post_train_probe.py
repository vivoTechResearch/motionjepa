#!/usr/bin/env python3
"""Frozen-encoder post-training probes for LeWorld benchmark checkpoints.

The state head always predicts one concatenated vector.  Scalar planar angles
are encoded as (cos, sin): this is a continuous loss at +/- pi and allows the
reported metric to be decoded back to an angle.  3-D rotations can be added as
``rotation_6d`` targets when a dataset exposes them (the helper is included
below); it follows the first-two-columns rotation representation.
"""
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils import get_img_preprocessor
from wood_augmentation import add_static_episode_wood_torch


# These are intentionally per-environment, rather than assuming a shared
# observation layout.  They match the fields saved by the benchmark loaders.
SPECS = {
    "pusht": {
        "dataset": "pusht_expert_train.h5", "keys": ["pixels", "state", "episode_idx"],
        "targets": [("agent_xy", "vector", "state", [0, 1]),
                    ("object_xy", "vector", "state", [2, 3]),
                    ("object_angle", "angle", "state", 4)],
    },
    "tworoom": {
        "dataset": "tworoom.h5", "keys": ["pixels", "proprio", "pos_target", "ep_idx"],
        # Appendix F.2 reports only agent position for TwoRoom.
        "targets": [("agent_xy", "vector", "proprio", None)],
    },
    "ogcube": {
        # This is the local benchmark filename (not the HF repo identifier).
        "dataset": "cube_single_expert.h5",
        "keys": ["pixels", "proprio_effector_pos", "proprio_effector_yaw",
                 "privileged_block_0_pos", "privileged_block_0_yaw",
                 "ep_idx"],
        "targets": [("agent_xyz", "vector", "proprio_effector_pos", None),
                    ("object_xyz", "vector", "privileged_block_0_pos", None),
                    ("agent_yaw", "angle", "proprio_effector_yaw", 0),
                    ("object_yaw", "angle", "privileged_block_0_yaw", 0)],
    },
    "reacher": {
        "dataset": "reacher.h5", "keys": ["pixels", "qpos", "finger_pos", "target_pos", "ep_idx"],
        "targets": [("joint_0_angle", "angle", "qpos", 0),
                    ("joint_1_angle", "angle", "qpos", 1),
                    ("agent_finger_xy", "vector", "finger_pos", None),
                    ("object_target_xy", "vector", "target_pos", None)],
    },
}


# Mirrors scripts/_method_benchmark_loop.sh for each wood-trained benchmark.
WOOD_CONFIGS = {
    "pusht": {"seed": 3072, "texture_path": "assets/wood_texture.png", "mask_mode": "white", "white_threshold": 245, "background_tolerance": 8, "static_background_path": None},
    "tworoom": {"seed": 3072, "texture_path": "assets/wood_texture.png", "mask_mode": "white", "white_threshold": 245, "background_tolerance": 8, "static_background_path": None},
    "ogcube": {"seed": 3072, "texture_path": "assets/wood_texture.png", "mask_mode": "static", "white_threshold": 245, "background_tolerance": 8, "static_background_path": "assets/cube_static_background.npy"},
    "reacher": {"seed": 3072, "texture_path": "assets/wood_texture.png", "mask_mode": "static", "white_threshold": 245, "background_tolerance": 8, "static_background_path": "assets/reacher_static_background.npy"},
}


class StateProbe(nn.Module):
    def __init__(self, dim, output_dim, kind, hidden_dim):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, output_dim)
        else:
            self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class ProbeImageDecoder(nn.Module):
    """Probe-only decoder: training decoder architecture without output sigmoid."""
    def __init__(self, input_dim, img_size=224, hidden_dim=512, base_channels=64):
        super().__init__()
        self.img_size = img_size
        self.start_size = 7
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, base_channels * self.start_size * self.start_size), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(base_channels // 8, 3, kernel_size=3, padding=1),
        )

    def forward(self, z):
        shape = z.shape[:-1]
        x = self.fc(z.reshape(-1, z.size(-1)))
        x = x.view(x.size(0), -1, self.start_size, self.start_size)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x.view(*shape, 3, self.img_size, self.img_size)


@dataclass
class TargetInfo:
    name: str
    kind: str
    start: int
    end: int
    raw: torch.Tensor


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--benchmark", choices=SPECS, required=True)
    p.add_argument("--checkpoint", required=True, help="save_pretrained run name, or a directory containing weights.pt")
    p.add_argument("--checkpoint-cache-dir", default=None, help="cache root for a save_pretrained run name")
    p.add_argument("--dataset-cache-dir", default=None, help="dataset cache root (defaults to LOCAL_DATASET_DIR, then STABLEWM_HOME)")
    p.add_argument("--output-root", default="probe_outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--probe", choices=("both",), default="both", help="trains linear and MLP probes together")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-loss-weight", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--log-every-n-steps", type=int, default=200)
    p.add_argument("--image-every-n-steps", type=int, default=1000)
    p.add_argument("--no-wood", action="store_true", help="disable the training-time wood augmentation")
    p.add_argument("--train-split", type=float, default=.9)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_encoder(args, device):
    checkpoint = Path(args.checkpoint).expanduser()
    if checkpoint.is_dir() and (checkpoint / "weights.pt").exists():
        model = swm.wm.utils.load_pretrained(checkpoint.name, cache_dir=str(checkpoint.parent.parent))
    else:
        model = swm.wm.utils.load_pretrained(args.checkpoint, cache_dir=args.checkpoint_cache_dir)
    model = model.to(device).eval().requires_grad_(False)
    # The paper defines z as the ViT CLS token *after* the learned projection
    # (Appendix F.2 / Sec. 3.1).  Keep the complete frozen model so probing
    # uses exactly that representation rather than the pre-projection CLS.
    return model


def rotation_6d(matrix):
    """Use this for future 3-D orientation targets: first two matrix columns."""
    return matrix[..., :3, :2].reshape(*matrix.shape[:-2], 6)


def display_pixels(pixels):
    """Convert ImageNet-normalized tensors to [0, 1] for TensorBoard."""
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (pixels * std + mean).clamp(0, 1)


def build_targets(batch, spec):
    values, info, offset = [], [], 0
    for name, kind, column, indices in spec["targets"]:
        raw = batch[column].float()
        if raw.ndim == 2:
            raw = raw.unsqueeze(-1)
        raw = raw[:, 0]  # probe the encoded first frame only
        if kind == "vector":
            raw = raw if indices is None else raw[:, indices]
            value = raw
        elif kind == "angle":
            raw = raw[:, int(indices)]
            value = torch.stack((raw.cos(), raw.sin()), dim=-1)
        elif kind == "rotation_6d":
            raw, value = raw, rotation_6d(raw.reshape(raw.size(0), 3, 3))
        else:
            raise ValueError(f"Unknown target kind: {kind}")
        values.append(value)
        info.append(TargetInfo(name, kind, offset, offset + value.size(-1), raw))
        offset += value.size(-1)
    return torch.cat(values, dim=-1), info


def target_statistics(dataset, spec):
    """Normalize vector targets, while preserving unit-circle angle targets."""
    pieces, angle_slices, offset = [], [], 0
    for _, kind, column, indices in spec["targets"]:
        raw = torch.as_tensor(np.asarray(dataset.get_col_data(column))).float()
        if raw.ndim == 1:
            raw = raw.unsqueeze(-1)
        if kind == "vector":
            pieces.append(raw if indices is None else raw[:, indices])
        elif kind == "angle":
            angle = raw[:, int(indices)]
            pieces.append(torch.stack((angle.cos(), angle.sin()), dim=-1))
            angle_slices.append(slice(offset, offset + 2))
        elif kind == "rotation_6d":
            pieces.append(rotation_6d(raw.reshape(raw.size(0), 3, 3)))
        else:
            raise ValueError(f"Unknown target kind: {kind}")
        offset += pieces[-1].size(-1)
    targets = torch.cat(pieces, dim=-1)
    finite = torch.isfinite(targets).all(dim=-1)
    if not finite.any():
        raise ValueError("No finite probe labels found")
    targets = targets[finite]
    mean, std = targets.mean(0), targets.std(0).clamp_min(1e-6)
    # cos/sin are already bounded, continuous, and geometrically meaningful.
    # Do not independently z-score them: use the direct unit-circle MSE.
    for angle_slice in angle_slices:
        mean[angle_slice] = 0.0
        std[angle_slice] = 1.0
    return mean, std


@torch.no_grad()
def encode(model, pixels):
    if hasattr(model, "_encode_pixels"):
        embedding, _ = model._encode_pixels(pixels)
        return embedding
    out = model.encoder(pixels, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    return model.projector(cls) if getattr(model, "projector", None) is not None else cls


def epoch(loader, encoder, state_probes, decoder, optimizer, target_mean, target_std, spec, device, image_weight, train,
          writer=None, global_step=0, log_every_n_steps=50, image_every_n_steps=250, epoch_index=None,
          wood_cfg=None):
    modules = (*state_probes.values(), decoder)
    for module in modules:
        module.train(train)
    sums, count, pearson_sums = {}, 0, {}
    split = "train" if train else "val"
    progress = tqdm(loader, desc=f"[probe] {split} {epoch_index}", unit="batch", mininterval=5.0, leave=True)
    for batch in progress:
        pixels = batch["pixels"][:, 0].to(device).float()
        if wood_cfg is not None:
            episode_idx = batch.get("episode_idx", batch.get("ep_idx"))
            pixels = add_static_episode_wood_torch(pixels, episode_idx, wood_cfg)
        target, info = build_targets(batch, spec)
        target = target.to(device)
        target_normalized = (target - target_mean) / target_std
        with torch.no_grad():
            z = encode(encoder, pixels)
        predictions = {name: probe(z) for name, probe in state_probes.items()}
        reconstruction = decoder(z)
        state_losses = {name: F.mse_loss(pred, target_normalized) for name, pred in predictions.items()}
        state_loss = torch.stack(tuple(state_losses.values())).mean()
        # The unconstrained probe decoder is supervised with ImageNet-
        # un-normalized pixels using MSE.
        image_target = display_pixels(pixels)
        image_loss = F.mse_loss(reconstruction, image_target)
        loss = state_loss + image_weight * image_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % log_every_n_steps == 0:
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=False)
            if writer is not None and global_step % image_every_n_steps == 0:
                # Each item is [input | decoded reconstruction].
                panel = torch.cat((image_target[:4], reconstruction[:4].clamp(0, 1)), dim=-1)
                writer.add_images("train/input_reconstruction", panel, global_step)
        n = pixels.size(0); count += n
        sums["loss"] = sums.get("loss", 0.) + loss.detach().item() * n
        sums["state_mse"] = sums.get("state_mse", 0.) + state_loss.detach().item() * n
        sums["image_mse"] = sums.get("image_mse", 0.) + image_loss.detach().item() * n
        batch_task_metrics = {}
        for probe_name, pred in predictions.items():
            sums[f"{probe_name}/state_mse"] = sums.get(f"{probe_name}/state_mse", 0.) + state_losses[probe_name].detach().item() * n
            pred_raw = pred * target_std + target_mean
            for item in info:
                key = f"{probe_name}/{item.name}"
                pp, tt = pred_raw[:, item.start:item.end], target[:, item.start:item.end]
                # This is the exact per-target contribution to the optimized
                # concatenated state MSE (angle targets occupy cos/sin channels).
                target_mse = F.mse_loss(
                    pred[:, item.start:item.end], target_normalized[:, item.start:item.end]
                )
                sums[f"{key}_mse"] = sums.get(f"{key}_mse", 0.) + target_mse.detach().item() * n
                batch_task_metrics[f"{key}_mse"] = target_mse.detach().item()
                # Preserve an interpretable raw-coordinate MSE alongside the
                # normalized MSE used for cross-property comparison.
                raw_mse = F.mse_loss(pp, tt)
                sums[f"{key}_raw_mse"] = sums.get(f"{key}_raw_mse", 0.) + raw_mse.detach().item() * n
                batch_task_metrics[f"{key}_raw_mse"] = raw_mse.detach().item()
                x, y = pp.detach().float().reshape(-1), tt.detach().float().reshape(-1)
                stat = pearson_sums.setdefault(key, [0, 0., 0., 0., 0., 0.])
                stat[0] += x.numel(); stat[1] += x.sum().item(); stat[2] += y.sum().item()
                stat[3] += x.square().sum().item(); stat[4] += y.square().sum().item(); stat[5] += (x * y).sum().item()
                if item.kind == "angle":
                    pangle = torch.atan2(pp[:, 1], pp[:, 0])
                    error = torch.atan2(torch.sin(pangle - item.raw.to(device)), torch.cos(pangle - item.raw.to(device))).abs()
                    sums[f"{key}_angle_mae_deg"] = sums.get(f"{key}_angle_mae_deg", 0.) + torch.rad2deg(error).sum().item()
                    batch_task_metrics[f"{key}_angle_mae_deg"] = torch.rad2deg(error).mean().item()
                else:
                    sums[f"{key}_mae"] = sums.get(f"{key}_mae", 0.) + (pp - tt).abs().mean().item() * n
                    # Report Euclidean distance for XY/XYZ agent/object positions.
                    sums[f"{key}_position_l2"] = sums.get(f"{key}_position_l2", 0.) + (pp - tt).norm(dim=-1).sum().item()
                    batch_task_metrics[f"{key}_position_l2"] = (pp - tt).norm(dim=-1).mean().item()
        if train and writer is not None and global_step % log_every_n_steps == 0:
            writer.add_scalar("train/loss", loss.detach(), global_step)
            writer.add_scalar("train/state_mse", state_loss.detach(), global_step)
            writer.add_scalar("train/image_mse", image_loss.detach(), global_step)
            for probe_name, probe_loss in state_losses.items():
                writer.add_scalar(f"train/{probe_name}/state_mse", probe_loss.detach(), global_step)
            for name, value in batch_task_metrics.items():
                writer.add_scalar(f"train/probe/{name}", value, global_step)
    for name, (numel, sum_x, sum_y, sum_x2, sum_y2, sum_xy) in pearson_sums.items():
        covariance = numel * sum_xy - sum_x * sum_y
        scale = max((numel * sum_x2 - sum_x * sum_x) * (numel * sum_y2 - sum_y * sum_y), 0.0) ** 0.5
        sums[f"{name}_pearson_r"] = covariance / scale if scale > 1e-12 else 0.0
    progress.close()
    return {key: value if key.endswith("_pearson_r") else value / count for key, value in sums.items()}, global_step


def format_task_metrics(metrics):
    """Compact physical-space metrics for the terminal epoch summary."""
    selected = [(name, value) for name, value in metrics.items()
                if name.endswith("_position_l2") or name.endswith("_angle_mae_deg")
                or name.endswith("_angle_norm_error") or name.endswith("_mse")
                or name.endswith("_pearson_r")]
    return " | ".join(f"{name}={value:.4f}" for name, value in selected)


def probe_lr(epoch):
    """Fixed six-epoch probe schedule: fit, then refine."""
    if epoch <= 4:
        return 1e-4
    return 1e-5


def main():
    args = parse_args(); spec = SPECS[args.benchmark]
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    dataset_cache_dir = args.dataset_cache_dir or os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME")
    dataset = swm.data.load_dataset(spec["dataset"], transform=None, cache_dir=dataset_cache_dir, num_steps=1, frameskip=1,
                                    keys_to_load=spec["keys"], keys_to_cache=[k for k in spec["keys"] if k != "pixels"])
    dataset.transform = spt.data.transforms.Compose(get_img_preprocessor("pixels", "pixels", args.image_size))
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [args.train_split, 1 - args.train_split], generator=generator)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    # Statistics and output dimensionality are derived from actual benchmark fields.
    target_mean, target_std = target_statistics(dataset, spec)
    target_mean = target_mean.to(device); target_std = target_std.to(device)
    encoder = load_encoder(args, device)
    first_batch = next(iter(train_loader))
    first_pixels = first_batch["pixels"][:, 0].to(device).float()
    wood_cfg = None if args.no_wood else WOOD_CONFIGS[args.benchmark]
    if wood_cfg is not None:
        first_episode_idx = first_batch.get("episode_idx", first_batch.get("ep_idx"))
        first_pixels = add_static_episode_wood_torch(first_pixels, first_episode_idx, wood_cfg)
    latent_dim = encode(encoder, first_pixels).size(-1)
    state_probes = nn.ModuleDict({
        "linear": StateProbe(latent_dim, target_mean.numel(), "linear", args.hidden_dim),
        "mlp": StateProbe(latent_dim, target_mean.numel(), "mlp", args.hidden_dim),
    }).to(device)
    decoder = ProbeImageDecoder(input_dim=latent_dim, img_size=args.image_size, hidden_dim=512, base_channels=64).to(device)
    optimizer = torch.optim.AdamW([
        {"params": state_probes.parameters(), "lr": 1e-4},
        {"params": decoder.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    run_name = args.run_name or f"{args.benchmark}_linear_mlp_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out = Path(args.output_root) / run_name; out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    writer = SummaryWriter(out / "tensorboard")
    history = []
    global_step = 0
    for index in range(1, args.epochs + 1):
        current_lr = probe_lr(index)
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.param_groups[1]["lr"] = 10.0 * current_lr
        train_metrics, global_step = epoch(
            train_loader, encoder, state_probes, decoder, optimizer, target_mean, target_std, spec, device,
            args.image_loss_weight, True, writer, global_step, args.log_every_n_steps, args.image_every_n_steps, index,
            wood_cfg,
        )
        with torch.no_grad():
            val_metrics, _ = epoch(
                val_loader, encoder, state_probes, decoder, optimizer, target_mean, target_std, spec, device,
                args.image_loss_weight, False,
                wood_cfg=wood_cfg,
            )
        record = {"epoch": index, "train": train_metrics, "val": val_metrics}; history.append(record)
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                writer.add_scalar(f"{split}/epoch_{name}", value, index)
                if (name.endswith("_position_l2") or name.endswith("_angle_mae_deg")
                        or name.endswith("_angle_norm_error") or name.endswith("_probe_mse")
                        or name.endswith("_mse") or name.endswith("_pearson_r")):
                    writer.add_scalar(f"{split}/probe/{name}", value, index)
        writer.add_scalar("train/lr", current_lr, index)
        writer.add_scalar("train/decoder_lr", 10.0 * current_lr, index)
        summary = (
            f"[probe] epoch {index}/{args.epochs} | lr={current_lr:.3e} | "
            f"train_loss={train_metrics['loss']:.5f} | val_loss={val_metrics['loss']:.5f} | "
            f"linear_val_mse={val_metrics['linear/state_mse']:.5f} | "
            f"mlp_val_mse={val_metrics['mlp/state_mse']:.5f}"
        )
        task_metrics = format_task_metrics(val_metrics)
        print(f"{summary}\n[probe] val metrics | {task_metrics}", flush=True)
    torch.save({"state_probes": state_probes.state_dict(), "image_decoder": decoder.state_dict(), "target_mean": target_mean.cpu(), "target_std": target_std.cpu(), "spec": spec, "args": vars(args), "lr_schedule": "probe epochs 1-4: 1e-4; 5+: 1e-5; decoder: 10x probe LR"}, out / "probe.pt")
    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
