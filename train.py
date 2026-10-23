import ast
import gc
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor
from wood_augmentation import add_static_episode_wood_torch, wood_enabled

try:
    mp.set_start_method("forkserver", force=True)
except RuntimeError:
    pass


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def make_warmup_cosine_scheduler(warmup_fraction):
    def scheduler(optimizer, module):
        total_steps = getattr(module.trainer, "estimated_stepping_batches", 1)
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=max(1, int(float(warmup_fraction) * total_steps)),
            max_steps=total_steps,
            warmup_start_lr=0.0,
            eta_min=0.0,
        )

    return scheduler


def _slug(text):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text)).strip("_")


def build_run_dir(cfg):
    if cfg.get("resume") and cfg.resume.get("enabled", False):
        if not cfg.resume.get("run_dir"):
            raise ValueError("resume.enabled=true requires resume.run_dir")
        run_dir = Path(cfg.resume.run_dir).expanduser()
        if not run_dir.exists():
            raise FileNotFoundError(f"resume.run_dir does not exist: {run_dir}")
        ensure_run_subdirs(run_dir, cfg)
        return run_dir

    run_name = cfg.experiment.name if cfg.experiment.name else "motion-jepa"
    enabled_options = []
    if cfg.get("wood") and cfg.wood.get("enabled", False):
        enabled_options.append("wood")
    if cfg.get("inverse_action_raw") and cfg.inverse_action_raw.get("enabled", False):
        enabled_options.append("invactionraw")
    if cfg.get("sigreg_flat") and cfg.sigreg_flat.get("enabled", False):
        enabled_options.append("sigregflat")
    if cfg.get("diff_image") and cfg.diff_image.get("enabled", False):
        enabled_options.append("diffimg")
    if enabled_options:
        run_name = f"{run_name}_{'_'.join(enabled_options)}"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(cfg.experiment.output_root) / f"{stamp}_{_slug(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "tensorboard", "visualizations", cfg.validation.success_rate_output_dir):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_run_subdirs(run_dir, cfg):
    for name in ("checkpoints", "tensorboard", "visualizations", cfg.validation.success_rate_output_dir):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def resolve_resume_checkpoint(cfg, run_dir):
    if not (cfg.get("resume") and cfg.resume.get("enabled", False)):
        return None
    if cfg.resume.get("checkpoint_path"):
        ckpt_path = Path(cfg.resume.checkpoint_path).expanduser()
    else:
        checkpoint_dir = Path(cfg.resume.get("checkpoint_dir") or (run_dir / "checkpoints")).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = run_dir / checkpoint_dir
        if cfg.resume.get("epoch") is not None:
            ckpt_path = checkpoint_dir / f"epoch_epoch={int(cfg.resume.epoch):04d}.ckpt"
            if not ckpt_path.exists():
                ckpt_path = checkpoint_dir / f"epoch_{int(cfg.resume.epoch):04d}.ckpt"
        else:
            ckpt_path = checkpoint_dir / "last.ckpt"
            if not ckpt_path.exists():
                candidates = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
                if not candidates:
                    raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
                ckpt_path = candidates[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_path}")
    return str(ckpt_path)


def save_code_snapshot(run_dir):
    src = Path(__file__).resolve().parent
    dst = run_dir / "code_snapshot"
    ignore = shutil.ignore_patterns(
        "wandb",
        "outputs",
        "logs",
        "checkpoints",
        "tensorboard",
        "code_snapshot",
        "__pycache__",
        ".pytest_cache",
        ".hydra",
        "*.ckpt",
        "*.pt",
        "*.pth",
        "events.out.tfevents.*",
        "*.h5",
        "*.hdf5",
        "*.lance",
    )
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def unnormalize_pixels(pixels):
    mean = IMAGENET_MEAN.to(device=pixels.device, dtype=pixels.dtype)
    std = IMAGENET_STD.to(device=pixels.device, dtype=pixels.dtype)
    return (pixels * std + mean).clamp(0.0, 1.0)


def log_latent_stats(module, stage, name, value):
    with torch.no_grad():
        is_train = stage in ("fit", "train")
        flat = value.detach().reshape(-1, value.size(-1))
        module.log(f"{stage}/stats/{name}_mean", flat.mean(), on_step=is_train, on_epoch=True, sync_dist=True)
        module.log(f"{stage}/stats/{name}_std", flat.std(unbiased=False), on_step=is_train, on_epoch=True, sync_dist=True)
        module.log(f"{stage}/stats/{name}_norm", flat.norm(dim=-1).mean(), on_step=is_train, on_epoch=True, sync_dist=True)


def attention_heatmap(rgb, attentions):
    if not attentions:
        return None
    attn = attentions[-1]
    if attn.ndim != 4 or attn.size(-1) < 2:
        return None
    cls_to_patch = attn[0].mean(dim=0)[0, 1:]
    grid = int(cls_to_patch.numel() ** 0.5)
    if grid * grid != cls_to_patch.numel():
        return None
    heat = cls_to_patch.view(1, 1, grid, grid)
    heat = F.interpolate(heat, size=rgb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heatmap = torch.zeros(3, *heat.shape, device=heat.device, dtype=heat.dtype)
    heatmap[0] = heat
    heatmap[1] = (1.0 - (heat - 0.5).abs() * 2.0).clamp(0, 1)
    heatmap[2] = (1.0 - heat).clamp(0, 1)
    return heatmap


def log_visuals(module, batch, output):
    if module.global_step % module.cfg.logging.visualize_every_n_steps != 0:
        return
    if module.logger is None:
        return
    writer = module.logger.experiment
    rgb = unnormalize_pixels(batch["pixels"][:1]).detach()
    target = rgb[0, 0]
    if "recon" in output:
        recon = output["recon"][0, 0].detach().clamp(0, 1)
    else:
        recon = torch.zeros_like(target)
    heatmap = attention_heatmap(rgb[:, 0], output.get("attentions"))
    if heatmap is None:
        heatmap = torch.zeros_like(target)
    panel = torch.cat([target, recon, heatmap.to(target.device, target.dtype)], dim=-1)
    writer.add_image("train/images/target_recon_attention", panel, module.global_step)


def _index_tensor(indices, device):
    return torch.as_tensor(list(indices), device=device, dtype=torch.long)


def _col_stats(cfg, col, device, dtype):
    stats = cfg.probe.stats[col]
    mean = torch.as_tensor(stats.mean, device=device, dtype=dtype)
    std = torch.as_tensor(stats.std, device=device, dtype=dtype).clamp_min(1e-6)
    return mean, std


def wrapped_angle_abs_error(pred_cos_sin, target_angle):
    pred_angle = torch.atan2(pred_cos_sin[..., 1], pred_cos_sin[..., 0])
    diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle))
    return diff.abs()


def z_sigreg_loss(sigreg, z, flatten_batch_time=False):
    """Apply SIGReg on T,B,D or on a single flattened B*T population."""
    if flatten_batch_time:
        # SIGReg pools dimension -3 (the sample/batch axis), so B*T must be
        # placed there while retaining a singleton time axis.
        return sigreg(z.reshape(1, -1, z.size(-1)))
    return sigreg(z.transpose(0, 1))


def build_probe_targets(batch, cfg):
    pieces = []
    slices = []
    offset = 0
    for target_cfg in cfg.probe.targets:
        col = target_cfg.col
        value = batch[col]
        if target_cfg.type == "vector":
            indices = target_cfg.get("indices")
            if indices is not None:
                value = value.index_select(-1, _index_tensor(indices, value.device))
            dim = value.size(-1)
            pieces.append(value)
            slices.append((target_cfg, offset, offset + dim))
            offset += dim
        elif target_cfg.type == "angle":
            index = int(target_cfg.index)
            mean, std = _col_stats(cfg, col, value.device, value.dtype)
            angle = value[..., index] * std[index] + mean[index]
            angle_target = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            pieces.append(angle_target)
            slices.append((target_cfg, offset, offset + 2))
            offset += 2
        else:
            raise ValueError(f"Unsupported probe target type: {target_cfg.type}")
    if not pieces:
        raise ValueError("probe.targets must contain at least one target")
    return torch.cat(pieces, dim=-1), slices


def log_probe_metrics(output, stage, probe_name, pred, target, slices, batch, cfg):
    output[f"{stage}/metric/probe_{probe_name}_mae"] = (pred - target).abs().mean()
    for target_cfg, start, end in slices:
        pred_part = pred[..., start:end]
        target_part = target[..., start:end]
        metric_base = f"{stage}/metric/probe_{probe_name}_{target_cfg.name}"
        if target_cfg.type == "vector":
            col = target_cfg.col
            indices = target_cfg.get("indices")
            if col in cfg.probe.stats:
                mean, std = _col_stats(cfg, col, pred.device, pred.dtype)
                if indices is not None:
                    idx = _index_tensor(indices, pred.device)
                    mean = mean.index_select(0, idx)
                    std = std.index_select(0, idx)
                pred_physical = pred_part * std + mean
                target_physical = target_part * std + mean
                output[f"{metric_base}_mae"] = (pred_physical - target_physical).abs().mean()
            else:
                output[f"{metric_base}_mae"] = (pred_part - target_part).abs().mean()
        elif target_cfg.type == "angle":
            col = target_cfg.col
            index = int(target_cfg.index)
            value = batch[col]
            mean, std = _col_stats(cfg, col, value.device, value.dtype)
            target_angle = value[..., index] * std[index] + mean[index]
            angle_error = wrapped_angle_abs_error(pred_part, target_angle)
            output[f"{metric_base}_angle_abs_error_rad"] = angle_error.mean()
            output[f"{metric_base}_angle_abs_error_deg"] = torch.rad2deg(angle_error).mean()


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations and compute the losses used by released variants."""
    self.cfg = cfg
    is_train = stage in ("fit", "train")
    weights = cfg.loss.weights

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    batch["return_attention"] = (
        is_train and self.global_step % cfg.logging.visualize_every_n_steps == 0
    )
    split = "train" if is_train else "val"
    if wood_enabled(cfg, split):
        batch = dict(batch)
        episode_idx = batch.get("episode_idx", batch.get("ep_idx"))
        batch["pixels"] = add_static_episode_wood_torch(
            batch["pixels"], episode_idx, cfg.wood
        )

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, : cfg.history_size]
    ctx_act = act_emb[:, : cfg.history_size]
    tgt_emb = emb[:, cfg.num_preds :]

    if cfg.diff_image.enabled and cfg.diff_image.detach_z_for_prediction:
        tgt_emb = tgt_emb.detach()
        pred_ctx_emb = ctx_emb.detach()
    else:
        pred_ctx_emb = ctx_emb
    pred_emb = self.model.predict(pred_ctx_emb, ctx_act)

    output["loss_prediction"] = F.mse_loss(pred_emb, tgt_emb)
    output["loss_sigreg_z"] = z_sigreg_loss(
        self.sigreg, emb, bool(cfg.sigreg_flat.enabled)
    )
    output[f"{stage}/metric/sigreg_z_flatten_batch_time"] = torch.as_tensor(
        float(cfg.sigreg_flat.enabled), device=emb.device
    )
    output[f"{stage}/metric/sigreg_z_input_norm"] = (
        emb.detach().norm(dim=-1).mean()
    )

    if cfg.diff_image.enabled:
        diff_pixels = batch["pixels"][:, 1:] - batch["pixels"][:, :-1]
        if cfg.diff_image.image_size != batch["pixels"].size(-1):
            shape = diff_pixels.shape
            diff_pixels = F.interpolate(
                diff_pixels.reshape(-1, *shape[-3:]),
                size=(cfg.diff_image.image_size, cfg.diff_image.image_size),
                mode="bilinear",
                align_corners=False,
            ).view(
                *shape[:-2],
                cfg.diff_image.image_size,
                cfg.diff_image.image_size,
            )
        diff_emb = self.model.encode_diff(diff_pixels)
        output["diff_emb"] = diff_emb
        output["loss_sigreg_diff_image"] = self.sigreg(diff_emb.transpose(0, 1))
        diff_from_latents = self.model.predict_diff_from_latents(
            emb[:, :-1], emb[:, 1:]
        )
        output["loss_diff_image_latent_prediction"] = F.mse_loss(
            diff_from_latents, diff_emb
        )

    if cfg.inverse_action_raw.enabled:
        delta_z = emb[:, 1:] - emb[:, :-1]
        action_target = batch["action"][:, : delta_z.size(1)]
        action_pred = self.model.predict_action_raw(delta_z)
        output["loss_inverse_action_raw_prediction"] = F.mse_loss(
            action_pred, action_target
        )

    recon_batch = max(1, emb.size(0) // 4)
    recon = self.model.reconstruct(emb[:recon_batch, :1])
    output["recon"] = recon
    output["loss_image_reconstruction"] = F.mse_loss(
        recon,
        unnormalize_pixels(batch["pixels"][:recon_batch, :1]),
    )

    probe_target, probe_slices = build_probe_targets(batch, cfg)
    probe_preds = self.model.probe_pose(emb)
    probe_losses = []
    for probe_name, probe_pred in probe_preds.items():
        output[f"probe_{probe_name}_pred"] = probe_pred
        probe_loss = F.mse_loss(probe_pred, probe_target)
        output[f"loss_pose_probe_{probe_name}"] = probe_loss
        probe_losses.append(probe_loss)
        log_probe_metrics(
            output,
            stage,
            probe_name,
            probe_pred,
            probe_target,
            probe_slices,
            batch,
            cfg,
        )
    output["loss_pose_probe"] = torch.stack(probe_losses).mean()

    output["loss"] = (
        weights.prediction * output["loss_prediction"]
        + weights.sigreg * output["loss_sigreg_z"]
        + weights.image_reconstruction * output["loss_image_reconstruction"]
        + weights.pose_probe * output["loss_pose_probe"]
    )
    if cfg.diff_image.enabled:
        output["loss"] = (
            output["loss"]
            + weights.sigreg_diff_image * output["loss_sigreg_diff_image"]
            + weights.diff_image_latent_prediction
            * output["loss_diff_image_latent_prediction"]
        )
    if cfg.inverse_action_raw.enabled:
        output["loss"] = (
            output["loss"]
            + weights.inverse_action_raw_prediction
            * output["loss_inverse_action_raw_prediction"]
        )

    losses_dict = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if key.startswith("loss")
    }
    losses_dict[f"{stage}/loss_total"] = output["loss"].detach()
    losses_dict.update(
        {
            key: value.detach()
            for key, value in output.items()
            if key.startswith(f"{stage}/metric/")
        }
    )
    self.log_dict(
        losses_dict,
        on_step=is_train,
        on_epoch=True,
        sync_dist=True,
    )
    log_latent_stats(self, stage, "z", emb)
    log_latent_stats(self, stage, "delta_z", emb[:, 1:] - emb[:, :-1])
    if cfg.diff_image.enabled:
        log_latent_stats(self, stage, "diff_image", output["diff_emb"])
    if is_train:
        log_visuals(self, batch, output)
        return output
    return {"loss": output["loss"].detach()}


class EvalSuccessCallback(Callback):
    def __init__(self, cfg, run_dir):
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.out_dir = self.run_dir / cfg.validation.success_rate_output_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self.cfg.validation.enable_epoch_eval_placeholder:
            return
        if not self.cfg.validation.get("eval_on_resume_start", False):
            return
        if not (self.cfg.get("resume") and self.cfg.resume.get("enabled", False)):
            return

        policy = self._latest_pretrained_policy()
        if policy is None:
            error_path = self.out_dir / "resume_start_eval_error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "error": "No pretrained_epoch_* export found for resume-start eval",
                        "search_dir": str(self.run_dir / "checkpoints" / "checkpoints"),
                    },
                    indent=2,
                )
            )
            return

        epoch, policy_name = policy
        evals = self.cfg.validation.get("evals") or [{"name": "default"}]
        original_device = self._offload_training_state(trainer, pl_module, epoch)
        try:
            for eval_cfg in evals:
                result = self._run_eval(epoch, policy_name, eval_cfg, prefix="resume_start")
                success_rate = result.get("success_rate")
                eval_name = result["eval_name"]
                if success_rate is not None and pl_module.logger is not None:
                    success_rate_normalized = success_rate / 100.0 if success_rate > 1.0 else success_rate
                    pl_module.logger.experiment.add_scalar(
                        f"val/resume_start/{eval_name}/success_rate",
                        success_rate_normalized,
                        trainer.global_step,
                    )
                    pl_module.logger.experiment.add_scalar(
                        f"val/resume_start/{eval_name}/success_rate_percent",
                        success_rate,
                        trainer.global_step,
                    )
        finally:
            self._restore_training_state(trainer, pl_module, original_device, epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self.cfg.validation.enable_epoch_eval_placeholder:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.cfg.validation.eval_every_n_epochs != 0:
            return
        policy_name = f"pretrained_epoch_{epoch:04d}"
        evals = self.cfg.validation.get("evals") or [{"name": "default"}]
        original_device = self._offload_training_state(trainer, pl_module, epoch)
        try:
            for eval_cfg in evals:
                result = self._run_eval(epoch, policy_name, eval_cfg)
                success_rate = result.get("success_rate")
                eval_name = result["eval_name"]
                if success_rate is not None and pl_module.logger is not None:
                    success_rate_normalized = success_rate / 100.0 if success_rate > 1.0 else success_rate
                    pl_module.logger.experiment.add_scalar(
                        f"val/{eval_name}/success_rate",
                        success_rate_normalized,
                        trainer.global_step,
                    )
                    pl_module.logger.experiment.add_scalar(
                        f"val/{eval_name}/success_rate_percent",
                        success_rate,
                        trainer.global_step,
                    )
                    if eval_name in ("default", "25_step"):
                        pl_module.logger.experiment.add_scalar(
                            "val/success_rate",
                            success_rate_normalized,
                            trainer.global_step,
                        )
                        pl_module.logger.experiment.add_scalar(
                            "val/success_rate_percent",
                            success_rate,
                            trainer.global_step,
                        )
        finally:
            self._restore_training_state(trainer, pl_module, original_device, epoch)

    @staticmethod
    def _move_optimizer_value(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = EvalSuccessCallback._move_optimizer_value(item, device)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = EvalSuccessCallback._move_optimizer_value(item, device)
            return value
        if isinstance(value, tuple):
            return tuple(EvalSuccessCallback._move_optimizer_value(item, device) for item in value)
        return value

    @classmethod
    def _move_optimizer_states(cls, trainer, device):
        for optimizer in trainer.optimizers:
            for state in optimizer.state.values():
                cls._move_optimizer_value(state, device)

    @staticmethod
    def _cuda_memory_snapshot(device, label):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot = {
            "label": label,
            "device": str(device),
            "free_gb": free_bytes / 2**30,
            "total_gb": total_bytes / 2**30,
            "parent_allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
            "parent_reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
        }
        print(f"[success-eval] cuda_memory={json.dumps(snapshot)}", flush=True)
        return snapshot

    def _offload_training_state(self, trainer, pl_module, epoch):
        original_device = pl_module.device
        if original_device.type != "cuda":
            return original_device
        snapshots = [self._cuda_memory_snapshot(original_device, "before_offload")]
        pl_module.to("cpu")
        self._move_optimizer_states(trainer, torch.device("cpu"))
        gc.collect()
        torch.cuda.empty_cache()
        snapshots.append(self._cuda_memory_snapshot(original_device, "after_offload"))
        (self.out_dir / f"epoch_{epoch:04d}_cuda_memory.json").write_text(
            json.dumps(snapshots, indent=2)
        )
        return original_device

    def _restore_training_state(self, trainer, pl_module, original_device, epoch):
        if original_device.type != "cuda":
            return
        pl_module.to(original_device)
        self._move_optimizer_states(trainer, original_device)
        snapshot = self._cuda_memory_snapshot(original_device, "after_restore")
        memory_path = self.out_dir / f"epoch_{epoch:04d}_cuda_memory.json"
        snapshots = json.loads(memory_path.read_text()) if memory_path.exists() else []
        snapshots.append(snapshot)
        memory_path.write_text(json.dumps(snapshots, indent=2))

    def _run_eval(self, epoch, policy_name, eval_cfg, prefix=None):
        script = Path(__file__).resolve().parent / self.cfg.validation.eval_script
        eval_name = str(eval_cfg.get("name", "default"))
        filename = f"epoch_{epoch:04d}_{eval_name}.json"
        if prefix:
            filename = f"{prefix}_{filename}"
        result_path = self.out_dir / filename
        eval_config_name = self._eval_config_name()
        cmd = [
            sys.executable,
            str(script),
            "--config-name",
            eval_config_name,
            f"policy={policy_name}",
            f"policy_cache_dir={self.run_dir / 'checkpoints'}",
            f"output.filename={result_path.name}",
        ]
        if "goal_offset_steps" in eval_cfg:
            cmd.append(f"eval.goal_offset_steps={eval_cfg.goal_offset_steps}")
        if "eval_budget" in eval_cfg:
            cmd.append(f"eval.eval_budget={eval_cfg.eval_budget}")
        if self.cfg.get("wood") and self.cfg.wood.enabled:
            cmd.extend(
                [
                    "wood.enabled=true",
                    f"wood.eval={str(bool(self.cfg.wood.get('eval', True))).lower()}",
                    f"wood.seed={int(self.cfg.wood.seed)}",
                    f"wood.texture_path={self.cfg.wood.texture_path}",
                    f"wood.mask_mode={self.cfg.wood.mask_mode}",
                    f"wood.white_threshold={int(self.cfg.wood.white_threshold)}",
                    f"wood.background_tolerance={int(self.cfg.wood.background_tolerance)}",
                ]
            )
            if self.cfg.wood.get("static_background_path"):
                cmd.append(f"wood.static_background_path={self.cfg.wood.static_background_path}")
        cmd.extend(str(x) for x in self.cfg.validation.get("extra_args", []))
        payload = {
            "epoch": epoch,
            "phase": prefix or "epoch_end",
            "eval_name": eval_name,
            "eval_cfg": OmegaConf.to_container(eval_cfg, resolve=True),
            "policy": policy_name,
            "command": cmd,
            "status": "started",
            "success_rate": None,
        }
        result_path.write_text(json.dumps(payload, indent=2))
        print(
            f"[success-eval] phase={payload['phase']} epoch={epoch} "
            f"name={eval_name} goal_offset={eval_cfg.get('goal_offset_steps', 'default')} "
            f"budget={eval_cfg.get('eval_budget', 'default')} started",
            flush=True,
        )
        eval_start_time = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=script.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            payload["status"] = "finished"
            payload["returncode"] = proc.returncode
            payload["output"] = proc.stdout[-8000:]
            payload["success_rate"] = parse_success_rate(proc.stdout)
            if payload["success_rate"] is not None:
                payload["success_rate_percent"] = payload["success_rate"]
                payload["success_rate_normalized"] = (
                    payload["success_rate"] / 100.0
                    if payload["success_rate"] > 1.0
                    else payload["success_rate"]
                )
        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = repr(exc)
        payload["elapsed_seconds"] = time.monotonic() - eval_start_time
        result_path.write_text(json.dumps(payload, indent=2))
        print(
            f"[success-eval] phase={payload['phase']} epoch={epoch} "
            f"name={eval_name} finished returncode={payload.get('returncode')} "
            f"success_rate={payload.get('success_rate')} "
            f"elapsed={payload['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return payload

    def _latest_pretrained_policy(self):
        export_root = self.run_dir / "checkpoints" / "checkpoints"
        candidates = sorted(export_root.glob("pretrained_epoch_*"))
        candidates = [p for p in candidates if p.is_dir() and (p / "weights.pt").exists()]
        if not candidates:
            return None
        latest = candidates[-1]
        try:
            epoch = int(latest.name.rsplit("_", 1)[-1])
        except ValueError:
            epoch = 0
        return epoch, latest.name

    def _eval_config_name(self):
        dataset_name = str(self.cfg.data.dataset.name)
        if "cube" in dataset_name:
            return "cube"
        if "reacher" in dataset_name:
            return "reacher"
        if "tworoom" in dataset_name:
            return "tworoom"
        if "pusht" in dataset_name:
            return "pusht"
        return self.cfg.validation.get("eval_config", "pusht")


class SavePretrainedCallback(Callback):
    def __init__(self, cfg, run_dir):
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        try:
            self._save(pl_module.model, epoch)
        except Exception as exc:
            out_dir = self.run_dir / self.cfg.validation.success_rate_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            error_path = out_dir / f"pretrained_export_epoch_{epoch:04d}_error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "status": "error",
                        "error": repr(exc),
                    },
                    indent=2,
                )
            )

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        save_pretrained(
            model,
            run_name=f"pretrained_epoch_{epoch:04d}",
            config=self.cfg.model,
            filename="weights.pt",
            cache_dir=str(self.run_dir / "checkpoints"),
        )


def parse_success_rate(text):
    for line in reversed(text.splitlines()):
        if "success_rate" not in line:
            continue
        try:
            parsed = ast.literal_eval(line.strip())
            if isinstance(parsed, dict) and "success_rate" in parsed:
                return float(parsed["success_rate"])
        except Exception:
            pass
        marker = "success_rate"
        tail = line[line.find(marker) + len(marker) :]
        for token in tail.replace(":", " ").replace(",", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
    return None


def configure_visible_gpus(cfg):
    gpus = cfg.hardware.get("gpus")
    if cfg.hardware.accelerator != "gpu" or gpus in (None, "auto"):
        return
    if isinstance(gpus, int):
        visible = [gpus]
    else:
        visible = list(gpus)
    if not visible:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible)
    with open_dict(cfg):
        cfg.trainer.devices = list(range(len(visible)))


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    configure_visible_gpus(cfg)
    run_dir = build_run_dir(cfg)
    resume_ckpt_path = resolve_resume_checkpoint(cfg, run_dir)

    with open_dict(cfg):
        cfg.run_dir = str(run_dir)
        if not cfg.data.get("probe"):
            raise ValueError(
                "Each training data config must define data.probe for benchmark probing"
            )
        cfg.probe.output_dim = cfg.data.probe.output_dim
        cfg.probe.targets = cfg.data.probe.targets
        cfg.model.action_encoder.norm_fn = None
        cfg.model.action_raw_predictor = None
        cfg.model.diff_encoder = None
        cfg.model.diff_projector = None
        cfg.model.diff_latent_predictor = None

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )
    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    with open_dict(cfg):
        cfg.probe.stats = {}
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col in (
                "episode_idx",
                "ep_idx",
                "step_idx",
            ):
                continue
            col_data = torch.from_numpy(dataset.get_col_data(col))
            if col_data.ndim == 1:
                col_data = col_data.unsqueeze(-1)
            finite = ~torch.isnan(col_data.reshape(col_data.size(0), -1)).any(dim=1)
            col_data = col_data[finite].float().reshape(finite.sum(), -1)
            cfg.probe.stats[col] = {
                "mean": col_data.mean(0).tolist(),
                "std": col_data.std(0).clamp_min(1e-6).tolist(),
            }
            transforms.append(get_column_normalizer(dataset, col, col))

        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim

        if cfg.diff_image.enabled:
            diff_latent_dim = cfg.diff_image.latent_dim
            cfg.model.diff_encoder = {
                "_target_": "stable_pretraining.backbone.utils.vit_hf",
                "size": cfg.diff_image.encoder_size,
                "patch_size": cfg.diff_image.encoder_patch_size,
                "image_size": cfg.diff_image.image_size,
                "pretrained": False,
                "use_mask_token": False,
                "output_attentions": False,
            }
            cfg.model.diff_projector = {
                "_target_": "module.MLP",
                "input_dim": 192,
                "output_dim": diff_latent_dim,
                "hidden_dim": 2048,
                "norm_fn": {
                    "_target_": "torch.nn.BatchNorm1d",
                    "_partial_": True,
                },
            }
            cfg.model.diff_latent_predictor = {
                "_target_": "module.DiffImageLatentPredictor",
                "z_dim": cfg.embed_dim,
                "diff_dim": diff_latent_dim,
                "hidden_dim": cfg.diff_image.latent_to_diff.hidden_dim,
                "depth": cfg.diff_image.latent_to_diff.depth,
            }

        if cfg.inverse_action_raw.enabled:
            cfg.model.action_raw_predictor = {
                "_target_": "module.InverseActionRawPredictor",
                "input_dim": cfg.embed_dim,
                "hidden_dim": cfg.inverse_action_raw.hidden_dim,
                "output_dim": action_dim,
                "depth": cfg.inverse_action_raw.depth,
            }

    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    world_model = hydra.utils.instantiate(cfg.model)
    optimizer_cfg = dict(cfg.optimizer)
    warmup_fraction = optimizer_cfg.pop("warmup_fraction", 0.005)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": optimizer_cfg,
            "scheduler": make_warmup_cosine_scheduler(warmup_fraction),
            "interval": "epoch",
        },
    }
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    world_model.cfg = cfg

    config_filename = "resume_config.yaml" if cfg.resume.enabled else "config.yaml"
    OmegaConf.save(cfg, run_dir / config_filename)
    if cfg.logging.save_code_snapshot and not cfg.resume.enabled:
        save_code_snapshot(run_dir)

    logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard", version="")
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch_{epoch:04d}",
        every_n_epochs=1,
        save_last=True,
        save_top_k=-1,
    )
    callbacks = [
        checkpoint_callback,
        SavePretrainedCallback(cfg, run_dir),
        LearningRateMonitor(logging_interval="step"),
        EvalSuccessCallback(cfg, run_dir),
    ]
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=run_dir,
    )
    trainer.fit(
        world_model,
        datamodule=data_module,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    run()
