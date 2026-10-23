from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def wood_enabled(cfg, split):
    wood_cfg = cfg_get(cfg, "wood")
    return bool(cfg_get(wood_cfg, "enabled", False) and cfg_get(wood_cfg, split, True))


def _asset_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


@lru_cache(maxsize=8)
def _load_rgb(path):
    path = _asset_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Wood augmentation asset does not exist: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


@lru_cache(maxsize=8)
def _load_static_background(path):
    path = _asset_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Static background asset does not exist: {path}")
    if path.suffix == ".npy":
        background = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            background = np.asarray(image.convert("RGB"))
    if background.ndim != 3 or background.shape[-1] != 3:
        raise ValueError(f"Expected H,W,3 static background, got {background.shape}")
    return background


def _crop_coordinates(episode_ids, texture_height, texture_width, height, width, seed):
    if texture_height < height or texture_width < width:
        raise ValueError(
            f"Wood texture must be at least {height}x{width}, got "
            f"{texture_height}x{texture_width}"
        )
    ids = np.asarray(episode_ids, dtype=np.uint64).reshape(-1)
    seed = np.uint64(seed)
    x_span = np.uint64(texture_width - width + 1)
    y_span = np.uint64(texture_height - height + 1)
    xs = (ids * np.uint64(1103515245) + seed * np.uint64(12345) + np.uint64(1013904223)) % x_span
    ys = (ids * np.uint64(214013) + seed * np.uint64(2531011) + np.uint64(1376312589)) % y_span
    return xs.astype(np.int64), ys.astype(np.int64)


def _episode_ids(episode_idx, batch_size):
    if episode_idx is None:
        raise KeyError("wood augmentation requires 'episode_idx' or 'ep_idx'")
    ids = np.asarray(
        episode_idx.detach().cpu() if torch.is_tensor(episode_idx) else episode_idx
    )
    if ids.ndim > 1:
        ids = ids[:, 0]
    ids = ids.reshape(-1)
    if len(ids) != batch_size:
        raise ValueError(f"Expected {batch_size} episode ids, got {len(ids)}")
    return ids


def _wood_crops_numpy(episode_ids, height, width, wood_cfg):
    texture = _load_rgb(cfg_get(wood_cfg, "texture_path", "assets/wood_texture.png"))
    xs, ys = _crop_coordinates(
        episode_ids,
        texture.shape[0],
        texture.shape[1],
        height,
        width,
        int(cfg_get(wood_cfg, "seed", 0)),
    )
    return np.stack(
        [texture[y : y + height, x : x + width] for x, y in zip(xs, ys)],
        axis=0,
    )


def _static_background_numpy(wood_cfg):
    path = cfg_get(wood_cfg, "static_background_path")
    if not path:
        raise ValueError(
            "wood.mask_mode=static requires wood.static_background_path. "
            "Generate it with scripts/compute_static_background.py."
        )
    return _load_static_background(path)


def _resize_nearest_numpy(image, height, width):
    if image.shape[:2] == (height, width):
        return image
    ys = np.rint(np.linspace(0, image.shape[0] - 1, height)).astype(np.int64)
    xs = np.rint(np.linspace(0, image.shape[1] - 1, width)).astype(np.int64)
    return image[ys[:, None], xs[None, :]]


def _raw_numpy_scale(pixels):
    if np.issubdtype(pixels.dtype, np.integer):
        return 255.0
    finite_max = float(np.nanmax(pixels)) if pixels.size else 1.0
    return 1.0 if finite_max <= 1.5 else 255.0


def add_static_episode_wood_numpy(pixels, episode_idx, wood_cfg):
    """Replace background pixels using a deterministic wood crop per episode."""
    out = np.array(pixels, copy=True)
    squeeze_time = False
    if out.ndim == 4:
        out = out[:, None]
        squeeze_time = True
    if out.ndim != 5 or out.shape[-1] != 3:
        raise ValueError(f"Expected B,T,H,W,C or B,H,W,C pixels, got {pixels.shape}")

    batch_size, _, height, width, _ = out.shape
    ids = _episode_ids(episode_idx, batch_size)
    crops = _wood_crops_numpy(ids, height, width, wood_cfg)
    scale = _raw_numpy_scale(out)
    source = out.astype(np.float32) * (255.0 / scale)

    mode = str(cfg_get(wood_cfg, "mask_mode", "white"))
    if mode == "white":
        threshold = float(cfg_get(wood_cfg, "white_threshold", 245))
        mask = np.all(source >= threshold, axis=-1)
    elif mode == "static":
        background = _static_background_numpy(wood_cfg)
        if background.shape[:2] != (height, width):
            background = _resize_nearest_numpy(background, height, width)
        valid_background = np.all(background >= 0, axis=-1)
        tolerance = float(cfg_get(wood_cfg, "background_tolerance", 8))
        matches_background = (
            np.max(np.abs(source - background.astype(np.float32)), axis=-1) <= tolerance
        )
        mask = valid_background[None, None] & matches_background
    else:
        raise ValueError(f"Unknown wood mask_mode: {mode}")

    replacement = crops[:, None].astype(np.float32) * (scale / 255.0)
    out = np.where(mask[..., None], replacement, out.astype(np.float32))
    if np.issubdtype(pixels.dtype, np.integer):
        out = np.rint(out).clip(0, np.iinfo(pixels.dtype).max).astype(pixels.dtype)
    else:
        out = out.astype(pixels.dtype)
    return out[:, 0] if squeeze_time else out


def add_static_episode_wood_torch(pixels, episode_idx, wood_cfg):
    """Replace backgrounds in normalized channel-first tensors."""
    out = pixels.clone()
    squeeze_time = False
    if out.ndim == 4:
        out = out.unsqueeze(1)
        squeeze_time = True
    if out.ndim != 5 or out.shape[2] != 3:
        raise ValueError(f"Expected B,T,C,H,W or B,C,H,W pixels, got {tuple(pixels.shape)}")

    batch_size, _, _, height, width = out.shape
    ids = _episode_ids(episode_idx, batch_size)
    crops = _wood_crops_numpy(ids, height, width, wood_cfg)
    crops = torch.from_numpy(crops.copy()).to(device=out.device, dtype=out.dtype)
    crops = crops.permute(0, 3, 1, 2).div_(255.0)

    mean = IMAGENET_MEAN.to(device=out.device, dtype=out.dtype).view(1, 1, 3, 1, 1)
    std = IMAGENET_STD.to(device=out.device, dtype=out.dtype).view(1, 1, 3, 1, 1)
    raw = out * std + mean
    mode = str(cfg_get(wood_cfg, "mask_mode", "white"))
    if mode == "white":
        threshold = float(cfg_get(wood_cfg, "white_threshold", 245)) / 255.0
        mask = torch.all(raw >= threshold, dim=2)
    elif mode == "static":
        background = _static_background_numpy(wood_cfg)
        background = torch.from_numpy(background.copy()).to(device=out.device, dtype=out.dtype)
        background = background.permute(2, 0, 1).unsqueeze(0)
        if background.shape[-2:] != (height, width):
            background = torch.nn.functional.interpolate(
                background, size=(height, width), mode="nearest"
            )
        valid_background = torch.all(background >= 0, dim=1)
        background = background.div_(255.0)
        tolerance = float(cfg_get(wood_cfg, "background_tolerance", 8)) / 255.0
        matches_background = (
            torch.amax(torch.abs(raw - background[:, None]), dim=2) <= tolerance
        )
        mask = valid_background[:, None] & matches_background
    else:
        raise ValueError(f"Unknown wood mask_mode: {mode}")

    normalized_crops = ((crops - mean[:, 0]) / std[:, 0])[:, None]
    out = torch.where(mask[:, :, None], normalized_crops, out)
    return out[:, 0] if squeeze_time else out
