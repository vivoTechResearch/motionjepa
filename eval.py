import os
import math
import json
from contextlib import contextmanager, nullcontext

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from wood_augmentation import add_static_episode_wood_numpy, wood_enabled


@contextmanager
def wood_video_panels(episode_ids, wood_cfg):
    """Render wood in evaluation videos as well as in planner inputs."""
    import stable_worldmodel.world.world as world_module

    original_save = world_module.save_panel_videos
    episode_ids = np.asarray(episode_ids)

    def apply_video(frames, episode_id):
        array = np.asarray(frames)
        return list(add_static_episode_wood_numpy(array[None], [episode_id], wood_cfg)[0])

    def save_with_wood(video_dir, panels, fps=15):
        panels = dict(panels)
        if "agent" in panels:
            panels["agent"] = {
                idx: apply_video(frames, episode_ids[idx])
                for idx, frames in panels["agent"].items()
            }
        if "dataset" in panels:
            panels["dataset"] = [
                apply_video(frames, episode_ids[idx])
                for idx, frames in enumerate(panels["dataset"])
            ]
        if "goal" in panels:
            panels["goal"] = add_static_episode_wood_numpy(
                np.asarray(panels["goal"]), episode_ids, wood_cfg
            )
        return original_save(video_dir, panels, fps=fps)

    world_module.save_panel_videos = save_with_wood
    try:
        yield
    finally:
        world_module.save_panel_videos = original_save


def attach_eval_progress(world, num_rollouts, eval_budget, enabled=True, progress_path=None):
    if not enabled:
        return

    try:
        from tqdm.auto import tqdm
    except Exception:
        return

    original_run = world._run
    started_at = time.time()
    completed = 0
    successes = 0

    def write_progress(event, **fields):
        if progress_path is None:
            return
        payload = {
            "event": event,
            "elapsed_seconds": round(time.time() - started_at, 3),
            **fields,
        }
        with Path(progress_path).open("a") as f:
            f.write(json.dumps(payload) + "\n")
            f.flush()

    write_progress(
        "started",
        num_rollouts=num_rollouts,
        eval_budget=eval_budget,
    )

    def run_with_progress(episodes=None, max_steps=None, seed=None, options=None, mode="auto", on_step=None, on_done=None):
        if max_steps is None:
            return original_run(
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
                options=options,
                mode=mode,
                on_step=on_step,
                on_done=on_done,
            )

        total = int(max_steps)
        desc = f"rollouts={num_rollouts}, budget={eval_budget}"
        with tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True) as pbar:
            def wrapped_on_step(inner_world):
                if on_step is not None:
                    on_step(inner_world)
                pbar.update(1)

            def wrapped_on_done(env_idx, ep_idx, inner_world):
                nonlocal completed, successes
                if on_done is not None:
                    on_done(env_idx, ep_idx, inner_world)
                completed += 1
                succeeded = bool(inner_world.terminateds[env_idx])
                successes += int(succeeded)
                write_progress(
                    "rollout_finished",
                    rollout=completed,
                    total_rollouts=num_rollouts,
                    env_idx=int(env_idx),
                    success=succeeded,
                    successes=successes,
                    success_rate=100.0 * successes / completed,
                    environment_steps=int(pbar.n),
                )

            return original_run(
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
                options=options,
                mode=mode,
                on_step=wrapped_on_step,
                on_done=wrapped_on_done,
            )

    world._run = run_with_progress


class CEMProgressCallback:
    """Show CEM optimization progress for each MPC planning call."""

    output_key = "cem_progress"

    def __init__(self, solver, enabled=True):
        self.solver = solver
        self.enabled = enabled
        self.history = []
        self._bar = None
        self._updates = 0
        self._solve_index = 0

    def reset(self):
        self.end_solve()
        self.history = []
        self._updates = 0
        self._solve_index += 1
        if not self.enabled:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:
            return

        batches = math.ceil(self.solver.n_envs / self.solver.batch_size)
        total = batches * self.solver.n_steps
        self._bar = tqdm(
            total=total,
            desc=f"CEM plan={self._solve_index} ({batches} batches × {self.solver.n_steps} iters)",
            unit="iter",
            dynamic_ncols=True,
        )

    def start_batch(self):
        pass

    def __call__(self, *, step, **_state):
        self._updates += 1
        if self._bar is not None:
            batch = (self._updates - 1) // self.solver.n_steps + 1
            self._bar.set_postfix_str(f"batch={batch}, iter={step + 1}/{self.solver.n_steps}")
            self._bar.update(1)

    def end_solve(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class WoodWorldModelPolicy(swm.policy.WorldModelPolicy):
    def __init__(self, *args, wood_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wood_cfg = wood_cfg
        self._wood_episode_idx = None

    def _prepare_info(self, info_dict):
        info_dict = dict(info_dict)
        episode_idx = info_dict.get("episode_idx", info_dict.get("ep_idx"))
        if episode_idx is not None:
            self._wood_episode_idx = np.asarray(episode_idx)
        goal_episode_idx = info_dict.get("goal_episode_idx", info_dict.get("goal_ep_idx"))
        if self._wood_episode_idx is None and goal_episode_idx is not None:
            self._wood_episode_idx = np.asarray(goal_episode_idx)
        if self._wood_episode_idx is not None and "pixels" in info_dict:
            info_dict["pixels"] = add_static_episode_wood_numpy(
                info_dict["pixels"], self._wood_episode_idx, self.wood_cfg
            )

        if goal_episode_idx is None:
            goal_episode_idx = self._wood_episode_idx
        if goal_episode_idx is not None and "goal" in info_dict:
            info_dict["goal"] = add_static_episode_wood_numpy(
                info_dict["goal"], goal_episode_idx, self.wood_cfg
            )
        return super()._prepare_info(info_dict)


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # # Make sure format of config file is correct
    # try:
    #     del cfg["predictor"]["_target_"]
    #     del cfg["action_encoder"]["_target_"]
    # except KeyError:
    #     print("Config file is already in the correct format.")

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels", "episode_idx", "ep_idx", "step_idx"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")
    if policy != "random":
        torch.cuda.reset_peak_memory_stats()
        model = swm.wm.utils.load_pretrained(
            cfg.policy,
            cache_dir=cfg.get("policy_cache_dir"),
        )
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if cfg.eval.get("show_solver_progress", False) and hasattr(solver, "callbacks"):
            solver.callbacks.append(CEMProgressCallback(solver))
        if wood_enabled(cfg, "eval"):
            policy_cls = WoodWorldModelPolicy
            kwargs = {"wood_cfg": cfg.wood}
        else:
            policy_cls = swm.policy.WorldModelPolicy
            kwargs = {}
        policy = policy_cls(
            solver=solver, config=config, process=process, transform=transform, **kwargs
        )

    else:
        policy = swm.policy.RandomPolicy()

    policy_cache_dir = cfg.get("policy_cache_dir") or swm.data.utils.get_cache_dir()
    results_path = (
        Path(policy_cache_dir, cfg.policy).parent if cfg.policy != "random" else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    progress_path = Path(
        cfg.eval.get("progress_path", results_path / f"{cfg.output.filename}.progress.jsonl")
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    attach_eval_progress(
        world,
        num_rollouts=cfg.eval.num_eval,
        eval_budget=cfg.eval.eval_budget,
        enabled=cfg.eval.get("show_progress", True),
        progress_path=progress_path,
    )
    video_context = (
        wood_video_panels(eval_episodes.tolist(), cfg.wood)
        if cfg.eval.get("save_video", False) and wood_enabled(cfg, "eval")
        else nullcontext()
    )
    try:
        with video_context:
            metrics = world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist(),
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist(),
                callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                video=results_path if cfg.eval.get("save_video", False) else None,
            )
    finally:
        world.close()
    end_time = time.time()
    if torch.cuda.is_available():
        print(
            {
                "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
