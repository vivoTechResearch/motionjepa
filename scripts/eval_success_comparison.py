#!/usr/bin/env python3
"""Run and summarize success-rate comparisons for LeWorld variants.

Default grid:
  4 tasks x 5 wood methods x 2 planning horizons x 250 eval episodes.

By default the script uses the current eval.py with each run's saved weights and
wood configuration. Pass --use-code-snapshot to use a run's saved evaluator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


TASKS = {
    "cube": "ogcube",
    "pusht": "pusht",
    "reacher": "reacher",
    "tworoom": "tworoom",
}

METHODS = {
    "ours": "ours",
    "ours_idm": "ours_idm",
    "idm": "idm",
    "original": "original_lewm",
    "lewm_flat": "lewm_flat",
}

HORIZONS = {
    "25_step": {"goal_offset_steps": 25, "eval_budget": 50},
    "50_step": {"goal_offset_steps": 50, "eval_budget": 100},
}


@dataclass(frozen=True)
class EvalJob:
    task: str
    method: str
    horizon: str
    seed: int
    run_dir: Path
    epoch: int
    policy: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=Path(os.environ.get("MOTION_JEPA_LOG_ROOT", "outputs")))
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path(os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for eval subprocesses.")
    parser.add_argument("--num-eval", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--horizons", nargs="+", choices=sorted(HORIZONS), default=list(HORIZONS))
    parser.add_argument(
        "--solver-num-samples",
        type=int,
        default=None,
        help="Override the solver population size (CEM: num_samples).",
    )
    parser.add_argument(
        "--solver-n-steps",
        type=int,
        default=None,
        help="Override the solver iteration count (CEM: n_steps).",
    )
    parser.add_argument(
        "--solver-topk",
        type=int,
        default=None,
        help="Override the number of elite CEM candidates retained per iteration.",
    )
    parser.add_argument(
        "--plan-horizon",
        type=int,
        default=None,
        help="Override plan_config.horizon (world-model transitions per plan).",
    )
    parser.add_argument(
        "--receding-horizon",
        type=int,
        default=None,
        help="Override plan_config.receding_horizon (transitions executed before replanning).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Label included in the output directory and per-evaluation result filenames.",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Use a specific exported epoch for all runs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run evals even if cached result JSON exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running evaluations.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-subprocess", action="store_true", help="Print full eval subprocess output.")
    parser.add_argument(
        "--use-code-snapshot",
        action="store_true",
        help="Use each run's saved code_snapshot/eval.py instead of this repository's evaluator.",
    )
    args = parser.parse_args()
    if (args.solver_num_samples is None) != (args.solver_n_steps is None):
        parser.error("--solver-num-samples and --solver-n-steps must be supplied together")
    if args.solver_num_samples is not None and (args.solver_num_samples <= 0 or args.solver_n_steps <= 0):
        parser.error("solver population size and iteration count must be positive")
    if args.solver_topk is not None and args.solver_topk <= 0:
        parser.error("--solver-topk must be positive")
    if args.solver_topk is not None and args.solver_num_samples is not None and args.solver_topk > args.solver_num_samples:
        parser.error("--solver-topk cannot exceed --solver-num-samples")
    if args.plan_horizon is not None and args.plan_horizon <= 0:
        parser.error("--plan-horizon must be positive")
    if args.receding_horizon is not None and args.receding_horizon <= 0:
        parser.error("--receding-horizon must be positive")
    return args


def task_token(task: str) -> str:
    return TASKS[task]


def method_token(method: str) -> str:
    return METHODS[method]


def discover_run(weights_root: Path, task: str, method: str) -> Path:
    token_task = task_token(task)
    token_method = method_token(method)
    exact_run_marker = f"_{token_task}_{token_method}_wood"
    candidates = [
        p
        for p in weights_root.iterdir()
        if p.is_dir()
        and exact_run_marker in p.name
        and (p / "code_snapshot" / "eval.py").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run found for task={task} method={method} under {weights_root}")
    return sorted(candidates)[-1]


def latest_exported_epoch(run_dir: Path) -> int:
    export_root = run_dir / "checkpoints" / "checkpoints"
    epochs = []
    for path in export_root.glob("pretrained_epoch_*/weights.pt"):
        match = re.search(r"pretrained_epoch_(\d+)", str(path))
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No exported weights.pt found under {export_root}")
    return max(epochs)


def read_distractor_args(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        wood = cfg.get("wood")
        if wood and wood.get("enabled"):
            result = [
                "wood.enabled=true",
                "wood.eval=true",
                f"wood.seed={int(wood.get('seed', 3072))}",
                f"wood.texture_path={wood.get('texture_path', 'assets/wood_texture.png')}",
                f"wood.mask_mode={wood.get('mask_mode', 'white')}",
                f"wood.white_threshold={int(wood.get('white_threshold', 245))}",
                f"wood.background_tolerance={int(wood.get('background_tolerance', 8))}",
            ]
            if wood.get("static_background_path"):
                result.append(f"wood.static_background_path={wood.static_background_path}")
            return result
    except Exception:
        return []
    return []


def eval_config_has_key(run_dir: Path, task: str, key: str) -> bool:
    cfg_path = run_dir / "code_snapshot" / "config" / "eval" / f"{task}.yaml"
    if not cfg_path.exists():
        return False
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        return key in cfg
    except Exception:
        pattern = re.compile(rf"^{re.escape(key)}\s*:", re.MULTILINE)
        return bool(pattern.search(cfg_path.read_text()))


def parse_success_rate(text: str):
    for line in reversed(text.splitlines()):
        if "success_rate" not in line:
            continue
        try:
            parsed = ast.literal_eval(line.strip())
            if isinstance(parsed, dict) and "success_rate" in parsed:
                return float(parsed["success_rate"])
        except Exception:
            pass
        tail = line[line.find("success_rate") + len("success_rate") :]
        for token in tail.replace(":", " ").replace(",", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
    return None


def build_command(job: EvalJob, args, result_path: Path):
    hcfg = HORIZONS[job.horizon]
    if args.use_code_snapshot:
        eval_script = job.run_dir / "code_snapshot" / "eval.py"
    else:
        eval_script = Path(__file__).resolve().parents[1] / "eval.py"
    policy_cache_key = "policy_cache_dir" if eval_config_has_key(job.run_dir, job.task, "policy_cache_dir") else "+policy_cache_dir"
    cmd = [
        args.python,
        str(eval_script),
        "--config-name",
        job.task,
        f"policy={job.policy}",
        f"{policy_cache_key}={job.run_dir / 'checkpoints'}",
        f"cache_dir={args.dataset_cache_dir}",
        f"output.filename={result_path.name}",
        f"+eval.progress_path={result_path.with_suffix('.progress.jsonl')}",
        f"eval.num_eval={args.num_eval}",
        f"eval.goal_offset_steps={hcfg['goal_offset_steps']}",
        f"eval.eval_budget={hcfg['eval_budget']}",
        f"seed={job.seed}",
        "+eval.show_progress=true",
        "+eval.show_solver_progress=true",
    ]
    if args.solver_num_samples is not None:
        cmd.extend(
            [
                f"solver.num_samples={args.solver_num_samples}",
                f"solver.n_steps={args.solver_n_steps}",
            ]
        )
    if args.solver_topk is not None:
        cmd.append(f"solver.topk={args.solver_topk}")
    if args.plan_horizon is not None:
        cmd.append(f"plan_config.horizon={args.plan_horizon}")
    if args.receding_horizon is not None:
        cmd.append(f"plan_config.receding_horizon={args.receding_horizon}")
    cmd.extend(read_distractor_args(job.run_dir))
    return cmd


def run_eval(job: EvalJob, args, raw_dir: Path):
    run_label = f"__{args.run_name}" if args.run_name else ""
    result_path = raw_dir / f"{job.task}__{job.method}__{job.horizon}{run_label}__seed{job.seed}.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text())

    cmd = build_command(job, args, result_path)
    payload = {
        "task": job.task,
        "method": job.method,
        "horizon": job.horizon,
        "seed": job.seed,
        "num_eval": args.num_eval,
        "run_dir": str(job.run_dir),
        "epoch": job.epoch,
        "policy": job.policy,
        "command": cmd,
        "status": "dry_run" if args.dry_run else "started",
        "success_rate": None,
    }
    if args.dry_run:
        result_path.write_text(json.dumps(payload, indent=2))
        return payload

    returncode, output = run_streaming_subprocess(
        cmd,
        cwd=(job.run_dir / "code_snapshot") if args.use_code_snapshot else Path(__file__).resolve().parents[1],
        verbose=args.verbose_subprocess,
    )
    payload["status"] = "finished"
    payload["returncode"] = returncode
    payload["output"] = output[-12000:]
    payload["success_rate"] = parse_success_rate(output)
    result_path.write_text(json.dumps(payload, indent=2))
    return payload


def run_streaming_subprocess(cmd, cwd: Path, verbose: bool = False):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    chunks = []
    progress_buf = []
    printed_progress = False

    def flush_progress_segment(segment: str):
        nonlocal printed_progress
        if verbose:
            sys.stdout.write(segment)
            sys.stdout.flush()
            return
        clean = segment.strip()
        if "rollouts=" not in clean and "CEM plan=" not in clean:
            return
        printed_progress = True
        sys.stdout.write("\r  " + clean[-160:])
        sys.stdout.flush()

    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(1)
            if chunk == "" and proc.poll() is not None:
                break
            if not chunk:
                continue
            chunks.append(chunk)
            progress_buf.append(chunk)
            if chunk in ("\r", "\n"):
                flush_progress_segment("".join(progress_buf))
                progress_buf.clear()
    except KeyboardInterrupt:
        print("\nInterrupt received; stopping the active evaluation…", flush=True)
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if progress_buf:
        flush_progress_segment("".join(progress_buf))
    if printed_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return proc.wait(), "".join(chunks)


def summarize(results, num_eval: int):
    grouped = defaultdict(list)
    for row in results:
        if row.get("success_rate") is None:
            continue
        grouped[(row["task"], row["method"], row["horizon"])].append(float(row["success_rate"]))

    summary = []
    for (task, method, horizon), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            std_kind = "across_seeds"
        else:
            p = max(0.0, min(1.0, mean / 100.0))
            std = 100.0 * math.sqrt(p * (1.0 - p))
            std_kind = "episode_bernoulli"
        stderr = std / math.sqrt(len(values) if std_kind == "across_seeds" else num_eval)
        summary.append(
            {
                "task": task,
                "method": method,
                "horizon": horizon,
                "n_runs": len(values),
                "num_eval_per_run": num_eval,
                "success_rate_mean": mean,
                "success_rate_std": std,
                "success_rate_stderr": stderr,
                "std_kind": std_kind,
            }
        )
    return summary


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary):
    lines = [
        "| task | horizon | method | success rate mean | std | stderr | n | std kind |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {task} | {horizon} | {method} | {mean:.2f} | {std:.2f} | {stderr:.2f} | {n} | {kind} |".format(
                task=row["task"],
                horizon=row["horizon"],
                method=row["method"],
                mean=row["success_rate_mean"],
                std=row["success_rate_std"],
                stderr=row["success_rate_stderr"],
                n=row["n_runs"],
                kind=row["std_kind"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def plot_summary(path: Path, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(METHODS)
    colors = {
        "ours": "#2ca02c",
        "idm": "#1f77b4",
        "ours_idm": "#9467bd",
        "original": "#d62728",
        "ours_nosigregz": "#ff7f0e",
        "lewm_flat": "#17becf",
    }
    by_horizon = {h: [r for r in summary if r["horizon"] == h] for h in HORIZONS}
    fig, axes = plt.subplots(1, len(by_horizon), figsize=(15, 5), sharey=True)
    if len(by_horizon) == 1:
        axes = [axes]

    for ax, (horizon, rows) in zip(axes, by_horizon.items()):
        xs = list(range(len(TASKS)))
        width = 0.16
        for mi, method in enumerate(methods):
            vals = []
            errs = []
            for task in TASKS:
                match = next((r for r in rows if r["task"] == task and r["method"] == method), None)
                vals.append(match["success_rate_mean"] if match else 0.0)
                errs.append(match["success_rate_stderr"] if match else 0.0)
            offset = (mi - (len(methods) - 1) / 2) * width
            ax.bar([x + offset for x in xs], vals, width, yerr=errs, capsize=3, label=method, color=colors[method])
        ax.set_title(horizon)
        ax.set_xticks(xs)
        ax.set_xticklabels(TASKS.keys(), rotation=20)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("success rate (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_dir_name = f"{args.run_name}__{timestamp}" if args.run_name else timestamp
    out_dir = args.out_dir or (args.weights_root / "success_rate_comparison" / default_dir_name)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task in args.tasks:
        for method in args.methods:
            run_dir = discover_run(args.weights_root, task, method)
            epoch = args.epoch if args.epoch is not None else latest_exported_epoch(run_dir)
            policy = f"pretrained_epoch_{epoch:04d}"
            for horizon in args.horizons:
                for seed in args.seeds:
                    jobs.append(EvalJob(task, method, horizon, seed, run_dir, epoch, policy))

    manifest = [
        {
            "task": job.task,
            "method": job.method,
            "horizon": job.horizon,
            "seed": job.seed,
            "run_dir": str(job.run_dir),
            "epoch": job.epoch,
            "policy": job.policy,
        }
        for job in jobs
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    results = []
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.task} {job.method} {job.horizon} seed={job.seed} epoch={job.epoch}")
        result = run_eval(job, args, raw_dir)
        results.append(result)
        if result.get("success_rate") is not None:
            print(f"  success_rate={result['success_rate']:.2f}")
        elif args.dry_run:
            print("  dry-run")
        else:
            print(f"  no success_rate parsed; status={result.get('status')} returncode={result.get('returncode')}")
            if result.get("output"):
                print(result["output"][-1200:])

    write_csv(out_dir / "raw_results.csv", results)
    summary = summarize(results, args.num_eval)
    write_csv(out_dir / "summary.csv", summary)
    write_markdown(out_dir / "summary.md", summary)
    if args.plot and summary:
        plot_summary(out_dir / "success_rate_comparison.png", summary)

    print(f"\nWrote results to: {out_dir}")
    print(f"Summary table: {out_dir / 'summary.md'}")
    if args.plot and summary:
        print(f"Plot: {out_dir / 'success_rate_comparison.png'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Evaluation comparison interrupted.", flush=True)
        sys.exit(130)
