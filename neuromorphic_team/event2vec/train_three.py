from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from .config import PROJECT_ROOT, Event2VecConfig, default_config, resolve_project_path
from .presets import apply_encoding_preset
from .train import run_training

ENCODINGS = ("rate", "latency", "delta")
DETACHED_CHILD_ENV = "EVENT2VEC_TRAIN_THREE_CHILD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3 Event2Vec models: rate, latency, delta")
    parser.add_argument(
        "--base-config",
        type=str,
        default="event2vec/default_config.json",
        help="Path to a base JSON config",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default=None,
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override the per-run output root directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch count for each encoding run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke run (1 epoch, tiny subset) for all encodings",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the foreground instead of spawning a detached background worker",
    )
    return parser.parse_args()


def load_config(path: str) -> Event2VecConfig:
    target = resolve_project_path(path)
    if target.exists():
        return Event2VecConfig.from_json(target)
    return default_config()


def apply_quick_overrides(cfg: Event2VecConfig) -> None:
    cfg.train.epochs = 1
    cfg.data.batch_size = 4
    cfg.data.max_tokens = min(cfg.data.max_tokens, 256)
    cfg.train.max_train_batches = 2
    cfg.train.max_val_batches = 1
    cfg.train.max_test_batches = 1
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False


def build_launch_artifacts() -> tuple[Path, Path]:
    launch_dir = PROJECT_ROOT / "runs" / "event2vec" / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        launch_dir / f"train_three_{stamp}.log",
        launch_dir / f"train_three_{stamp}.pid",
    )


def launch_background(args: argparse.Namespace) -> None:
    log_path, pid_path = build_launch_artifacts()
    cmd = [sys.executable, "-u", "-m", "event2vec.train_three", "--foreground"]

    if args.base_config != "event2vec/default_config.json":
        cmd.extend(["--base-config", args.base_config])
    if args.out_summary is not None:
        cmd.extend(["--out-summary", args.out_summary])
    if args.out_dir is not None:
        cmd.extend(["--out-dir", args.out_dir])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.quick:
        cmd.append("--quick")

    env = os.environ.copy()
    env[DETACHED_CHILD_ENV] = "1"

    with log_path.open("w") as log_file:
        child = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    pid_path.write_text(f"{child.pid}\n")
    print(f"Started background training with PID {child.pid}")
    print(f"Log file: {log_path}")
    print(f"PID file: {pid_path}")
    print(f"Tail logs with: tail -f {log_path}")


def main() -> None:
    args = parse_args()

    if not args.foreground and os.environ.get(DETACHED_CHILD_ENV) != "1":
        launch_background(args)
        return

    base_cfg = load_config(args.base_config)
    if args.out_dir is not None:
        base_cfg.result.out_dir = args.out_dir
    if args.epochs is not None:
        base_cfg.train.epochs = args.epochs

    out_summary = args.out_summary or f"{base_cfg.result.out_dir}/encoding_comparison/summary.json"
    records: list[dict] = []

    for encoding in ENCODINGS:
        cfg = Event2VecConfig.from_dict(base_cfg.to_dict())
        cfg.data.encoding = encoding
        apply_encoding_preset(cfg)
        cfg.result.run_name = encoding if not args.quick else f"{encoding}_quick"

        if args.quick:
            apply_quick_overrides(cfg)

        print(f"\n=== Training encoding: {encoding} ===")
        result = run_training(cfg)
        records.append(
            {
                "encoding": encoding,
                "run_dir": result["run_dir"],
                "selection": result["selection"],
                "test": result["test"],
                "test_last": result["test_last"],
            }
        )

    out_path = resolve_project_path(out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": records}, indent=2))
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
