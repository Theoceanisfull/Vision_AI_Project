from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from .config import PROJECT_ROOT, SNNConfig, default_config, resolve_project_path
from .presets import apply_encoding_preset
from .t_loop import run_training

ENCODINGS = ("rate", "latency", "delta")
DETACHED_CHILD_ENV = "SCNN_TRAIN_THREE_BASE_CHILD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3 base Conv-SNN models: rate, latency, delta")
    parser.add_argument(
        "--base-config",
        type=str,
        default="models/scnn/default_config.json",
        help="Path to a base JSON config",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="runs/scnn/base_encoding_comparison/summary.json",
        help="Output summary JSON path",
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


def load_config(path: str) -> SNNConfig:
    p = resolve_project_path(path)
    if p.exists():
        return SNNConfig.from_json(p)
    return default_config()


def apply_quick_overrides(cfg: SNNConfig) -> None:
    cfg.train.epochs = 1
    cfg.data.batch_size = 4
    cfg.data.num_steps = 8
    cfg.train.max_train_batches = 2
    cfg.train.max_val_batches = 1
    cfg.train.max_test_batches = 1


def build_launch_artifacts() -> tuple[Path, Path]:
    launch_dir = PROJECT_ROOT / "runs" / "scnn" / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        launch_dir / f"train_three_base_{stamp}.log",
        launch_dir / f"train_three_base_{stamp}.pid",
    )


def launch_background(args: argparse.Namespace) -> None:
    log_path, pid_path = build_launch_artifacts()
    cmd = [sys.executable, "-u", "-m", "models.scnn.train_three_base", "--foreground"]

    if args.base_config != "models/scnn/default_config.json":
        cmd.extend(["--base-config", args.base_config])
    if args.out_summary != "runs/scnn/base_encoding_comparison/summary.json":
        cmd.extend(["--out-summary", args.out_summary])
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

    records: list[dict] = []

    for enc in ENCODINGS:
        cfg = SNNConfig.from_dict(base_cfg.to_dict())
        cfg.model.architecture = "base"
        cfg.data.encoding = enc
        apply_encoding_preset(cfg)
        cfg.result.run_name = f"base_{enc}" if not args.quick else f"base_{enc}_quick"

        if args.quick:
            apply_quick_overrides(cfg)

        print(f"\n=== Training encoding: {enc} ===")
        result = run_training(cfg)

        record = {
            "encoding": enc,
            "run_dir": result["run_dir"],
            "selection": result["selection"],
            "test": result["test"],
            "test_last": result["test_last"],
        }
        records.append(record)

    out_path = resolve_project_path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
