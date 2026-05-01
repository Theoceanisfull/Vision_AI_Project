from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["image.cmap"] = "coolwarm"

SENSOR_HEIGHT = 180
SENSOR_WIDTH = 240


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "ASLDVS").exists():
            return candidate
    raise FileNotFoundError("Could not find data/ASLDVS from the current working directory.")


def flip_y_for_display(raw_y: np.ndarray) -> np.ndarray:
    return SENSOR_HEIGHT - 1 - raw_y.astype(np.int64)


def load_asldvs_mat(path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(path)
    return {name: mat[name].reshape(-1) for name in ("x", "y", "ts", "pol")}


def make_time_frames(event_dict: dict[str, np.ndarray], n_frames: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x = event_dict["x"].astype(np.int64)
    display_y = flip_y_for_display(event_dict["y"])
    ts = event_dict["ts"].astype(np.int64)
    signed_pol = np.where(event_dict["pol"] > 0, 1, -1).astype(np.int8)

    edges = np.linspace(ts.min(), ts.max() + 1, n_frames + 1, dtype=np.int64)
    frames = np.zeros((n_frames, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.int16)

    for frame_idx in range(n_frames):
        mask = (ts >= edges[frame_idx]) & (ts < edges[frame_idx + 1])
        np.add.at(frames[frame_idx], (display_y[mask], x[mask]), signed_pol[mask])

    return frames, edges


def render_animation_mp4(
    *,
    sample_path: Path,
    output_path: Path,
    n_frames: int,
    fps: int,
    hold_last_frames: int,
) -> None:
    events = load_asldvs_mat(sample_path)
    anim_frames, anim_edges = make_time_frames(events, n_frames=n_frames)
    anim_limit = float(np.abs(anim_frames).max())
    anim_limit = anim_limit if anim_limit > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    image = ax.imshow(anim_frames[0], vmin=-anim_limit, vmax=anim_limit)
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("display y")
    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(output_path.stem + "._tmp_h264" + output_path.suffix)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(temp_output_path),
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        total_frames = n_frames + max(0, hold_last_frames)
        assert process.stdin is not None
        for frame_idx in range(total_frames):
            src_idx = min(frame_idx, n_frames - 1)
            image.set_data(anim_frames[src_idx])
            title.set_text(
                f"Frame {src_idx + 1}/{n_frames}: {int(anim_edges[src_idx])} - {int(anim_edges[src_idx + 1])} us"
            )
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            process.stdin.write(rgb.astype(np.uint8).tobytes())
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        plt.close(fig)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}: {stderr}")
        temp_output_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the last ASL-DVS viewer notebook cell as an MP4.")
    parser.add_argument("--class-name", default="f", help="ASL-DVS class folder name.")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based sample index inside the class folder.")
    parser.add_argument("--frames", type=int, default=24, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=6, help="Video frames per second.")
    parser.add_argument(
        "--hold-last-frames",
        type=int,
        default=6,
        help="Number of extra frames to repeat the last frame for easier slide embedding.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults to Analytics/asldvs_viewer_<class>_<index>.mp4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path.cwd().resolve())
    data_root = project_root / "data" / "ASLDVS"

    sample_paths = sorted((data_root / args.class_name).glob("*.mat"))
    if not sample_paths:
        raise FileNotFoundError(f"No .mat files found for class {args.class_name!r} in {data_root}.")
    if args.sample_index < 0 or args.sample_index >= len(sample_paths):
        raise IndexError(
            f"sample_index {args.sample_index} is out of range for class {args.class_name!r} "
            f"with {len(sample_paths)} samples."
        )

    sample_path = sample_paths[args.sample_index]
    output_path = (
        Path(args.output)
        if args.output
        else project_root / "Analytics" / f"asldvs_viewer_{args.class_name}_{args.sample_index:04d}.mp4"
    )

    render_animation_mp4(
        sample_path=sample_path,
        output_path=output_path,
        n_frames=args.frames,
        fps=args.fps,
        hold_last_frames=args.hold_last_frames,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
