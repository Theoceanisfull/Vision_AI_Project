"""
Downloads directly into data/ inside this repo.

Run:
  env/python.exe scripts/download_data.py --dataset abd0kamel/asl-citizen
  env/python.exe scripts/download_data.py --dataset risangbaskoro/wlasl-processed
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["KAGGLEHUB_CACHE"] = os.path.join(REPO_ROOT, "data")

KNOWN_DATASETS = {
    "abd0kamel/asl-citizen":        ("ASL-Citizen", "~26 GB"),
    "risangbaskoro/wlasl-processed": ("WLASL processed", "~4.8 GB"),
}


def find_video_dir(root):
    """Return the directory containing the most .mp4 files."""
    best = (0, None)
    for dirpath, _, files in os.walk(root):
        count = sum(1 for f in files if f.lower().endswith('.mp4'))
        if count > best[0]:
            best = (count, dirpath)
    return best[1]


def find_file(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='abd0kamel/asl-citizen',
        help='Kaggle dataset slug, e.g. abd0kamel/asl-citizen'
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError:
        print("kagglehub not found. Run: pip install kagglehub")
        sys.exit(1)

    label, size = KNOWN_DATASETS.get(args.dataset, (args.dataset, "unknown size"))
    print(f"Dataset   : {label} ({args.dataset})")
    print(f"Size      : {size}")
    print(f"Cache dir : {os.environ['KAGGLEHUB_CACHE']}")
    print("Downloading...\n")

    try:
        dl_path = kagglehub.dataset_download(args.dataset)
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nMake sure your Kaggle credentials are configured:")
        print("  Option 1: ~/.kaggle/kaggle.json")
        print("  Option 2: env vars KAGGLE_USERNAME and KAGGLE_KEY")
        sys.exit(1)

    print(f"\nDownloaded to: {dl_path}")

    video_dir = find_video_dir(dl_path)
    if video_dir:
        mp4_count = sum(1 for f in os.listdir(video_dir) if f.lower().endswith('.mp4'))
        print(f"Videos    : {video_dir}  ({mp4_count} .mp4 files)")

    # WLASL-specific: show JSON path for config
    if 'wlasl' in args.dataset.lower():
        json_file = find_file(dl_path, "WLASL_v0.3.json")
        if json_file:
            print(f"JSON      : {json_file}")
        print("\n--- Paste into configs/config.yaml ---")
        if json_file:
            print(f"  json_path: \"{json_file.replace(os.sep, '/')}\"")
        if video_dir:
            print(f"  video_dir: \"{video_dir.replace(os.sep, '/')}\"")
        print("--------------------------------------")

    print("\nDone.")


if __name__ == "__main__":
    main()
