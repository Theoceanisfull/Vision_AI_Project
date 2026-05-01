#!/usr/bin/env python3
"""Download and unpack ASL-DVS into a layout that ``tonic.datasets.ASLDVS`` can reuse."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


EXPECTED_MAT_FILES = 100_800
OPENI_REPO_ID = "OpenI/ASLDVS"
OPENI_FILENAME = "ASLDVS/ICCV2019_DVS_dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-to",
        default="data",
        help="Root directory passed to tonic.datasets.ASLDVS(save_to=...).",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Where the OpenI mirror archive should be stored before extraction.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel downloads for the OpenI client.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an already-downloaded ICCV2019_DVS_dataset.zip.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip instantiating tonic.datasets.ASLDVS at the end.",
    )
    return parser.parse_args()


def count_mat_files(target_dir: Path) -> int:
    return sum(1 for _ in target_dir.rglob("*.mat"))


def tonic_cache_ready(target_dir: Path) -> bool:
    return (target_dir / "ASLDVS.zip").exists() and count_mat_files(target_dir) >= EXPECTED_MAT_FILES


def ensure_openi_archive(download_dir: Path, max_workers: int) -> Path:
    try:
        from openi.refactor.sdk import notice, openi_download_file
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the OpenI client first with `python3 -m pip install --user openi`."
        ) from exc

    notice(off=True)
    openi_download_file(
        repo_id=OPENI_REPO_ID,
        filename=OPENI_FILENAME,
        local_dir=download_dir,
        max_workers=max_workers,
    )

    archive = find_openi_archive(download_dir)
    if archive is None:
        raise SystemExit(f"Could not find ICCV2019_DVS_dataset.zip under {download_dir}.")
    return archive


def find_openi_archive(download_dir: Path) -> Path | None:
    exact = sorted(download_dir.rglob("ICCV2019_DVS_dataset.zip"))
    if exact:
        return exact[0]

    cached = sorted(download_dir.rglob(".openi--cache--ICCV2019_DVS_dataset.zip"))
    if cached:
        return cached[0]

    return None


def extract_outer_zip(outer_zip: Path, target_dir: Path) -> Path:
    temp_dir = target_dir / "_openi_outer"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(outer_zip) as zf:
        zf.extractall(temp_dir)

    return temp_dir


def extract_inner_archives(outer_dir: Path, target_dir: Path) -> int:
    inner_archives = sorted(outer_dir.rglob("*.zip"))
    if not inner_archives:
        raise SystemExit(f"No per-class zip files were found under {outer_dir}.")

    for archive in inner_archives:
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)

    return len(inner_archives)


def ensure_tonic_sentinel(target_dir: Path, source_archive: Path) -> None:
    sentinel = target_dir / "ASLDVS.zip"
    if sentinel.exists() or sentinel.is_symlink():
        return

    try:
        sentinel.symlink_to(source_archive.resolve())
    except OSError:
        sentinel.touch()


def verify_with_tonic(save_to_root: Path) -> None:
    try:
        from tonic.datasets import ASLDVS
    except ImportError as exc:
        raise SystemExit("Could not import tonic to verify the dataset.") from exc

    dataset = ASLDVS(str(save_to_root))
    print(f"tonic verification OK: {len(dataset)} samples")


def main() -> int:
    args = parse_args()

    save_to_root = Path(args.save_to).expanduser().resolve()
    target_dir = save_to_root / "ASLDVS"
    download_dir = (
        Path(args.download_dir).expanduser().resolve()
        if args.download_dir
        else save_to_root / "openi_asldvs"
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    if tonic_cache_ready(target_dir):
        print(f"ASL-DVS already looks ready in {target_dir}.")
        if not args.skip_verify:
            verify_with_tonic(save_to_root)
        return 0

    archive = find_openi_archive(download_dir)
    if archive is None and args.skip_download:
        raise SystemExit(
            f"--skip-download was set, but no ICCV2019_DVS_dataset.zip was found under {download_dir}."
        )

    if archive is None:
        archive = ensure_openi_archive(download_dir, args.max_workers)

    print(f"Using archive: {archive}")
    outer_dir = extract_outer_zip(archive, target_dir)
    inner_count = extract_inner_archives(outer_dir, target_dir)
    ensure_tonic_sentinel(target_dir, archive)

    mat_count = count_mat_files(target_dir)
    print(f"Extracted {inner_count} inner archives and found {mat_count} .mat files.")

    if mat_count < EXPECTED_MAT_FILES:
        raise SystemExit(
            f"Expected at least {EXPECTED_MAT_FILES} .mat files, but found only {mat_count}."
        )

    shutil.rmtree(outer_dir)

    if not args.skip_verify:
        verify_with_tonic(save_to_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
