#!/usr/bin/env python3
"""Keep only the newest N checkpoint files in a directory; delete the rest.

Checkpoints are matched by extension (.pt, .pth, .ckpt). Files are ranked by
creation time when available (st_birthtime), otherwise by last modification time.

Examples:
  python cleanup_checkpoints.py imagenet_es_v2_outputs/checkpoints 3
  python cleanup_checkpoints.py checkpoints 5 --dry-run
  python cleanup_checkpoints.py runs/ --keep 2 --recursive
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def file_sort_time(path: Path) -> float:
    st = path.stat()
    if hasattr(st, "st_birthtime"):
        return st.st_birthtime
    return st.st_mtime


def iter_checkpoints(directory: Path, recursive: bool) -> list[Path]:
    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files = [
        p
        for p in candidates
        if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    files.sort(key=file_sort_time, reverse=True)
    return files


def human_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete old checkpoint files, keeping only the newest N by creation time."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "keep",
        type=int,
        help="Number of newest checkpoints to keep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for checkpoint files",
    )
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    keep = args.keep

    if keep < 0:
        print("error: keep must be >= 0", file=sys.stderr)
        return 1
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    checkpoints = iter_checkpoints(directory, recursive=args.recursive)
    if not checkpoints:
        print(f"No checkpoint files found in {directory}")
        return 0

    keep_files = checkpoints[:keep]
    delete_files = checkpoints[keep:]

    print(f"Directory: {directory}")
    print(f"Found {len(checkpoints)} checkpoint file(s); keeping {len(keep_files)}, deleting {len(delete_files)}")

    if keep_files:
        print("\nKeeping:")
        for path in keep_files:
            ts = file_sort_time(path)
            print(f"  {path.name}  ({human_size(path.stat().st_size)}, {ts:.0f})")

    if not delete_files:
        print("\nNothing to delete.")
        return 0

    freed = 0
    print("\nDeleting:" if not args.dry_run else "\nWould delete:")
    for path in delete_files:
        size = path.stat().st_size
        print(f"  {path}  ({human_size(size)})")
        if not args.dry_run:
            path.unlink()
            freed += size

    if args.dry_run:
        total = sum(p.stat().st_size for p in delete_files)
        print(f"\nDry run: would free {human_size(total)}")
    else:
        print(f"\nFreed {human_size(freed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
