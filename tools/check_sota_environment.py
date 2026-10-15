"""Validate the local data layout and optional NVIDIA runtime for SOTA notebooks.

This script never downloads competition data. The competition rules restrict the
data to registered participants, so each user must place their own authorized
copy in DATA_DIR.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path


REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "customers.csv",
    "products.csv",
    "stores.csv",
    "calendar.csv",
    "sample_submission.csv",
)


def check_data(data_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} required competition file(s) in {data_dir}: "
            + ", ".join(missing)
        )
    return list(REQUIRED_FILES)


def check_modules() -> None:
    required = ("numpy", "pandas", "sklearn", "catboost", "xgboost", "torch")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing Python package(s): "
            + ", ".join(missing)
            + ". Install requirements-gpu.txt in the selected environment."
        )


def gpu_info() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "nvidia-smi was not available. Use an NVIDIA runtime or run the "
            "CPU-compatible baseline instead."
        ) from exc
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "data")),
        help="Directory containing the seven authorized competition CSV files.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip NVIDIA and GPU-package checks.",
    )
    args = parser.parse_args()

    check_data(args.data_dir)
    print(f"Data layout: OK ({args.data_dir})")
    if args.cpu_only:
        print("GPU checks: skipped (--cpu-only)")
        return
    check_modules()
    print("Python packages: OK")
    print("NVIDIA GPU:")
    print(gpu_info())


if __name__ == "__main__":
    main()
