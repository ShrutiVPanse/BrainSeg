#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path


def require_env_path(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        print(f"ERROR: environment variable {var} is required.", file=sys.stderr)
        sys.exit(1)
    return Path(value).expanduser().resolve()


def collect_post_scans(data_dir: Path):
    scans = []
    for path in data_dir.rglob("*.nii.gz"):
        if path.name.lower().endswith("post.nii.gz"):
            scans.append(path.resolve())
    return sorted(scans)


def main() -> None:
    root = require_env_path("ROOT")
    data = require_env_path("DATA")

    script_dir = Path(__file__).resolve().parent
    checkpoint = root / "weights" / "sam_vit_h_4b8939.pth"
    if not checkpoint.exists():
        print(f"ERROR: missing checkpoint {checkpoint}", file=sys.stderr)
        sys.exit(1)

    out_root = root / "runs_sam"
    out_root.mkdir(parents=True, exist_ok=True)

    scans = collect_post_scans(data)
    if not scans:
        print(f"No POST scans found under {data}", file=sys.stderr)
        return

    for nii_path in scans:
        parent = nii_path.parent
        grandparent = parent.parent

        if parent == grandparent:
            print(f"Skipping {nii_path}: expected subject/date folders", file=sys.stderr)
            continue

        subj = grandparent.name
        date = parent.name
        if not subj or not date:
            print(f"Skipping {nii_path}: unable to determine subject/date", file=sys.stderr)
            continue

        out_slices = root / "data" / "slices" / subj / date / "POST"
        out_run = out_root / subj / date / "POST"
        out_slices.mkdir(parents=True, exist_ok=True)
        out_run.mkdir(parents=True, exist_ok=True)

        print(f">> {subj} {date}", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(script_dir / "prep_slices.py"),
                "--nifti",
                str(nii_path),
                "--outdir",
                str(out_slices),
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(script_dir / "run_sam_on_slices.py"),
                "--slices_dir",
                str(out_slices),
                "--checkpoint",
                str(checkpoint),
                "--model_type",
                "vit_h",
                "--outdir",
                str(out_run),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
