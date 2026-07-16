from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dropbox", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split-groups", nargs="+", required=True)
    p.add_argument("--gpus", nargs="+", default=["0"])
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for i, group in enumerate(args.split_groups):
        gpu = args.gpus[i % len(args.gpus)]
        out = args.output_root / "runs" / group
        subprocess.Popen(["python", "-m", "emulator_bench.train", "--dropbox", str(args.dropbox), "--save-dir", str(out), "--gpu", gpu], cwd=repo)


if __name__ == "__main__":
    main()
