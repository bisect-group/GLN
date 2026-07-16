"""Configure process-wide PyTorch runtime settings before starting GLN."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch


def configure_torch_runtime() -> tuple[int, int]:
    """Cap CPU thread pools before GLN imports model code."""
    intra_threads = int(os.environ.get("EMULATOR_BENCH_TORCH_THREADS", "1"))
    interop_threads = int(os.environ.get("EMULATOR_BENCH_TORCH_INTEROP_THREADS", "1"))
    if intra_threads < 1 or interop_threads < 1:
        raise ValueError("EMULATOR_BENCH_TORCH_THREADS and EMULATOR_BENCH_TORCH_INTEROP_THREADS must be positive")
    torch.set_num_threads(intra_threads)
    torch.set_num_interop_threads(interop_threads)
    return intra_threads, interop_threads


def main() -> None:
    intra_threads, interop_threads = configure_torch_runtime()
    print(f"[emulator_bench] torch CPU threads: intra-op={intra_threads}, inter-op={interop_threads}", flush=True)
    repo = Path(__file__).resolve().parents[1]
    sys.argv[0] = str(repo / "gln" / "training" / "main.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
