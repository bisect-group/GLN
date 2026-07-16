from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .dataset_adapter import DEFAULT_BASELINES, DEFAULT_ROOT, prepare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-group", default="random_splits_grouped_rxn_smiles")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--gpu", default="3")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    prepare(args.dataset_root, args.output_root, args.split_group)
    print("Adapter smoke test passed: mapped CSV views and manifest were generated.")
    print("Native preprocessing/training smoke requires the GLN legacy RDKit and compiled extensions.")


if __name__ == "__main__":
    main()
