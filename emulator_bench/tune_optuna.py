"""Safe Optuna launcher; architecture and preprocessing remain fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-group", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("optuna_search_space.json"))
    args = p.parse_args()
    # Keep the search space explicit and serializable. Training/evaluation can
    # consume these proposals on machines where the optional optuna package is
    # installed, without changing GLN architecture or preprocessing.
    space = {"split_group": args.split_group, "gpus": args.gpus.split(","), "trials": args.trials,
             "learning_rate": [1e-4, 3e-4, 1e-3], "batch_size": [32, 64],
             "neg_num": [32, 64], "dropout": [0.0, 0.1]}
    args.output.write_text(json.dumps(space, indent=2) + "\n")
    print(f"Wrote safe Optuna search space to {args.output}; launch trials with train.py.")


if __name__ == "__main__":
    main()
