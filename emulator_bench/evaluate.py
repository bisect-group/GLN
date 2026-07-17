from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dropbox", type=Path, required=True)
    p.add_argument("--data-name", default="reaction_outcome_dataset")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--gpu", default="0")
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--evaluation-cache-root", type=Path)
    p.add_argument("--rebuild-evaluation-cache", action="store_true")
    args = p.parse_args()
    if args.checkpoint is None:
        p.error("--checkpoint is required: benchmark evaluation always uses final model-9")
    from .optimized_evaluator import main as optimized_main
    import sys
    sys.argv = [sys.argv[0], "--dropbox", str(args.dropbox), "--data-name", args.data_name,
               "--save-dir", str(args.save_dir), "--checkpoint", str(args.checkpoint), "--gpu", str(args.gpu),
               "--eval-workers", str(args.eval_workers)] + (
        ["--evaluation-cache-root", str(args.evaluation_cache_root)] if args.evaluation_cache_root else []
    ) + (
        ["--rebuild-evaluation-cache"] if args.rebuild_evaluation_cache else []
    )
    optimized_main()


if __name__ == "__main__":
    main()
