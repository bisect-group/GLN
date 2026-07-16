from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dropbox", type=Path, required=True)
    p.add_argument("--data-name", default="reaction_outcome_dataset")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "gln/test/main_test.py", "-dropbox", str(args.dropbox), "-data_name", args.data_name, "-save_dir", str(args.save_dir), "-tpl_name", "default", "-f_atoms", str(args.dropbox / f"cooked_{args.data_name}" / "atom_list.txt"), "-topk", "50", "-beam_size", "50", "-gpu", str(args.gpu), "-num_parts", "1"]
    if args.checkpoint:
        cmd.extend(["-model_for_test", str(args.checkpoint)])
    args.save_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    subprocess.run(cmd, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
