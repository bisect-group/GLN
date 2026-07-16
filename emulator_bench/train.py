from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CPU_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def configure_child_env() -> dict[str, str]:
    """Keep concurrent GLN workers from creating oversized CPU thread pools."""
    env = dict(os.environ)
    for name in CPU_THREAD_ENV:
        env.setdefault(name, "1")
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dropbox", type=Path, required=True)
    p.add_argument("--data-name", default="reaction_outcome_dataset")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=19260817)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--neg-num", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data = args.data_name
    cmd = [sys.executable, "-m", "emulator_bench.training_entrypoint", "-gm", "mean_field", "-fp_degree", "2", "-neg_sample", "all", "-att_type", "bilinear", "-gnn_out", "last", "-tpl_enc", "deepset", "-subg_enc", "mean_field", "-latent_dim", "128", "-bn", "True", "-gen_method", "weighted", "-retro_during_train", "True", "-neg_num", str(args.neg_num), "-embed_dim", "256", "-readout_agg_type", "max", "-act_func", "relu", "-act_last", "True", "-max_lv", "3", "-dropbox", str(args.dropbox), "-data_name", data, "-save_dir", str(args.save_dir), "-tpl_name", "default", "-f_atoms", str(args.dropbox / f"cooked_{data}" / "atom_list.txt"), "-iters_per_val", "3000", "-gpu", str(args.gpu), "-topk", "50", "-beam_size", "50", "-num_parts", "1", "-num_epochs", str(args.num_epochs), "-learning_rate", str(args.learning_rate), "-batch_size", str(args.batch_size), "-dropout", str(args.dropout), "-seed", str(args.seed)]
    if args.resume_from:
        cmd.extend(["-init_model_dump", str(args.resume_from)])
    args.save_dir.mkdir(parents=True, exist_ok=True)
    env = configure_child_env()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    subprocess.run(cmd, cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
