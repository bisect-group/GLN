"""Resilient GLN center mapping with isolated workers and failure reporting."""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import pickle as cp
import signal
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

from gln.common.cmd_args import cmd_args
from gln.common.mol_utils import smarts_has_useless_parentheses

smiles_cano_map = {}
prod_center_mols = []
smarts_type_set = defaultdict(set)
TASK_TIMEOUT = int(os.environ.get("EMULATOR_BENCH_CENTER_TIMEOUT", "60"))


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _find_edges(task):
    idx, rxn_type, original_smiles = task
    try:
        smiles = smiles_cano_map[original_smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ("failed", idx, rxn_type, original_smiles, "invalid molecule")
        centers = []
        for center_idx, (sm_center, center_mol) in enumerate(prod_center_mols):
            if center_mol is None or rxn_type not in smarts_type_set[sm_center]:
                continue
            if mol.HasSubstructMatch(center_mol):
                centers.append(str(center_idx))
        return ("ok", idx, rxn_type, smiles, " ".join(centers) if centers else None)
    except BaseException as exc:  # keep one bad/interrupting task from killing the pool
        return ("failed", idx, rxn_type, original_smiles, f"{type(exc).__name__}: {exc}")
def _supervised_map(tasks, workers):
    """Map centers in one persistent worker pool.

    Constructing a new RDKit process pool for every handful of reactions made
    center mapping spend most of its time in process startup.  Tasks catch
    their own errors, so a persistent pool is both faster and reports useful
    progress continuously through the caller's tqdm iterator.
    """
    if not tasks:
        return []
    with mp.Pool(max(1, min(workers, len(tasks))), initializer=_ignore_sigint, maxtasksperchild=500) as pool:
        yield from pool.imap_unordered(_find_edges, tasks, chunksize=1)


def main() -> None:
    global smiles_cano_map, prod_center_mols, smarts_type_set
    save_dir = Path(cmd_args.save_dir)
    with (save_dir / "../cano_smiles.pkl").resolve().open("rb") as f:
        smiles_cano_map = cp.load(f)
    with (save_dir / "cano_smarts.pkl").open("rb") as f:
        smarts_cano_map = cp.load(f)
    with (save_dir / "prod_cano_smarts.txt").open() as f:
        prod_cano_smarts = [line.strip() for line in f if line.strip()]
    prod_center_mols = [(sm, Chem.MolFromSmarts(sm)) for sm in tqdm(
        prod_cano_smarts, desc="Build product center molecules", unit="SMARTS")]

    with (save_dir / "templates.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        tpl_idx, type_idx = header.index("retro_templates"), header.index("class")
        for row in reader:
            sm_prod, _, _ = row[tpl_idx].split(">")
            if smarts_has_useless_parentheses(sm_prod):
                sm_prod = sm_prod[1:-1]
            smarts_type_set[smarts_cano_map[sm_prod]].add(row[type_idx])

    raw_root = Path(cmd_args.dropbox) / cmd_args.data_name
    out_folder = save_dir / f"np-{cmd_args.num_parts}"
    out_folder.mkdir(parents=True, exist_ok=True)
    completion_marker = save_dir / "find_centers.complete"
    completion_marker.unlink(missing_ok=True)
    failed_path = save_dir / "failed_centers.csv"
    failed = failed_path.open("w", newline="")
    failed_writer = csv.writer(failed)
    failed_writer.writerow(["phase", "index", "class", "smiles", "error"])
    try:
        for phase in ("train", "val", "test"):
            with (raw_root / f"raw_{phase}.csv").open(newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rxn_idx, type_idx = header.index("reactants>reagents>production"), header.index("class")
                tasks = [(idx, row[type_idx], row[rxn_idx].split(">", 2)[2]) for idx, row in enumerate(reader)]
            local = [None] * len(tasks)
            for result in tqdm(_supervised_map(tasks, max(1, cmd_args.num_cores)), total=len(tasks), desc=f"Find centers ({phase})", unit="reaction"):
                status, idx, rxn_type, smiles, centers = result
                if status == "ok":
                    # ``load_center_maps`` indexes (class, canonical_smiles)
                    # after reading the on-disk ``smiles,class,centers`` CSV.
                    # Preserve that column order; reversing these values makes
                    # every later reaction lookup miss silently.
                    local[idx] = (smiles, rxn_type, centers)
                else:
                    failed_writer.writerow([phase, idx, rxn_type, smiles, centers])
            failed.flush()
            output = out_folder / f"{phase}-prod_center_maps-part-0.csv"
            with output.open("w", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["smiles", "class", "centers"])
                writer.writerows(row for row in local if row is not None and row[2] is not None)
    finally:
        failed.close()
    completion_marker.write_text("completed\n")
    with failed_path.open() as report:
        failure_count = max(0, sum(1 for _ in report) - 1)
    print(f"[centers] completed; failures={failure_count}", flush=True)


if __name__ == "__main__":
    main()
