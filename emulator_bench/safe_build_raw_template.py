"""Exception-safe, parallel replacement for GLN's raw template driver."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import os
import warnings
from pathlib import Path

from tqdm import tqdm

from gln.mods.rdchiral.template_extractor import extract_from_reaction

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_row(item: tuple[int, list[str]]) -> tuple[int, dict | None, str | None]:
    idx, row = item
    try:
        reactants, _, products = row[2].split(">")
        reaction = {"_id": row[0], "reactants": reactants, "products": products}
        with contextlib.redirect_stdout(io.StringIO()):
            template = extract_from_reaction(reaction)
        if template is None:
            return idx, None, "template extractor returned None"
        return idx, template, None
    except Exception as exc:  # row-local RDKit failures must not kill the pool
        return idx, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-dropbox", required=True)
    p.add_argument("-data_name", required=True)
    p.add_argument("-save_dir", required=True)
    p.add_argument("-num_cores", type=int, default=1)
    args = p.parse_args()
    raw = Path(args.dropbox) / args.data_name / "raw_train.csv"
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    with raw.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    results: list[tuple[int, dict | None, str | None] | None] = [None] * len(rows)
    workers = max(1, min(args.num_cores, len(rows)))
    with mp.Pool(workers, maxtasksperchild=500) as pool:
        tasks = ((idx, row) for idx, row in enumerate(rows))
        for result in tqdm(pool.imap_unordered(extract_row, tasks, chunksize=1), total=len(rows), desc="Extract reaction templates", unit="reaction"):
            results[result[0]] = result
    proc_path = save / "proc_train_singleprod.csv"
    failed_path = save / "failed_template.csv"
    with proc_path.open("w", newline="") as good, failed_path.open("w", newline="") as bad:
        good_writer = csv.writer(good)
        bad_writer = csv.writer(bad)
        good_writer.writerow(["id", "class", "rxn_smiles", "retro_templates"])
        bad_writer.writerow(["id", "class", "rxn_smiles", "err_msg"])
        good_count = failed_count = 0
        for idx, template, error in results:
            row = rows[idx]
            if template is not None and "reaction_smarts" in template:
                good_writer.writerow([row[0], row[1], row[2], template["reaction_smarts"]])
                good_count += 1
            else:
                bad_writer.writerow([row[0], row[1], row[2], error or template.get("err_msg", "template extraction failed")])
                failed_count += 1
    print(f"[templates] successful={good_count} failed={failed_count} workers={workers}", flush=True)


if __name__ == "__main__":
    main()
