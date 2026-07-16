from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .utils import REQUIRED, SPLITS, sha256, write_manifest

MAPPING_FIELDS = (
    "atom_mapped_rxn_smiles",
    "atom_mapping_template",
    "atom_mapping_confident",
    "atom_mapping_source",
    "atom_mapping_status",
    "atom_mapping_error",
    "atom_mapping_input_hash",
    "atom_mapping_model_version",
)


DEFAULT_ROOT = Path("/home/adhil/github/EMULaToR/data/processed/datasets/reaction_outcome_dataset")
DEFAULT_BASELINES = Path("/home/adhil/github/EMULaToR/data/processed/baselines/GLN")


def resolve_split_dir(root: Path, group: str) -> Path:
    if group == "drfp_tanimoto_splits":
        return root / group / "threshold_0.11"
    return root / group


def resolve_master_path(root: Path) -> Path:
    for name in ("reaction_outcome_dataset_atom_mapped.parquet", "reaction_outcome_dataset.parquet"):
        path = root / name
        if path.exists():
            columns = set(pd.read_parquet(path, engine="pyarrow").columns)
            if set(MAPPING_FIELDS).issubset(columns):
                return path
    raise FileNotFoundError(f"no mapped master parquet found under {root}")


def resolve_split_path(root: Path, group: str, phase: str) -> Path:
    directory = resolve_split_dir(root, group)
    for name in (f"{phase}_atom_mapped.parquet", f"{phase}.parquet"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no mapped {phase} split found under {directory}")


def normalize_split(path: Path, master: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    frame = pd.read_parquet(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["rxn_smiles"].duplicated().any():
        raise ValueError(f"duplicate rxn_smiles in {path}")
    master_index = master.set_index("rxn_smiles", drop=False)
    matched = frame["rxn_smiles"].isin(master_index.index)
    for column in MAPPING_FIELDS:
        if column not in frame:
            frame[column] = pd.NA
        # A matching master row wins even when its mapping is null/failed.
        frame.loc[matched, column] = frame.loc[matched, "rxn_smiles"].map(master_index[column])
    audit: list[dict] = []
    for position, row in frame.iterrows():
        resolution = "master_mapping" if bool(matched.loc[position]) else "split_only_mapping"
        if resolution == "split_only_mapping":
            audit.append({"row": int(position), "rxn_smiles": row["rxn_smiles"], "resolution": resolution})
        mapping = row["atom_mapped_rxn_smiles"]
        status = row["atom_mapping_status"]
        reason = None
        if pd.isna(mapping):
            reason = "null_mapping"
        elif str(status) == "failed":
            reason = "failed_mapping_status"
        elif str(mapping).count(">") != 2:
            reason = "invalid_mapped_reaction"
        if reason:
            audit.append({"row": int(position), "rxn_smiles": row["rxn_smiles"], "resolution": resolution, "status": status, "error": row["atom_mapping_error"], "reason": reason})
    return frame, audit


def prepare(dataset_root: Path, output_root: Path, split_group: str) -> dict:
    master_path = resolve_master_path(dataset_root)
    master = pd.read_parquet(master_path)
    split_dir = resolve_split_dir(dataset_root, split_group)
    namespace = output_root / "namespaces" / split_group
    dropbox = namespace / "dropbox"
    data_name = "reaction_outcome_dataset"
    data_dir = dropbox / data_name
    data_dir.mkdir(parents=True, exist_ok=True)
    copied_dir = namespace / "splits"
    copied_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": str(master_path), "split_group": split_group, "master_rows": len(master), "dropbox": str(dropbox), "data_name": data_name, "splits": {}}
    for split in SPLITS:
        src = resolve_split_path(dataset_root, split_group, split)
        frame, audit = normalize_split(src, master)
        dst = copied_dir / f"{split}.parquet"
        frame.to_parquet(dst, index=False)
        raw = data_dir / f"raw_{split}.csv"
        # Only atom-mapping failures are excluded. Downstream template failures
        # are audits, never a reason to mutate a validated raw split view.
        excluded_rows = {item["row"] for item in audit if item.get("reason")}
        with raw.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "class", "reactants>reagents>production"])
            for position, row in frame.iterrows():
                if position in excluded_rows:
                    continue
                writer.writerow([position, "UNK", row["atom_mapped_rxn_smiles"]])
        audit_path = namespace / f"{split}_mapping_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
        report["splits"][split] = {"source": str(src), "rows": len(frame), "valid_rows": len(frame) - len(excluded_rows), "excluded_rows": len(excluded_rows), "split_only_rows": sum(item.get("resolution") == "split_only_mapping" for item in audit), "audit": str(audit_path), "sha256": sha256(dst)}
    write_manifest(namespace / "manifest.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--split-group", required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_root, args.split_group), indent=2))


if __name__ == "__main__":
    main()
