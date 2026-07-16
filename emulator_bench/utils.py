from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED = {"rxn_smiles", "atom_mapped_rxn_smiles", "reactants", "products"}
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_validate_split(path: Path, master: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["rxn_smiles"].duplicated().any():
        raise ValueError(f"duplicate rxn_smiles in {path}")
    master_keys = set(master["rxn_smiles"].astype(str))
    unknown = set(frame["rxn_smiles"].astype(str)) - master_keys
    if unknown:
        raise ValueError(f"{path} contains {len(unknown)} reactions absent from refreshed master")
    failed = frame.get("atom_mapping_status", pd.Series(index=frame.index, dtype=object)).eq("failed")
    if failed.any():
        frame = frame.loc[~failed].copy()
    if frame["atom_mapped_rxn_smiles"].isna().any():
        raise ValueError(f"{path} contains null atom_mapped_rxn_smiles")
    return frame


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
