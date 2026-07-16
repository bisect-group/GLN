"""Content-addressed inventory for reaction-intrinsic preprocessing inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .dataset_adapter import MAPPING_FIELDS, resolve_master_path, resolve_split_path
from .stage_cache import CACHE_SCHEMA
from .utils import SPLITS, sha256

def cache_fingerprint(dataset_root: Path, split_groups: list[str], repo: Path | None = None) -> tuple[str, dict]:
    """Fingerprint all inputs that can affect reaction-intrinsic preprocessing."""
    extractor = (repo / "gln/mods/rdchiral/template_extractor.py") if repo else None
    graph_builder = (repo / "gln/mods/mol_gnn/mol_utils.py") if repo else None
    try:
        import rdkit
        rdkit_version = rdkit.__version__
    except Exception:
        rdkit_version = "unavailable"
    sources = []
    master_path = resolve_master_path(dataset_root)
    sources.append({"group": "master", "phase": "master", "path": str(master_path), "sha256": sha256(master_path)})
    for group in split_groups:
        for phase in SPLITS:
            path = resolve_split_path(dataset_root, group, phase)
            sources.append({"group": group, "phase": phase, "path": str(path), "sha256": sha256(path)})
    payload = {
        "schema": CACHE_SCHEMA,
        "split_groups": sorted(split_groups),
        "sources": sources,
        "extractor_sha256": sha256(extractor) if extractor and extractor.exists() else None,
        "graph_builder_sha256": sha256(graph_builder) if graph_builder and graph_builder.exists() else None,
        "rdkit": rdkit_version,
        "native": _native_identity(repo),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _native_identity(repo: Path | None) -> dict:
    if repo is None:
        return {"library": None}
    lib = repo / "gln/mods/mol_gnn/mg_clib/build/dll/libmolgnn.so"
    return {"library": str(lib), "sha256": sha256(lib) if lib.exists() else None}


def build_inventory(dataset_root: Path, cache_root: Path, split_groups: list[str], rebuild: bool = False, repo: Path | None = None) -> Path:
    master_path = resolve_master_path(dataset_root)
    master = pd.read_parquet(master_path)
    if "atom_mapped_rxn_smiles" not in master:
        raise ValueError(f"missing atom_mapped_rxn_smiles in {master_path}; shared GLN preprocessing requires the refreshed mapped master")
    values_frame = master[["atom_mapped_rxn_smiles"]].dropna()
    frames = []
    for group in split_groups:
        for phase in SPLITS:
            path = resolve_split_path(dataset_root, group, phase)
            frame = pd.read_parquet(path)
            if not set(MAPPING_FIELDS).issubset(frame.columns):
                raise ValueError(f"missing atom_mapped_rxn_smiles in {path}")
            frames.append(frame[["atom_mapped_rxn_smiles"]].assign(split_group=group, phase=phase))
    all_rows = values_frame.assign(split_group="master", phase="master")
    membership_rows = pd.concat(frames, ignore_index=True)
    values = sorted(set(all_rows["atom_mapped_rxn_smiles"].astype(str)) | set(membership_rows["atom_mapped_rxn_smiles"].dropna().astype(str)))
    digest, fingerprint_payload = cache_fingerprint(dataset_root, split_groups, repo)
    root = cache_root / digest
    manifest = root / "manifest.json"
    if manifest.exists() and not rebuild:
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    with (root / "reactions.jsonl").open("w") as f:
        for value in values:
            f.write(json.dumps({"key": hashlib.sha256(value.encode()).hexdigest(), "atom_mapped_rxn_smiles": value}) + "\n")
    memberships = {}
    for row in membership_rows.itertuples(index=False):
        memberships.setdefault(str(row.atom_mapped_rxn_smiles), []).append({"split_group": row.split_group, "phase": row.phase})
    with (root / "memberships.jsonl").open("w") as f:
        for value in values:
            f.write(json.dumps({"key": hashlib.sha256(value.encode()).hexdigest(), "memberships": memberships.get(value, [])}) + "\n")
    payload = {**fingerprint_payload, "dataset_root": str(dataset_root), "reaction_count": len(values), "source_rows": len(all_rows), "split_membership_rows": len(membership_rows), "fingerprint": digest}
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest
