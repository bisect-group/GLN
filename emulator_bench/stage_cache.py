"""Schema-5 durable manifests and atomic publication for GLN preprocessing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Iterable


CACHE_SCHEMA = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def implementation_identity(repo: Path, sources: Iterable[str], graph_abi: bool = False) -> dict:
    payload = {"python": platform.python_version(), "sources": {}}
    for source in sources:
        path = repo / source
        payload["sources"][source] = sha256(path) if path.exists() else None
    try:
        import rdkit
        payload["rdkit"] = rdkit.__version__
    except Exception:
        payload["rdkit"] = "unavailable"
    if graph_abi:
        library = repo / "gln/mods/mol_gnn/mg_clib/build/dll/libmolgnn.so"
        payload["native_graph_library"] = sha256(library) if library.exists() else None
    return payload


def stage_key(name: str, direct_inputs: dict, implementation: dict, parameters: dict) -> str:
    """Key only semantic inputs; worker count, paths and CUDA are deliberately absent."""
    payload = {"schema": CACHE_SCHEMA, "stage": name, "direct_inputs": direct_inputs,
               "implementation": implementation, "parameters": parameters}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_records(outputs: Iterable[Path]) -> list[dict]:
    records = []
    for path in outputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return records


def validate_manifest(path: Path, key: str, outputs: Iterable[Path]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing manifest"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("key") != key:
        return False, "schema or key mismatch"
    expected = {str(item): item for item in outputs}
    records = {item.get("path"): item for item in manifest.get("outputs", [])}
    if set(records) != set(expected):
        return False, "output set mismatch"
    for name, output in expected.items():
        record = records[name]
        if not output.is_file():
            return False, f"missing output: {output}"
        if output.stat().st_size != record.get("bytes") or sha256(output) != record.get("sha256"):
            return False, f"corrupt output: {output}"
    return True, "verified"


def publish_manifest(path: Path, name: str, key: str, direct_inputs: dict,
                     implementation: dict, parameters: dict, outputs: Iterable[Path],
                     row_count: int | None = None, error_count: int = 0,
                     elapsed_seconds: float | None = None) -> None:
    atomic_json(path, {"schema": CACHE_SCHEMA, "stage": name, "key": key,
                       "direct_inputs": direct_inputs, "implementation": implementation,
                       "parameters": parameters, "outputs": artifact_records(outputs),
                       "row_count": row_count, "error_count": error_count,
                       "elapsed_seconds": elapsed_seconds, "completed_at": time.time()})
