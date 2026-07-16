"""Validate/build GLN's native molecular-graph library before preprocessing."""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path


def library_path(repo: Path) -> Path:
    return repo / "gln" / "mods" / "mol_gnn" / "mg_clib" / "build" / "dll" / "libmolgnn.so"


def ensure_native(repo: Path, log_path: Path | None = None) -> Path:
    """Build and load libmolgnn.so, failing before expensive preprocessing."""
    mg_dir = repo / "gln" / "mods" / "mol_gnn" / "mg_clib"
    lib = library_path(repo)
    log_path = log_path or (repo / "native_preflight.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"native preflight: repo={repo} library={lib}\n")
        if not lib.exists():
            command = ["make", "-j", str(max(1, min(os.cpu_count() or 1, 8)))]
            log.write("library missing; running: " + " ".join(command) + "\n")
            result = subprocess.run(command, cwd=mg_dir,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(f"failed to build {lib} with {' '.join(command)}; see {log_path}")
        try:
            ctypes.CDLL(str(lib))
        except OSError as exc:
            raise RuntimeError(f"cannot load {lib}: {exc}; see {log_path}") from exc
        # Importing this module exercises the same path used by DataInfo/build_all_reactions.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        check = subprocess.run([sys.executable, "-c", "from gln.mods.mol_gnn.mg_clib import MGLIB; assert MGLIB is not None"],
                               cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if check.returncode:
            raise RuntimeError(f"native GLN import failed; see {log_path}")
        log.write("native preflight complete\n")
    return lib


def write_manifest(path: Path, lib: Path) -> None:
    path.write_text(json.dumps({"library": str(lib), "size": lib.stat().st_size, "mtime_ns": lib.stat().st_mtime_ns}, indent=2) + "\n")


def validate_datainfo(repo: Path, atom_file: Path, log_path: Path) -> None:
    """Exercise the exact native import with the generated atom vocabulary."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    code = "from gln.data_process.data_info import DataInfo; from gln.mods.mol_gnn.mg_clib import MGLIB; assert MGLIB is not None; assert MGLIB.NUM_NODE_FEATS > 23"
    with log_path.open("a") as log:
        log.write(f"validating DataInfo with atom file {atom_file}\n")
        result = subprocess.run([sys.executable, "-c", code, "-f_atoms", str(atom_file)], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"DataInfo/native import failed with generated atom configuration; see {log_path}")
