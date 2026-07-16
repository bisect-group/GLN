from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import selectors
import subprocess
import sys
import time
import warnings
import fcntl
from pathlib import Path

from .dataset_adapter import DEFAULT_BASELINES, DEFAULT_ROOT, prepare
from .native_preflight import ensure_native, validate_datainfo, write_manifest
from .shared_cache import build_inventory, cache_fingerprint
from .stage_cache import CACHE_SCHEMA, implementation_identity, publish_manifest, stage_key, validate_manifest
from gln.mods.rdchiral.template_extractor import extract_from_reaction

warnings.filterwarnings("ignore", category=FutureWarning)


@contextlib.contextmanager
def preprocessing_lock(namespace: Path):
    """Allow only one preprocessing process per split namespace."""
    namespace.mkdir(parents=True, exist_ok=True)
    path = namespace / "preprocessing.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(f"preprocessing already active for {namespace.name}; {owner}")
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started": time.time(), "namespace": str(namespace), "command": " ".join(sys.argv)}) + "\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validation_signature(raw_root: Path, extractor_path: Path) -> str:
    h = hashlib.sha256()
    for phase in ("train", "val", "test"):
        h.update((raw_root / f"raw_{phase}.csv").read_bytes())
    h.update(str(extractor_path.stat().st_mtime_ns).encode())
    try:
        import rdkit
        h.update(rdkit.__version__.encode())
    except Exception:
        pass
    return h.hexdigest()


def run(cmd: list[str], cwd: Path, log_path: Path, verbose: bool = False) -> None:
    print(f"[stage] {cmd[1] if len(cmd) > 1 else cmd[0]} -> {log_path}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("$", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=cwd, env=env)
    else:
        with log_path.open("a") as stream:
            stream.write("$ " + " ".join(cmd) + "\n")
            stream.flush()
            process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            buffer = ""
            last_update = time.monotonic()
            last_display = 0.0
            while selector.get_map():
                events = selector.select(timeout=20.0)
                if not events:
                    elapsed = int(time.monotonic() - last_update)
                    print(f"[stage running] {cmd[1] if len(cmd) > 1 else cmd[0]} | waiting for output | +{elapsed}s", flush=True)
                    continue
                for key, _ in events:
                    chunk = key.fileobj.read1(8192).decode(errors="replace")
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stream.write(chunk)
                    stream.flush()
                    buffer += chunk.replace("\r", "\n")
                    parts = buffer.split("\n")
                    buffer = parts.pop()
                    for line in parts:
                        clean = line.strip()
                        if "%|" in clean or re.search(r"\d+\.\d+.*(?:/s|reaction|template|molecule|SMARTS|iteration)", clean):
                            now = time.monotonic()
                            if now - last_display >= 0.5 or "100%" in clean:
                                print("\r" + clean, end="", flush=True)
                                last_display = now
                            last_update = time.monotonic()
            result = subprocess.CompletedProcess(cmd, process.wait())
    if result.returncode:
        print(f"[stage failed] exit={result.returncode}; see {log_path}", flush=True)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print("\n[stage complete]", flush=True)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def _stage_marker_path(namespace: Path, name: str) -> Path:
    return namespace / "stages" / f"{name}.json"


def _stage_digest(split_fingerprint: str, name: str, command: list[str], dependency: str | None) -> tuple[str, dict, dict, dict]:
    repo = Path(__file__).resolve().parents[1]
    relevant_sources = {
        "canonical_smiles": ["gln/data_process/get_canonical_smiles.py"],
        "raw_templates": ["emulator_bench/safe_build_raw_template.py", "gln/mods/rdchiral/template_extractor.py"],
        "filtered_templates": ["gln/data_process/filter_template.py"],
        "canonical_smarts": ["gln/data_process/get_canonical_smarts.py"],
        "center_maps": ["emulator_bench/safe_find_centers.py", "gln/data_process/data_info.py"],
        "all_reactions": ["gln/data_process/build_all_reactions.py", "gln/data_process/data_info.py"],
        "molecular_graphs": ["gln/data_process/dump_graphs.py", "gln/mods/mol_gnn/mol_utils.py"],
        "smarts_graphs": ["gln/data_process/dump_graphs.py", "gln/mods/mol_gnn/mol_utils.py"],
        "negative_graphs": ["gln/data_process/dump_graphs.py", "gln/mods/mol_gnn/mol_utils.py"],
    }
    direct_inputs = {"split_fingerprint": split_fingerprint, "dependency_key": dependency}
    # Commands contain namespace paths and worker counts, neither of which is
    # semantic cache input. Keep only stage-specific semantic options here.
    parameters = {"template_threshold": 1} if name == "filtered_templates" else {}
    implementation = implementation_identity(repo, relevant_sources.get(name, []), graph_abi=name.endswith("graphs"))
    return stage_key(name, direct_inputs, implementation, parameters), direct_inputs, implementation, parameters


def _stage_valid(marker_path: Path, digest: str, outputs: list[Path]) -> bool:
    return validate_manifest(marker_path, digest, outputs)[0]


def _write_stage_marker(marker_path: Path, name: str, digest: str, direct_inputs: dict, implementation: dict, parameters: dict, outputs: list[Path], elapsed: float) -> None:
    publish_manifest(marker_path, name, digest, direct_inputs, implementation, parameters, outputs, elapsed_seconds=elapsed)


def _validate_row(item: tuple[str, list[str]]) -> tuple[list[str] | None, str | None]:
    phase, row = item
    try:
        reactants, _, products = row[2].split(">")
        # The legacy extractor prints full SMARTS diagnostics directly to
        # stdout. Capture them here; native stage logs retain diagnostics when
        # build_raw_template runs, while compact mode stays readable.
        with contextlib.redirect_stdout(io.StringIO()):
            result = extract_from_reaction({"_id": row[0], "reactants": reactants, "products": products})
        if result is None:
            raise ValueError("template extractor returned None")
        return row, None
    except Exception as exc:
        return None, json.dumps({"split": phase, "id": row[0], "error": f"{type(exc).__name__}: {exc}"})


def validate_template_inputs(dropbox: Path, data: str, namespace: Path, workers: int) -> None:
    marker = namespace / "template_input_validation.json"
    raw_root = dropbox / data
    signature = validation_signature(raw_root, Path(__file__).resolve().parents[1] / "gln/mods/rdchiral/template_extractor.py")
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:
        report = json.loads(marker.read_text())
        return
    report = {"signature": signature, "excluded": [], "splits": {}}
    for phase in ("train", "val", "test"):
        path = raw_root / f"raw_{phase}.csv"
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        with mp.Pool(max(1, workers)) as pool:
            for row, error in pool.imap_unordered(_validate_row, ((phase, row) for row in rows), chunksize=1):
                if row is None:
                    report["excluded"].append(json.loads(error))
        report["splits"][phase] = {"input_rows": len(rows), "kept_rows": len(rows), "template_failures": sum(item["split"] == phase for item in report["excluded"])}
    marker.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Template preflight excluded {len(report['excluded'])} invalid reactions; report: {marker}", flush=True)


def _run_preprocessing(args: argparse.Namespace, namespace: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    lib = ensure_native(Path(__file__).resolve().parents[1], namespace / "logs" / "native_extension.log")
    write_manifest(namespace / "native_extension.json", lib)
    if not args.verify_cache_only:
        prepare(args.dataset_root, args.output_root, args.split_group)
    if args.shared_cache_manifest is not None:
        (namespace / "shared_cache.json").write_text(args.shared_cache_manifest.read_text())
    elif args.shared_cache_root is not None:
        manifest = build_inventory(args.dataset_root, args.shared_cache_root, [args.split_group], args.rebuild_shared_cache, repo)
        (namespace / "shared_cache.json").write_text(manifest.read_text())
    os.environ["EMULATOR_BENCH_CENTER_TIMEOUT"] = str(args.center_timeout)
    dropbox = namespace / "dropbox"
    data = "reaction_outcome_dataset"
    cooked = dropbox / f"cooked_{data}"
    logs = namespace / "logs"
    validate_template_inputs(dropbox, data, namespace, args.num_cores)
    split_fingerprint, _ = cache_fingerprint(args.dataset_root, [args.split_group], repo)
    common = [sys.executable]
    commands = [
        common + ["gln/data_process/get_canonical_smiles.py", "-dropbox", str(dropbox), "-data_name", data, "-save_dir", str(cooked)],
        common + ["-m", "emulator_bench.safe_build_raw_template", "-dropbox", str(dropbox), "-data_name", data, "-save_dir", str(cooked), "-num_cores", str(args.num_cores)],
        common + ["gln/data_process/filter_template.py", "-dropbox", str(dropbox), "-data_name", data, "-tpl_name", "default", "-save_dir", str(cooked / "tpl-default")],
        common + ["gln/data_process/get_canonical_smarts.py", "-dropbox", str(dropbox), "-data_name", data, "-tpl_name", "default", "-save_dir", str(cooked / "tpl-default")],
        common + ["-m", "emulator_bench.safe_find_centers", "-dropbox", str(dropbox), "-data_name", data, "-tpl_name", "default", "-save_dir", str(cooked / "tpl-default"), "-num_cores", str(args.num_cores), "-num_parts", "1"],
        common + ["gln/data_process/build_all_reactions.py", "-dropbox", str(dropbox), "-phase", "cooking", "-data_name", data, "-save_dir", str(cooked / "tpl-default"), "-tpl_name", "default", "-f_atoms", str(cooked / "atom_list.txt"), "-num_cores", str(args.num_cores), "-num_parts", "1", "-gpu", "-1"],
    ]
    names = ["canonical_smiles", "raw_templates", "filtered_templates", "canonical_smarts", "center_maps", "all_reactions"]
    output_sets = [
        [cooked / "atom_list.txt", cooked / "cano_smiles.pkl"],
        [cooked / "proc_train_singleprod.csv", cooked / "failed_template.csv"],
        [cooked / "tpl-default" / "templates.csv"],
        [cooked / "tpl-default" / "prod_cano_smarts.txt", cooked / "tpl-default" / "react_cano_smarts.txt", cooked / "tpl-default" / "cano_smarts.pkl"],
        [cooked / "tpl-default" / "find_centers.complete", cooked / "tpl-default" / "np-1" / "train-prod_center_maps-part-0.csv", cooked / "tpl-default" / "np-1" / "val-prod_center_maps-part-0.csv", cooked / "tpl-default" / "np-1" / "test-prod_center_maps-part-0.csv"],
        [cooked / "tpl-default" / "np-1" / "pos_tpls-part-0.csv", cooked / "tpl-default" / "np-1" / "neg_reacts-part-0.csv"],
    ]
    descendants = {
        "canonical_smiles": set(names + ["molecular_graphs", "smarts_graphs", "negative_graphs"]),
        "raw_templates": {"raw_templates", "filtered_templates", "canonical_smarts", "center_maps", "all_reactions", "smarts_graphs", "negative_graphs"},
        "filtered_templates": {"filtered_templates", "canonical_smarts", "center_maps", "all_reactions", "smarts_graphs", "negative_graphs"},
        "canonical_smarts": {"canonical_smarts", "center_maps", "all_reactions", "smarts_graphs", "negative_graphs"},
        "center_maps": {"center_maps", "all_reactions", "negative_graphs"},
        "all_reactions": {"all_reactions", "negative_graphs"},
        "molecular_graphs": {"molecular_graphs"}, "smarts_graphs": {"smarts_graphs"}, "negative_graphs": {"negative_graphs"},
    }
    if args.force_stage:
        for stale in descendants[args.force_stage]:
            _stage_marker_path(namespace, stale).unlink(missing_ok=True)
        (namespace / "preprocess.complete").unlink(missing_ok=True)
        print(f"[force] invalidated {args.force_stage} and true descendants", flush=True)
    dependency = None
    stage_keys: dict[str, str] = {}
    for name, cmd in zip(names, commands):
        outputs = output_sets[names.index(name)]
        marker_path = _stage_marker_path(namespace, name)
        digest, direct_inputs, implementation, parameters = _stage_digest(split_fingerprint, name, cmd, dependency)
        if _stage_valid(marker_path, digest, outputs):
            print(f"[verified] stage {name}" if args.verify_cache_only else f"[cache hit] stage {name}", flush=True)
            dependency = digest
            stage_keys[name] = digest
            continue
        if args.verify_cache_only:
            raise RuntimeError(f"[verify failed] {name}: " + validate_manifest(marker_path, digest, outputs)[1])
        # A stale stage invalidates itself and every descendant marker, but a
        # failed downstream stage never damages an already-valid ancestor.
        downstream = names[names.index(name):] + ["molecular_graphs", "smarts_graphs", "negative_graphs"]
        for stale in downstream:
            _stage_marker_path(namespace, stale).unlink(missing_ok=True)
        (namespace / "preprocess.complete").unlink(missing_ok=True)
        print(f"[compute] stage {name}", flush=True)
        stage_started = time.monotonic()
        run(cmd, repo, logs / f"{name}.log", args.verbose)
        if name == "canonical_smiles":
            atom_file = cooked / "atom_list.txt"
            if not atom_file.exists():
                raise RuntimeError(f"canonical SMILES stage did not create atom list: {atom_file}")
            validate_datainfo(repo, atom_file, namespace / "logs" / "native_extension.log")
        if name == "raw_templates":
            print(f"[templates] successful={count_rows(cooked / 'proc_train_singleprod.csv')} failed_validation={count_rows(cooked / 'failed_template.csv')}", flush=True)
        if name == "center_maps":
            marker = cooked / "tpl-default" / "find_centers.complete"
            if not marker.exists():
                raise RuntimeError(f"find_centers exited without completion marker: {marker}")
            for path in outputs[1:]:
                with path.open(newline="") as f:
                    header = next(csv.reader(f), [])
                if header != ["smiles", "class", "centers"]:
                    raise RuntimeError(f"invalid center-map header in {path}: {header}")
        if name == "all_reactions":
            np_dir = cooked / "tpl-default" / "np-1"
            required = (np_dir / "pos_tpls-part-0.csv", np_dir / "neg_reacts-part-0.csv")
            missing = [str(path) for path in required if not path.exists() or count_rows(path) == 0]
            if missing:
                raise RuntimeError("build_all_reactions completed without required outputs: " + ", ".join(missing))
        if any(not path.exists() for path in outputs):
            raise RuntimeError(f"stage {name} completed without required outputs: " + ", ".join(map(str, outputs)))
        _write_stage_marker(marker_path, name, digest, direct_inputs, implementation, parameters, outputs, time.monotonic() - stage_started)
        dependency = digest
        stage_keys[name] = digest
    graph_stages = [
        ("molecular_graphs", "molecules", "False", [cooked / "graph_smiles.names", cooked / "graph_smiles.bin"]),
        ("smarts_graphs", "smarts", "False", [cooked / "tpl-default" / "graph_smarts.names", cooked / "tpl-default" / "graph_smarts.bin"]),
        ("negative_graphs", "negative", "True", [cooked / "tpl-default" / "np-1" / "neg_graphs-part-0.names", cooked / "tpl-default" / "np-1" / "neg_graphs-part-0.bin"]),
    ]
    graph_dependencies = {"molecular_graphs": stage_keys["canonical_smiles"], "smarts_graphs": stage_keys["canonical_smarts"], "negative_graphs": stage_keys["all_reactions"]}
    for name, kind, retro, outputs in graph_stages:
        cmd = common + ["gln/data_process/dump_graphs.py", "-dropbox", str(dropbox), "-data_name", data, "-tpl_name", "default", "-save_dir", str(cooked / "tpl-default"), "-f_atoms", str(cooked / "atom_list.txt"), "-num_parts", "1", "-fp_degree", "2", "-retro_during_train", retro, "-graph_dump_kind", kind]
        marker_path = _stage_marker_path(namespace, name)
        digest, direct_inputs, implementation, parameters = _stage_digest(split_fingerprint, name, cmd, graph_dependencies[name])
        if _stage_valid(marker_path, digest, outputs):
            print(f"[verified] stage {name}" if args.verify_cache_only else f"[cache hit] stage {name}", flush=True)
            continue
        if args.verify_cache_only:
            raise RuntimeError(f"[verify failed] {name}: " + validate_manifest(marker_path, digest, outputs)[1])
        print(f"[compute] stage {name}", flush=True)
        stage_started = time.monotonic()
        run(cmd, repo, logs / f"{name}.log", args.verbose)
        if any(not path.exists() for path in outputs):
            raise RuntimeError(f"stage {name} completed without required outputs: " + ", ".join(map(str, outputs)))
        _write_stage_marker(marker_path, name, digest, direct_inputs, implementation, parameters, outputs, time.monotonic() - stage_started)
    if args.verify_cache_only:
        print("[verified] all schema-5 stage manifests and artifacts", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--split-group", required=True)
    p.add_argument("--num-cores", type=int, default=4)
    p.add_argument("--verbose", action="store_true", help="stream native GLN output instead of compact stage logging")
    p.add_argument("--lock-held", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--shared-cache-root", type=Path, default=DEFAULT_BASELINES / "shared_cache")
    p.add_argument("--shared-cache-manifest", type=Path)
    p.add_argument("--rebuild-shared-cache", action="store_true")
    p.add_argument("--center-timeout", type=int, default=60)
    p.add_argument("--resume", action="store_true", help="reuse only hash- and semantics-validated stage manifests")
    p.add_argument("--force-stage", choices=("canonical_smiles", "raw_templates", "filtered_templates", "canonical_smarts", "center_maps", "all_reactions", "molecular_graphs", "smarts_graphs", "negative_graphs"))
    p.add_argument("--verify-cache-only", action="store_true", help="validate schema-5 manifests and outputs without computation")
    p.add_argument("--legacy-cache-policy", choices=("fresh", "adopt"), default="fresh")
    args = p.parse_args()
    namespace = args.output_root / "namespaces" / args.split_group
    if args.lock_held:
        _run_preprocessing(args, namespace)
    else:
        with preprocessing_lock(namespace):
            _run_preprocessing(args, namespace)


if __name__ == "__main__":
    main()
