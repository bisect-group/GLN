"""One-command, resumable multi-GPU GLN experiment orchestrator."""
from __future__ import annotations

import argparse
import csv
import contextlib
import fcntl
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .dataset_adapter import DEFAULT_BASELINES, DEFAULT_ROOT, prepare
from .native_preflight import ensure_native, write_manifest
from .shared_cache import CACHE_SCHEMA, build_inventory, cache_fingerprint

ALL_SPLITS = ("random_splits_grouped_rxn_smiles", "uniprot_time_splits", "drfp_tanimoto_splits")

_active_children: dict[int, subprocess.Popen] = {}
_children_lock = threading.RLock()
_interrupt_requested = threading.Event()
_TQDM_PROGRESS = re.compile(r"(?P<percent>\d{1,3})%\|.*?(?P<completed>\d+)\s*/\s*(?P<total>\d+)")


def _register_child(process: subprocess.Popen) -> None:
    with _children_lock:
        _active_children[process.pid] = process


def _unregister_child(process: subprocess.Popen) -> None:
    with _children_lock:
        _active_children.pop(process.pid, None)


def _interrupt_children(signum=None, frame=None) -> None:
    if _interrupt_requested.is_set():
        return
    _interrupt_requested.set()
    print("\n[interrupt] stopping active GLN subprocesses; waiting for cleanup...", flush=True)
    with _children_lock:
        children = list(_active_children.values())
    for process in children:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass


def _install_signal_handlers():
    previous = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    signal.signal(signal.SIGINT, _interrupt_children)
    signal.signal(signal.SIGTERM, _interrupt_children)
    return previous


def _restore_signal_handlers(previous) -> None:
    signal.signal(signal.SIGINT, previous[0])
    signal.signal(signal.SIGTERM, previous[1])


def _stop_all_children(force: bool = False) -> None:
    with _children_lock:
        children = list(_active_children.values())
    for process in children:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGINT)
        except (OSError, ProcessLookupError):
            pass


def prepare_schema5_root(output_root: Path, legacy_policy: str) -> Path:
    """Create an isolated schema-5 root; never mistake legacy markers for hits."""
    root = output_root / "schema-5"
    if legacy_policy == "adopt":
        return root
    legacy = output_root / "namespaces"
    if legacy.exists():
        quarantine = output_root / "quarantine" / f"schema-4-namespaces-{time.strftime('%Y%m%d-%H%M%S')}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(quarantine))
        print(f"[legacy cache] quarantined {legacy} -> {quarantine}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextlib.contextmanager
def split_lock(namespace: Path):
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


def parse_ints(value: str) -> list[int]:
    out: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(item))
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return sorted(out)


@dataclass(frozen=True)
class Job:
    split: str
    seed: int
    gpu: str


@dataclass(frozen=True)
class StageInfo:
    lifecycle: str
    stage: str
    resource: str


def parse_tqdm_progress(line: str) -> tuple[int, int, int] | None:
    """Return percent, completed, and total from a standard tqdm frame."""
    match = _TQDM_PROGRESS.search(line)
    if match is None:
        return None
    return tuple(int(match.group(name)) for name in ("percent", "completed", "total"))


def initial_stage(phase: str) -> StageInfo:
    if phase == "evaluate":
        return StageInfo("Evaluate", "subprocess startup", "CPU / startup")
    return StageInfo("Train", "subprocess startup", "CPU / startup")


def classify_stage(phase: str, line: str) -> StageInfo | None:
    """Translate GLN's native status text into an operator-facing stage."""
    text = line.lower()
    if "worker" in text and ("is dead" in text or "restarts" in text):
        return StageInfo("Train", "data-generator worker recovery", "CPU")
    if "load negative reaction map" in text or "loading negative reactions" in text:
        return StageInfo("Train", "loading negative-reaction map", "CPU")
    if "load product center maps" in text or "loading training prod center maps" in text:
        lifecycle = "Evaluate" if phase == "evaluate" else "Train"
        return StageInfo(lifecycle, "loading product-center maps", "CPU")
    if "load unique templates" in text or "loading templates" in text:
        return StageInfo("Train", "loading template index", "CPU")
    if "loading smiles feature dump" in text:
        return StageInfo("Train", "loading molecular graph features", "CPU / disk")
    if "# raw train loaded" in text or "loading positive tpls" in text:
        return StageInfo("Train", "initializing training data generator", "CPU")
    if "loading data info" in text:
        return StageInfo("Train", "loading dataset metadata", "CPU / disk")
    if re.search(r"(?:train epoch|\bepoch\s+\d+(?:\.\d+)?[, ])", text):
        return StageInfo("Train", "optimization", "GPU")

    if "evaluate val reactions" in text or "evaluate test reactions" in text:
        return StageInfo("Evaluate", "reaction inference", "GPU")
    if re.search(r"load (?:val|test) reactions", text) or re.search(r"loading raw (?:val|test)", text):
        return StageInfo("Evaluate", "loading evaluation reactions", "CPU / disk")
    if text.startswith("testing ") or "model_for_test" in text:
        return StageInfo("Evaluate", "loading checkpoint", "CPU / disk")
    return None


class JobReporter(Protocol):
    def start(self, key: str, label: str, phase: str) -> None: ...
    def update(self, key: str, phase: str, line: str, *, verbose: bool = False) -> None: ...
    def waiting(self, key: str, phase: str) -> None: ...
    def finish(self, key: str, status: str) -> None: ...


class PlainReporter:
    """Thread-safe, redirect-friendly fallback for the interactive dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._labels: dict[str, str] = {}
        self._last_progress: dict[str, float] = {}
        self._stages: dict[str, StageInfo] = {}

    def __enter__(self) -> "PlainReporter":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def _write(self, key: str, message: str) -> None:
        with self._lock:
            print(f"[{self._labels.get(key, key)}] {message}", flush=True)

    def start(self, key: str, label: str, phase: str) -> None:
        self._labels[key] = label
        stage = initial_stage(phase)
        self._stages[key] = stage
        self._write(key, f"{stage.lifecycle} | {stage.stage} | {stage.resource}")

    def update(self, key: str, phase: str, line: str, *, verbose: bool = False) -> None:
        stage = classify_stage(phase, line)
        if stage is not None and stage != self._stages.get(key):
            self._stages[key] = stage
            self._last_progress.pop(key, None)
            self._write(key, f"{stage.lifecycle} | {stage.stage} | {stage.resource}")
        stage = self._stages.get(key, initial_stage(phase))
        progress = parse_tqdm_progress(line)
        if progress is not None:
            now = time.monotonic()
            if now - self._last_progress.get(key, 0.0) >= 0.5 or progress[0] >= 100:
                self._write(key, f"{stage.lifecycle} | {stage.stage} | {stage.resource}: {progress[0]}% ({progress[1]}/{progress[2]})")
                self._last_progress[key] = now
        elif verbose and line:
            self._write(key, f"{stage.lifecycle} | {stage.stage}: {line}")

    def waiting(self, key: str, phase: str) -> None:
        stage = self._stages.get(key, initial_stage(phase))
        self._write(key, f"{stage.lifecycle} | {stage.stage} | {stage.resource}: waiting for output")

    def finish(self, key: str, status: str) -> None:
        self._write(key, status)


class RichReporter:
    """One Rich display shared safely by every concurrent GLN job."""

    def __init__(self) -> None:
        from rich.console import Console
        from rich.live import Live
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

        self._lock = threading.RLock()
        self._task_ids: dict[str, int] = {}
        self._last_progress: dict[str, float] = {}
        self._stages: dict[str, StageInfo] = {}
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[label]}", justify="left"),
            TextColumn("{task.fields[lifecycle]}", justify="left"),
            TextColumn("{task.fields[stage]}", justify="left"),
            TextColumn("{task.fields[resource]}", justify="left"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[progress]}", justify="right"),
            TimeElapsedColumn(),
            console=Console(),
            expand=True,
        )
        self._live = Live(self._progress, console=self._progress.console, refresh_per_second=8)

    def __enter__(self) -> "RichReporter":
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._live.__exit__(*exc_info)

    def start(self, key: str, label: str, phase: str) -> None:
        with self._lock:
            stage = initial_stage(phase)
            self._stages[key] = stage
            fields = dict(label=label, lifecycle=stage.lifecycle, stage=stage.stage, resource=stage.resource, progress="starting")
            if key not in self._task_ids:
                self._task_ids[key] = self._progress.add_task("", total=None, **fields)
            else:
                self._progress.update(self._task_ids[key], total=None, completed=0, **fields)

    def update(self, key: str, phase: str, line: str, *, verbose: bool = False) -> None:
        with self._lock:
            task_id = self._task_ids[key]
            stage = classify_stage(phase, line)
            if stage is not None and stage != self._stages.get(key):
                self._stages[key] = stage
                self._last_progress.pop(key, None)
                self._progress.update(
                    task_id, total=None, completed=0, lifecycle=stage.lifecycle,
                    stage=stage.stage, resource=stage.resource, progress="starting",
                )
            stage = self._stages.get(key, initial_stage(phase))
            progress = parse_tqdm_progress(line)
            if progress is not None:
                percent, completed, total = progress
                now = time.monotonic()
                if now - self._last_progress.get(key, 0.0) >= 0.1 or percent >= 100:
                    self._progress.update(task_id, total=total, completed=completed, progress=f"{completed:,}/{total:,}")
                    self._last_progress[key] = now
            elif verbose and line:
                self._progress.update(task_id, progress=line[-80:])

    def waiting(self, key: str, phase: str) -> None:
        with self._lock:
            self._progress.update(self._task_ids[key], progress="waiting for output")

    def finish(self, key: str, status: str) -> None:
        with self._lock:
            task_id = self._task_ids.get(key)
            if task_id is not None:
                self._progress.update(task_id, stage=status, resource="", progress="")


def make_reporter() -> JobReporter:
    if sys.stdout.isatty():
        try:
            return RichReporter()
        except ImportError:
            pass
    return PlainReporter()


def run_cmd(cmd: list[str], cwd: Path, log: Path, env: dict[str, str], reporter: JobReporter, key: str, phase: str, dry: bool = False, verbose: bool = False) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print("DRY:", " ".join(cmd))
        return 0
    with log.open("a") as stream:
        stream.write("$ " + " ".join(cmd) + "\n")
        stream.flush()
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        _register_child(process)
        try:
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            buffer = ""
            while selector.get_map():
                events = selector.select(timeout=20.0)
                if not events:
                    reporter.waiting(key, phase)
                    continue
                for event_key, _ in events:
                    chunk = event_key.fileobj.read1(8192).decode(errors="replace")
                    if not chunk:
                        selector.unregister(event_key.fileobj)
                        continue
                    stream.write(chunk)
                    stream.flush()
                    buffer += chunk.replace("\r", "\n")
                    parts = buffer.split("\n")
                    buffer = parts.pop()
                    for line in parts:
                        clean = line.strip()
                        reporter.update(key, phase, clean, verbose=verbose)
            selector.close()
            process.stdout.close()
            code = process.wait()
            return code
        finally:
            _unregister_child(process)


def job_worker(job: Job, args: argparse.Namespace, repo: Path, reporter: JobReporter) -> dict:
    namespace = args.output_root / "namespaces" / job.split
    dropbox = namespace / "dropbox"
    run_dir = args.output_root / "runs" / job.split / f"seed_{job.seed}"
    status_path = run_dir / "status.json"
    log = run_dir / "run.log"
    if args.resume and status_path.exists():
        try:
            old = json.loads(status_path.read_text())
            if old.get("status") == "complete":
                return old
        except json.JSONDecodeError:
            pass
    if _interrupt_requested.is_set():
        return {"split_group": job.split, "seed": job.seed, "gpu": job.gpu, "status": "interrupted"}
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {"split_group": job.split, "seed": job.seed, "gpu": job.gpu, "status": "running", "started": started}
    key = f"{job.split}/seed_{job.seed}/gpu_{job.gpu}"
    label = f"{job.split} | seed {job.seed} | GPU {job.gpu}"
    status_path.write_text(json.dumps(record, indent=2) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    common = [sys.executable, "-m"]
    try:
        if not args.eval_only:
            train = common + ["emulator_bench.train", "--dropbox", str(dropbox), "--data-name", "reaction_outcome_dataset", "--save-dir", str(run_dir / "train"), "--gpu", "0", "--seed", str(job.seed)]
            checkpoints = sorted((run_dir / "train").glob("model-*/model.dump"))
            if args.resume and checkpoints:
                train.extend(["--resume-from", str(checkpoints[-1])])
            reporter.start(key, label, "train")
            rc = run_cmd(train, repo, log, env, reporter, key, "train", args.dry_run, args.verbose)
            if _interrupt_requested.is_set():
                raise KeyboardInterrupt
            if rc:
                raise RuntimeError(f"training exited with {rc}")
        if not args.train_only:
            evaluate = common + ["emulator_bench.evaluate", "--dropbox", str(dropbox), "--data-name", "reaction_outcome_dataset", "--save-dir", str(run_dir / "train"), "--gpu", "0"]
            reporter.start(key, label, "evaluate")
            rc = run_cmd(evaluate, repo, log, env, reporter, key, "evaluate", args.dry_run, args.verbose)
            if _interrupt_requested.is_set():
                raise KeyboardInterrupt
            if rc:
                raise RuntimeError(f"evaluation exited with {rc}")
        best_val = None
        for phase in ("val", "test"):
            summaries = sorted((run_dir / "train").glob(f"{phase}-*.summary"))
            if summaries:
                text = summaries[-1].read_text()
                for k in (1, 10):
                    match = re.search(rf"top {k}: ([0-9.]+)", text)
                    if match:
                        record[f"{phase}_top{k}"] = float(match.group(1))
                if phase == "val":
                    candidates = []
                    for summary in summaries:
                        match = re.search(r"top 1: ([0-9.]+)", summary.read_text())
                        if match:
                            candidates.append((float(match.group(1)), summary.stem.replace("val-", "model-") + ".dump"))
                    if candidates:
                        best_val = max(candidates)[1]
        if best_val:
            record["selected_checkpoint"] = str(run_dir / "train" / best_val)
        record.update(status="complete", elapsed_seconds=time.time() - started)
    except KeyboardInterrupt:
        record.update(status="interrupted", error="interrupt requested", elapsed_seconds=time.time() - started)
    except Exception as exc:
        record.update(status="failed", error=str(exc), elapsed_seconds=time.time() - started)
        if args.fail_fast:
            raise
    status_path.write_text(json.dumps(record, indent=2) + "\n")
    reporter.finish(key, record["status"])
    return record


def aggregate(output_root: Path, records: list[dict]) -> None:
    result_dir = output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "per_job_metrics.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    with (result_dir / "summary.csv").open("w", newline="") as f:
        fields = ["split_group", "seed", "gpu", "status", "elapsed_seconds", "error", "val_top1", "val_top10", "test_top1", "test_top10"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    complete = [r for r in records if r.get("status") == "complete"]
    groups = sorted({r["split_group"] for r in complete})
    with (result_dir / "summary_by_split.csv").open("w", newline="") as f:
        fields = ["split_group", "seed_count", "test_top1_mean", "test_top1_std", "test_top10_mean", "test_top10_std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            rows = [r for r in complete if r["split_group"] == group]
            vals1 = [r["test_top1"] for r in rows if "test_top1" in r]
            vals10 = [r["test_top10"] for r in rows if "test_top10" in r]
            import statistics
            writer.writerow({"split_group": group, "seed_count": len(rows), "test_top1_mean": statistics.mean(vals1) if vals1 else "", "test_top1_std": statistics.stdev(vals1) if len(vals1) > 1 else 0.0 if vals1 else "", "test_top10_mean": statistics.mean(vals10) if vals10 else "", "test_top10_std": statistics.stdev(vals10) if len(vals10) > 1 else 0.0 if vals10 else ""})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--runs-per-gpu", type=int, default=2)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--split-groups", nargs="+", default=["all"])
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_BASELINES)
    p.add_argument("--num-cores", type=int, default=4, help="CPU workers for GLN preprocessing and template validation")
    p.add_argument("--verbose", action="store_true", help="stream native GLN preprocessing output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shared-cache-root", type=Path, default=DEFAULT_BASELINES / "shared_cache")
    p.add_argument("--rebuild-shared-cache", action="store_true")
    p.add_argument("--center-timeout", type=int, default=60, help="hard timeout in seconds for a center-search task")
    p.add_argument("--legacy-cache-policy", choices=("fresh", "adopt"), default="fresh")
    p.add_argument("--force-stage", choices=("canonical_smiles", "raw_templates", "filtered_templates", "canonical_smarts", "center_maps", "all_reactions", "molecular_graphs", "smarts_graphs", "negative_graphs"))
    p.add_argument("--verify-cache-only", action="store_true")
    args = p.parse_args()
    if args.runs_per_gpu < 1:
        p.error("--runs-per-gpu must be positive")
    if args.num_cores < 1:
        p.error("--num-cores must be positive")
    if args.train_only and args.eval_only:
        p.error("--train-only and --eval-only are mutually exclusive")
    groups = list(ALL_SPLITS) if "all" in args.split_groups else args.split_groups
    unknown = set(groups) - set(ALL_SPLITS)
    if unknown:
        p.error(f"unknown split groups: {sorted(unknown)}")
    gpus = [str(x) for x in parse_ints(args.gpus)]
    seeds = parse_ints(args.seeds)
    repo = Path(__file__).resolve().parents[1]
    previous_signals = _install_signal_handlers()
    try:
        args.output_root = prepare_schema5_root(args.output_root, args.legacy_cache_policy)
        preflight_log = args.output_root / "logs" / "native_extension.log"
        lib = ensure_native(repo, preflight_log)
        write_manifest(args.output_root / "native_extension.json", lib)
        shared_manifest = None
        if args.shared_cache_root is not None:
            shared_manifest = build_inventory(args.dataset_root, args.shared_cache_root, groups, args.rebuild_shared_cache, repo)
            print(f"[shared cache] reactions indexed from {shared_manifest}", flush=True)
        for split in groups:
            if _interrupt_requested.is_set():
                break
            namespace = args.output_root / "namespaces" / split
            with split_lock(namespace):
                manifest = namespace / "manifest.json"
                if args.resume and manifest.exists():
                    print(f"[resume] prepared {split}")
                else:
                    print(f"[prepare] {split}")
                    prepare(args.dataset_root, args.output_root, split)
                cooked_marker = namespace / "preprocess.complete"
                marker_valid = False
                split_fingerprint, _ = cache_fingerprint(args.dataset_root, [split], repo)
                if args.resume and cooked_marker.exists():
                    try:
                        marker = json.loads(cooked_marker.read_text())
                        marker_valid = marker.get("cache_schema") == CACHE_SCHEMA and marker.get("fingerprint") == split_fingerprint
                    except (OSError, json.JSONDecodeError):
                        marker_valid = False
                if not marker_valid or args.verify_cache_only:
                    cache_cmd = [sys.executable, "-m", "emulator_bench.cache_features", "--dataset-root", str(args.dataset_root), "--output-root", str(args.output_root), "--split-group", split, "--num-cores", str(args.num_cores), "--center-timeout", str(args.center_timeout), "--lock-held"]
                    cache_cmd.extend(["--legacy-cache-policy", args.legacy_cache_policy])
                    if args.force_stage:
                        cache_cmd.extend(["--force-stage", args.force_stage])
                    if args.verify_cache_only:
                        cache_cmd.append("--verify-cache-only")
                    if args.shared_cache_root is not None:
                        cache_cmd.extend(["--shared-cache-root", str(args.shared_cache_root)])
                    if shared_manifest is not None:
                        cache_cmd.extend(["--shared-cache-manifest", str(shared_manifest)])
                    if args.rebuild_shared_cache:
                        cache_cmd.append("--rebuild-shared-cache")
                    if args.verbose:
                        cache_cmd.append("--verbose")
                    if args.dry_run:
                        print("DRY:", " ".join(cache_cmd))
                    else:
                        process = subprocess.Popen(cache_cmd, cwd=repo, start_new_session=True)
                        _register_child(process)
                        try:
                            rc = process.wait()
                        finally:
                            _unregister_child(process)
                        if rc:
                            raise SystemExit(f"preprocessing failed for {split} with exit code {rc}")
                        if not args.verify_cache_only:
                            cache_manifest = json.loads((namespace / "shared_cache.json").read_text()) if (namespace / "shared_cache.json").exists() else {}
                            cooked_marker.write_text(json.dumps({"split_group": split, "completed_at": time.time(), "cache_schema": CACHE_SCHEMA, "fingerprint": split_fingerprint, "shared_fingerprint": cache_manifest.get("fingerprint")}, indent=2) + "\n")
                else:
                    print(f"[verified] preprocessed {split}" if args.verify_cache_only else f"[cache hit] preprocessed {split}")
        if args.prepare_only or _interrupt_requested.is_set():
            return
        slots = [gpu for gpu in gpus for _ in range(args.runs_per_gpu)]
        jobs = [Job(split, seed, slots[i % len(slots)]) for i, (split, seed) in enumerate(((s, seed) for s in groups for seed in seeds))]
        if args.dry_run:
            for job in jobs:
                print(f"JOB split={job.split} seed={job.seed} gpu={job.gpu}")
            return
        queues = [[] for _ in slots]
        for i, job in enumerate(jobs):
            queues[i % len(slots)].append(job)

        reporter = make_reporter()

        def worker(queue: list[Job]) -> list[dict]:
            return [job_worker(job, args, repo, reporter) for job in queue]

        records: list[dict] = []
        with reporter, ThreadPoolExecutor(max_workers=len(slots)) as pool:
            for result in pool.map(worker, queues):
                records.extend(result)
        aggregate(args.output_root, records)
        failed = [r for r in records if r.get("status") in {"failed", "interrupted"}]
        print(f"Completed {len(records) - len(failed)}/{len(records)} jobs; results in {args.output_root / 'results'}")
        if failed:
            raise SystemExit(1)
    except KeyboardInterrupt:
        _interrupt_children()
        _stop_all_children(force=False)
        time.sleep(2)
        _stop_all_children(force=True)
        raise SystemExit(130)
    finally:
        _restore_signal_handlers(previous_signals)


if __name__ == "__main__":
    main()
