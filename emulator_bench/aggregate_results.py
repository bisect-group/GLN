"""Verify and aggregate GLN schema-5 final-checkpoint test evaluations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rich.console import Console
from rich.table import Table


DEFAULT_BASE_DIR = Path("/home/adhil/github/EMULaToR/data/processed/baselines/GLN")
SUMMARY_TOLERANCE = 0.000051  # Native summaries are rounded to four decimals.
console = Console()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{__import__('os').getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def parse_summary(path: Path) -> dict[int, float]:
    scores = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"top (\d+): ([0-9]+(?:\.[0-9]+)?)", line.strip())
        if match:
            scores[int(match.group(1))] = float(match.group(2))
    if 1 not in scores or 10 not in scores:
        raise ValueError("test.summary must contain top 1 and top 10")
    return scores


def load_journal(path: Path, fingerprint: str, expected_reactions: int, topk: int) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"test.partial.jsonl line {line_number} is invalid JSON: {error.msg}") from error
        if record.get("fingerprint") != fingerprint:
            raise ValueError(f"journal fingerprint mismatch at line {line_number}")
        if record.get("index") != len(records):
            raise ValueError(f"journal index is not contiguous at line {line_number}")
        hits = record.get("hits")
        if not isinstance(hits, list) or len(hits) < topk or any(value not in (0, 1, 0.0, 1.0) for value in hits[:topk]):
            raise ValueError(f"journal hits are invalid at line {line_number}")
        if any(float(hits[index]) < float(hits[index - 1]) for index in range(1, topk)):
            raise ValueError(f"journal hits are not cumulative at line {line_number}")
        records.append(record)
    if len(records) != expected_reactions:
        raise ValueError(f"journal contains {len(records)} reactions; manifest declares {expected_reactions}")
    return records


def validate_prediction_file(path: Path, records: list[dict[str, Any]]) -> None:
    lines = path.read_text().splitlines()
    cursor = 0
    for record in records:
        if cursor >= len(lines):
            raise ValueError("test.pred ended before all journal reactions")
        prefix = f"{record.get('rxn_type')} {record.get('rxn')} "
        header = lines[cursor]
        if not header.startswith(prefix):
            raise ValueError(f"test.pred header mismatch at reaction {record['index']}")
        try:
            count = int(header[len(prefix):])
        except ValueError as error:
            raise ValueError(f"test.pred prediction count is invalid at reaction {record['index']}") from error
        if count != len(record.get("predictions", [])):
            raise ValueError(f"test.pred prediction count disagrees with journal at reaction {record['index']}")
        cursor += 1 + count
        if cursor > len(lines):
            raise ValueError(f"test.pred is truncated at reaction {record['index']}")
    if cursor != len(lines):
        raise ValueError("test.pred contains trailing records not present in the journal")


def discover_runs(schema_root: Path, split_groups: list[str] | None) -> list[dict[str, Any]]:
    runs_root = schema_root / "runs"
    if not runs_root.is_dir():
        return []
    records = []
    for seed_dir in sorted(path for path in runs_root.rglob("seed_*") if path.is_dir() and (path / "train").is_dir()):
        split_root = seed_dir.parent
        split_name = split_root.relative_to(runs_root).as_posix()
        split_group = split_name.split("/", 1)[0]
        if split_groups and split_name not in split_groups and split_group not in split_groups:
            continue
        seed_text = seed_dir.name.removeprefix("seed_")
        try:
            seed: int | str = int(seed_text)
        except ValueError:
            seed = seed_text
        records.append({"split_group": split_group, "split_name": split_name, "seed": seed, "run_dir": seed_dir / "train"})
    return records


def inspect_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    run_dir = result["run_dir"]
    required = {"manifest": run_dir / "evaluation.complete.json", "summary": run_dir / "test.summary", "predictions": run_dir / "test.pred", "journal": run_dir / "test.partial.jsonl"}
    result.update(run_dir=str(run_dir), status="valid", error="", metrics={})
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        result.update(status="missing", error=f"missing required artifact(s): {', '.join(missing)}")
        return result
    try:
        manifest = load_json(required["manifest"])
        fingerprint = manifest.get("fingerprint")
        reactions, topk = manifest.get("reactions"), manifest.get("topk")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("evaluation manifest has no fingerprint")
        if not isinstance(reactions, int) or reactions <= 0 or not isinstance(topk, int) or topk < 10:
            raise ValueError("evaluation manifest has invalid reactions or topk")
        journal = load_journal(required["journal"], fingerprint, reactions, topk)
        validate_prediction_file(required["predictions"], journal)
        native_scores = parse_summary(required["summary"])
        recomputed = {rank: sum(float(item["hits"][rank - 1]) for item in journal) / reactions for rank in (1, 10)}
        mismatches = [f"top {rank}: native={native_scores[rank]:.4f}, recomputed={recomputed[rank]:.6f}" for rank in (1, 10) if abs(native_scores[rank] - recomputed[rank]) > SUMMARY_TOLERANCE]
        cache = manifest.get("cache") if isinstance(manifest.get("cache"), dict) else {}
        result["metrics"] = {
            "test_top1": recomputed[1], "test_top10": recomputed[10], "reactions": reactions, "topk": topk,
            "cache_hits": cache.get("hits"), "cache_misses": cache.get("misses"), "cache_waits": cache.get("waits"), "cache_computed": cache.get("computed"), "cache_failures": cache.get("failures"),
        }
        if mismatches:
            result.update(status="invalid", error="; ".join(mismatches))
    except (OSError, ValueError, TypeError) as error:
        result.update(status="invalid", error=str(error))
    return result


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split_group"], record["split_name"])].append(record)
    rows, payload = [], []
    for identity, members in sorted(groups.items()):
        valid = [member for member in members if member["status"] == "valid"]
        row = {"split_group": identity[0], "split_name": identity[1], "runs_discovered": len(members), "runs_completed": len(valid), "runs_missing": sum(member["status"] == "missing" for member in members), "runs_invalid": sum(member["status"] == "invalid" for member in members), "completed_seeds": ",".join(str(member["seed"]) for member in valid)}
        metrics = {}
        for name in ("test_top1", "test_top10"):
            values = [member["metrics"][name] for member in valid]
            metrics[name] = {"count": len(values), "mean": mean(values) if values else None, "sample_std": stdev(values) if len(values) > 1 else 0.0 if values else None}
            row[f"{name}_mean"] = metrics[name]["mean"]
            row[f"{name}_sample_std"] = metrics[name]["sample_std"]
        rows.append(row)
        payload.append({**{key: value for key, value in row.items() if not key.endswith("_mean") and not key.endswith("_sample_std")}, "metrics": metrics})
    return rows, payload


def write_reports(output_dir: Path, records: list[dict[str, Any]], summaries: list[dict[str, Any]], summary_json: list[dict[str, Any]]) -> None:
    metrics = ["test_top1", "test_top10", "reactions", "topk", "cache_hits", "cache_misses", "cache_waits", "cache_computed", "cache_failures"]
    per_run = [{key: value for key, value in record.items() if key != "metrics"} | record["metrics"] for record in records]
    fields = ["split_group", "split_name", "seed", "run_dir", "status", "error"] + metrics
    summary_fields = ["split_group", "split_name", "runs_discovered", "runs_completed", "runs_missing", "runs_invalid", "completed_seeds", "test_top1_mean", "test_top1_sample_std", "test_top10_mean", "test_top10_sample_std"]
    diagnostics = [{key: value for key, value in record.items() if key != "metrics"} for record in records if record["status"] != "valid"]
    atomic_write(output_dir / "per_run_metrics.csv", csv_text(per_run, fields))
    atomic_write(output_dir / "summary_metrics.csv", csv_text(summaries, summary_fields))
    atomic_write(output_dir / "summary_metrics.json", json.dumps({"groups": summary_json}, indent=2, sort_keys=True) + "\n")
    atomic_write(output_dir / "diagnostics.json", json.dumps({"runs": diagnostics}, indent=2, sort_keys=True) + "\n")


def render_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    table = Table(title="GLN aggregated final-checkpoint test metrics")
    for column in ("split", "completed", "missing", "invalid", "top-1 mean", "top-10 mean"):
        table.add_column(column)
    for row in rows:
        table.add_row(row["split_name"], str(row["runs_completed"]), str(row["runs_missing"]), str(row["runs_invalid"]), f"{row['test_top1_mean']:.4f}" if row["test_top1_mean"] is not None else "", f"{row['test_top10_mean']:.4f}" if row["test_top10_mean"] is not None else "")
    console.print(table)
    console.print(f"Reports written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--schema-dir", default="schema-5")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-groups", nargs="*")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    schema_root = args.base_dir / args.schema_dir
    output_dir = args.output_dir or schema_root / "aggregated_metrics"
    records = [inspect_record(record) for record in discover_runs(schema_root, args.split_groups)]
    if not records:
        parser.error(f"No GLN seed runs found under {schema_root / 'runs'}")
    summaries, summary_json = aggregate(records)
    write_reports(output_dir, records, summaries, summary_json)
    render_summary(summaries, output_dir)
    if args.strict and any(record["status"] != "valid" for record in records):
        raise SystemExit("Strict aggregation failed: see diagnostics.json.")


if __name__ == "__main__":
    main()
