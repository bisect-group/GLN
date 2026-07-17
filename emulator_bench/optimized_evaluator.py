"""Exact-output GLN final-checkpoint evaluator with RDChiral cache and resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from scipy.special import softmax

from .evaluation_cache import SQLiteReactionCache, _CLAIMED


EVALUATOR_VERSION = "gln-final-checkpoint-cache-v1"


def _apply_template(raw_product: str, template: str) -> list[str] | None:
    """Worker entrypoint; each process owns native Reactor's local caches."""
    from gln.common.reactor import Reactor
    return Reactor.run_reaction(raw_product, template)


def _fingerprint(checkpoint: Path, test_csv: Path, beam_size: int, topk: int) -> str:
    payload = {
        "version": EVALUATOR_VERSION,
        "checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_csv": hashlib.sha256(test_csv.read_bytes()).hexdigest(),
        "beam_size": beam_size,
        "topk": topk,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def native_import_args(dropbox: Path, data_name: str, checkpoint: Path, gpu: str) -> list[str]:
    """Arguments GLN reads at import time, before ``RetroGLN`` loads args.pkl.

    GLN's C++ molecule library takes ``-f_atoms`` from ``sys.argv`` while
    ``gln.common.consts`` selects a device from ``-gpu``.  Both happen before
    the checkpoint's saved argument object is applied, so omitting them builds
    a different graph shape (and defaults to CPU).
    """
    with (checkpoint.parent / "args.pkl").open("rb") as stream:
        checkpoint_args = pickle.load(stream)
    return [
        "-gpu", str(gpu),
        "-f_atoms", str(dropbox / f"cooked_{data_name}" / "atom_list.txt"),
        "-fp_degree", str(getattr(checkpoint_args, "fp_degree", 0)),
    ]


def configure_cuda_visibility(requested_gpu: str) -> str:
    """Preserve runner-assigned physical visibility and return GLN's local GPU.

    ``run_all`` restricts every child to one physical device through
    ``CUDA_VISIBLE_DEVICES``.  Within that child the only usable CUDA device is
    index 0.  A standalone evaluator gets the requested physical GPU exposed
    first, then also uses local index 0.
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
    return "0"


def _quarantine(path: Path, reason: str) -> None:
    if path.exists():
        target = path.with_name(f"{path.name}.{reason}.{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.move(path, target)


def load_journal(path: Path, fingerprint: str) -> list[dict]:
    """Return only a valid contiguous prefix; quarantine semantic corruption."""
    if not path.exists():
        return []
    records: list[dict] = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, raw in enumerate(lines):
        complete = raw.endswith(b"\n")
        if index == len(lines) - 1 and not complete:
            # Appends are newline terminated; a final unterminated record may have been cut off.
            path.write_bytes(b"".join(lines[:index]))
            return records
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            _quarantine(path, "corrupt")
            return []
        if record.get("fingerprint") != fingerprint or record.get("index") != index:
            _quarantine(path, "mismatch")
            return []
        records.append(record)
    return records


class CachedRetroGLN:
    """Native ranking/scoring with a bounded, ordered parallel reactor stage."""

    def __init__(self, native, cache: SQLiteReactionCache, workers: int, status: Callable[[str], None] | None = None):
        self.native = native
        self.cache = cache
        # Forking after CUDA initialization is unsafe even when workers only run RDChiral.
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        self.workers = workers
        self.status = status or (lambda _message: None)

    def close(self, *, interrupted: bool = False) -> None:
        """Release workers; never block shutdown after Ctrl-C/SIGTERM."""
        if interrupted:
            # ``shutdown(wait=True)`` can hang in a native RDChiral call.  The
            # journal has already flushed completed reactions, so terminate the
            # disposable worker processes and let --resume reclaim unfinished work.
            for process in self.pool._processes.values():
                if process.is_alive():
                    process.terminate()
        self.pool.shutdown(wait=not interrupted, cancel_futures=True)

    def _template_outcomes(self, raw_product: str, templates: list[str], progress_label: str) -> Iterator[tuple[str, list[str] | None]]:
        """Compute in bounded batches, yielding strictly in ranked-template order."""
        for start in range(0, len(templates), self.workers):
            batch = templates[start:start + self.workers]
            batch_number = start // self.workers + 1
            batch_total = (len(templates) + self.workers - 1) // self.workers
            resolved: dict[int, list[str] | None] = {}
            pending: list[tuple[int, object, str]] = []
            hit_count = claim_count = wait_count = 0
            for offset, template in enumerate(batch):
                state, outcomes = self.cache.acquire(raw_product, template)
                if state == "ready":
                    resolved[offset] = outcomes
                    hit_count += 1
                elif state == "claimed":
                    pending.append((offset, self.pool.submit(_apply_template, raw_product, template), template))
                    claim_count += 1
                else:
                    pending.append((offset, None, template))
                    wait_count += 1
            self.status(
                f"Evaluation CPU cache lookup: {progress_label}; template batch {batch_number}/{batch_total}; "
                f"{hit_count} cached, {claim_count} RDChiral tasks, {wait_count} shared-cache waits"
            )
            for offset, future, template in pending:
                if future is None:
                    self.status(
                        f"Evaluation cache wait: {progress_label}; template batch {batch_number}/{batch_total}; "
                        f"waiting for another evaluator's RDChiral result"
                    )
                    waited = self.cache.wait_ready(raw_product, template)
                    if waited is _CLAIMED:
                        self.status(
                            f"Evaluation CPU RDChiral: {progress_label}; template batch {batch_number}/{batch_total}; "
                            "reclaimed expired cache lease"
                        )
                        future = self.pool.submit(_apply_template, raw_product, template)
                    else:
                        resolved[offset] = waited
                        continue
                try:
                    self.status(
                        f"Evaluation CPU RDChiral: {progress_label}; template batch {batch_number}/{batch_total}; "
                        f"applying {len(pending)} templates in {self.workers} workers"
                    )
                    outcomes = future.result()
                except Exception:
                    outcomes = None
                self.cache.store(raw_product, template, outcomes)
                resolved[offset] = outcomes
            for offset, template in enumerate(batch):
                yield template, resolved[offset]

    def run(self, raw_product: str, beam_size: int, topk: int, rxn_type: str = "UNK", progress_label: str = "") -> dict | None:
        # Keep native center/template ordering and all GPU scoring exactly intact.
        from gln.data_process.data_info import DataInfo
        from gln.mods.mol_gnn.mol_utils import SmilesMols

        with torch.inference_mode():
            self.status(f"Evaluation GPU center/template ranking: {progress_label}; scoring reaction-center and template candidates")
            cano_prod = DataInfo.get_cano_smiles(raw_product)
            prod_mol = SmilesMols.get_mol_graph(cano_prod)
            tpl_with_scores = self.native._ordered_tpls(cano_prod, beam_size, rxn_type)
            if tpl_with_scores is None:
                return None
            successful = 0
            list_of_list_reacts = []
            list_tpls = []
            for (prod_tpl_score, expected_template), (template, outcomes) in zip(
                tpl_with_scores,
                self._template_outcomes(raw_product, [pair[1] for pair in tpl_with_scores], progress_label),
            ):
                assert template == expected_template
                if outcomes:
                    list_of_list_reacts.append(outcomes)
                    list_tpls.append((prod_tpl_score, template))
                    successful += 1
                    if successful >= beam_size:
                        break
            list_rxns = [
                [DataInfo.get_cano_smiles(reactants) + ">>" + cano_prod for reactants in alternatives]
                for alternatives in list_of_list_reacts
            ]
            if not list_rxns or not list_tpls:
                return {"template": [], "reactants": [], "scores": []}
            candidate_count = sum(len(reactions) for reactions in list_of_list_reacts)
            self.status(
                f"Evaluation GPU reactant scoring: {progress_label}; scoring {candidate_count} candidate reactants "
                f"from {len(list_tpls)} successful templates"
            )
            react_scores = self.native.gln.reaction_predicate.inference([prod_mol] * len(list_tpls), list_rxns)
            react_scores = react_scores.view(-1).cpu().numpy()
            final_joint = []
            index = 0
            for (prod_tpl_score, template), alternatives in zip(list_tpls, list_of_list_reacts):
                for reactants in alternatives:
                    final_joint.append((prod_tpl_score + react_scores[index], template, reactants))
                    index += 1
            final_joint = sorted(final_joint, key=lambda item: -item[0])[:topk]
            return {
                "template": [item[1] for item in final_joint],
                "reactants": [item[2] for item in final_joint],
                "scores": softmax([item[0] for item in final_joint]),
            }


def _prediction_record(index: int, total: int, fingerprint: str, rxn_type: str, rxn: str, model: CachedRetroGLN, beam_size: int, topk: int) -> dict:
    from gln.common.evaluate import canonicalize

    raw_product = rxn.split(">")[2]
    pred_struct = model.run(raw_product, beam_size, topk, rxn_type=rxn_type, progress_label=f"{index + 1}/{total} reactions")
    reactants, _, product = rxn.split(">")
    predictions = list(pred_struct["reactants"]) if pred_struct is not None and pred_struct["reactants"] else [product]
    expected = canonicalize(reactants)
    hits: list[float] = []
    score = 0.0
    for rank in range(topk):
        if rank < len(predictions):
            predictions[rank] = canonicalize(predictions[rank])
            score = max(float(predictions[rank] == expected), score)
        hits.append(score)
    if pred_struct is None or not pred_struct["reactants"]:
        predictions = []
        templates: list[str] = []
    else:
        templates = list(pred_struct["template"])
    return {"fingerprint": fingerprint, "index": index, "rxn_type": rxn_type, "rxn": rxn,
            "predictions": predictions, "templates": templates, "hits": hits}


def _write_outputs(save_dir: Path, records: list[dict], topk: int, fingerprint: str, cache: SQLiteReactionCache) -> None:
    pred_lines: list[str] = []
    scores = [0.0] * topk
    for record in records:
        predictions = record["predictions"]
        pred_lines.append(f"{record['rxn_type']} {record['rxn']} {len(predictions)}\n")
        pred_lines.extend(f"{template} {prediction}\n" for template, prediction in zip(record["templates"], predictions))
        scores = [total + value for total, value in zip(scores, record["hits"])]
    count = len(records)
    _atomic_text(save_dir / "test.pred", "".join(pred_lines))
    summary = "type overall\n" + "".join(f"top {rank}: {scores[rank - 1] / count:.4f}\n" for rank in range(1, topk + 1))
    _atomic_text(save_dir / "test.summary", summary)
    _atomic_text(save_dir / "evaluation.complete.json", json.dumps({
        "fingerprint": fingerprint, "reactions": count, "topk": topk,
        "cache": cache.stats.as_dict(), "cache_manifest": str(cache.write_manifest()),
        "completed_at": time.time(),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropbox", type=Path, required=True)
    parser.add_argument("--data-name", default="reaction_outcome_dataset")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--evaluation-cache-root", type=Path)
    parser.add_argument("--rebuild-evaluation-cache", action="store_true")
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()
    if args.eval_workers < 1:
        parser.error("--eval-workers must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluation_cache_root is None:
        schema_root = next((parent for parent in args.save_dir.parents if parent.name == "schema-5"), None)
        args.evaluation_cache_root = (schema_root / "evaluation_cache") if schema_root else (args.save_dir / "evaluation_cache")
    local_gpu = configure_cuda_visibility(args.gpu)
    # These must be visible before GLN import-time global configuration.
    sys.argv.extend(native_import_args(args.dropbox, args.data_name, args.checkpoint, local_gpu))
    # Imports after CUDA visibility and native import-time args are fixed.
    from gln.data_process.data_info import load_center_maps
    from gln.test.model_inference import RetroGLN

    test_csv = args.dropbox / args.data_name / "raw_test.csv"
    fingerprint = _fingerprint(args.checkpoint, test_csv, args.beam_size, args.topk)
    journal = args.save_dir / "test.partial.jsonl"
    records = load_journal(journal, fingerprint)
    with test_csv.open() as stream:
        reader = csv.reader(stream)
        next(reader)
        cases = [(row[1], row[2]) for row in reader]
    center_maps = load_center_maps(args.dropbox / f"cooked_{args.data_name}" / "tpl-default" / "np-1" / "test-prod_center_maps-part-0.csv")
    if records:
        print(f"Evaluation resumed: {len(records)}/{len(cases)} reactions", flush=True)
    cache = SQLiteReactionCache(args.evaluation_cache_root, rebuild=args.rebuild_evaluation_cache)
    native = RetroGLN(str(args.dropbox), str(args.checkpoint.parent))
    native.prod_center_maps = center_maps
    model = CachedRetroGLN(native, cache, args.eval_workers, status=lambda message: print(message, flush=True))
    completed = False
    try:
        with journal.open("a") as stream:
            for index in range(len(records), len(cases)):
                rxn_type, rxn = cases[index]
                record = _prediction_record(index, len(cases), fingerprint, rxn_type, rxn, model, args.beam_size, args.topk)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                stats = cache.stats
                total_cache = stats.hits + stats.misses
                rate = 100.0 * stats.hits / total_cache if total_cache else 0.0
                print(f"Evaluation progress: {index + 1}/{len(cases)} reactions; cache hits={stats.hits} misses={stats.misses} waits={stats.waits} hit_rate={rate:.1f}%", flush=True)
        _write_outputs(args.save_dir, records, args.topk, fingerprint, cache)
        print(f"Evaluation complete: {len(records)} reactions; cache={cache.stats.as_dict()}", flush=True)
        completed = True
    except KeyboardInterrupt:
        print("[interrupt] evaluation stopped; completed reactions are journaled for --resume", flush=True)
        raise
    finally:
        model.close(interrupted=not completed)
        cache.close()


if __name__ == "__main__":
    main()
