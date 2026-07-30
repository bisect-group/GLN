import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emulator_bench.aggregate_results import aggregate, discover_runs, inspect_record, main


class AggregateResultsTest(unittest.TestCase):
    def write_run(self, schema_root, split, seed, state="valid"):
        run = schema_root / "runs" / split / f"seed_{seed}" / "train"
        run.mkdir(parents=True)
        records = [
            {"fingerprint": "f" * 64, "index": 0, "rxn_type": "UNK", "rxn": "a>>b", "predictions": ["a"], "templates": ["t"], "hits": [1] * 10},
            {"fingerprint": "f" * 64, "index": 1, "rxn_type": "UNK", "rxn": "c>>d", "predictions": [], "templates": [], "hits": [0] * 10},
        ]
        (run / "test.partial.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
        (run / "test.pred").write_text("UNK a>>b 1\nt a\nUNK c>>d 0\n")
        (run / "test.summary").write_text("type overall\ntop 1: 0.5000\ntop 10: 0.5000\n")
        if state == "malformed":
            (run / "evaluation.complete.json").write_text("{")
        elif state != "missing":
            manifest = {"fingerprint": "f" * 64, "reactions": 2, "topk": 10, "cache": {"hits": 1, "misses": 1, "waits": 0, "computed": 1, "failures": 0}}
            if state == "mismatch":
                (run / "test.summary").write_text("type overall\ntop 1: 0.0000\ntop 10: 0.5000\n")
            (run / "evaluation.complete.json").write_text(json.dumps(manifest))

    def test_discovers_and_aggregates_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / "schema-5"
            self.write_run(schema, "random_splits_grouped_rxn_smiles", 666)
            self.write_run(schema, "uniprot_time_splits", 777)
            self.write_run(schema, "reaction_drfp_tanimoto_splits/threshold_0.11", 888)
            records = [inspect_record(record) for record in discover_runs(schema, None)]
            self.assertEqual(len(records), 3)
            summaries, _ = aggregate(records)
            self.assertEqual(sum(item["runs_completed"] for item in summaries), 3)
            self.assertIn("reaction_drfp_tanimoto_splits/threshold_0.11", {item["split_name"] for item in summaries})

    def test_marks_bad_manifest_and_summary_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / "schema-5"
            self.write_run(schema, "random_splits_grouped_rxn_smiles", 666)
            self.write_run(schema, "random_splits_grouped_rxn_smiles", 777, "malformed")
            self.write_run(schema, "random_splits_grouped_rxn_smiles", 888, "mismatch")
            records = [inspect_record(record) for record in discover_runs(schema, None)]
            self.assertEqual({item["seed"]: item["status"] for item in records}, {666: "valid", 777: "invalid", 888: "invalid"})

    def test_strict_writes_diagnostics_then_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, output = Path(tmp) / "base", Path(tmp) / "out"
            self.write_run(base / "schema-5", "random_splits_grouped_rxn_smiles", 666, "missing")
            with patch("sys.argv", ["aggregate_results", "--base-dir", str(base), "--output-dir", str(output), "--strict"]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertTrue((output / "diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
