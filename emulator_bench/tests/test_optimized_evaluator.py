import json
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from emulator_bench.evaluation_cache import SQLiteReactionCache, _CLAIMED
from emulator_bench.optimized_evaluator import configure_cuda_visibility, load_journal, native_import_args


class OptimizedEvaluatorTest(unittest.TestCase):
    def cache(self, root, *, rebuild=False):
        return SQLiteReactionCache(root, rebuild=rebuild)

    @patch("emulator_bench.evaluation_cache.cache_namespace", return_value="test-namespace")
    def test_cache_stores_hits_and_deterministic_failures(self, _namespace):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.cache(Path(tmp))
            self.assertEqual(cache.acquire("P", "T")[0], "claimed")
            cache.store("P", "T", None)
            self.assertEqual(cache.acquire("P", "T"), ("ready", None))
            cache.close()

    @patch("emulator_bench.evaluation_cache.cache_namespace", return_value="test-namespace")
    def test_concurrent_claim_wait_and_expired_lease_recovery(self, _namespace):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.cache(Path(tmp), rebuild=True)
            second = self.cache(Path(tmp))
            self.assertEqual(first.acquire("P", "T")[0], "claimed")
            self.assertEqual(second.acquire("P", "T")[0], "wait")
            second.connection.execute("UPDATE reactions SET lease_until=?", (time.time() - 1,))
            self.assertEqual(second.wait_ready("P", "T"), _CLAIMED)
            second.store("P", "T", ["result"])
            self.assertEqual(first.acquire("P", "T"), ("ready", ["result"]))
            first.close()
            second.close()

    @patch("emulator_bench.evaluation_cache.cache_namespace", return_value="test-namespace")
    @patch("emulator_bench.evaluation_cache._owner_is_alive", return_value=False)
    def test_dead_lease_owner_is_reclaimed_without_waiting_for_expiry(self, _alive, _namespace):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.cache(Path(tmp), rebuild=True)
            second = self.cache(Path(tmp))
            self.assertEqual(first.acquire("P", "T")[0], "claimed")
            # The lease is deliberately still valid; only the dead-owner check
            # should allow this immediate reclaim.
            self.assertEqual(second.acquire("P", "T")[0], "claimed")
            first.close()
            second.close()

    def test_journal_resume_and_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.partial.jsonl"
            first = {"fingerprint": "f", "index": 0, "rxn": "a"}
            path.write_bytes(json.dumps(first).encode() + b"\n" + b'{"fingerprint": "f"')
            self.assertEqual(load_journal(path, "f"), [first])
            self.assertEqual(path.read_text(), json.dumps(first) + "\n")

    def test_journal_fingerprint_mismatch_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.partial.jsonl"
            path.write_text(json.dumps({"fingerprint": "old", "index": 0}) + "\n")
            self.assertEqual(load_journal(path, "new"), [])
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(tmp).glob("test.partial.jsonl.mismatch.*"))), 1)

    def test_native_import_args_use_checkpoint_features_and_local_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "train" / "model-9.dump" / "model.dump"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"model")
            args = Namespace(fp_degree=2)
            import pickle
            with (checkpoint.parent / "args.pkl").open("wb") as stream:
                pickle.dump(args, stream)
            result = native_import_args(root / "dropbox", "reaction_outcome_dataset", checkpoint, "0")
            self.assertEqual(result, [
                "-gpu", "0", "-f_atoms",
                str(root / "dropbox" / "cooked_reaction_outcome_dataset" / "atom_list.txt"),
                "-fp_degree", "2",
            ])

    def test_cuda_visibility_preserves_runner_gpu_and_uses_local_zero(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            self.assertEqual(configure_cuda_visibility("0"), "0")
            self.assertEqual(__import__("os").environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_cuda_visibility_exposes_requested_gpu_for_standalone_evaluator(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(configure_cuda_visibility("2"), "0")
            self.assertEqual(__import__("os").environ["CUDA_VISIBLE_DEVICES"], "2")


if __name__ == "__main__":
    unittest.main()
