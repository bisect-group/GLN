from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emulator_bench.stage_cache import CACHE_SCHEMA, implementation_identity, publish_manifest, stage_key, validate_manifest


class StageCacheTest(unittest.TestCase):
    def test_manifest_detects_corrupt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact.txt"
            output.write_text("good\n")
            key = stage_key("canonical_smiles", {"source": "a"}, {"code": "b"}, {})
            manifest = root / "stage.json"
            publish_manifest(manifest, "canonical_smiles", key, {"source": "a"}, {"code": "b"}, {}, [output], row_count=1)
            self.assertEqual(validate_manifest(manifest, key, [output]), (True, "verified"))
            output.write_text("corrupt\n")
            self.assertFalse(validate_manifest(manifest, key, [output])[0])

    def test_execution_settings_do_not_change_key(self) -> None:
        # Worker count, logging/output paths and CUDA visibility are never
        # supplied as semantic parameters, so identical inputs produce one key.
        inputs = {"source": "a", "dependency_key": "b"}
        implementation = {"code": "c"}
        self.assertEqual(stage_key("center_maps", inputs, implementation, {}),
                         stage_key("center_maps", inputs, implementation, {}))

    def test_implementation_identity_is_stable(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        identity = implementation_identity(repo, ["emulator_bench/stage_cache.py"])
        self.assertIn("emulator_bench/stage_cache.py", identity["sources"])
        self.assertEqual(CACHE_SCHEMA, 5)


if __name__ == "__main__":
    unittest.main()
