import os
import sys
import tempfile
import unittest
from pathlib import Path

from emulator_bench.run_all import classify_stage, parse_tqdm_progress, run_cmd


class RecordingReporter:
    def __init__(self):
        self.events = []

    def start(self, key, label, phase):
        self.events.append(("start", key, phase))

    def update(self, key, phase, line, *, verbose=False):
        self.events.append(("update", key, phase, line, verbose))

    def waiting(self, key, phase):
        self.events.append(("waiting", key, phase))

    def finish(self, key, status):
        self.events.append(("finish", key, status))


class RunAllProgressTest(unittest.TestCase):
    def test_parses_tqdm_frame(self):
        self.assertEqual(
            parse_tqdm_progress("14%|#4        | 744491/5187088 [00:15<01:24, 52571.68it/s]"),
            (14, 744491, 5187088),
        )

    def test_ignores_non_progress_line(self):
        self.assertIsNone(parse_tqdm_progress("# raw train loaded 36996"))

    def test_classifies_cpu_setup_and_gpu_optimization(self):
        self.assertEqual(
            classify_stage("train", "Load negative reaction map: 100record [00:01, 100record/s]").stage,
            "loading negative-reaction map",
        )
        self.assertEqual(classify_stage("train", "Train epoch 0: 10%").resource, "GPU")

    def test_classifies_evaluation_loading_and_inference(self):
        self.assertEqual(
            classify_stage("evaluate", "Load val reactions: 100reaction [00:01]").stage,
            "loading evaluation reactions",
        )
        self.assertEqual(
            classify_stage("evaluate", "Load product center maps: 100record [00:01]").lifecycle,
            "Evaluate",
        )
        self.assertEqual(
            classify_stage("evaluate", "Evaluate test reactions: 10reaction [00:01]").stage,
            "reaction inference",
        )

    def test_run_cmd_preserves_raw_log_and_reports_complete_frames(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.log"
            script = "import sys; sys.stdout.write('50%|#####| 5/10 [00:01<00:01, 5it/s]\\r'); sys.stdout.flush()"
            code = run_cmd(
                [sys.executable, "-c", script], root, log, dict(os.environ),
                reporter, "job", "train",
            )
            self.assertEqual(code, 0)
            self.assertIn("50%|#####| 5/10", log.read_text())
            updates = [event for event in reporter.events if event[0] == "update"]
            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0][1:4], ("job", "train", "50%|#####| 5/10 [00:01<00:01, 5it/s]"))


if __name__ == "__main__":
    unittest.main()
