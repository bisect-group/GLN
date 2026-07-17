import os
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from emulator_bench.run_all import (
    Job,
    completed_training_checkpoint,
    classify_stage,
    job_worker,
    mark_training_complete,
    parse_evaluation_progress,
    parse_tqdm_progress,
    run_cmd,
)
import emulator_bench.run_all as run_all


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

    def test_parses_unknown_total_evaluation_frame(self):
        line = "average score, t1: 0.1742, t10: 0.2921: : 178reaction [30:31, 74.24s/reaction]"
        self.assertEqual(parse_evaluation_progress(line), (178, None))

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
        self.assertEqual(
            classify_stage("evaluate", "average score, t1: 0.1742: : 178reaction [30:31]").stage,
            "reaction inference",
        )

    def test_classifies_evaluation_cpu_cache_and_rdchiral_substages(self):
        self.assertEqual(
            classify_stage("evaluate", "Evaluation CPU cache lookup: 12/100 reactions; template batch 1/4").stage,
            "template cache lookup",
        )
        self.assertEqual(
            classify_stage("evaluate", "Evaluation cache wait: 12/100 reactions; template batch 1/4").stage,
            "waiting for shared template cache",
        )
        self.assertEqual(
            classify_stage("evaluate", "Evaluation CPU RDChiral: 12/100 reactions; template batch 1/4").stage,
            "RDChiral template application",
        )

    def test_adopts_completed_legacy_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "train" / "model-9.dump" / "model.dump"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(completed_training_checkpoint(run_dir, adopt_legacy=True), checkpoint)
            self.assertEqual(completed_training_checkpoint(run_dir, adopt_legacy=False), checkpoint)

    def test_resume_skips_completed_training_and_retries_evaluation(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            run_dir = output_root / "runs" / "split" / "seed_1"
            checkpoint = run_dir / "train" / "model-9.dump" / "model.dump"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            mark_training_complete(run_dir, checkpoint, seed=1)
            args = Namespace(output_root=output_root, resume=True, eval_only=False, train_only=False, dry_run=False, verbose=False, fail_fast=False)
            commands = []

            def fake_run_cmd(cmd, *unused_args):
                commands.append(cmd)
                return 0

            with patch("emulator_bench.run_all.run_cmd", side_effect=fake_run_cmd):
                result = job_worker(Job("split", 1, "0"), args, root, reporter)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(commands), 1)
            self.assertIn("emulator_bench.evaluate", commands[0])
            self.assertEqual(commands[0][commands[0].index("--checkpoint") + 1], str(checkpoint))

    def test_new_training_evaluates_only_final_checkpoint(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            run_dir = output_root / "runs" / "split" / "seed_1"
            args = Namespace(output_root=output_root, resume=False, eval_only=False, train_only=False, dry_run=False, verbose=False, fail_fast=False)
            commands = []

            def fake_run_cmd(cmd, *unused_args):
                commands.append(cmd)
                if "emulator_bench.train" in cmd:
                    checkpoint = run_dir / "train" / "model-9.dump" / "model.dump"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"checkpoint")
                return 0

            with patch("emulator_bench.run_all.run_cmd", side_effect=fake_run_cmd):
                result = job_worker(Job("split", 1, "0"), args, root, reporter)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[1][commands[1].index("--checkpoint") + 1], str(run_dir / "train" / "model-9.dump" / "model.dump"))

    def test_eval_only_requires_final_checkpoint(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(output_root=root / "output", resume=True, eval_only=True, train_only=False, dry_run=False, verbose=False, fail_fast=False)
            result = job_worker(Job("split", 1, "0"), args, root, reporter)
            self.assertEqual(result["status"], "failed")
            self.assertIn("evaluation requires completed final checkpoint", result["error"])

    def test_single_checkpoint_summary_populates_test_metrics(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "output"
            run_dir = output_root / "runs" / "split" / "seed_1"
            checkpoint = run_dir / "train" / "model-9.dump" / "model.dump"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            mark_training_complete(run_dir, checkpoint, seed=1)
            args = Namespace(output_root=output_root, resume=True, eval_only=False, train_only=False, dry_run=False, verbose=False, fail_fast=False)

            def fake_run_cmd(cmd, *unused_args):
                (run_dir / "train" / "test.summary").write_text("top 1: 0.1234\ntop 10: 0.5678\n")
                return 0

            with patch("emulator_bench.run_all.run_cmd", side_effect=fake_run_cmd):
                result = job_worker(Job("split", 1, "0"), args, root, reporter)
            self.assertEqual(result["test_top1"], 0.1234)
            self.assertEqual(result["test_top10"], 0.5678)
            self.assertEqual(result["selected_checkpoint"], str(checkpoint))

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

    def test_run_cmd_interrupt_escalates_stuck_child_group(self):
        reporter = RecordingReporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.log"
            script = "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
            run_all._interrupt_requested.clear()
            timer = threading.Timer(0.2, run_all._interrupt_requested.set)
            timer.start()
            try:
                started = time.monotonic()
                code = run_cmd([sys.executable, "-c", script], root, log, dict(os.environ), reporter, "job", "evaluate")
                self.assertNotEqual(code, 0)
                self.assertLess(time.monotonic() - started, 8)
            finally:
                timer.cancel()
                run_all._interrupt_requested.clear()


if __name__ == "__main__":
    unittest.main()
