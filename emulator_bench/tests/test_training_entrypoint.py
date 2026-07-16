import os
import subprocess
import sys
import unittest

from emulator_bench.train import configure_child_env


class TrainingEntrypointTest(unittest.TestCase):
    def test_child_environment_defaults_cpu_thread_pools_to_one(self):
        env = configure_child_env()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(env[name], os.environ.get(name, "1"))

    def test_sets_cpu_thread_caps_before_gln_imports(self):
        code = "from emulator_bench.training_entrypoint import configure_torch_runtime; import torch; assert configure_torch_runtime() == (1, 1); assert torch.get_num_threads() == 1; assert torch.get_num_interop_threads() == 1"
        env = dict(os.environ)
        env["EMULATOR_BENCH_TORCH_THREADS"] = "1"
        env["EMULATOR_BENCH_TORCH_INTEROP_THREADS"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
