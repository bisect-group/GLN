import unittest

import torch
from torch_scatter import scatter_max as native_scatter_max

from gln.mods.mol_gnn.gnn_family.utils import scatter_max


class TorchScatterCompatibilityTest(unittest.TestCase):
    def test_scatter_max_matches_current_torch_scatter_signature(self):
        src = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
        index = torch.tensor([0, 0, 1])

        expected = native_scatter_max(src, index, dim=0, dim_size=2)[0]
        actual = scatter_max(src, index, dim=0, dim_size=2)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
