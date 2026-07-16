"""PyTorch fallback for the legacy GLN jagged log-softmax operation."""
from __future__ import annotations

import torch


def jagged_log_softmax(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    chunks = torch.split(values, lengths.detach().cpu().tolist())
    return torch.cat([torch.log_softmax(c, dim=0) for c in chunks], dim=0)
