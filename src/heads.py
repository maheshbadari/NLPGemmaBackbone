"""NER head: Linear emission layer + CRF decoder."""

import torch
import torch.nn as nn
from typing import List, Optional

from .crf import CRF


class NERHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, bio_labels: Optional[List[str]] = None):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)
        self.crf    = CRF(num_labels, bio_labels)

    def loss(self, h: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.crf(self.linear(h), tags, mask)

    def decode(self, h: torch.Tensor, mask: torch.BoolTensor) -> List[List[int]]:
        return self.crf.decode(self.linear(h), mask)

    def emissions(self, h: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities [B, T, num_labels] — used for confidence scoring."""
        return torch.softmax(self.linear(h), dim=-1)
