"""Linear-chain CRF layer (pure PyTorch)."""

import torch
import torch.nn as nn
from typing import List, Optional


class CRF(nn.Module):
    def __init__(self, num_tags: int, bio_labels: Optional[List[str]] = None):
        """
        Args:
            num_tags:   number of tag classes
            bio_labels: label list in index order; if provided, BIO-impossible
                        transitions are initialised to -1e4 and clamped during
                        training to remain strongly negative.
        """
        super().__init__()
        self.num_tags = num_tags

        self.transitions     = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_trans     = nn.Parameter(torch.empty(num_tags))
        self.end_trans       = nn.Parameter(torch.empty(num_tags))

        nn.init.uniform_(self.transitions,  -0.1, 0.1)
        nn.init.uniform_(self.start_trans,  -0.1, 0.1)
        nn.init.uniform_(self.end_trans,    -0.1, 0.1)

        if bio_labels is not None:
            self._apply_bio_constraints(bio_labels)

    # ------------------------------------------------------------------
    def _apply_bio_constraints(self, labels: List[str]) -> None:
        """Set structurally impossible BIO transitions to -1e4."""
        with torch.no_grad():
            for j, to_label in enumerate(labels):
                if to_label.startswith("I-"):
                    entity = to_label[2:]
                    for i, from_label in enumerate(labels):
                        if from_label not in (f"B-{entity}", f"I-{entity}"):
                            self.transitions[i, j] = -1e4
                    self.start_trans[j] = -1e4

    # ------------------------------------------------------------------
    def forward(
        self,
        emissions: torch.Tensor,    # [B, T, C]
        tags:      torch.Tensor,    # [B, T]  — padded positions may hold any valid idx
        mask:      torch.BoolTensor # [B, T]
    ) -> torch.Tensor:
        """Return mean negative log-likelihood."""
        return -self._log_likelihood(emissions, tags, mask).mean()

    def decode(
        self,
        emissions: torch.Tensor,    # [B, T, C]
        mask:      torch.BoolTensor # [B, T]
    ) -> List[List[int]]:
        return self._viterbi(emissions, mask)

    # ------------------------------------------------------------------
    def _log_likelihood(self, emissions, tags, mask):
        B, T, C = emissions.shape

        # Score along the gold path
        score = self.start_trans[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, T):
            active     = mask[:, t].float()
            trans_sc   = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_sc    = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            score      = score + (trans_sc + emit_sc) * active

        seq_lens  = mask.long().sum(1)
        last_idx  = (seq_lens - 1).clamp(min=0)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score     = score + self.end_trans[last_tags]

        return score - self._partition(emissions, mask)

    def _partition(self, emissions, mask):
        # [B, C]
        alpha = self.start_trans + emissions[:, 0]

        for t in range(1, emissions.size(1)):
            # scores[b, from, to] = alpha[b,from] + trans[from,to] + emit[b,to]
            scores     = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)                         # [B, C]
            alpha      = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

        return torch.logsumexp(alpha + self.end_trans, dim=1)                   # [B]

    def _viterbi(self, emissions, mask):
        B, T, C   = emissions.shape
        viterbi   = self.start_trans + emissions[:, 0]                          # [B, C]
        backptrs: List[torch.Tensor] = []

        for t in range(1, T):
            # scores[b, from, to]
            scores       = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_scores, best_from = scores.max(dim=1)                          # [B, C] each
            backptrs.append(best_from)
            viterbi = torch.where(mask[:, t].unsqueeze(1), best_scores, viterbi)

        viterbi     = viterbi + self.end_trans
        _, best_last = viterbi.max(dim=1)                                       # [B]
        seq_lens    = mask.long().sum(1).tolist()

        paths = []
        for b in range(B):
            length = int(seq_lens[b])
            tag    = best_last[b].item()
            path   = [tag]
            for bp_idx in range(length - 2, -1, -1):
                tag = backptrs[bp_idx][b, tag].item()
                path.append(tag)
            path.reverse()
            paths.append(path)

        return paths
