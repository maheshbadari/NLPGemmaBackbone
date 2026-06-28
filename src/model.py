"""GemmaBackbone NER model.

Frozen Gemma 3 270M → BiLSTM → LayerNorm + Dropout → 4 parallel CRF heads.
Only the BiLSTM and heads (~2.4M params) are trained.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Optional, Tuple

from .heads import NERHead
from .labels import IDENTITY_LABELS, LOCATION_LABELS, TEMPORAL_LABELS, DOMAIN_LABELS


class GemmaBackboneNER(nn.Module):
    def __init__(self, cfg):   # cfg: ModelConfig
        super().__init__()

        # ── Frozen Gemma backbone ─────────────────────────────────────
        gemma_cfg   = AutoConfig.from_pretrained(cfg.gemma_model_id)
        self.gemma  = AutoModel.from_pretrained(cfg.gemma_model_id)
        for p in self.gemma.parameters():
            p.requires_grad = False
        self.gemma.eval()

        hidden   = gemma_cfg.hidden_size        # 1024 for gemma-3-270m
        lstm_out = cfg.bilstm_hidden * 2        # 512

        # ── Trainable BiLSTM ─────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size   = hidden,
            hidden_size  = cfg.bilstm_hidden,
            num_layers   = cfg.bilstm_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = cfg.bilstm_dropout if cfg.bilstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(lstm_out)
        self.drop = nn.Dropout(cfg.post_dropout)

        # ── Four parallel CRF heads ───────────────────────────────────
        bio = lambda labels: labels if cfg.bio_constraints else None
        self.identity_head = NERHead(lstm_out, cfg.identity_labels, bio(IDENTITY_LABELS))
        self.location_head = NERHead(lstm_out, cfg.location_labels, bio(LOCATION_LABELS))
        self.temporal_head = NERHead(lstm_out, cfg.temporal_labels, bio(TEMPORAL_LABELS))
        self.domain_head   = NERHead(lstm_out, cfg.domain_labels,   bio(DOMAIN_LABELS))

    # ------------------------------------------------------------------
    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        max_len = input_ids.size(1)

        with torch.no_grad():
            h = self.gemma(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = h.float()   # Gemma outputs BFloat16; BiLSTM + CRF require Float32

        lengths = attention_mask.sum(1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        h, _    = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)

        return self.drop(self.norm(h))          # [B, T, 512]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        identity_tags:  Optional[torch.Tensor] = None,
        location_tags:  Optional[torch.Tensor] = None,
        temporal_tags:  Optional[torch.Tensor] = None,
        domain_tags:    Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        h    = self._encode(input_ids, attention_mask)
        mask = attention_mask.bool()

        losses = []
        if identity_tags is not None:
            losses.append(self.identity_head.loss(h, identity_tags, mask))
        if location_tags is not None:
            losses.append(self.location_head.loss(h, location_tags, mask))
        if temporal_tags is not None:
            losses.append(self.temporal_head.loss(h, temporal_tags, mask))
        if domain_tags is not None:
            losses.append(self.domain_head.loss(h, domain_tags, mask))

        return (sum(losses) if losses else None), h

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List, List, List, List, torch.Tensor]:
        """Returns (identity_preds, location_preds, temporal_preds, domain_preds, h)."""
        h    = self._encode(input_ids, attention_mask)
        mask = attention_mask.bool()
        return (
            self.identity_head.decode(h, mask),
            self.location_head.decode(h, mask),
            self.temporal_head.decode(h, mask),
            self.domain_head.decode(h, mask),
            h,
        )

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
