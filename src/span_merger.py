"""Extract entity spans from BIO predictions and resolve cross-head conflicts.

After all four heads produce independent BIO sequences, this module:
  1. Converts each sequence to a list of (start, end, label, confidence) spans.
  2. Merges all spans from all heads.
  3. Resolves token-level conflicts greedily by confidence (highest wins).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch

from .labels import IDENTITY_LABELS, LOCATION_LABELS, TEMPORAL_LABELS, DOMAIN_LABELS

_HEAD_META: List[Tuple[str, List[str]]] = [
    ("identity", IDENTITY_LABELS),
    ("location", LOCATION_LABELS),
    ("temporal", TEMPORAL_LABELS),
    ("domain",   DOMAIN_LABELS),
]


@dataclass
class EntitySpan:
    start:        int               # inclusive token index
    end:          int               # exclusive token index
    label:        str               # e.g. "identity:PER"
    category:     str               # identity | location | temporal | domain
    confidence:   float
    sub_entities: List["EntitySpan"] = field(default_factory=list)


def _extract(
    tag_ids:    List[int],
    labels:     List[str],
    probs:      torch.Tensor,       # [T, C] softmax probs
    category:   str,
) -> List[EntitySpan]:
    spans: List[EntitySpan] = []
    i = 0
    while i < len(tag_ids):
        name = labels[tag_ids[i]]
        if name.startswith("B-"):
            entity_type = name[2:]
            start, conf = i, probs[i, tag_ids[i]].item()
            j = i + 1
            while j < len(tag_ids) and labels[tag_ids[j]] == f"I-{entity_type}":
                conf += probs[j, tag_ids[j]].item()
                j += 1
            spans.append(EntitySpan(start, j, f"{category}:{entity_type}", category, conf / (j - start)))
            i = j
        else:
            i += 1
    return spans


def _overlaps(a: EntitySpan, b: EntitySpan) -> bool:
    return a.start < b.end and b.start < a.end


def _resolve(spans: List[EntitySpan]) -> List[EntitySpan]:
    """Greedy conflict resolution: process spans highest-confidence-first."""
    kept: List[EntitySpan] = []
    for span in sorted(spans, key=lambda s: -s.confidence):
        if not any(_overlaps(span, k) for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def decode_batch(
    predictions:    tuple,              # (identity, location, temporal, domain) lists of per-seq lists
    representations: torch.Tensor,     # [B, T, 512]
    heads,                              # list of NERHead in same order as _HEAD_META
    attention_mask: torch.Tensor,       # [B, T]
) -> List[List[EntitySpan]]:
    """Produce merged entity spans for every example in the batch."""
    B         = len(predictions[0])
    all_probs = [h.emissions(representations) for h in heads]  # [B, T, C] each

    results = []
    for b in range(B):
        seq_len   = int(attention_mask[b].sum().item())
        all_spans: List[EntitySpan] = []

        for idx, (category, label_names) in enumerate(_HEAD_META):
            pred  = predictions[idx][b][:seq_len]
            probs = all_probs[idx][b, :seq_len]
            all_spans.extend(_extract(pred, label_names, probs, category))

        results.append(_resolve(all_spans))

    return results
