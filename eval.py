"""Evaluation script — span-level F1 per head and per entity type.

Usage:
  python eval.py                       # uses checkpoints/best.pt
  python eval.py checkpoints/best.pt  # explicit checkpoint path
"""

import sys
import torch
from transformers import AutoTokenizer

from config import Config
from src.model import GemmaBackboneNER
from src.data import read_conll, make_loader
from src.labels import (
    IDENTITY_LABELS, LOCATION_LABELS, TEMPORAL_LABELS, DOMAIN_LABELS,
)

HEAD_LABELS = {
    "Identity": IDENTITY_LABELS,
    "Location": LOCATION_LABELS,
    "Temporal": TEMPORAL_LABELS,
    "Domain":   DOMAIN_LABELS,
}
HEAD_TAG_KEYS = {
    "Identity": "identity_tags",
    "Location": "location_tags",
    "Temporal": "temporal_tags",
    "Domain":   "domain_tags",
}


# ── Span utilities ────────────────────────────────────────────────────────────

def extract_spans(tag_ids, id_to_label, valid_len):
    """BIO tag sequence → list of (start, end, type) spans."""
    spans = []
    cur_type, cur_start = None, None
    for i in range(valid_len):
        label = id_to_label[tag_ids[i]]
        if label == "O":
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
                cur_type, cur_start = None, None
        elif label.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
            cur_type, cur_start = label[2:], i
        elif label.startswith("I-"):
            etype = label[2:]
            if cur_type != etype:
                if cur_type is not None:
                    spans.append((cur_start, i, cur_type))
                cur_type, cur_start = etype, i
    if cur_type is not None:
        spans.append((cur_start, valid_len, cur_type))
    return spans


def compute_f1(pred_all, gold_all):
    tp = fp = fn = 0
    for pred, gold in zip(pred_all, gold_all):
        p, g = set(pred), set(gold)
        tp += len(p & g);  fp += len(p - g);  fn += len(g - p)
    prec  = tp / (tp + fp) if (tp + fp) else 0.0
    rec   = tp / (tp + fn) if (tp + fn) else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, tp + fp, tp + fn


def per_type_f1(pred_all, gold_all):
    types = {s[2] for spans in gold_all for s in spans}
    out = {}
    for etype in sorted(types):
        tp = fp = fn = 0
        for pred, gold in zip(pred_all, gold_all):
            p = {s for s in pred if s[2] == etype}
            g = {s for s in gold if s[2] == etype}
            tp += len(p & g);  fp += len(p - g);  fn += len(g - p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[etype] = (prec, rec, f1, tp + fn)   # gold count
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ckpt   = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint : {ckpt}")
    print(f"Device     : {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.gemma_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_samples = read_conll("data/test.txt")
    test_loader  = make_loader(test_samples, tokenizer, cfg.model.max_seq_len,
                               batch_size=cfg.train.batch_size, shuffle=False)
    print(f"Test samples : {len(test_samples)}\n")

    model = GemmaBackboneNER(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Build id → label maps
    id_to_label = {
        name: {i: l for i, l in enumerate(labels)}
        for name, labels in HEAD_LABELS.items()
    }

    # Collect predictions and gold spans per head
    pred_spans = {name: [] for name in HEAD_LABELS}
    gold_spans = {name: [] for name in HEAD_LABELS}

    head_decoders = {
        "Identity": model.identity_head,
        "Location": model.location_head,
        "Temporal": model.temporal_head,
        "Domain":   model.domain_head,
    }

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            mask      = batch_dev["attention_mask"]
            valid_lens = mask.sum(dim=1).tolist()

            ident_pred, loc_pred, temp_pred, dom_pred, h = model.predict(
                batch_dev["input_ids"], mask
            )
            preds = {
                "Identity": ident_pred,
                "Location": loc_pred,
                "Temporal": temp_pred,
                "Domain":   dom_pred,
            }

            for name in HEAD_LABELS:
                gold_tags = batch_dev[HEAD_TAG_KEYS[name]].cpu().tolist()
                for i, (pred_seq, gold_seq) in enumerate(zip(preds[name], gold_tags)):
                    vl = int(valid_lens[i])
                    pred_spans[name].append(extract_spans(pred_seq,  id_to_label[name], vl))
                    gold_spans[name].append(extract_spans(gold_seq,  id_to_label[name], vl))

    # ── Print results ──────────────────────────────────────────────────────────
    SEP = "-" * 62
    all_pred, all_gold = [], []

    for name in HEAD_LABELS:
        p, r, f1, tp, n_pred, n_gold = compute_f1(pred_spans[name], gold_spans[name])
        type_rows = per_type_f1(pred_spans[name], gold_spans[name])

        print(SEP)
        print(f"  {name} Head")
        print(SEP)
        print(f"  {'Type':<12}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Gold':>6}")
        print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
        for etype, (ep, er, ef1, ecnt) in type_rows.items():
            print(f"  {etype:<12}  {ep:7.4f}  {er:7.4f}  {ef1:7.4f}  {ecnt:6}")
        print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
        print(f"  {'ALL':<12}  {p:7.4f}  {r:7.4f}  {f1:7.4f}  {n_gold:6}")
        print()

        all_pred.extend(pred_spans[name])
        all_gold.extend(gold_spans[name])

    # Overall across all heads
    op, or_, of1, _, _, _ = compute_f1(all_pred, all_gold)
    print(SEP)
    print(f"  OVERALL  Prec={op:.4f}  Rec={or_:.4f}  F1={of1:.4f}")
    print(SEP)


if __name__ == "__main__":
    main()
