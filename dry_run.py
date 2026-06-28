"""Dry run — validates the full pipeline without running a real training loop.

Checks (in order):
  1. Imports
  2. Config instantiation
  3. Data loading  (first 8 samples from data/train.txt)
  4. Tokenizer load
  5. DataLoader + batch shapes
  6. Model instantiation + param count
  7. Forward pass (loss + hidden shape)
  8. Backward pass (gradients only on trainable params)
  9. Gradient check (Gemma params must stay frozen)

Usage:
  python dry_run.py
"""

import sys
import traceback

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SEP  = "-" * 60


def check(label):
    print(f"\n{SEP}\n{label}")


# ── 1. Imports ────────────────────────────────────────────────
check("1. Imports")
try:
    import torch
    from transformers import AutoTokenizer
    from config import Config
    from src.model import GemmaBackboneNER
    from src.data import read_conll, make_loader
    print(f"{PASS}  torch={torch.__version__}  cuda={'yes' if torch.cuda.is_available() else 'no (CPU)'}")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 2. Config ────────────────────────────────────────────────
check("2. Config")
try:
    cfg = Config()
    print(f"{PASS}  model_id={cfg.model.gemma_model_id}  max_seq_len={cfg.model.max_seq_len}")
    print(f"         bilstm_hidden={cfg.model.bilstm_hidden}  layers={cfg.model.bilstm_layers}")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 3. Data loading ───────────────────────────────────────────
check("3. Data loading  (data/train.txt)")
try:
    samples = read_conll("data/train.txt")
    mini    = samples[:8]
    print(f"{PASS}  total samples in train: {len(samples)}  using first: {len(mini)}")
    s = mini[0]
    print(f"         sample[0]: {len(s.words)} words  |  words[:5]: {s.words[:5]}")
    print(f"         identity[:5]: {s.identity[:5]}")
    print(f"         location[:5]: {s.location[:5]}")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 4. Tokenizer ──────────────────────────────────────────────
check(f"4. Tokenizer  ({cfg.model.gemma_model_id})")
try:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.gemma_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{PASS}  vocab_size={tokenizer.vocab_size}  pad='{tokenizer.pad_token}'")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 5. DataLoader + batch shapes ──────────────────────────────
check("5. DataLoader + batch shapes")
try:
    loader = make_loader(mini, tokenizer, cfg.model.max_seq_len, batch_size=4, shuffle=False)
    batch  = next(iter(loader))
    for k, v in batch.items():
        print(f"         {k:20s} {tuple(v.shape)}  dtype={v.dtype}")
    print(PASS)
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 6. Model instantiation ────────────────────────────────────
check(f"6. Model instantiation  ({cfg.model.gemma_model_id})")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GemmaBackboneNER(cfg.model).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"{PASS}")
    print(f"         total params   : {total:>12,}")
    print(f"         frozen  params : {frozen:>12,}  (Gemma backbone)")
    print(f"         trainable      : {trainable:>12,}  (BiLSTM + 4 CRF heads)")
    print(f"         device         : {device}")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 7. Forward pass ───────────────────────────────────────────
check("7. Forward pass (loss + hidden state shape)")
try:
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    loss, h   = model(**batch_dev)

    print(f"{PASS}")
    print(f"         loss  : {loss.item():.4f}")
    print(f"         h     : {tuple(h.shape)}  (batch, seq_len, bilstm_out)")
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 8. Backward pass ──────────────────────────────────────────
check("8. Backward pass")
try:
    loss.backward()
    print(PASS)
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── 9. Gradient check ─────────────────────────────────────────
check("9. Gradient check (Gemma must stay frozen)")
try:
    gemma_grads    = [p for p in model.gemma.parameters() if p.grad is not None]
    bilstm_grads   = [p for p in model.bilstm.parameters() if p.grad is not None]

    gemma_ok   = len(gemma_grads) == 0
    bilstm_ok  = len(bilstm_grads) > 0

    print(f"         Gemma params with grad  : {len(gemma_grads)}   {'OK (frozen)' if gemma_ok else 'ERROR — should be 0'}")
    print(f"         BiLSTM params with grad : {len(bilstm_grads)}  {'OK' if bilstm_ok else 'ERROR — should be > 0'}")

    if gemma_ok and bilstm_ok:
        print(PASS)
    else:
        print(FAIL); sys.exit(1)
except Exception:
    print(FAIL); traceback.print_exc(); sys.exit(1)


# ── Summary ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("All checks passed. Pipeline is ready for training.")
print(f"  Run:  python train.py")
print(SEP)
