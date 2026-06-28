"""Training entry point.

Usage:
  python train.py

Data: place CoNLL-2003 split files at data/train.txt, data/valid.txt.
Requires `huggingface-cli login` to download the gated Gemma backbone.
"""

import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from src.model import GemmaBackboneNER
from src.data import read_conll, make_loader


def make_scheduler(optimizer, warmup_steps, total_steps):
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    decay  = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(**batch)
        if loss is not None:
            total += loss.item(); n += 1
    model.train()
    return total / n if n else float("inf")


def main():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer ({cfg.model.gemma_model_id}) …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.gemma_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples = read_conll("data/train.txt")
    val_samples   = read_conll("data/valid.txt")
    print(f"Samples — train: {len(train_samples)}  val: {len(val_samples)}")

    train_loader = make_loader(train_samples, tokenizer, cfg.model.max_seq_len, cfg.train.batch_size, shuffle=True)
    val_loader   = make_loader(val_samples,   tokenizer, cfg.model.max_seq_len, cfg.train.batch_size, shuffle=False)

    print("Building model …")
    model = GemmaBackboneNER(cfg.model).to(device)
    trainable = model.trainable_params()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    total_steps = len(train_loader) * cfg.train.epochs
    optimizer   = AdamW(trainable, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler   = make_scheduler(optimizer, cfg.train.warmup_steps, total_steps)

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_val, step = float("inf"), 0

    for epoch in range(cfg.train.epochs):
        model.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")

        for batch in bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(**batch)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % cfg.train.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nstep {step}  val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"{cfg.train.save_dir}/best.pt")
                    print(f"  ↑ saved best  (val_loss={val_loss:.4f})")

        torch.save(model.state_dict(), f"{cfg.train.save_dir}/epoch_{epoch + 1}.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
