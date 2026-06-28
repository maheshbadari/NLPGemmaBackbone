"""Inference entry point.

Usage:
  python predict.py [path/to/checkpoint.pt]

If no checkpoint is given, loads checkpoints/best.pt.
"""

import re
import sys
import torch
from transformers import AutoTokenizer

from config import Config
from src.model import GemmaBackboneNER
from src.span_merger import decode_batch


def load_model(ckpt: str, cfg: Config, device: torch.device) -> GemmaBackboneNER:
    model = GemmaBackboneNER(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def word_tokenize(text: str):
    """Split text into words + punctuation tokens, preserving original strings."""
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)


def run(text: str, model: GemmaBackboneNER, tokenizer, device, max_len: int):
    words = word_tokenize(text)
    enc   = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    ident, loc, temp, dom, h = model.predict(input_ids, attention_mask)

    heads   = [model.identity_head, model.location_head, model.temporal_head, model.domain_head]
    with torch.no_grad():
        entities = decode_batch((ident, loc, temp, dom), h, heads, attention_mask)

    # map subword token spans → word spans using word_ids()
    word_ids  = enc.word_ids()
    tok_to_word = {i: wid for i, wid in enumerate(word_ids) if wid is not None}

    results = []
    for e in entities[0]:
        w_start = tok_to_word.get(e.start)
        w_end   = tok_to_word.get(e.end - 1)
        if w_start is None or w_end is None:
            continue
        surface = " ".join(words[w_start: w_end + 1])
        results.append((surface, w_start, w_end + 1, e.label, e.confidence))

    return results, words


def main():
    cfg   = Config()
    ckpt  = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.gemma_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(ckpt, cfg, device)
    print(f"Loaded {ckpt}  |  device={device}\nType text and press Enter (Ctrl-C to quit).\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        results, words = run(text, model, tokenizer, device, cfg.model.max_seq_len)
        if not results:
            print("  (no entities)")
        for surface, w_start, w_end, label, conf in results:
            print(f"  [{w_start}:{w_end}]  '{surface}'  →  {label:<28}  conf={conf:.3f}")


if __name__ == "__main__":
    main()
