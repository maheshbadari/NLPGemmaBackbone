Input text (max 128 tokens)
    ↓
Gemma SentencePiece tokenizer  (256K vocab)
    ↓
Gemma 3 270M backbone  ← FROZEN, no gradients
    12 transformer layers · hidden dim 1024 · causal attention
    ~268M total params · 256K vocab
    ↓
Last hidden layer  →  h ∈ ℝ^(seq_len × 1152)
    ↓
BiLSTM  ← TRAINABLE
    hidden=256 each direction · concat→512 · 2 layers · dropout=0.2
    Adds bidirectionality that causal Gemma lacks
    ~2.4M parameters
    ↓
LayerNorm + Dropout(0.15)
    ↓
┌─────────────────┬──────────────────┬──────────────────┬──────────────┐
│  Identity head  │  Location head   │  Temporal head   │  Domain head │
│  Linear(512→7)  │  Linear(512→7)   │  Linear(512→9)   │ Linear(512→7)│
│  + CRF          │  + CRF           │  + CRF           │ + CRF        │
└─────────────────┴──────────────────┴──────────────────┴──────────────┘
    ↓ (all 4 heads run in parallel on same h)
Span merger + conflict resolution  (confidence tiebreak)
    ↓
Structured entity output  (span + label + confidence + nested sub-entities)