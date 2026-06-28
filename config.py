from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Gemma 3 270M backbone — gated model, run `huggingface-cli login` first.
    # google/gemma-3-270m: 12 transformer layers · hidden 1024 · 256K vocab · ~268M params
    # For on-device deployment, export the trained BiLSTM+CRF heads separately
    # and run quantized Gemma (INT8/INT4) as the frozen feature extractor.
    gemma_model_id: str = "google/gemma-3-270m"
    max_seq_len: int = 128

    # BiLSTM: hidden per direction; output = bilstm_hidden * 2 = 512
    bilstm_hidden: int = 256
    bilstm_layers: int = 2
    bilstm_dropout: float = 0.2

    # Post-BiLSTM regularisation
    post_dropout: float = 0.15

    # Head output sizes — must match label counts in src/labels.py
    identity_labels: int = 7
    location_labels: int = 9   # LOC GPE FAC ADDR (B+I each) + O
    temporal_labels: int = 9
    domain_labels: int = 7

    # Enforce BIO impossibility constraints in the CRF transition matrix
    bio_constraints: bool = True


@dataclass
class TrainConfig:
    learning_rate: float = 2e-3
    batch_size: int = 16
    epochs: int = 10
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 500       # steps between validation passes
    save_dir: str = "checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
