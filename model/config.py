"""
Model configuration for MicroLM — Stock Exchange Language Model (TensorFlow / Keras).
Supports serialization to/from JSON so checkpoints load with their exact architecture.
"""
import json
import os
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Hyperparameters for the MicroLM transformer model."""

    # --- Model Architecture ---
    vocab_size: int = 5000          # Dynamically set from tokenizer
    embed_dim: int = 256           # Embedding dimension
    num_heads: int = 4             # Number of attention heads
    num_layers: int = 4            # Number of transformer blocks
    ff_dim: int = 1024             # Feed-forward hidden dimension
    seq_length: int = 128          # Context window length (128 for efficient Mac training/serving)
    dropout: float = 0.1           # Dropout rate (regularization)

    # --- Training ---
    batch_size: int = 64           # Batch size
    epochs: int = 4                # Target epochs (meets >= 3 threshold within 60 min)
    learning_rate: float = 3.5e-4  # Peak LR (warmup + cosine-decayed)
    warmup_steps: int = 100        # Warmup steps
    min_lr: float = 1e-5           # Min LR at end of cosine decay
    weight_decay: float = 0.01     # AdamW weight decay
    val_split: float = 0.1         # 10% held out for non-overlapping validation
    patience: int = 4              # Early stopping patience (epochs)
    max_minutes: float = 60.0      # Cap training time in minutes

    # --- Generation & Sampler Defaults (Task C) ---
    default_temperature: float = 0.65
    default_top_k: int = 30
    default_top_p: float = 0.85
    repetition_penalty: float = 1.08  # Balanced penalty (windowed)
    no_repeat_ngram_size: int = 3     # Block 3-gram repetitions
    repetition_window: int = 128      # Penalize last 128 tokens
    default_max_length: int = 120

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    history_path: str = "training_history.json"

    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Instantiate config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Config] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.from_dict(data)
        print(f"[Config] Loaded from {path}")
        return cfg

    def summary(self) -> str:
        """Return a formatted summary of the config."""
        lines = [
            "+==========================================+",
            "|     MicroLM - Stock Exchange Model       |",
            "+==========================================+",
            f"|  Framework       : TensorFlow / Keras    |",
            f"|  Embedding dim   : {self.embed_dim:<21}|",
            f"|  Attention heads : {self.num_heads:<21}|",
            f"|  Transformer layers: {self.num_layers:<19}|",
            f"|  FF hidden dim   : {self.ff_dim:<21}|",
            f"|  Sequence length : {self.seq_length:<21}|",
            f"|  Dropout         : {self.dropout:<21}|",
            f"|  Vocab size      : {self.vocab_size:<21}|",
            "+==========================================+",
            f"|  Batch size      : {self.batch_size:<21}|",
            f"|  Epochs          : {self.epochs:<21}|",
            f"|  Learning rate   : {self.learning_rate:<21}|",
            f"|  Min LR          : {self.min_lr:<21}|",
            f"|  Weight decay    : {self.weight_decay:<21}|",
            f"|  Warmup steps    : {self.warmup_steps:<21}|",
            f"|  Val split       : {self.val_split:<21}|",
            f"|  Early stop patience: {self.patience:<18}|",
            "+==========================================+",
            f"|  Temperature     : {self.default_temperature:<21}|",
            f"|  Top-K / Top-P   : {self.default_top_k} / {self.default_top_p:<15}|",
            f"|  Rep. penalty    : {self.repetition_penalty:<21}|",
            f"|  Rep. window     : {self.repetition_window:<21}|",
            "+==========================================+",
        ]
        return "\n".join(lines)
