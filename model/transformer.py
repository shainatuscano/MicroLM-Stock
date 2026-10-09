"""
MicroLM — A micro GPT-style decoder-only transformer for stock exchange text.
Built with TensorFlow / Keras (tf.keras).

Architecture:
- Shared embedding (weight tying between token embedding and output projection)
- Learned positional embeddings
- Pre-LayerNorm Transformer decoder blocks (causal multi-head attention + GELU feed-forward)
- Final layer normalization
- Logits projection via tied embedding weights
"""
import math
import numpy as np
import tensorflow as tf
from .config import ModelConfig


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class SharedEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer whose weight matrix is shared between the input token
    lookup and the final linear projection to vocabulary logits (weight tying).
    """

    def __init__(self, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.vocab_size, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mode="embedding"):
        """
        mode == 'embedding': inputs of shape (batch, seq_len) -> (batch, seq_len, embed_dim)
        mode == 'linear':    inputs of shape (batch, seq_len, embed_dim) -> (batch, seq_len, vocab_size)
        """
        if mode == "embedding":
            return tf.nn.embedding_lookup(self.embeddings, inputs)
        elif mode == "linear":
            return tf.matmul(inputs, self.embeddings, transpose_b=True)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'embedding' or 'linear'.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class PositionalEmbedding(tf.keras.layers.Layer):
    """Learned positional embedding layer."""

    def __init__(self, seq_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = int(seq_length)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        self.pos_embeddings = self.add_weight(
            name="pos_embeddings",
            shape=(self.seq_length, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len)
        seq_len = tf.shape(inputs)[1]
        return self.pos_embeddings[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_length": self.seq_length,
            "embed_dim": self.embed_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class CausalSelfAttention(tf.keras.layers.Layer):
    """Multi-head causal (masked) self-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.head_dim = self.embed_dim // self.num_heads

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            dropout=self.dropout_rate,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            name="mha",
        )
        self.proj_dropout = tf.keras.layers.Dropout(self.dropout_rate, name="proj_dropout")

    def call(self, x, training=False):
        attn_out = self.mha(x, x, use_causal_mask=True, training=training)
        return self.proj_dropout(attn_out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class FeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.ff_dim = int(ff_dim)
        self.dropout_rate = float(dropout)

        self.dense1 = tf.keras.layers.Dense(
            self.ff_dim,
            activation="gelu",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            name="dense1",
        )
        self.dense2 = tf.keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            name="dense2",
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="ffn_dropout")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer decoder block: Attention + FFN with pre-LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout_rate = float(dropout)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln1")
        self.attn = CausalSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout_rate, name="attn")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln2")
        self.ffn = FeedForward(embed_dim=self.embed_dim, ff_dim=self.ff_dim, dropout=self.dropout_rate, name="ffn")

    def call(self, x, training=False):
        # Pre-LayerNorm residual connections
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.ffn(self.ln2(x), training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MicroLM")
class MicroLM(tf.keras.Model):
    """
    Micro Language Model — GPT-style decoder-only transformer.
    Built with tf.keras and weight-tied token embeddings.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.token_embedding = SharedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            name="token_embedding",
        )
        self.pos_embedding = PositionalEmbedding(
            seq_length=config.seq_length,
            embed_dim=config.embed_dim,
            name="pos_embedding",
        )
        self.embed_dropout = tf.keras.layers.Dropout(config.dropout, name="embed_dropout")

        self.blocks = [
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                name=f"block_{i}",
            )
            for i in range(config.num_layers)
        ]
        self.ln_final = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_final")

    def call(self, inputs, training=False):
        """
        Forward pass.
        Args:
            inputs: Tensor of integer token IDs with shape (batch_size, seq_len)
            training: boolean
        Returns:
            logits with shape (batch_size, seq_len, vocab_size)
        """
        tok_emb = self.token_embedding(inputs, mode="embedding")
        pos_emb = self.pos_embedding(inputs)
        x = self.embed_dropout(tok_emb + pos_emb, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_final(x)
        logits = self.token_embedding(x, mode="linear")
        return logits

    def get_num_params(self) -> int:
        """Count total trainable parameters (weight tying accounted for)."""
        if not self.built:
            dummy = tf.zeros((1, self.config.seq_length), dtype=tf.int32)
            self(dummy, training=False)
        return int(sum(int(tf.size(v).numpy()) for v in self.trainable_variables))

    def summary_str(self) -> str:
        """Return formatted architecture summary string."""
        num_params = self.get_num_params()
        lines = [
            "=" * 50,
            "  MicroLM - Stock Exchange Language Model",
            "=" * 50,
            f"  Architecture: GPT-style Decoder-Only Transformer",
            f"  Framework:    TensorFlow 2.x / Keras",
            f"  Layers:       {self.config.num_layers}",
            f"  Heads:        {self.config.num_heads}",
            f"  Embed dim:    {self.config.embed_dim}",
            f"  FF dim:       {self.config.ff_dim}",
            f"  Seq length:   {self.config.seq_length}",
            f"  Vocab size:   {self.config.vocab_size}",
            f"  Dropout:      {self.config.dropout}",
            f"  Parameters:   {num_params:,}",
            f"  Size (est):   ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)",
            "=" * 50,
        ]
        return "\n".join(lines)


def build_micro_lm(config: ModelConfig) -> MicroLM:
    """Helper to instantiate and build a MicroLM model."""
    model = MicroLM(config)
    dummy_input = tf.zeros((1, config.seq_length), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    return model


if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=5000, embed_dim=256, num_heads=4, num_layers=4, ff_dim=1024)
    m = build_micro_lm(cfg)
    print(m.summary_str())
