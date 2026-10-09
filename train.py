"""
Training pipeline for MicroLM — Stock Exchange Language Model (TensorFlow / Keras).
Trains a byte-level BPE decoder-only transformer on financial text data.

Features:
  - Positional train/val split (zero token overlap)
  - Non-overlapping validation windows, stride seq_length//2 for training
  - tf.data pipeline with shuffling and prefetching
  - Custom warmup + cosine-decay LearningRateSchedule
  - AdamW with weight decay and gradient clipping (clipnorm=1.0)
  - TimeLimitCallback (--max-minutes)
  - SampleGenerationCallback printing 3 generations every few epochs
  - Early stopping and ModelCheckpoint on val_loss
  - Evaluation reporting loss and perplexity
  - training_history.json logging
  - Post-training sanity check on fixed prompts
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import build_micro_lm, MicroLM
from data.corpus import get_corpus
from generate import generate_text


# ---------------------------------------------------------------------------
# Learning Rate Schedule (Warmup + Cosine Decay)
# ---------------------------------------------------------------------------
class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, initial_learning_rate: float, min_lr: float, warmup_steps: int, total_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup from 0 to initial_learning_rate
        warmup_lr = self.initial_learning_rate * (step / tf.maximum(1.0, self.warmup_steps))

        # Cosine decay from initial_learning_rate to min_lr
        decay_steps = tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value((step - self.warmup_steps) / decay_steps, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(float(math.pi) * progress))
        decay_lr = self.min_lr + (self.initial_learning_rate - self.min_lr) * cosine_decay

        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "initial_learning_rate": float(self.initial_learning_rate),
            "min_lr": float(self.min_lr),
            "warmup_steps": float(self.warmup_steps),
            "total_steps": float(self.total_steps),
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class TimeLimitCallback(tf.keras.callbacks.Callback):
    """Stops training if elapsed time exceeds max_minutes."""

    def __init__(self, max_minutes: float):
        super().__init__()
        self.max_minutes = max_minutes
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.max_minutes and self.max_minutes > 0:
            elapsed_min = (time.time() - self.start_time) / 60.0
            if elapsed_min >= self.max_minutes:
                print(f"\n[TimeLimit] ⏹ Reached time limit of {self.max_minutes:.1f} minutes ({elapsed_min:.1f} min elapsed). Stopping training.")
                self.model.stop_training = True


class SampleGenerationCallback(tf.keras.callbacks.Callback):
    """Generates sample completions every few epochs to monitor text quality."""

    def __init__(self, tokenizer: BPETokenizer, sample_prompts: List[str], every_n_epochs: int = 2):
        super().__init__()
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num % self.every_n_epochs == 0 or epoch_num == 1:
            print(f"\n{'-' * 60}")
            print(f"  Sample Generations (Epoch {epoch_num})")
            print(f"{'-' * 60}")
            for prompt in self.sample_prompts:
                try:
                    text = generate_text(
                        self.model,
                        self.tokenizer,
                        prompt,
                        max_length=60,
                        temperature=0.8,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.05,
                        stop_at_eos=True,
                    )
                    preview = text[:180] + ("..." if len(text) > 180 else "")
                    print(f'  Prompt: "{prompt}"')
                    print(f"  Output: {preview}\n")
                except Exception as e:
                    print(f"  Generation error: {e}")
            print(f"{'-' * 60}\n")


class MetricsTrackerCallback(tf.keras.callbacks.Callback):
    """Tracks training and validation metrics including perplexity per epoch."""

    def __init__(self):
        super().__init__()
        self.history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_perplexity": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        acc = float(logs.get("accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0)) if "val_loss" in logs else None
        val_acc = float(logs.get("val_accuracy", 0.0)) if "val_accuracy" in logs else None

        self.history["loss"].append(loss)
        self.history["accuracy"].append(acc)

        if val_loss is not None:
            ppl = float(math.exp(min(val_loss, 20.0)))
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_perplexity"].append(ppl)
            print(f"[Epoch {epoch+1:2d}] train_loss={loss:.4f} acc={acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={ppl:.2f}")
        else:
            print(f"[Epoch {epoch+1:2d}] train_loss={loss:.4f} acc={acc:.4f}")


class LRMonitorCallback(tf.keras.callbacks.Callback):
    """Monitors and prints the actual learning rate at steps 0, 100, 1000 and at the end of epoch 1."""

    def __init__(self, steps_per_epoch: int):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.global_step = 0

    def _get_current_lr(self, step: int) -> float:
        try:
            sched = self.model.optimizer.learning_rate
            if callable(sched):
                return float(sched(step))
            return float(sched)
        except Exception:
            return float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def on_train_batch_begin(self, batch, logs=None):
        if self.global_step in (0, 100, 1000):
            lr = self._get_current_lr(self.global_step)
            print(f"\n[LR Monitor] Step {self.global_step:5d}: learning_rate = {lr:.6e}")

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:  # End of epoch 1
            step = self.steps_per_epoch
            lr = self._get_current_lr(step)
            print(f"\n[LR Monitor] End of Epoch 1 (step {step:5d}): learning_rate = {lr:.6e}\n")


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def prepare_datasets(
    encoded_tokens: List[int],
    seq_length: int,
    batch_size: int,
    val_split: float = 0.1,
    stride: int = None,
    max_train_samples: int = None,
    max_val_samples: int = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Split token stream into train and validation portions by position:
    - The last val_split portion of tokens is held out for validation (no leakage!).
    - Non-overlapping windows for validation (stride = seq_length).
    - Configurable stride for training (defaults to seq_length for full token coverage without overlap).
    """
    total_tokens = len(encoded_tokens)
    val_tokens_count = int(total_tokens * val_split) if val_split > 0 else 0
    train_tokens = encoded_tokens[:-val_tokens_count] if val_tokens_count > 0 else encoded_tokens
    val_tokens = encoded_tokens[-val_tokens_count:] if val_tokens_count > 0 else []

    # Training windows (50% overlapping windows for rich transition learning)
    if stride is None or stride <= 0:
        stride = max(1, seq_length // 2)
    stride = max(1, stride)
    train_x, train_y = [], []
    for i in range(0, len(train_tokens) - seq_length, stride):
        train_x.append(train_tokens[i : i + seq_length])
        train_y.append(train_tokens[i + 1 : i + seq_length + 1])
        if max_train_samples and len(train_x) >= max_train_samples:
            break

    train_x = np.array(train_x, dtype=np.int32)
    train_y = np.array(train_y, dtype=np.int32)

    # Validation windows (non-overlapping: stride = seq_length)
    val_x, val_y = [], []
    for j in range(0, len(val_tokens) - seq_length, seq_length):
        val_x.append(val_tokens[j : j + seq_length])
        val_y.append(val_tokens[j + 1 : j + seq_length + 1])
        if max_val_samples and len(val_x) >= max_val_samples:
            break

    val_x = np.array(val_x, dtype=np.int32) if len(val_x) else np.empty((0, seq_length), dtype=np.int32)
    val_y = np.array(val_y, dtype=np.int32) if len(val_y) else np.empty((0, seq_length), dtype=np.int32)

    num_train = len(train_x)
    num_val = len(val_x)
    print(f"[Dataset] Sequences created: Train={num_train:,} (stride={stride}) | Val={num_val:,} (stride={seq_length}, non-overlapping)")

    # Build tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(buffer_size=min(num_train, 10000), seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = None
    if num_val > 0:
        val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_ds = val_ds.batch(batch_size, drop_remainder=False)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_train, num_val


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------
def run_post_training_sanity_check(model: MicroLM, tokenizer: BPETokenizer, config: ModelConfig) -> bool:
    """Check that generation produces readable financial text without degenerate repetition."""
    test_prompts = [
        "The stock market",
        "The Federal Reserve announced",
        "Investors are closely watching",
        "Earnings reports indicate that",
    ]

    print(f"\n{'=' * 60}")
    print("  Post-Training Sanity Check")
    print(f"{'=' * 60}")

    all_passed = True
    for prompt in test_prompts:
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=100,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=config.repetition_penalty,
            repetition_window=config.repetition_window,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            stop_at_eos=True,
        )

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(text)
        new_ids = full_ids[len(prompt_ids):]

        if new_ids:
            counter = Counter(new_ids)
            most_common_count = counter.most_common(1)[0][1]
            rep_ratio = most_common_count / len(new_ids)
        else:
            rep_ratio = 1.0

        is_degenerate = rep_ratio >= 0.45 or len(new_ids) < 5
        status = "✅ PASS" if not is_degenerate else "❌ FAIL"
        if is_degenerate:
            all_passed = False

        preview = text[:280] + ("..." if len(text) > 280 else "")
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: {preview}")
        print(f"  Repetition ratio: {rep_ratio:.2%} — {status}")

    print(f"\n{'=' * 60}")
    overall = "✅ ALL CHECKS PASSED" if all_passed else "⚠️ SOME CHECKS NEED ATTENTION"
    print(f"  Sanity check result: {overall}")
    print(f"{'=' * 60}\n")
    return all_passed


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train(config: ModelConfig = None, stride: int = None, max_train_samples: int = None, max_val_samples: int = None):
    """Run full training pipeline in TensorFlow / Keras."""
    if config is None:
        config = ModelConfig()

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Detect devices
    gpus = tf.config.list_physical_devices("GPU")
    print("\n" + "=" * 60)
    print("  MicroLM - Stock Exchange Language Model")
    print("  Training Pipeline (TensorFlow 2.x / Keras)")
    print(f"  Hardware: {'GPU (' + str(len(gpus)) + ' available)' if gpus else 'CPU / Apple Silicon'}")
    print("=" * 60 + "\n")

    # 1. Load corpus
    print("Step 1: Loading clean corpus...")
    corpus = get_corpus(use_cache=True)
    print(f"[Corpus] Loaded {len(corpus):,} characters")

    # 2. Build or load tokenizer
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    tokenizer = BPETokenizer()
    if os.path.exists(tokenizer_path):
        print(f"Step 2: Loading tokenizer from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
    else:
        print("Step 2: Fitting byte-level BPE tokenizer...")
        tokenizer.fit(corpus, vocab_size=config.vocab_size)
        tokenizer.save(tokenizer_path)

    config.vocab_size = tokenizer.vocab_size
    tokenizer.log_vocab_utilization(corpus[:2_000_000])

    # 3. Encode corpus
    print("Step 3: Encoding corpus into token IDs...")
    t0 = time.time()
    encoded = tokenizer.encode(corpus)
    num_tokens = len(encoded)
    print(f"[Encode] {len(corpus):,} chars -> {num_tokens:,} tokens ({time.time() - t0:.1f}s)")

    # 4. Model size and tokens/param ratio check
    temp_model = build_micro_lm(config)
    num_params = temp_model.get_num_params()
    tokens_per_param = num_tokens / max(num_params, 1)

    print(f"\n[Model & Data Ratio]")
    print(f"  Corpus tokens:        {num_tokens:,}")
    print(f"  Model parameters:     {num_params:,}")
    print(f"  Tokens per parameter: {tokens_per_param:.2f}")
    print(f"  Layers / Heads / Dim: {config.num_layers} layers / {config.num_heads} heads / {config.embed_dim} dim (FF: {config.ff_dim})")
    print(f"  Sequence length:      {config.seq_length}")
    del temp_model

    # 5. Build tf.data pipelines (with positional split)
    print("\nStep 4: Creating tf.data pipelines (positional train/val split)...")
    train_ds, val_ds, num_train, num_val = prepare_datasets(
        encoded,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        val_split=config.val_split,
        stride=stride,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    steps_per_epoch = num_train // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    print(f"[Steps] {steps_per_epoch} steps/epoch | {total_steps} total steps ({config.epochs} epochs)")

    # 6. Build model
    print("\nStep 5: Building Keras model...")
    model = build_micro_lm(config)
    best_weights_path = os.path.join(config.checkpoint_dir, "best_model.weights.h5")
    if not getattr(config, "from_scratch", False) and os.path.exists(best_weights_path):
        try:
            model.load_weights(best_weights_path)
            print(f"[Model] Initialized weights from {best_weights_path} for continuous fine-tuning on clean corpus.")
        except Exception as e:
            print(f"[Model] Could not load prior weights ({e}), initializing fresh weights.")
    else:
        print("[Model] Initialized fresh model weights from scratch.")
    print(model.summary_str())

    # 7. Optimizer with warmup + cosine decay
    print("\nStep 6: Setting up AdamW optimizer with warmup + cosine decay...")
    warmup_steps = min(config.warmup_steps, max(1, total_steps // 5))
    lr_schedule = WarmupCosineDecaySchedule(
        initial_learning_rate=config.learning_rate,
        min_lr=config.min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Print the learning rate at steps 0, 100, 1000 and at the end of epoch 1
    lr_0 = float(lr_schedule(0))
    lr_100 = float(lr_schedule(min(100, total_steps)))
    lr_warmup = float(lr_schedule(warmup_steps))
    lr_1000 = float(lr_schedule(min(1000, total_steps)))
    lr_epoch1 = float(lr_schedule(min(steps_per_epoch, total_steps)))

    print(f"\n{'=' * 60}")
    print("  Learning Rate Schedule Verification")
    print(f"{'=' * 60}")
    print(f"  Warmup steps:      {int(warmup_steps)}")
    print(f"  Total steps:       {total_steps} ({steps_per_epoch} steps/epoch * {config.epochs} epochs)")
    print(f"  Peak LR:           {float(config.learning_rate):.6e} (reached at step {int(warmup_steps)})")
    print(f"  Min LR:            {float(config.min_lr):.6e}")
    print(f"  Step 0:            lr = {lr_0:.6e}")
    print(f"  Step 100:          lr = {lr_100:.6e}")
    print(f"  Step {int(warmup_steps)} (Peak):   lr = {lr_warmup:.6e}")
    print(f"  Step 1000:         lr = {lr_1000:.6e}")
    print(f"  End of Epoch 1:    lr = {lr_epoch1:.6e} (step {steps_per_epoch})")
    print(f"{'=' * 60}\n")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,  # Gradient clipping
    )
    try:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            jit_compile=True,
        )
    except Exception:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    # 8. Callbacks
    best_weights_path = os.path.join(config.checkpoint_dir, "best_model.weights.h5")
    final_weights_path = os.path.join(config.checkpoint_dir, "final_model.weights.h5")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )
    sample_prompts = [
        "The stock market",
        "The Federal Reserve announced",
        "Investors are closely watching",
    ]
    sample_gen_cb = SampleGenerationCallback(tokenizer, sample_prompts, every_n_epochs=2)
    time_limit_cb = TimeLimitCallback(max_minutes=config.max_minutes)
    metrics_cb = MetricsTrackerCallback()
    lr_monitor_cb = LRMonitorCallback(steps_per_epoch=steps_per_epoch)

    callbacks = [lr_monitor_cb, checkpoint_cb, early_stop_cb, sample_gen_cb, time_limit_cb, metrics_cb]

    # Save initial config JSON
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    config.save(config_path)

    # 9. Train
    print("\nStep 7: Training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time

    # Save best model weights to final_weights_path as well
    if os.path.exists(best_weights_path):
        print(f"\n[Training] Restoring best model weights from {best_weights_path} (best val_loss)...")
        model.load_weights(best_weights_path)
    model.save_weights(final_weights_path)
    print(f"[Training] Saved model weights to {best_weights_path} and {final_weights_path}")

    # Final evaluation
    val_loss, val_ppl = None, None
    if val_ds is not None:
        val_eval = model.evaluate(val_ds, verbose=0)
        val_loss = float(val_eval[0])
        val_ppl = float(math.exp(min(val_loss, 20.0)))
        print(f"[Training] Final validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

    best_val_loss = min(metrics_cb.history["val_loss"]) if metrics_cb.history["val_loss"] else val_loss

    # Save training history JSON
    history_dict = {
        "loss": [float(x) for x in metrics_cb.history["loss"]],
        "accuracy": [float(x) for x in metrics_cb.history["accuracy"]],
        "val_loss": [float(x) for x in metrics_cb.history["val_loss"]],
        "val_accuracy": [float(x) for x in metrics_cb.history["val_accuracy"]],
        "val_perplexity": round(val_ppl, 2) if val_ppl else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss else None,
        "epochs_completed": len(metrics_cb.history["loss"]),
        "training_time_seconds": round(elapsed, 1),
        "corpus_tokens": num_tokens,
        "model_params": num_params,
        "tokens_per_param": round(tokens_per_param, 2),
        "config": config.to_dict(),
    }
    with open(config.history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    checkpoint_history = os.path.join(config.checkpoint_dir, "training_history.json")
    with open(checkpoint_history, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Training] History written to {config.history_path} and {checkpoint_history}")

    # 10. Post-training sanity check
    run_post_training_sanity_check(model, tokenizer, config)

    print(f"{'=' * 60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes!")
    print(f"  Run 'python app.py' to launch the web demo.")
    print(f"{'=' * 60}\n")
    return model, tokenizer, history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroLM (TensorFlow)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None, help="Sequence window stride for training")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training sequences for faster training")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild corpus from scratch")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize fresh weights without loading prior checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch smoke test on a slice")

    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.lr is not None: overrides["learning_rate"] = args.lr
    if args.seq_length is not None: overrides["seq_length"] = args.seq_length
    if args.embed_dim is not None: overrides["embed_dim"] = args.embed_dim
    if args.num_layers is not None: overrides["num_layers"] = args.num_layers
    if args.num_heads is not None: overrides["num_heads"] = args.num_heads
    if args.ff_dim is not None: overrides["ff_dim"] = args.ff_dim
    if args.max_minutes is not None: overrides["max_minutes"] = args.max_minutes

    config = ModelConfig(**overrides)
    if args.from_scratch:
        config.from_scratch = True

    if args.no_cache:
        from data.corpus import CACHE_FILE
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            print("[Train] Deleted corpus cache.")

    if args.smoke_test:
        config.epochs = 1
        config.batch_size = 16
        train(config, stride=args.stride, max_train_samples=64, max_val_samples=32)
    else:
        # Full training: full dataset without smoke-test slices
        train(config, stride=args.stride, max_train_samples=args.max_train_samples, max_val_samples=None)
