"""
CLI and programmatic text generation with MicroLM (TensorFlow 2.x / Keras).
Loads trained model and tokenizer from checkpoint directory and generates
high-quality stock exchange text using autoregressive sampling.
"""
import argparse
import json
import os
import string
import sys
import numpy as np
import tensorflow as tf

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import build_micro_lm, MicroLM


def _get_punctuation_and_space_tokens(tokenizer: BPETokenizer) -> set:
    """Precompute set of token IDs that represent whitespace or punctuation to exempt them from repetition penalty."""
    exempt = set()
    for token_id in range(tokenizer.vocab_size):
        try:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            cleaned = token_text.strip()
            if not cleaned or all(ch in string.punctuation for ch in cleaned):
                exempt.add(token_id)
        except Exception:
            pass
    return exempt


def check_model_qualification(
    checkpoint_dir: str = "checkpoints",
    history_path: str = "training_history.json",
    raise_on_failure: bool = True,
):
    """
    Startup qualification check:
    Refuses to serve when history shows epochs < 3 or val loss > 6.
    """
    candidates = [
        history_path,
        os.path.join(checkpoint_dir, "training_history.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_history.json"),
    ]
    resolved_path = None
    for p in candidates:
        if os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        msg = (
            "Model startup check failed: No 'training_history.json' found. "
            "Refusing to serve: The model must be trained for at least 3 epochs with val loss <= 6.0."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        return False, msg, None

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        msg = f"Model startup check failed: Could not read training history ({e})."
        if raise_on_failure:
            raise RuntimeError(msg)
        return False, msg, None

    epochs_completed = history.get("epochs_completed", len(history.get("loss", [])))
    val_losses = history.get("val_loss", [])

    if not val_losses:
        msg = "Model startup check failed: No validation loss found in training history."
        if raise_on_failure:
            raise RuntimeError(msg)
        return False, msg, history

    best_val_loss = min(val_losses)

    if epochs_completed < 3:
        msg = (
            f"Model startup check failed: Refusing to serve model! "
            f"Training history shows epochs completed = {epochs_completed} (< 3 required). "
            f"Smoke-test or incomplete models cannot be served. "
            f"Please run full training ('python train.py') for at least 3 epochs."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        return False, msg, history

    if best_val_loss > 6.0:
        msg = (
            f"Model startup check failed: Refusing to serve model! "
            f"Best validation loss is {best_val_loss:.4f} (> 6.0 max allowed threshold). "
            f"The model has not sufficiently converged. "
            f"Please retrain until validation loss falls well below 5.0."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        return False, msg, history

    return True, f"Qualification passed: {epochs_completed} epochs, best val_loss={best_val_loss:.4f}", history


def load_model(checkpoint_dir: str = "checkpoints", check_qualification: bool = True):
    """
    Load trained model, tokenizer, and config from checkpoint directory.
    Rebuilds architecture from the saved config JSON to prevent breaking
    on non-default configurations.
    """
    # 0. Check qualification criteria
    if check_qualification:
        check_model_qualification(checkpoint_dir=checkpoint_dir, raise_on_failure=True)

    # 1. Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Please run train.py first."
        )
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # 2. Load config from JSON if present, otherwise fallback to defaults
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        config = ModelConfig.load(config_path)
        config.vocab_size = tokenizer.vocab_size
    else:
        config = ModelConfig(vocab_size=tokenizer.vocab_size)

    # 3. Build model architecture
    model = build_micro_lm(config)

    # 4. Load weights (.weights.h5 or .h5)
    best_weights = os.path.join(checkpoint_dir, "best_model.weights.h5")
    final_weights = os.path.join(checkpoint_dir, "final_model.weights.h5")
    legacy_best = os.path.join(checkpoint_dir, "best_model.h5")
    legacy_final = os.path.join(checkpoint_dir, "final_model.h5")

    if os.path.exists(best_weights):
        model.load_weights(best_weights)
        print(f"[Model] Loaded best weights from {best_weights}")
    elif os.path.exists(final_weights):
        model.load_weights(final_weights)
        print(f"[Model] Loaded final weights from {final_weights}")
    elif os.path.exists(legacy_best):
        model.load_weights(legacy_best)
        print(f"[Model] Loaded best weights from {legacy_best}")
    elif os.path.exists(legacy_final):
        model.load_weights(legacy_final)
        print(f"[Model] Loaded final weights from {legacy_final}")
    else:
        raise FileNotFoundError(
            f"No model weights (.weights.h5) found in {checkpoint_dir}. "
            "Please run train.py first."
        )

    return model, tokenizer, config


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> int:
    """Sample next token ID from unnormalized logits using temperature, top-k, and top-p (nucleus)."""
    logits = np.array(logits, dtype=np.float64)

    # 1. Temperature
    temperature = max(1e-5, temperature)
    logits = logits / temperature

    # 2. Top-K filtering
    if 0 < top_k < len(logits):
        k_val = np.partition(logits, -top_k)[-top_k]
        logits[logits < k_val] = -np.inf

    # 3. Softmax probabilities
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)

    # 4. Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)
        # Keep tokens with cumulative probability below top_p
        cutoff = cum_probs > top_p
        cutoff[0] = False  # Always keep at least the most probable token
        sorted_probs[cutoff] = 0.0
        prob_sum = np.sum(sorted_probs)
        if prob_sum > 0:
            sorted_probs /= prob_sum
            return int(np.random.choice(sorted_indices, p=sorted_probs))

    return int(np.random.choice(len(probs), p=probs))


def generate_text(
    model: MicroLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_length: int = 150,
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.20,
    repetition_window: int = 128,
    no_repeat_ngram_size: int = 3,
    stop_at_eos: bool = True,
    return_continuation_only: bool = False,
    trim_sentence: bool = True,
) -> str:
    """
    Autoregressive text generation with fixed-length input to avoid tf.function retracing.
    Applies temperature, top-k, top-p, windowed repetition penalty, n-gram blocking,
    stop phrase detection, and optional clean sentence boundary trimming.
    """
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_id]

    generated = list(prompt_ids)
    seq_length = model.config.seq_length
    vocab_size = model.config.vocab_size
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    # Precompute exempt punctuation and whitespace tokens
    exempt_tokens = _get_punctuation_and_space_tokens(tokenizer)

    # Fixed-length buffer for tf.function to avoid any retracing
    buf = np.full((1, seq_length), pad_id, dtype=np.int32)

    # Compile the forward pass once with fixed shape signature
    predict_fn = getattr(model, "_predict_step_compiled", None)
    if predict_fn is None:
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, seq_length], dtype=tf.int32)])
        def _compiled_predict(x):
            return model(x, training=False)
        predict_fn = _compiled_predict
        model._predict_step_compiled = predict_fn

    stop_phrases = [
        "http://", "https://", "www.", "view source", "view original",
        "about pcn", "dial-", "dial in", "dialing", "a replay will be",
        "replay will be", "replay of the", "replay of the webcast",
        "webcast and replay", "conference call to discuss", "conference call",
        "webcast", "passcode", "participant", "press release contains",
        "for more information", "view original content",
        "reporting by", "editing by", "rough cut", "source:", "source text",
        "nasdaq:", "nyse:"
    ]

    stopped_early = False
    for step_i in range(max_length):
        # Crop context to seq_length
        context = generated[-seq_length:]
        curr_len = len(context)

        # Place context into fixed-length buffer
        buf[0, :curr_len] = context
        if curr_len < seq_length:
            buf[0, curr_len:] = pad_id

        # Forward pass (runs in ~4ms with zero retracing)
        logits_tensor = predict_fn(buf)
        next_logits = logits_tensor[0, curr_len - 1, :].numpy()

        # Repetition penalty over recent window (Task C)
        if repetition_penalty != 1.0:
            window = generated[-repetition_window:]
            for token_id in set(window):
                if token_id in exempt_tokens:
                    continue
                if 0 <= token_id < vocab_size:
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

        # No-repeat n-gram blocking (Task C)
        if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size:
            prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
            for i in range(len(generated) - no_repeat_ngram_size + 1):
                candidate = tuple(generated[i : i + no_repeat_ngram_size - 1])
                if candidate == prefix:
                    banned = generated[i + no_repeat_ngram_size - 1]
                    if 0 <= banned < vocab_size:
                        next_logits[banned] = -np.inf

        # Sample next token
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Stop at <EOS> if produced (Task C)
        if stop_at_eos and next_id == eos_id:
            stopped_early = True
            break

        generated.append(next_id)

        # Check for stop phrases periodically
        if step_i > 10 and step_i % 5 == 0:
            recent_text = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).lower()
            if any(sp in recent_text for sp in stop_phrases):
                stopped_early = True
                break

    # Decode
    if return_continuation_only:
        continuation_ids = generated[len(prompt_ids):]
        raw_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    else:
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Filter out any trailing stop phrase fragments
    raw_lower = raw_text.lower()
    for sp in stop_phrases:
        if sp in raw_lower:
            idx = raw_lower.find(sp)
            raw_text = raw_text[:idx].rstrip()
            raw_lower = raw_text.lower()

    # If requested, ensure text finishes cleanly on sentence boundary
    if trim_sentence:
        import re
        # Strip trailing ellipsis (2 or more dots) or trailing newlines/spaces
        cleaned = re.sub(r"[\s\n]*\.{2,}[\s\n]*$", "", raw_text)
        # Find all sentence-ending punctuation preceded by a word char or symbol
        sentence_ends = [m.end() for m in re.finditer(r"[\w%\"\'\)\]\}][\.\!\?](?!\.)", cleaned)]
        min_cutoff = 10 if return_continuation_only else len(prompt) + 10
        if sentence_ends:
            last_end = sentence_ends[-1]
            if last_end >= min_cutoff:
                raw_text = cleaned[:last_end]
        else:
            raw_text = cleaned.rstrip()

    if return_continuation_only:
        continuation = raw_text.rstrip()
        if (
            continuation
            and not prompt.endswith(" ")
            and not continuation.startswith(" ")
            and continuation[0] not in ".,!?;:'\")]}"
        ):
            continuation = " " + continuation
        return continuation
    else:
        return raw_text.strip()


def interactive_mode(model, tokenizer, config):
    """Interactive REPL for text generation."""
    print("\n" + "=" * 60)
    print("  MicroLM - Interactive Text Generation (TensorFlow)")
    print("  Type a prompt and press Enter to generate text.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("📈 Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not prompt:
                continue

            print("Generating...\n")
            text = generate_text(
                model, tokenizer, prompt,
                max_length=config.default_max_length,
                temperature=config.default_temperature,
                top_k=config.default_top_k,
                top_p=config.default_top_p,
                repetition_penalty=config.repetition_penalty,
                repetition_window=config.repetition_window,
            )
            print(f"📝 Generated:\n{text}\n")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MicroLM (TensorFlow)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt to complete")
    parser.add_argument("--length", type=int, default=120,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.65,
                        help="Sampling temperature (higher = more creative)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.85,
                        help="Top-p nucleus sampling (1.0 = disabled)")
    parser.add_argument("--rep-penalty", type=float, default=1.08,
                        help="Repetition penalty (1.0 = disabled)")
    parser.add_argument("--rep-window", type=int, default=128,
                        help="Recent window size for repetition penalty")
    parser.add_argument("--no-repeat-ngram", type=int, default=3,
                        help="Block repeated n-grams of this size (0 = disabled)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    args = parser.parse_args()

    try:
        model, tokenizer, config = load_model(args.checkpoint, check_qualification=True)
    except RuntimeError as e:
        print(f"\n[Startup Refusal] {e}\n", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}\n", file=sys.stderr)
        sys.exit(1)

    print(model.summary_str())

    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, config)
    else:
        text = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            repetition_window=args.rep_window,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )
        print(f"\n{text}")
