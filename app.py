"""
Flask web application for MicroLM — Interactive stock exchange text generation.
Built with TensorFlow 2.x / Keras.
"""
import json
import os
import time
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Global model state ──
_model = None
_tokenizer = None
_config = None
_history = None


def get_model():
    """Lazy-load the trained TensorFlow model on first request."""
    global _model, _tokenizer, _config, _history

    if _model is None:
        from generate import load_model, check_model_qualification

        checkpoint_dir = "checkpoints"
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                "No checkpoints directory found. "
                "Please run 'python train.py' first to train the model."
            )

        # Enforce startup qualification check
        check_model_qualification(checkpoint_dir, raise_on_failure=True)

        print("[App] Loading TensorFlow model...")
        t0 = time.time()
        _model, _tokenizer, _config = load_model(checkpoint_dir, check_qualification=False)
        print(f"[App] Model loaded in {time.time() - t0:.2f}s")

        # Load training history if available
        history_path = _config.history_path if _config else "training_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                _history = json.load(f)
            print("[App] Training history loaded.")

    return _model, _tokenizer, _config, _history


@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate text completion from a prompt."""
    try:
        from generate import generate_text

        model, tokenizer, config, _ = get_model()

        data = request.get_json() or {}
        prompt = data.get("prompt", "The stock market")
        temperature = float(data.get("temperature", config.default_temperature))
        top_k = int(data.get("top_k", config.default_top_k))
        top_p = float(data.get("top_p", config.default_top_p))
        max_length = int(data.get("max_length", config.default_max_length))
        repetition_penalty = float(data.get("repetition_penalty", config.repetition_penalty))
        repetition_window = int(data.get("repetition_window", config.repetition_window))
        no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", config.no_repeat_ngram_size))

        # Clamp values safely
        temperature = max(0.05, min(2.0, temperature))
        top_k = max(0, min(200, top_k))
        top_p = max(0.1, min(1.0, top_p))
        max_length = max(10, min(1000, max_length))
        repetition_penalty = max(1.0, min(2.0, repetition_penalty))
        no_repeat_ngram_size = max(0, min(5, no_repeat_ngram_size))

        t0 = time.time()
        # Generate continuation separately
        continuation = generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stop_at_eos=True,
            return_continuation_only=True,
            trim_sentence=True,
        )
        # Ensure continuation has a leading space if needed
        if (
            continuation
            and not prompt.endswith(" ")
            and not continuation.startswith(" ")
            and continuation[0] not in ".,!?;:'\")]}"
        ):
            continuation = " " + continuation

        gen_time = time.time() - t0
        full_text = prompt + continuation

        return jsonify({
            "success": True,
            "prompt": prompt,
            "continuation": continuation,
            "generated_text": full_text,
            "generation_time": round(gen_time, 2),
            "settings": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            }
        })

    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-info")
def api_model_info():
    """Return model architecture and training information."""
    try:
        model, tokenizer, config, history = get_model()

        info = {
            "architecture": "Decoder-Only Transformer (MicroLM)",
            "framework": "TensorFlow 2.x / Keras",
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "ff_dim": config.ff_dim,
            "seq_length": config.seq_length,
            "parameters": f"{model.get_num_params():,}",
            "tokenizer": "Byte-Level BPE",
            "model_type": "Text-Completion Model",
        }

        if history:
            info["training"] = {
                "epochs": history.get("epochs_completed", 0),
                "final_loss": round(history["loss"][-1], 4) if history.get("loss") else None,
                "final_val_loss": round(history["val_loss"][-1], 4) if history.get("val_loss") else None,
                "val_perplexity": history.get("val_perplexity"),
                "training_time": round(history.get("training_time_seconds", 0), 1),
                "tokens_per_param": history.get("tokens_per_param"),
            }

        return jsonify({"success": True, "info": info})

    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/training-history")
def api_training_history():
    """Return training loss/accuracy history for chart visualization."""
    try:
        _, _, _, history = get_model()

        if history:
            return jsonify({"success": True, "history": history})
        else:
            return jsonify({"success": False, "error": "No training history found"})

    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MicroLM - Stock Exchange Language Model")
    print("  Web Demo (TensorFlow / Keras)")
    print("=" * 60)

    # Startup qualification check
    from generate import check_model_qualification
    import sys
    ok, msg, _ = check_model_qualification(raise_on_failure=False)
    if not ok:
        print("\n" + "!" * 70)
        print("  [STARTUP CHECK FAILED - REFUSING TO SERVE]")
        print(f"  {msg}")
        print("!" * 70 + "\n")
        sys.exit(1)

    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Starting server at http://localhost:{port}")
    print("  Model qualification verified successfully.")
    print("  Model will load on first request...\n")

    app.run(host="0.0.0.0", port=port, debug=False)
