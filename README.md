# MicroLM — Stock Exchange Language Model

A micro GPT-style decoder-only transformer language model trained on financial and stock market text, built with **TensorFlow 2.x / Keras (`tf.keras`)**.

> ⚠️ **Important Note**: MicroLM is a causal **text-completion** model, not a conversational or question-answering assistant. Provide prompts that start a financial sentence or headline (e.g., *"The Federal Reserve announced"*, *"The stock market"*), rather than asking questions.
>
> ⚠️ **Disclaimer**: This is an educational demonstration. Generated text should **never** be used for financial or trading decisions.

---

## Architecture & Specifications

| Component            | Details                                                          |
|----------------------|------------------------------------------------------------------|
| **Model Type**       | GPT-style Decoder-Only Transformer (pre-LayerNorm)               |
| **Framework**        | TensorFlow 2.x / Keras (`tf.keras`)                              |
| **Parameters**       | **4,472,320** (~4.47M parameters)                                |
| **Embedding Dim**    | 256                                                              |
| **Attention Heads**  | 4 (head dimension 64)                                            |
| **Layers**           | 4 Transformer decoder blocks                                     |
| **FFN Dimension**    | 1024 with GELU activation                                        |
| **Weight Tying**     | Tied token embedding and linear output projection (`SharedEmbedding`) |
| **Positional Embed** | Learned positional embeddings (`seq_length = 128` context)        |
| **Tokenizer**        | Byte-Level BPE (HuggingFace tokenizers)                          |
| **Vocabulary Size**  | 5,000 tokens (with `<PAD>`, `<UNK>`, `<EOS>`)                    |
| **Causal Masking**   | Strictly causal attention (0.0 future-token leakage verified)    |

---

## Sampler & Generation Features

- **Compilation**: Autoregressive loop wrapped in fixed-length `@tf.function` with right-padding for zero retracing overhead (~4ms per token).
- **Sampling**: Temperature, Top-K, and Top-P (Nucleus) sampling.
- **Windowed Repetition Penalty**: Multiplicative penalty applied only over the recent 64 tokens, completely exempting whitespace and punctuation tokens.
- **Early Stop**: Generation halts cleanly upon emitting the `<EOS>` token.

### Default Generation Parameters
- `temperature`: `0.80`
- `top_k`: `40`
- `top_p`: `0.90`
- `repetition_penalty`: `1.05` (window: `64` tokens)
- `no_repeat_ngram_size`: `0` (disabled)

---

## Quick Start

### 1. Environment Setup
```bash
# Automated setup (creates .venv, installs dependencies & verifies)
bash setup.sh

# Activate virtual environment
source .venv/bin/activate

# Or install manually
pip install -r requirements.txt
python check_env.py
```

### 2. Train the Model
```bash
# Run training with default configuration
python train.py

# Optional: Run quick smoke test
python train.py --smoke-test

# Optional: Cap training time or sample counts
python train.py --epochs 6 --batch-size 64 --max-minutes 30
```

### 3. Generate Text (CLI)
```bash
# Single prompt completion
python generate.py --prompt "The Federal Reserve announced" --length 200

# Interactive REPL mode
python generate.py --interactive
```

### 4. Launch the Web Interface
```bash
python app.py
```
Open **http://localhost:5000** in your browser. The Web UI provides interactive parameter sliders, typing animation, live architecture inspector, and loss/accuracy curves.

---

## Project Structure

```
MicroLM-Stock/
├── checkpoints/               # Model weights (.weights.h5) & tokenizer.json
│   ├── best_model.weights.h5
│   ├── final_model.weights.h5
│   ├── config.json
│   └── tokenizer.json
├── data/
│   ├── __init__.py
│   ├── corpus.py              # Financial dataset downloader, cleaner & cache
│   └── corpus_cache.txt       # Cleaned text corpus with <EOS> separation
├── model/
│   ├── __init__.py
│   ├── config.py              # ModelConfig dataclass with JSON save/load
│   ├── tokenizer.py           # Byte-Level BPE tokenizer & round-trip validator
│   └── transformer.py         # MicroLM tf.keras architecture & custom layers
├── templates/
│   └── index.html             # Web UI HTML template
├── static/
│   ├── style.css              # Dark theme styling & animations
│   └── script.js              # Frontend generation & metrics logic
├── train.py                   # TensorFlow training pipeline
├── generate.py                # Autoregressive generation & CLI
├── app.py                     # Flask web server
├── check_env.py               # Environment & device verification
├── run_train.sh               # Convenient training script
├── setup.sh                   # Environment setup script
├── requirements.txt           # Python package requirements
└── README.md
```

---

## Dataset Pipeline

The corpus combines multiple curated financial sources:
1. **Embedded Financial Corpus**: Foundational stock exchange theory (upweighted 3x).
2. **Financial News Articles** (`ashraq/financial-news-articles`): Full news article bodies.
3. **Financial PhraseBank** (`warwickai/financial_phrasebank_mirror`): Expert sentiment statements.
4. **Market Headlines** (`ashraq/financial-news`): Cleaned market updates.

All text is processed with `ftfy` to resolve mojibake, whitespace-normalized, deduplicated, formatted with one document per line, and delimited with `<EOS>`.

---

## License

MIT
