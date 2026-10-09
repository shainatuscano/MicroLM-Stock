"""
Byte-level BPE tokenizer for MicroLM using HuggingFace tokenizers.
Maps text to token IDs and back with exact round-trip preservation
of whitespace, punctuation, and newlines.
"""
import os
import random
from collections import Counter
from typing import List, Iterable, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    """Byte-level BPE Tokenizer using HuggingFace tokenizers."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        self.tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDec()
        self.vocab_size: int = 0
        self._fitted: bool = False

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id(self.UNK_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    def fit(self, corpus: Union[str, Iterable[str]], vocab_size: int = 5000, min_frequency: int = 2) -> "BPETokenizer":
        """
        Train byte-level BPE tokenizer on a text corpus or an iterable of strings.
        Avoids creating temporary files by streaming chunks to train_from_iterator.
        """
        vocab_size = max(1000, min(8000, vocab_size))

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN],
            initial_alphabet=ByteLevelPre.alphabet(),
        )

        def _iterator():
            if isinstance(corpus, str):
                chunk_size = 100_000
                for i in range(0, len(corpus), chunk_size):
                    yield corpus[i : i + chunk_size]
            else:
                batch = []
                for item in corpus:
                    batch.append(item)
                    if len(batch) >= 1000:
                        yield "\n".join(batch)
                        batch = []
                if batch:
                    yield "\n".join(batch)

        self.tokenizer.train_from_iterator(_iterator(), trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._fitted = True

        print(f"[Tokenizer] Fitted byte-level BPE: target {vocab_size} | actual {self.vocab_size} tokens")
        return self

    def encode(self, text: str) -> List[int]:
        """Convert text to integer token IDs."""
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert integer token IDs back to string."""
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding.")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def log_vocab_utilization(self, text: str, min_count: int = 5) -> dict:
        """Encode sample text and log how many tokens are utilized."""
        encoded = self.encode(text)
        token_counts = Counter(encoded)

        total_tokens = self.vocab_size
        used_tokens = len(token_counts)
        frequent_tokens = sum(1 for c in token_counts.values() if c >= min_count)
        rare_tokens = sum(1 for c in token_counts.values() if c < min_count)

        stats = {
            "total_vocab": total_tokens,
            "tokens_used": used_tokens,
            "tokens_frequent": frequent_tokens,
            "tokens_rare": rare_tokens,
            "utilization_pct": 100.0 * used_tokens / max(total_tokens, 1),
        }

        print("\n[Tokenizer] === Vocab Utilization ===")
        print(f"[Tokenizer]   Total vocab:      {total_tokens:,}")
        print(f"[Tokenizer]   Tokens used:      {used_tokens:,} ({stats['utilization_pct']:.1f}%)")
        print(f"[Tokenizer]   Frequent (≥{min_count}x): {frequent_tokens:,}")
        print(f"[Tokenizer]   Rare (<{min_count}x):     {rare_tokens:,}")
        return stats

    def save(self, path: str) -> None:
        """Save tokenizer configuration to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str) -> "BPETokenizer":
        """Load tokenizer configuration from a JSON file."""
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._fitted = True
        print(f"[Tokenizer] Loaded from {path} ({self.vocab_size} tokens)")
        return self

    def test_roundtrip(self, snippets: List[str]) -> bool:
        """
        Test that encode -> decode round-trips exactly for each snippet,
        including newlines, spaces, and punctuation.
        """
        all_passed = True
        print(f"[Tokenizer] Running round-trip test on {len(snippets)} snippets...")
        for i, snip in enumerate(snippets):
            ids = self.encode(snip)
            decoded = self.decode(ids, skip_special_tokens=False)
            if decoded != snip:
                all_passed = False
                print(f"  ❌ Snippet {i+1} failed roundtrip!")
                print(f"     Expected: {repr(snip[:80])}")
                print(f"     Got:      {repr(decoded[:80])}")
            else:
                pass
        if all_passed:
            print(f"[Tokenizer] ✅ All {len(snippets)} snippets passed round-trip test exactly!")
        return all_passed

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"BPETokenizer(vocab_size={self.vocab_size}, {status})"


if __name__ == "__main__":
    # Smoke test with representative text
    sample_text = (
        "The New York Stock Exchange (NYSE) was founded in 1792.\n"
        "Revenue grew by 14.5% year-over-year to $4.2 billion!\n"
        "Can we handle quotes, 'single' and \"double\", and <EOS>?\n"
        "Line 1.\nLine 2.\n\nParagraph 2 with symbols: @#$%^&*()_+.\n"
    )
    tok = BPETokenizer()
    tok.fit(sample_text * 10, vocab_size=300)
    snippets = [
        "The stock market is fluctuating.",
        "Revenue grew by 14.5%!",
        "Special chars: $100.50, AAPL, MSFT.",
        "Newline test:\nLine 1\nLine 2\n",
        "Punctuation: !?,.:;\"'()[]{}",
    ]
    tok.test_roundtrip(snippets)
