from collections import defaultdict
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable


DEFAULT_SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>",
    "<SYSTEM>",
    "<TOOL>",
    "<THOUGHT>",
]


class VeyraTokenizer:
    """Byte-level BPE Tokenizer built from scratch for VEYRA-LM.

    Features:
    - 100% byte-level coverage: every UTF-8 byte [0..255] is in the base vocabulary,
      guaranteeing zero out-of-vocabulary crashes for arbitrary text.
    - Explicit special tokens for conversation, role tagging, system, and tool invocation.
    - Deterministic merge application with priority ordering.
    - Batch encoding, padding, truncation, attention mask generation.
    - JSON serialization.
    """

    def __init__(
        self,
        special_tokens: list[str] | None = None,
        vocab_size: int = 4096,
    ) -> None:
        self.target_vocab_size = vocab_size
        self.special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)

        # ID mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Special token IDs
        self.special_token_ids: set[int] = set()

        # Initialize base vocab
        self._init_base_vocab()

        # Regex pre-tokenizer pattern (splits punctuation, words, numbers, whitespace)
        self.pre_tokenize_regex = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
        )

    def _init_base_vocab(self) -> None:
        """Initializes special tokens and 256 raw byte representations."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        self.special_token_ids.clear()

        # 1. Register special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.special_token_ids.add(idx)

        # 2. Register 256 individual UTF-8 byte tokens
        base_offset = len(self.special_tokens)
        for b in range(256):
            token_str = self._byte_to_str(b)
            token_id = base_offset + b
            self.token_to_id[token_str] = token_id
            self.id_to_token[token_id] = token_str

    @staticmethod
    def _byte_to_str(b: int) -> str:
        return f"<0x{b:02X}>"

    @staticmethod
    def _str_to_byte(s: str) -> int | None:
        if s.startswith("<0x") and s.endswith(">") and len(s) == 6:
            try:
                return int(s[3:5], 16)
            except ValueError:
                return None
        return None

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id.get("<PAD>", 0)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id.get("<UNK>", 1)

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id.get("<BOS>", 2)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id.get("<EOS>", 3)

    @property
    def user_token_id(self) -> int:
        return self.token_to_id.get("<USER>", 4)

    @property
    def assistant_token_id(self) -> int:
        return self.token_to_id.get("<ASSISTANT>", 5)

    @property
    def system_token_id(self) -> int:
        return self.token_to_id.get("<SYSTEM>", 6)

    @property
    def tool_token_id(self) -> int:
        return self.token_to_id.get("<TOOL>", 7)

    def train_from_texts(
        self,
        texts: Iterable[str],
        vocab_size: int | None = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Trains BPE merge rules from an iterable of training texts using a fast inverted-index algorithm."""
        if vocab_size is not None:
            self.target_vocab_size = vocab_size

        self._init_base_vocab()
        self.merges.clear()
        self.merge_ranks.clear()

        # Pre-tokenize all texts into sequences of byte token strings
        word_counts: dict[tuple[str, ...], int] = {}
        for text in texts:
            for piece in self.pre_tokenize_regex.findall(text):
                if not piece:
                    continue
                byte_tokens = tuple(self._byte_to_str(b) for b in piece.encode("utf-8"))
                word_counts[byte_tokens] = word_counts.get(byte_tokens, 0) + 1

        num_merges_to_learn = self.target_vocab_size - len(self.token_to_id)
        if num_merges_to_learn <= 0:
            return

        vocab: list[tuple[str, ...]] = list(word_counts.keys())
        freqs: list[int] = list(word_counts.values())

        # Inverted index: pair -> frequency count, pair -> set of word indices
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set[int]] = defaultdict(set)

        for w_idx, (w, freq) in enumerate(zip(vocab, freqs)):
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(w_idx)

        t_start = time.time()
        for merge_idx in range(num_merges_to_learn):
            if not pair_counts:
                break

            # Deterministic best pair selection
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best_pair]
            if best_count < min_frequency:
                break

            # Register merge
            merged_token = best_pair[0] + best_pair[1]
            new_id = len(self.token_to_id)
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = merge_idx

            del pair_counts[best_pair]
            affected_words = list(pair_to_words[best_pair])
            del pair_to_words[best_pair]

            p0, p1 = best_pair
            for w_idx in affected_words:
                w = vocab[w_idx]
                f = freqs[w_idx]

                # Decrement counts of old pairs in this word
                for i in range(len(w) - 1):
                    pair = (w[i], w[i + 1])
                    if pair in pair_counts:
                        pair_counts[pair] -= f
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(w_idx)

                # Merge occurrences of best_pair in this word
                new_w: list[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == p0 and w[i + 1] == p1:
                        new_w.append(merged_token)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_tuple = tuple(new_w)
                vocab[w_idx] = new_tuple

                # Increment counts of newly formed pairs
                for i in range(len(new_tuple) - 1):
                    pair = (new_tuple[i], new_tuple[i + 1])
                    pair_counts[pair] += f
                    pair_to_words[pair].add(w_idx)

            if show_progress and ((merge_idx + 1) % 250 == 0 or (merge_idx + 1) == num_merges_to_learn):
                elapsed = time.time() - t_start
                rate = (merge_idx + 1) / max(elapsed, 0.001)
                pct = ((merge_idx + 1) / num_merges_to_learn) * 100
                remaining = (num_merges_to_learn - (merge_idx + 1)) / max(rate, 1)
                print(
                    f"    [BPE Progress] Merge {merge_idx + 1:,}/{num_merges_to_learn:,} ({pct:.1f}%) | "
                    f"Vocab: {len(self.token_to_id):,} | {rate:.0f} merges/s | ETA: {remaining:.0f}s",
                    flush=True,
                )

    def _apply_merges_to_word(self, word_tokens: list[str]) -> list[str]:
        """Iteratively applies known BPE merges to a sequence of tokens in priority order."""
        if len(word_tokens) <= 1:
            return word_tokens

        tokens = list(word_tokens)
        while len(tokens) >= 2:
            # Find the best merge available in the current word tokens
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            mergeable_pairs = [p for p in pairs if p in self.merge_ranks]
            if not mergeable_pairs:
                break

            best_pair = min(mergeable_pairs, key=lambda p: self.merge_ranks[p])
            p0, p1 = best_pair
            merged_token = p0 + p1

            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> list[int]:
        """Encodes text into token IDs."""
        if not text:
            ids: list[int] = []
            if add_bos:
                ids.insert(0, self.bos_token_id)
            if add_eos:
                ids.append(self.eos_token_id)
            return ids

        # Handle special tokens by splitting text into special and non-special segments
        special_pattern = "|".join(re.escape(st) for st in self.special_tokens)
        segments = re.split(f"({special_pattern})", text)

        token_ids: list[int] = []
        for segment in segments:
            if not segment:
                continue
            if segment in self.token_to_id:
                token_ids.append(self.token_to_id[segment])
            else:
                # Pre-tokenize regular text
                words = self.pre_tokenize_regex.findall(segment)
                for word in words:
                    byte_tokens = [self._byte_to_str(b) for b in word.encode("utf-8")]
                    merged_tokens = self._apply_merges_to_word(byte_tokens)
                    for t in merged_tokens:
                        token_ids.append(self.token_to_id.get(t, self.unk_token_id))

        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)

        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_eos and token_ids[-1] != self.eos_token_id:
                token_ids[-1] = self.eos_token_id

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back into string."""
        raw_bytes: bytearray = bytearray()
        out_segments: list[str] = []

        for tid in token_ids:
            if tid in self.special_token_ids:
                if raw_bytes:
                    out_segments.append(raw_bytes.decode("utf-8", errors="replace"))
                    raw_bytes = bytearray()
                if not skip_special_tokens:
                    out_segments.append(self.id_to_token.get(tid, "<UNK>"))
                continue

            token_str = self.id_to_token.get(tid, "")
            # Decompose token_str back into byte sequences
            # Each byte token is formatted as '<0xNN>' (6 characters)
            byte_indices = [
                int(token_str[i + 3 : i + 5], 16)
                for i in range(0, len(token_str), 6)
                if token_str[i : i + 3] == "<0x" and token_str[i + 5 : i + 6] == ">"
            ]
            for b in byte_indices:
                raw_bytes.append(b)

        if raw_bytes:
            out_segments.append(raw_bytes.decode("utf-8", errors="replace"))

        return "".join(out_segments)

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> dict[str, list[list[int]]]:
        """Encodes a batch of texts with padding and attention masks."""
        encoded_list = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length, truncation=truncation)
            for t in texts
        ]

        max_len = max_length or (max(len(ids) for ids in encoded_list) if encoded_list else 0)

        padded_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for ids in encoded_list:
            if truncation and len(ids) > max_len:
                ids = ids[:max_len]

            pad_amount = max(0, max_len - len(ids)) if padding else 0
            curr_ids = ids + [self.pad_token_id] * pad_amount
            mask = [1] * len(ids) + [0] * pad_amount

            padded_ids.append(curr_ids)
            attention_masks.append(mask)

        return {"input_ids": padded_ids, "attention_mask": attention_masks}

    def save(self, directory: str | Path) -> None:
        """Saves tokenizer state to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        vocab_path = dir_path / "vocab.json"
        merges_path = dir_path / "merges.json"
        config_path = dir_path / "tokenizer_config.json"

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=2, ensure_ascii=False)

        config = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.token_to_id),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> VeyraTokenizer:
        """Loads tokenizer from directory."""
        dir_path = Path(directory)
        config_path = dir_path / "tokenizer_config.json"
        vocab_path = dir_path / "vocab.json"
        merges_path = dir_path / "merges.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = cls(
            special_tokens=config.get("special_tokens"),
            vocab_size=config.get("target_vocab_size", 4096),
        )

        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.token_to_id = json.load(f)
            tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges_raw = json.load(f)
            tokenizer.merges = [tuple(m) for m in merges_raw]
            tokenizer.merge_ranks = {tuple(m): idx for idx, m in enumerate(merges_raw)}

        tokenizer.special_token_ids = {
            tokenizer.token_to_id[st] for st in tokenizer.special_tokens if st in tokenizer.token_to_id
        }

        return tokenizer
