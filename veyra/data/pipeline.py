from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any, Iterator
import torch
from torch.utils.data import Dataset

from ..config.dataset_config import DatasetMixConfig, DatasetSourceConfig
from ..tokenizer.bpe import VeyraTokenizer
from .cleaner import TextCleaner


class ShardedTokenDataset(Dataset):
    """PyTorch Dataset that loads tokenized shards for training."""

    def __init__(self, shard_paths: list[Path], seq_length: int) -> None:
        self.seq_length = seq_length
        self.samples: list[torch.Tensor] = []
        for p in shard_paths:
            data = torch.load(p, weights_only=True)
            if isinstance(data, torch.Tensor):
                # Shape: [num_sequences, seq_length]
                for seq in data:
                    self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Target is input shifted by 1 for autoregressive next-token prediction
        seq = self.samples[idx]
        input_ids = seq[:-1]
        labels = seq[1:]
        return {"input_ids": input_ids, "labels": labels}


class DatasetPipeline:
    """End-to-end dataset preprocessing, tokenization, sharding, and mixing pipeline."""

    def __init__(self, tokenizer: VeyraTokenizer) -> None:
        self.tokenizer = tokenizer
        self.cleaner = TextCleaner()

    def load_raw_source(self, source_cfg: DatasetSourceConfig) -> list[str]:
        """Loads raw text from file path or identifier."""
        p = Path(source_cfg.path_or_identifier)
        if not p.exists():
            raise FileNotFoundError(f"Dataset source file not found: {p}")

        docs: list[str] = []
        fmt = source_cfg.file_format.lower()

        if fmt == "txt":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                # Split by double newline or delimiter if multi-doc, otherwise each paragraph is a doc
                raw_docs = [d.strip() for d in content.split("\n\n") if d.strip()]
                docs.extend(raw_docs if raw_docs else [content])

        elif fmt == "jsonl":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        val = record.get(source_cfg.text_column, "")
                        if val:
                            docs.append(str(val))
                    except json.JSONDecodeError:
                        continue

        elif fmt == "csv":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val = row.get(source_cfg.text_column, "")
                    if val:
                        docs.append(str(val))
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

        return docs

    def process_and_shard(
        self,
        mix_config: DatasetMixConfig,
        output_dir: str | Path,
        shard_size: int = 5000,
        include_private: bool = False,
    ) -> dict[str, Any]:
        """Processes configured sources, applies cleaning, tokenizes, and saves shards."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "train").mkdir(exist_ok=True)
        (out_path / "val").mkdir(exist_ok=True)
        (out_path / "metadata").mkdir(exist_ok=True)

        all_cleaned_docs: list[tuple[str, float]] = []  # (doc, weight)
        source_summaries: list[dict[str, Any]] = []

        for source in mix_config.sources:
            if "private" in source.path_or_identifier.lower() and not include_private:
                continue

            raw_docs = self.load_raw_source(source)
            cleaned_docs, stats = self.cleaner.clean_corpus(raw_docs)
            for doc in cleaned_docs:
                all_cleaned_docs.append((doc, source.sampling_weight))

            source_summaries.append({
                "source_name": source.name,
                "license": source.license,
                "total_raw": stats.total_docs,
                "kept": stats.kept_docs,
                "dropped_empty": stats.dropped_empty,
                "dropped_length": stats.dropped_length,
                "dropped_repetitive": stats.dropped_repetitive,
                "dropped_duplicate": stats.dropped_duplicate,
            })

        if not all_cleaned_docs:
            raise ValueError("No valid documents remained after preprocessing.")

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(all_cleaned_docs)

        # Tokenize and pack sequences into fixed length
        # Target sequence length + 1 (for input_ids and labels offset)
        chunk_len = mix_config.max_sequence_length + 1
        all_token_stream: list[int] = []

        for doc, _ in all_cleaned_docs:
            tokens = self.tokenizer.encode(doc, add_bos=True, add_eos=True)
            all_token_stream.extend(tokens)

        # Slice into chunks of size chunk_len
        total_tokens = len(all_token_stream)
        num_chunks = total_tokens // chunk_len
        if num_chunks == 0:
            # Pad the remaining stream if shorter than single chunk
            pad_needed = chunk_len - total_tokens
            all_token_stream.extend([self.tokenizer.pad_token_id] * pad_needed)
            num_chunks = 1

        packed_tensors = torch.tensor(
            all_token_stream[: num_chunks * chunk_len], dtype=torch.long
        ).view(num_chunks, chunk_len)

        # Train / val split
        val_count = max(1, int(num_chunks * mix_config.val_ratio))
        train_count = num_chunks - val_count

        train_chunks = packed_tensors[:train_count]
        val_chunks = packed_tensors[train_count:]

        # Save train shards
        train_shards: list[str] = []
        for i in range(0, train_count, shard_size):
            shard_tensor = train_chunks[i : i + shard_size]
            shard_file = out_path / "train" / f"shard_{i // shard_size:05d}.pt"
            torch.save(shard_tensor, shard_file)
            train_shards.append(str(shard_file))

        # Save val shards
        val_shards: list[str] = []
        for i in range(0, val_count, shard_size):
            shard_tensor = val_chunks[i : i + shard_size]
            shard_file = out_path / "val" / f"shard_{i // shard_size:05d}.pt"
            torch.save(shard_tensor, shard_file)
            val_shards.append(str(shard_file))

        # Metadata & statistics
        stats_meta = {
            "version": mix_config.version,
            "total_documents": len(all_cleaned_docs),
            "total_tokens": total_tokens,
            "sequence_length": mix_config.max_sequence_length,
            "train_sequences": train_count,
            "val_sequences": val_count,
            "num_train_shards": len(train_shards),
            "num_val_shards": len(val_shards),
            "sources": source_summaries,
        }

        with open(out_path / "metadata" / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_meta, f, indent=2)

        return stats_meta
