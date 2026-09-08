import shutil
import tempfile
import unittest
from pathlib import Path

from veyra.config.dataset_config import DatasetMixConfig, DatasetSourceConfig
from veyra.data.cleaner import TextCleaner
from veyra.data.pipeline import DatasetPipeline, ShardedTokenDataset
from veyra.tokenizer.bpe import VeyraTokenizer


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tokenizer = VeyraTokenizer(vocab_size=300)
        self.tokenizer.train_from_texts([
            "Hello VEYRA",
            "This is a test corpus for dataset pipeline testing.",
            "Algorithms and systems programming in Python.",
        ])

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_text_cleaner_filtering_and_dedup(self) -> None:
        cleaner = TextCleaner(min_chars=15, max_chars=100)

        # Too short
        cleaned, reason = cleaner.clean_document("Short")
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "too_short")

        # Repetitive
        repetitive = "a" * 30
        cleaned, reason = cleaner.clean_document(repetitive)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "repetitive")

        # Normal valid document
        valid_text = "This is a clean and valid document for training testing."
        cleaned, reason = cleaner.clean_document(valid_text)
        self.assertIsNotNone(cleaned)
        self.assertIsNone(reason)

        # Duplicate document
        cleaned_dup, reason_dup = cleaner.clean_document(valid_text)
        self.assertIsNone(cleaned_dup)
        self.assertEqual(reason_dup, "duplicate")

    def test_pipeline_sharding_and_loading(self) -> None:
        raw_file = self.temp_dir / "sample.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(
                "Document number one with sufficient length to pass the minimum character filter.\n\n"
                "Document number two discussing machine learning and neural network training loops.\n\n"
                "Document number three about operating systems and data structures in software.\n\n"
                "Document number four detailing the implementation of Transformer attention layers."
            )

        source = DatasetSourceConfig(
            name="test_sample",
            source_type="local",
            path_or_identifier=str(raw_file),
            file_format="txt",
            min_chars=10,
        )

        mix_cfg = DatasetMixConfig(
            max_sequence_length=16,
            train_ratio=0.75,
            val_ratio=0.25,
            sources=[source],
        )

        pipeline = DatasetPipeline(self.tokenizer)
        stats = pipeline.process_and_shard(mix_cfg, self.temp_dir / "shards")

        self.assertGreater(stats["total_documents"], 0)
        self.assertGreater(stats["total_tokens"], 0)

        # Test ShardedTokenDataset loading
        train_shards = list((self.temp_dir / "shards" / "train").glob("*.pt"))
        self.assertGreater(len(train_shards), 0)

        dataset = ShardedTokenDataset(train_shards, seq_length=16)
        self.assertGreater(len(dataset), 0)

        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)
        self.assertEqual(len(item["input_ids"]), 16)
        self.assertEqual(len(item["labels"]), 16)


if __name__ == "__main__":
    unittest.main()
