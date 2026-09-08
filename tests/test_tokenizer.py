import shutil
import tempfile
import unittest
from pathlib import Path

from veyra.tokenizer.bpe import VeyraTokenizer


class TestVeyraTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_corpus = [
            "Hello, VEYRA! VEYRA is a serious personal artificial intelligence system.",
            "VEYRA-LM is trained from scratch with its own tokenizer and neural architecture.",
            "Tokenization uses byte-pair encoding to represent words and subwords efficiently.",
            "Testing special tokens: <USER> Hello! <ASSISTANT> Greetings, Sensei. <TOOL> read_file",
            "Unicode and code: def add(x: int, y: int) -> int: return x + y  # 🚀 α + β = γ",
        ]

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_base_vocab(self) -> None:
        tok = VeyraTokenizer(vocab_size=512)
        # Should have special tokens (9) + 256 byte tokens = 265 minimum
        self.assertEqual(tok.vocab_size, 265)
        self.assertEqual(tok.pad_token_id, 0)
        self.assertEqual(tok.unk_token_id, 1)
        self.assertEqual(tok.bos_token_id, 2)
        self.assertEqual(tok.eos_token_id, 3)

    def test_train_and_encode_decode(self) -> None:
        tok = VeyraTokenizer(vocab_size=350)
        tok.train_from_texts(self.sample_corpus, min_frequency=2)
        self.assertGreater(tok.vocab_size, 265)

        for text in self.sample_corpus:
            ids = tok.encode(text)
            self.assertIsInstance(ids, list)
            self.assertTrue(all(isinstance(i, int) for i in ids))
            # Roundtrip decoding should match exactly (without adding bos/eos)
            decoded = tok.decode(ids)
            self.assertEqual(decoded, text)

    def test_special_tokens_handling(self) -> None:
        tok = VeyraTokenizer(vocab_size=300)
        text = "<USER> What is VEYRA? <ASSISTANT>"
        ids = tok.encode(text)
        self.assertIn(tok.user_token_id, ids)
        self.assertIn(tok.assistant_token_id, ids)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, text)

        # Decode skipping special tokens
        decoded_no_spec = tok.decode(ids, skip_special_tokens=True)
        self.assertNotIn("<USER>", decoded_no_spec)
        self.assertNotIn("<ASSISTANT>", decoded_no_spec)
        self.assertIn("What is VEYRA?", decoded_no_spec)

    def test_zero_oov_arbitrary_unicode(self) -> None:
        tok = VeyraTokenizer(vocab_size=300)
        unseen = "Unseen text with Chinese: 你好, Japanese: こんにちは, Arabic: مرحبا, Math: ∑(x^2), Emoji: 🧠⚡"
        ids = tok.encode(unseen)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, unseen)

    def test_batch_encoding(self) -> None:
        tok = VeyraTokenizer(vocab_size=300)
        batch = ["Short", "This is a longer sentence for batch testing."]
        res = tok.encode_batch(batch, max_length=12, padding=True, truncation=True)
        self.assertEqual(len(res["input_ids"]), 2)
        self.assertEqual(len(res["attention_mask"]), 2)
        self.assertEqual(len(res["input_ids"][0]), 12)
        self.assertEqual(len(res["input_ids"][1]), 12)
        # First one should be padded with pad_token_id (0)
        self.assertIn(tok.pad_token_id, res["input_ids"][0])
        self.assertEqual(res["attention_mask"][0][-1], 0)

    def test_save_and_load(self) -> None:
        tok = VeyraTokenizer(vocab_size=320)
        tok.train_from_texts(self.sample_corpus, min_frequency=2)
        tok.save(self.temp_dir)

        loaded_tok = VeyraTokenizer.load(self.temp_dir)
        self.assertEqual(tok.vocab_size, loaded_tok.vocab_size)
        self.assertEqual(tok.merges, loaded_tok.merges)

        sample = "VEYRA-LM test sentence."
        self.assertEqual(tok.encode(sample), loaded_tok.encode(sample))
        self.assertEqual(tok.decode(tok.encode(sample)), loaded_tok.decode(loaded_tok.encode(sample)))


if __name__ == "__main__":
    unittest.main()
