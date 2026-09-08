import shutil
import tempfile
import unittest
from pathlib import Path
import torch

from veyra.config.model_config import ModelConfig
from veyra.data.instruction import InstructionDataset
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer
from veyra.training.sft import SFTTrainer


class TestInstructionTuning(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tokenizer = VeyraTokenizer(vocab_size=350)
        self.tokenizer.train_from_texts([
            "Hello VEYRA",
            "Explain binary search algorithm in Python.",
            "Binary search divides search space by half.",
        ])

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_prompt_loss_masking(self) -> None:
        samples = [
            {
                "prompt": "What is binary search?",
                "response": "Binary search is an efficient divide and conquer algorithm.",
            }
        ]
        dataset = InstructionDataset(samples, self.tokenizer, max_length=256)
        self.assertEqual(len(dataset), 1)

        item = dataset[0]
        input_ids = item["input_ids"]
        labels = item["labels"]

        # First tokens (prompt) must be masked with -100 in labels
        self.assertEqual(labels[0].item(), -100)
        # Later tokens (assistant response) must NOT be -100
        non_masked = (labels != -100).nonzero()
        self.assertGreater(len(non_masked), 0)

    def test_sft_micro_train(self) -> None:
        cfg = ModelConfig(
            model_name="test-sft",
            vocab_size=350,
            context_length=256,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            ffn_size=172,
            tie_embeddings=True,
        )
        model = VeyraLM(cfg)

        samples = [
            {"prompt": "Hello", "response": "Greetings! I am VEYRA."},
            {"prompt": "Calculate 2+2", "response": "The answer is 4."},
        ]
        dataset = InstructionDataset(samples, self.tokenizer, max_length=256)

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            output_dir=self.temp_dir / "sft_ckpt",
            learning_rate=1e-4,
            max_steps=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            warmup_steps=1,
            device="cpu",
        )
        res = trainer.train()
        self.assertEqual(res["final_step"], 2)
        self.assertTrue(Path(res["checkpoint"]).exists())


if __name__ == "__main__":
    unittest.main()
