import unittest
import torch

from veyra.config.model_config import ModelConfig
from veyra.inference.engine import VeyraInferenceEngine
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer


class TestInferenceEngine(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = ModelConfig(
            model_name="test-inf",
            vocab_size=350,
            context_length=64,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            ffn_size=172,
            tie_embeddings=True,
        )
        self.model = VeyraLM(self.config)
        self.tokenizer = VeyraTokenizer(vocab_size=350)
        self.tokenizer.train_from_texts([
            "Hello VEYRA",
            "This is a test prompt for inference engine.",
            "Testing response generation.",
        ])

    def test_format_prompt(self) -> None:
        engine = VeyraInferenceEngine(self.model, self.tokenizer)
        context = [("user", "Hello"), ("assistant", "Greetings")]
        prompt = engine.format_prompt("How are you?", context)

        self.assertIn("<SYSTEM>", prompt)
        self.assertIn("<USER> Hello", prompt)
        self.assertIn("<ASSISTANT> Greetings", prompt)
        self.assertIn("<USER> How are you? <ASSISTANT>", prompt)

    def test_generation_and_streaming(self) -> None:
        engine = VeyraInferenceEngine(
            self.model,
            self.tokenizer,
            max_new_tokens=10,
            temperature=0.0,
        )

        streamed_pieces: list[str] = []
        response = engine.generate(
            "What is VEYRA?",
            context=[],
            callback=lambda p: streamed_pieces.append(p),
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertGreater(len(streamed_pieces), 0)


if __name__ == "__main__":
    unittest.main()
