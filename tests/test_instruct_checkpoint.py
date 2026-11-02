import unittest
from pathlib import Path
import torch

from veyra.checkpoints.manager import CheckpointManager
from veyra.config.model_config import ModelConfig
from veyra.inference.engine import VeyraInferenceEngine
from veyra.memory.database import MemoryDB
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer
from veyra.app import get_active_engine as get_veyra_engine
from zonareth.app import get_active_engine as get_zonareth_engine


class TestInstructCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.veyra_dir = Path("checkpoints/veyra_instruct_125m")
        cls.zonareth_dir = Path("checkpoints/zonareth_instruct_125m")

    def test_checkpoint_files_exist(self) -> None:
        for d in [self.veyra_dir, self.zonareth_dir]:
            self.assertTrue(d.exists(), f"Directory not found: {d}")
            self.assertTrue((d / "model.pt").exists(), f"model.pt missing in {d}")
            self.assertTrue((d / "model_config.yaml").exists(), f"model_config.yaml missing in {d}")
            self.assertTrue((d / "tokenizer" / "vocab.json").exists(), f"vocab.json missing in {d}")
            self.assertTrue((d / "tokenizer" / "merges.json").exists(), f"merges.json missing in {d}")

    def test_load_checkpoint_and_weights(self) -> None:
        model, payload = CheckpointManager.load_checkpoint(self.veyra_dir / "model.pt", device="cpu")
        self.assertIsInstance(model, VeyraLM)
        self.assertIn(model.config.vocab_size, (796, 8192))
        self.assertEqual(model.config.hidden_size, 768)
        self.assertEqual(model.config.num_layers, 12)
        self.assertEqual(model.config.num_heads, 12)
        self.assertTrue(model.config.is_instruct)
        self.assertGreater(model.get_num_params(), 80_000_000)

    def test_tokenizer_encode_decode(self) -> None:
        tok = VeyraTokenizer.load(self.veyra_dir / "tokenizer")
        self.assertIn(tok.vocab_size, (796, 8192))

        text = "Hello, VEYRA AI Assistant!"
        ids = tok.encode(text, add_bos=True)
        self.assertTrue(len(ids) > 0)
        self.assertEqual(ids[0], tok.bos_token_id)

        decoded = tok.decode(ids, skip_special_tokens=True)
        self.assertIn("Hello", decoded)

    def test_model_forward_pass(self) -> None:
        model, _ = CheckpointManager.load_checkpoint(self.veyra_dir / "model.pt", device="cpu")
        model.eval()

        input_ids = torch.tensor([[2, 4, 100, 101, 5, 41]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits

        self.assertEqual(logits.shape, (1, 6, model.config.vocab_size))
        self.assertFalse(torch.isnan(logits).any())

    def test_inference_generation(self) -> None:
        engine = VeyraInferenceEngine.from_checkpoint(
            self.veyra_dir / "model.pt",
            self.veyra_dir / "tokenizer",
            device="cpu",
            temperature=0.3,
            max_new_tokens=15,
        )
        response = engine.generate("Explain virtual memory in one sentence.", context=[])
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_app_engine_auto_detection(self) -> None:
        db_path = Path("checkpoints/test_mem.db")
        memory = MemoryDB(db_path)
        try:
            v_engine, v_desc, v_model = get_veyra_engine(memory)
            self.assertIn("125m", v_desc.lower())
            self.assertIn("veyra_instruct_125m", v_model.lower())

            z_engine, z_desc, z_model = get_zonareth_engine(memory)
            self.assertIn("125m", z_desc.lower())
            self.assertIn("zonareth_instruct_125m", z_model.lower())
        finally:
            memory.close()
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    unittest.main()
