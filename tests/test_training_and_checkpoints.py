import shutil
import tempfile
import unittest
from pathlib import Path
import torch

from veyra.checkpoints.manager import CheckpointManager
from veyra.config.model_config import ModelConfig
from veyra.config.training_config import TrainingConfig
from veyra.data.pipeline import ShardedTokenDataset
from veyra.evaluation.evaluator import ModelEvaluator
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer
from veyra.training.trainer import Trainer


class TestTrainingAndCheckpoints(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = ModelConfig(
            model_name="test-micro",
            vocab_size=64,
            context_length=16,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            ffn_size=88,
            tie_embeddings=True,
        )
        self.model = VeyraLM(self.config)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_save_and_load(self) -> None:
        mgr = CheckpointManager(self.temp_dir / "ckpts")
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        save_path = mgr.save_checkpoint(
            model=self.model,
            optimizer=opt,
            step=42,
            loss=2.345,
            model_config=self.config,
            tag="test_step.pt",
        )
        self.assertTrue(save_path.exists())

        # Inspect checkpoint
        info = mgr.inspect_checkpoint(save_path)
        self.assertEqual(info["step"], 42)
        self.assertAlmostEqual(info["loss"], 2.345, places=3)

        # Load checkpoint
        loaded_model, payload = mgr.load_checkpoint(save_path)
        self.assertEqual(payload["step"], 42)
        # Compare weights
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # List checkpoints
        ckpts = mgr.list_checkpoints()
        self.assertEqual(len(ckpts), 1)

    def test_micro_training_and_eval(self) -> None:
        # Create synthetic shard
        shard_path = self.temp_dir / "shard.pt"
        # 16 sequences of length 17 (16 tokens + 1 target offset)
        dummy_tokens = torch.randint(0, self.config.vocab_size, (16, 17))
        torch.save(dummy_tokens, shard_path)

        dataset = ShardedTokenDataset([shard_path], seq_length=16)

        train_cfg = TrainingConfig(
            run_name="test-run",
            output_dir=str(self.temp_dir / "ckpts"),
            device="cpu",
            batch_size=4,
            gradient_accumulation_steps=1,
            max_steps=5,
            eval_interval=5,
            save_interval=5,
            log_interval=1,
            learning_rate=1e-3,
            warmup_steps=1,
        )

        trainer = Trainer(
            model=self.model,
            training_config=train_cfg,
            train_dataset=dataset,
            val_dataset=dataset,
        )

        results = trainer.train()
        self.assertEqual(results["final_step"], 5)
        self.assertTrue(Path(results["latest_checkpoint"]).exists())

        # Evaluation
        tok = VeyraTokenizer(vocab_size=64)
        evaluator = ModelEvaluator(self.model, tok, device="cpu")
        eval_metrics = evaluator.evaluate_loss_and_perplexity(dataset, batch_size=4)
        self.assertIn("loss", eval_metrics)
        self.assertIn("perplexity", eval_metrics)


if __name__ == "__main__":
    unittest.main()
