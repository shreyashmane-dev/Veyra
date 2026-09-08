import unittest
from pathlib import Path

from veyra.config.dataset_config import DatasetMixConfig
from veyra.config.model_config import ModelConfig
from veyra.config.training_config import TrainingConfig


class TestModelConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        config = ModelConfig()
        self.assertEqual(config.model_name, "veyra-tiny")
        self.assertEqual(config.head_dim, 32)
        params = config.calculate_parameter_count()
        self.assertGreater(params["total"], 1_000_000)
        self.assertLess(params["total"], 10_000_000)

    def test_invalid_heads(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=256, num_heads=7)

    def test_load_yaml_configs(self) -> None:
        configs_dir = Path("configs/models")
        tiny = ModelConfig.from_yaml(configs_dir / "veyra_tiny.yaml")
        self.assertEqual(tiny.hidden_size, 128)

        m125 = ModelConfig.from_yaml(configs_dir / "veyra_125m.yaml")
        params125 = m125.calculate_parameter_count()["total"]
        # Should be between 100M and 150M
        self.assertTrue(100_000_000 <= params125 <= 150_000_000, f"Got {params125}")

        m350 = ModelConfig.from_yaml(configs_dir / "veyra_350m.yaml")
        params350 = m350.calculate_parameter_count()["total"]
        # Should be between 300M and 400M
        self.assertTrue(300_000_000 <= params350 <= 400_000_000, f"Got {params350}")

        m1b = ModelConfig.from_yaml(configs_dir / "veyra_1b.yaml")
        params1b = m1b.calculate_parameter_count()["total"]
        # Should be between 900M and 1.3B (1B class, similar to Llama-3.2 1B at 1.23B)
        self.assertTrue(900_000_000 <= params1b <= 1_300_000_000, f"Got {params1b}")

    def test_training_config_yaml(self) -> None:
        t_cfg = TrainingConfig.from_yaml("configs/training/pretrain_tiny.yaml")
        self.assertEqual(t_cfg.batch_size, 2)
        self.assertEqual(t_cfg.gradient_accumulation_steps, 2)

    def test_dataset_config_yaml(self) -> None:
        d_cfg = DatasetMixConfig.from_yaml("configs/datasets/starter_corpus.yaml")
        self.assertEqual(len(d_cfg.sources), 1)
        self.assertEqual(d_cfg.sources[0].name, "veyra_bootstrap")


if __name__ == "__main__":
    unittest.main()
