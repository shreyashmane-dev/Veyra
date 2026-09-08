import math
import unittest
import torch

from veyra.config.model_config import ModelConfig
from veyra.model.attention import CausalSelfAttention
from veyra.model.embeddings import apply_rotary_emb, precompute_rope_frequencies
from veyra.model.generation import GenerationEngine, sample_next_token
from veyra.model.mlp import SwiGLUFeedForward
from veyra.model.model import VeyraLM
from veyra.model.normalization import RMSNorm


class TestVeyraModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.tiny_config = ModelConfig(
            model_name="test-tiny",
            vocab_size=128,
            context_length=32,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            ffn_size=172,
            activation="swiglu",
            normalization="rmsnorm",
            tie_embeddings=True,
        )

    def test_rmsnorm(self) -> None:
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Verify RMS of output is approximately 1.0 (with weight=1.0)
        rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_rope_frequencies_and_rotation(self) -> None:
        cos, sin = precompute_rope_frequencies(dim=16, max_seq_len=32)
        self.assertEqual(cos.shape, (32, 8))
        self.assertEqual(sin.shape, (32, 8))

        q = torch.randn(2, 4, 10, 16)
        rotated = apply_rotary_emb(q, cos, sin)
        self.assertEqual(rotated.shape, q.shape)

    def test_causal_masking_property(self) -> None:
        """Crucial test: token at position t must NOT depend on any token at position > t."""
        attn = CausalSelfAttention(self.tiny_config)
        attn.eval()
        cos, sin = precompute_rope_frequencies(
            dim=self.tiny_config.head_dim,
            max_seq_len=self.tiny_config.context_length,
        )

        x1 = torch.randn(1, 4, self.tiny_config.hidden_size)
        x2 = x1.clone()
        # Modify only the last token in x2
        x2[:, -1, :] = torch.randn(1, self.tiny_config.hidden_size)

        out1, _ = attn(x1, cos=cos, sin=sin)
        out2, _ = attn(x2, cos=cos, sin=sin)

        # Output at positions 0, 1, 2 must be EXACTLY identical between x1 and x2
        self.assertTrue(
            torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6),
            "Future tokens leaked into past positions! Causal mask violation.",
        )
        # Output at position 3 must differ
        self.assertFalse(torch.allclose(out1[:, -1, :], out2[:, -1, :]))

    def test_swiglu_mlp(self) -> None:
        mlp = SwiGLUFeedForward(self.tiny_config)
        x = torch.randn(2, 8, self.tiny_config.hidden_size)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_model_forward_and_loss(self) -> None:
        model = VeyraLM(self.tiny_config)
        input_ids = torch.randint(0, self.tiny_config.vocab_size, (2, 16))

        # Forward without labels
        out = model(input_ids)
        self.assertEqual(out.logits.shape, (2, 16, self.tiny_config.vocab_size))
        self.assertIsNone(out.loss)

        # Forward with labels
        labels = torch.randint(0, self.tiny_config.vocab_size, (2, 16))
        out_with_loss = model(input_ids, labels=labels)
        self.assertIsNotNone(out_with_loss.loss)
        self.assertFalse(torch.isnan(out_with_loss.loss))
        self.assertGreater(out_with_loss.loss.item(), 0.0)

        # Test backward pass
        out_with_loss.loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_generation_engine(self) -> None:
        model = VeyraLM(self.tiny_config)
        gen = GenerationEngine(model)

        prompt = [10, 20, 30]
        out_tokens = gen.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.0,  # greedy deterministic
            stop_token_ids=[99],
        )
        self.assertEqual(len(out_tokens), len(prompt) + 5)
        self.assertEqual(out_tokens[:3], prompt)

        # Test stop token behavior (deterministic greedy)
        out_stopped = gen.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            stop_token_ids=[out_tokens[3]],  # first generated token as stop token
        )
        self.assertEqual(len(out_stopped), len(prompt) + 1)


if __name__ == "__main__":
    unittest.main()
