import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine


class TokenizerConfigTest(unittest.TestCase):

    def test_deepseek_yarn_is_not_rejected_by_qwen_validation(self):
        hf_config = SimpleNamespace(
            architectures=["DeepseekV2ForCausalLM"],
            model_type="deepseek_v2",
            max_position_embeddings=4096,
            rope_scaling={"rope_type": "yarn", "factor": 4.0},
            quantization_config=None,
            dtype=torch.bfloat16,
        )

        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=hf_config,
            ) as load_config:
                config = Config(model_dir, trust_remote_code=True)

        load_config.assert_called_once_with(model_dir, trust_remote_code=True)
        self.assertEqual(config.runtime_dtype, torch.bfloat16)
        self.assertEqual(config.deepseek_mla_backend, "latent")
        self.assertEqual(hf_config.deepseek_mla_backend, "latent")

    def test_deepseek_mla_backend_is_forwarded_and_validated(self):
        hf_config = SimpleNamespace(max_position_embeddings=128)
        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=hf_config,
            ):
                config = Config(
                    model_dir,
                    deepseek_mla_backend="EXPANDED",
                )
        self.assertEqual(config.deepseek_mla_backend, "expanded")
        self.assertEqual(hf_config.deepseek_mla_backend, "expanded")

        with tempfile.TemporaryDirectory() as model_dir:
            with patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(max_position_embeddings=128),
            ):
                with self.assertRaisesRegex(ValueError, "expanded.*latent"):
                    Config(model_dir, deepseek_mla_backend="unknown")

    def test_trust_remote_code_is_explicitly_forwarded_to_tokenizer(self):
        hf_config = SimpleNamespace(max_position_embeddings=128)
        tokenizer = Mock(eos_token_id=2)

        with tempfile.TemporaryDirectory() as model_dir:
            with (
                patch(
                    "nanovllm.config.AutoConfig.from_pretrained",
                    return_value=hf_config,
                ),
                patch("nanovllm.engine.llm_engine.ModelRunner"),
                patch("nanovllm.engine.llm_engine.Scheduler"),
                patch(
                    "nanovllm.engine.llm_engine.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
                patch("nanovllm.engine.llm_engine.atexit.register"),
            ):
                LLMEngine(model_dir, trust_remote_code=True)

        load_tokenizer.assert_called_once_with(
            model_dir,
            use_fast=True,
            trust_remote_code=True,
        )


if __name__ == "__main__":
    unittest.main()
