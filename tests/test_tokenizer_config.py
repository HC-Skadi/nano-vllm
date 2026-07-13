import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nanovllm.engine.llm_engine import LLMEngine


class TokenizerConfigTest(unittest.TestCase):

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
