import unittest
from types import SimpleNamespace

from nanovllm.models import get_model_class
from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM


class ModelRegistryTest(unittest.TestCase):

    def test_selects_model_by_architecture(self):
        config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            model_type="unknown",
        )

        self.assertIs(get_model_class(config), Qwen3ForCausalLM)

    def test_selects_qwen2_model_implementation(self):
        config = SimpleNamespace(
            architectures=["Qwen2ForCausalLM"],
            model_type="qwen2",
        )

        self.assertIs(get_model_class(config), Qwen2ForCausalLM)

    def test_selects_llama_by_architecture(self):
        config = SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
        )

        self.assertIs(get_model_class(config), LlamaForCausalLM)

    def test_selects_llama_by_model_type(self):
        config = SimpleNamespace(architectures=None, model_type="llama")

        self.assertIs(get_model_class(config), LlamaForCausalLM)

    def test_accepts_single_architecture_string(self):
        config = SimpleNamespace(
            architectures="Qwen3ForCausalLM",
            model_type=None,
        )

        self.assertIs(get_model_class(config), Qwen3ForCausalLM)

    def test_falls_back_to_model_type(self):
        config = SimpleNamespace(
            architectures=["UnregisteredForCausalLM"],
            model_type="qwen3",
        )

        self.assertIs(get_model_class(config), Qwen3ForCausalLM)

    def test_selects_deepseek_by_architecture(self):
        config = SimpleNamespace(
            architectures=["DeepseekV2ForCausalLM"],
            model_type="deepseek_v2",
        )

        self.assertIs(get_model_class(config), DeepseekV2ForCausalLM)

    def test_selects_deepseek_by_model_type(self):
        config = SimpleNamespace(architectures=None, model_type="deepseek_v2")

        self.assertIs(get_model_class(config), DeepseekV2ForCausalLM)

    def test_unsupported_model_has_actionable_error(self):
        config = SimpleNamespace(
            architectures=["UnknownForCausalLM"],
            model_type="unknown",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"UnknownForCausalLM.*model_type='unknown'.*"
            r"DeepseekV2ForCausalLM.*Qwen3ForCausalLM.*deepseek_v2.*qwen3",
        ):
            get_model_class(config)


if __name__ == "__main__":
    unittest.main()
