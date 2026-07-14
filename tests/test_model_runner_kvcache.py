import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.engine.model_runner import ModelRunner


class CacheModule(nn.Module):

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_count: int = 2,
    ):
        super().__init__()
        # Simulate an AWQ module whose first stored parameter is packed INT32.
        self.packed_weight = nn.Parameter(
            torch.empty(1, dtype=torch.int32),
            requires_grad=False,
        )
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_cache_count = kv_cache_count
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])


class CacheModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = CacheModule(1, 4)
        self.second = CacheModule(2, 8)
        self.latent = CacheModule(1, 12, kv_cache_count=1)


class ModelRunnerKVCacheTest(unittest.TestCase):

    def test_runtime_dtype_overrides_packed_parameter_dtype(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            gpu_memory_utilization=0.5,
            num_kvcache_blocks=-1,
            runtime_dtype=torch.float16,
        )
        runner.block_size = 256
        runner.world_size = 1
        runner.model = CacheModel()

        memory_stats = {
            "allocated_bytes.all.peak": 0,
            "allocated_bytes.all.current": 0,
        }
        with (
            patch("torch.cuda.mem_get_info", return_value=(200_000, 200_000)),
            patch("torch.cuda.memory_stats", return_value=memory_stats),
        ):
            runner.allocate_kv_cache()

        self.assertGreater(runner.config.num_kvcache_blocks, 0)
        self.assertEqual(runner.config.num_kvcache_blocks, 3)
        self.assertEqual(len(runner.kv_cache), 3)
        for module in (runner.model.first, runner.model.second):
            self.assertEqual(module.k_cache.dtype, torch.float16)
            self.assertEqual(module.v_cache.dtype, torch.float16)
            self.assertEqual(module.k_cache.shape[0], runner.config.num_kvcache_blocks)
        self.assertEqual(runner.model.latent.k_cache.dtype, torch.float16)
        self.assertEqual(runner.model.latent.v_cache.dtype, torch.float16)
        self.assertEqual(runner.model.latent.v_cache.numel(), 0)
        self.assertEqual(
            runner.model.latent.k_cache.shape,
            (runner.config.num_kvcache_blocks, 256, 1, 12),
        )

    def test_explicit_cache_block_count_is_respected(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            gpu_memory_utilization=0.5,
            num_kvcache_blocks=2,
            runtime_dtype=torch.float16,
        )
        runner.block_size = 256
        runner.world_size = 1
        runner.model = CacheModel()
        memory_stats = {
            "allocated_bytes.all.peak": 0,
            "allocated_bytes.all.current": 0,
        }
        with (
            patch("torch.cuda.mem_get_info", return_value=(200_000, 200_000)),
            patch("torch.cuda.memory_stats", return_value=memory_stats),
        ):
            runner.allocate_kv_cache()
        self.assertEqual(runner.config.num_kvcache_blocks, 2)


if __name__ == "__main__":
    unittest.main()
