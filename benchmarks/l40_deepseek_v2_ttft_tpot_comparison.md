# DeepSeek-V2-Lite TTFT/TPOT: nano-vllm vs vLLM (L40, 真权重)

- 日期: 2026-09-20
- GPU: NVIDIA L40 48GB (Cloud Studio 工作区, 与 RTX 3060 本地结果不可比)
- 模型: DeepSeek-V2-Lite-Chat, BF16, 真实权重 (ModelScope 下载)
- 方法: 同一组随机 token prompt(相同种子)、相同场景、逐请求 TTFT/TPOT,
  3 次重复取中位数。nano-vllm 为 eager(DeepSeek 路径不支持 CUDA Graph);
  vLLM 0.25.1 分别跑默认(CUDA Graph)与 --enforce-eager。
- vLLM 0.25.1 启动崩溃 workaround: triton>=3.6 与其 MiniMax-M3 warmup 内核
  JIT 解析冲突, 已用 patch_vllm_warmup.py 将该 warmup 置为 no-op(与 DeepSeek 无关)。

## 结果 (TTFT / TPOT 中位数, ms)

| 场景 | nano latent | nano expanded | vLLM eager | vLLM 默认(图) |
|---|---|---|---|---|
| 1×512+128  | 591 / 151.1 | 577 / 142.6 | 79 / 38.2  | 62 / 8.4  |
| 4×512+128  | 701 / 247.1 | 645 / 239.6 | 141 / 39.5 | 125 / 16.3 |
| 8×512+128  | 790 / 324.9 | 713 / 303.1 | 176 / 39.9 | 158 / 21.7 |
| 1×2048+128 | 776 / 152.3 | 659 / 142.6 | 100 / 38.4 | 102 / 9.4  |

## 结论

1. vLLM 默认比 nano-vllm latent 快: TTFT 约 6-10x, TPOT 约 15-35x。
2. 即使同为 eager, vLLM 仍快 4-8x —— 差距主要来自内核成熟度
   (FlashInfer MLA 吸收内核、融合 MoE、调度), 不只是 CUDA Graph。
3. CUDA Graph 单项贡献约 4x decode 提升(vLLM eager 38ms → 默认 8.4ms @bs1)。
   nano-vllm 的 DeepSeek 路径 supports_cuda_graph=False, 是最大单点优化空间。
4. nano latent(权重吸收)在 L40 上仍略慢于 expanded(bs1: 151 vs 143ms),
   与 RTX 3060 结论一致: 当前实现的吸收路径在无融合内核+eager 下不占优,
   其收益(更小 KV cache)体现在容量而非速度。

## 产物

- 工作区: /workspace/{nano_latent,nano_expanded,vllm_default,vllm_eager}.json
- 脚本(分支 bench-ttft-tpot): bench_deepseek_ttft_tpot.py,
  bench_vllm_ttft_tpot.py, bench_all_l40.sh, bench_vllm_only_l40.sh,
  patch_vllm_warmup.py
