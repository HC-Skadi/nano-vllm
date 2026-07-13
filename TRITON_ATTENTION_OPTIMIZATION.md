# nano-vLLM Triton Attention 精度与性能优化记录

本文记录 nano-vLLM 自定义 Triton Attention 的适配、正确性修复、性能测试方法、优化结果和后续方向。

## 1. 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6 GB |
| 模型 | Qwen3-0.6B |
| 数据类型 | BF16 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| FlashAttention | 2.8.3 |
| Q heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| KV cache block size | 256 |

## 2. 实现场景

自定义实现覆盖三条 Attention 路径。

### 2.1 Packed Prefill

用于首次处理没有前缀缓存的完整 prompt。不同请求的 Q/K/V 按 token 维拼接，通过 `cu_seqlens_q` 和 `cu_seqlens_k` 描述序列边界。

```text
Q/K/V: [total_tokens, num_heads, head_dim]
Q length = K length
causal = True
```

该路径主要影响首 token 延迟 TTFT。

### 2.2 Prefix Prefill

用于已有前缀 KV cache、只计算新增 prompt token 的场景。

```text
Q: packed new tokens
K/V: paged KV cache
Q length < K/V length
logical block → block_table → physical block
```

该路径用于 prefix caching 和 chunked prefill。

### 2.3 Paged Decode

用于自回归生成，每个序列每一步只有一个 query token。

```text
Q length = 1
K/V length = current context length
K/V stored in paged cache
```

该路径主要影响每输出 token 时间 TPOT。

## 3. 已修复的正确性问题

### 3.1 模型路径不存在

最初示例指定：

```text
~/huggingface/Qwen3-0.6B/
```

目录不存在时，Transformers 将该路径误判为 Hugging Face repo ID。模型已下载至：

```text
models/Qwen3-0.6B
```

示例改为基于脚本目录解析模型路径。

### 3.2 BF16/FP16 类型不匹配

旧 Triton kernel 将 softmax probability 固定转换为 FP16：

```python
p.to(tl.float16)
```

Qwen3-0.6B 使用 BF16，导致 `tl.dot` 两侧分别为 FP16 和 BF16。修复为根据 V 的数据类型转换：

```python
p.to(v.dtype)
```

### 3.3 完全 masked block 产生 NaN

当某个 causal query 在当前 K block 中没有任何合法 key 时，旧实现会计算：

```text
exp(-inf - -inf) = NaN
```

NaN 随后污染在线 softmax accumulator。修复方式是显式构造 `score_mask`，masked probability 直接置零：

```python
p = tl.where(score_mask, tl.exp(scores - m_new[:, None]), 0.0)
```

### 3.4 在线 softmax 数值稳定性

旧实现每处理一个 K tile 都对 accumulator 归一化。现改为标准的未归一化 FP32 在线 softmax：

```text
m_new = max(m_old, block_max)
alpha = exp(m_old - m_new)
p = exp(scores - m_new)
acc = acc * alpha + p @ V
l = l * alpha + sum(p)
output = acc / l
```

整个循环只在最终写回时除以 softmax denominator。

### 3.5 非连续 V stride

这是造成模型输出出现 `LEELEE...`、错误 HTML 和异常首 token 的主要原因。

框架中的 V 来自：

```python
qkv.split(...).view(...)
```

V 的最后一维连续，但 token stride 仍包含完整 QKV projection width，因此 V 不是连续 tensor。旧 packed kernel 假设：

```text
token stride = num_kv_heads * head_dim
```

导致读取了错误的 V 地址。同一 prompt 的首 token logits 对比为：

```text
Triton top token: 'itionally'
FlashAttention top token: '<think>'
最大 logits 差异: 22.40625
```

修复后，packed kernel 显式接收并使用：

```python
q.stride(0), q.stride(1)
k.stride(0), k.stride(1)
v.stride(0), v.stride(1)
```

并加入非连续 V 的回归测试。修复后完整模型输出恢复正常。

## 4. M=1 Decode 优化

### 4.1 原 Tensor Core 方案

Triton 3.2 的 `tl.dot` 要求矩阵维度至少为 16，但 decode 的真实 query 数量为 1。旧实现把同一个 query 复制为 16 行：

```text
真实 Q: [1, head_dim]
计算 Q: [16, head_dim]
```

只写回第 0 行，但 16 行都会执行 QK、softmax 和 PV，产生大量冗余计算。

### 4.2 专用 M=1 向量 kernel

新增单 query 向量 kernel，不再使用 `tl.dot`：

```python
scores = tl.sum(k * q[None, :], axis=1)
pv = tl.sum(p[:, None] * v, axis=0)
```

实现特征：

- 每个 program 处理一个 `(batch, q_head)`。
- Q/K/V 转 FP32 后参与向量乘加。
- 在线 softmax 和 output accumulator 使用 FP32。
- 支持 GQA、paged cache、非连续物理 block 和跨 block context。

### 4.3 配置扫描

扫描空间：

```text
BLOCK_N = 16, 32, 64
num_warps = 4, 8
```

代表性结果：

| Batch | KV length | B_N=16/W4 | B_N=32/W4 | B_N=64/W4 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.1062 ms | 0.0655 ms | 0.0572 ms |
| 1 | 512 | 0.0647 ms | 0.0590 ms | 0.0591 ms |
| 1 | 1024 | 0.1447 ms | 0.0891 ms | 0.0574 ms |
| 1 | 2048 | 0.2096 ms | 0.1242 ms | 0.0877 ms |
| 8 | 1024 | 0.2877 ms | 0.1799 ms | 0.1223 ms |

最终使用：

```python
BLOCK_N = 64
num_warps = 4
```

### 4.4 当前分派策略

低 batch 使用 M=1 vector kernel，高 batch 保留 Tensor Core kernel：

```python
if batch < 16:
    use_vector_decode_kernel()
else:
    use_tensorcore_decode_kernel()
```

### 4.5 融合 KV 写入与 Decode

原 decode 路径需要两个独立 kernel launch：

```text
store_kvcache → paged_decode_attention
```

现已将本轮 `new_k/new_v` 写入融合到 vector 和 Tensor Core decode kernel。为避免依赖不同 Triton program 之间不可用的全局同步，融合 kernel 采用：

1. 当前最后一个 KV position 直接从 `new_k/new_v` 输入读取并参与 attention。
2. 每个 KV head 选择一个 Q-head program，将相同数据写入 `slot_mapping` 指定的物理 cache slot。
3. 历史 token 继续通过 block table 从 paged KV cache 读取。
4. kernel 结束后，新 KV 已持久化，供下一 decode step 使用。

框架 decode 路径不再单独调用 `store_kvcache`，prefill 路径仍保留原批量 KV 写入 kernel。

新增融合回归测试覆盖：

- cache 初始不包含当前 token；
- `new_v` 为非连续 stride；
- 融合输出与“独立 store + FlashAttention decode”比较；
- 融合后的 K/V cache 与独立 store 逐元素完全一致。

融合性能测试脚本：

```bash
python bench_fused_decode.py
```

该脚本比较相同 Triton Attention 下的 separate 与 fused 延迟，覆盖 B=1/8/16/32 和 KV=128/1024。

## 5. 正确性测试

测试文件：

```text
tests/test_triton_attention.py
```

覆盖：

1. BF16 + GQA packed prefill。
2. 非连续 V stride。
3. prefix-cache paged prefill。
4. 300-token 跨物理 KV block decode。
5. `store_kvcache → decode` 完整链路。

执行命令：

```bash
python -m unittest -v tests/test_triton_attention.py
```

当前结果：

```text
test_decode_paged_gqa_bf16 ... ok
test_fused_store_kvcache_and_decode ... ok
test_store_kvcache_then_decode ... ok
test_varlen_packed_gqa_bf16 ... ok
test_varlen_prefix_cache_gqa_bf16 ... ok

Ran 5 tests
OK
```

经典矩阵中的最大绝对误差约为：

```text
0.000488281 ～ 0.0078125
```

平均绝对误差约为：

```text
3.8e-5 ～ 1.6e-4
```

误差处于 BF16 量化范围内。

## 6. 基准测试方法

### 6.1 初始单点基准的局限

初始测试混合 `[128, 512]` 等不同序列长度，并固定先测 FlashAttention、后测 Triton，只得到一个平均延迟。该方法不能展示长度和 batch 拐点，也可能受到 GPU 动态频率、cache 和测试顺序影响。

### 6.2 经典测试方法

经典矩阵脚本：

```text
bench_attention_classic.py
```

每个点采用：

```text
同质 batch
预热 5 次
每轮重复 50 次
测试 5 轮
交替后端执行顺序
报告轮次延迟中位数
```

执行命令：

```bash
python bench_attention_classic.py \
  --warmup 5 \
  --repeats 50 \
  --rounds 5
```

## 7. 最终经典性能结果

### 7.1 Packed Prefill

| Batch | Q | KV | FlashAttention | Triton | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 0.1203 ms | 0.0697 ms | 1.73x |
| 1 | 512 | 512 | 0.1259 ms | 0.2303 ms | 0.55x |
| 1 | 1024 | 1024 | 0.2493 ms | 0.6924 ms | 0.36x |
| 1 | 2048 | 2048 | 0.8298 ms | 2.6489 ms | 0.31x |
| 8 | 128 | 128 | 0.1201 ms | 0.1146 ms | 1.05x |

结论：当前固定 `BLOCK_M=16, BLOCK_N=32` 适合短 prefill，但长序列会产生过多 Q tile、K 循环和 K/V 重复读取。

### 7.2 Prefix Prefill

| Batch | Q | KV | FlashAttention | Triton | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1024 | 0.1706 ms | 0.1100 ms | 1.55x |
| 1 | 64 | 1024 | 0.1664 ms | 0.1087 ms | 1.53x |
| 1 | 256 | 1024 | 0.1393 ms | 0.1919 ms | 0.73x |
| 8 | 16 | 1024 | 0.2477 ms | 0.1647 ms | 1.50x |

结论：专用 paged prefix kernel 在短 Q、长 KV 时有稳定优势；Q 增大到 256 后固定小 tile 开始落后。

### 7.3 Paged Decode

| Batch | Q | KV | FlashAttention | Triton | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 128 | 0.0379 ms | 0.0808 ms | 0.47x |
| 1 | 1 | 512 | 0.0367 ms | 0.0818 ms | 0.45x |
| 1 | 1 | 1024 | 0.0404 ms | 0.0855 ms | 0.47x |
| 1 | 1 | 2048 | 0.0481 ms | 0.0903 ms | 0.53x |
| 8 | 1 | 1024 | 0.1494 ms | 0.1251 ms | 1.19x |
| 16 | 1 | 1024 | 0.2641 ms | 0.2603 ms | 1.01x |
| 32 | 1 | 1024 | 0.5089 ms | 0.4658 ms | 1.09x |

结论：

- M=1 vector kernel 明显降低了自定义 Triton 的低 batch 延迟。
- Batch 1 仍受 kernel launch、output allocation、paged 地址计算和 FP32 reduction 影响，未超过成熟的 FlashAttention。
- Batch 8 时 Triton 快 19%。
- Batch 16 基本持平。
- Batch 32 使用 Tensor Core 路径，Triton 快 9%。

## 8. M=1 优化前后

以下对比均为 Triton 自身优化前后：

| Batch | KV | 复制 16 行 Q | M=1 vector | 延迟降低 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.1283 ms | 0.0808 ms | 37% |
| 1 | 512 | 0.1204 ms | 0.0818 ms | 32% |
| 1 | 1024 | 0.1517 ms | 0.0855 ms | 44% |
| 1 | 2048 | 0.1564 ms | 0.0903 ms | 42% |
| 8 | 1024 | 0.1692 ms | 0.1251 ms | 26% |

## 9. 完整模型结果

在 M=1 优化前，16 个请求、每请求固定输出 128 tokens 的 Qwen3-0.6B 结果为：

| 后端 | 总输出 tokens | 时间 | 吞吐 |
|---|---:|---:|---:|
| FlashAttention | 2048 | 8.1582 s | 251.04 tok/s |
| Triton | 2048 | 8.2917 s | 246.99 tok/s |

由于 batch 为 16，当前分派会走 Tensor Core decode 路径，因此 M=1 优化主要改善 batch 小于 16 的在线推理负载，不应使用上述 B=16 结果评价 vector kernel。

## 10. 当前结论

不能简单判断 Triton 全局更快或更慢，必须按 shape 分析：

```text
短 packed prefill：Triton 有优势
长 packed prefill：FlashAttention 明显更快
短 Q prefix prefill：Triton 有稳定优势
大 Q prefix prefill：FlashAttention 更快
低 batch decode：M=1 优化显著，但 B=1 仍落后 FlashAttention
中高 batch decode：Triton 持平或更快
```

当前最合理的工程策略是 shape-aware dispatch，而不是所有 shape 强制使用同一个 kernel。

## 11. 后续优化方向

### 11.1 Packed Prefill

1. 按序列长度选择 `BLOCK_M/BLOCK_N`。
2. 引入 Triton autotune。
3. 对 causal K 循环做上界裁剪。
4. 长序列增大 tile，减少 K/V 重复加载。
5. 调整 `num_warps` 和 `num_stages`。

### 11.2 Prefix Prefill

1. Q 小于等于 64 时保留当前 Triton kernel。
2. Q 较大时使用更大 `BLOCK_M` 或回退 FlashAttention。
3. 针对固定 KV block size 256 特化地址计算。

### 11.3 Decode

1. 减少 Python wrapper 和 output allocation 开销。
2. 将 KV 写入与 decode attention 融合。
3. 按 context length 选择 `BLOCK_N`。
4. 长 context 引入 split-K，两阶段合并局部 softmax。
5. 进一步减少 block table 和物理地址重复计算。

### 11.4 测试体系

后续应继续增加：

- FP32 PyTorch reference。
- FP16 输入。
- 更多 head dimension 和 GQA 比例。
- 混合长度动态 batch。
- TTFT、TPOT、P50/P95/P99。
- 峰值显存和实际带宽。
- CUDA Graph 模式。

## 12. 相关文件

| 文件 | 用途 |
|---|---|
| `nanovllm/layers/attention.py` | Triton Attention 实现和运行时分派 |
| `tests/test_triton_attention.py` | 正确性回归测试 |
| `bench_attention.py` | 初始代表性 kernel 对比 |
| `bench_attention_classic.py` | 经典 shape 性能矩阵 |
| `bench_decode_vector_configs.py` | M=1 vector 配置扫描 |
| `bench_fused_decode.py` | 独立 KV 写入与融合 Decode 性能对比 |
| `bench_model_backend.py` | 完整模型后端 A/B |
