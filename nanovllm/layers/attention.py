import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    block_d = 256
    store_kvcache_kernel[(N, triton.cdiv(D, block_d))](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        BLOCK_D=block_d,
    )


@triton.jit
def flash_attn_varlen_packed_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh - batch_id * num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    q_start = tl.load(cu_seqlens_q_ptr + batch_id)
    q_end = tl.load(cu_seqlens_q_ptr + batch_id + 1)
    k_start = tl.load(cu_seqlens_k_ptr + batch_id)
    k_end = tl.load(cu_seqlens_k_ptr + batch_id + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_idxs = q_start + offs_m

    q = tl.load(
        q_ptr + q_idxs[:, None] * q_stride_t + q_head_id * q_stride_h + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, max_seqlen_k, BLOCK_N):
        n_idxs = n_start + offs_n
        k_idxs = k_start + n_idxs
        k = tl.load(
            k_ptr + k_idxs[:, None] * k_stride_t + kv_head_id * k_stride_h + offs_d[None, :],
            mask=(n_idxs[:, None] < k_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_ptr + k_idxs[:, None] * v_stride_t + kv_head_id * v_stride_h + offs_d[None, :],
            mask=(n_idxs[:, None] < k_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        score_mask = n_idxs[None, :] < k_len
        if causal:
            q_pos = offs_m + k_len - q_len
            score_mask = score_mask & (n_idxs[None, :] <= q_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(score_mask, tl.exp(scores - m_new[:, None]), 0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    tl.store(
        out_ptr + (q_idxs[:, None] * num_q_heads + q_head_id) * head_dim + offs_d[None, :],
        acc / l_i[:, None],
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
    )


@triton.jit
def flash_attn_varlen_paged_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    block_table_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    max_num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh - batch_id * num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    q_start = tl.load(cu_seqlens_q_ptr + batch_id)
    q_end = tl.load(cu_seqlens_q_ptr + batch_id + 1)
    k_start = tl.load(cu_seqlens_k_ptr + batch_id)
    k_end = tl.load(cu_seqlens_k_ptr + batch_id + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_idxs = q_start + offs_m

    q = tl.load(
        q_ptr + (q_idxs[:, None] * num_q_heads + q_head_id) * head_dim + offs_d[None, :],
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, max_seqlen_k, BLOCK_N):
        n_idxs = n_start + offs_n
        logical_block_ids = n_idxs // block_size
        block_offsets = n_idxs - logical_block_ids * block_size
        physical_block_ids = tl.load(
            block_table_ptr + batch_id * max_num_blocks + logical_block_ids,
            mask=n_idxs < k_len,
            other=0,
        )

        cache_offsets = (
            ((physical_block_ids[:, None] * block_size + block_offsets[:, None]) * num_kv_heads + kv_head_id)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            k_cache_ptr + cache_offsets,
            mask=(n_idxs[:, None] < k_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr + cache_offsets,
            mask=(n_idxs[:, None] < k_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        score_mask = n_idxs[None, :] < k_len
        if causal:
            q_pos = offs_m + k_len - q_len
            score_mask = score_mask & (n_idxs[None, :] <= q_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(score_mask, tl.exp(scores - m_new[:, None]), 0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    tl.store(
        out_ptr + (q_idxs[:, None] * num_q_heads + q_head_id) * head_dim + offs_d[None, :],
        acc / l_i[:, None],
        mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim),
    )


@triton.jit
def flash_attn_decode_vector_kernel(
    q_ptr,
    new_k_ptr,
    new_v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    slot_mapping_ptr,
    new_k_stride_t: tl.constexpr,
    new_k_stride_h: tl.constexpr,
    new_v_stride_t: tl.constexpr,
    new_v_stride_h: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    HAS_NEW_KV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-query paged attention without duplicating Q for tl.dot."""
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh - batch_id * num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    cache_len = tl.load(cache_seqlens_ptr + batch_id)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + (batch_id * num_q_heads + q_head_id) * head_dim + offs_d,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    if HAS_NEW_KV:
        new_k = tl.load(
            new_k_ptr + batch_id * new_k_stride_t + kv_head_id * new_k_stride_h + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        new_v = tl.load(
            new_v_ptr + batch_id * new_v_stride_t + kv_head_id * new_v_stride_h + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        slot = tl.load(slot_mapping_ptr + batch_id)
        group_size = num_q_heads // num_kv_heads
        cache_write_offsets = slot * num_kv_heads * head_dim + kv_head_id * head_dim + offs_d
        write_mask = (slot >= 0) & (q_head_id % group_size == 0) & (offs_d < head_dim)
        tl.store(k_cache_ptr + cache_write_offsets, new_k, mask=write_mask)
        tl.store(v_cache_ptr + cache_write_offsets, new_v, mask=write_mask)

    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(0, max_num_blocks * block_size, BLOCK_N):
        n_idxs = n_start + offs_n
        logical_block_ids = n_idxs // block_size
        block_offsets = n_idxs - logical_block_ids * block_size
        physical_block_ids = tl.load(
            block_table_ptr + batch_id * max_num_blocks + logical_block_ids,
            mask=n_idxs < cache_len,
            other=0,
        )
        cache_offsets = (
            ((physical_block_ids[:, None] * block_size + block_offsets[:, None]) * num_kv_heads + kv_head_id)
            * head_dim
            + offs_d[None, :]
        )
        token_mask = n_idxs < cache_len
        element_mask = token_mask[:, None] & (offs_d[None, :] < head_dim)
        k = tl.load(k_cache_ptr + cache_offsets, mask=element_mask, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + cache_offsets, mask=element_mask, other=0.0).to(tl.float32)
        if HAS_NEW_KV:
            current_token = n_idxs == cache_len - 1
            k = tl.where(current_token[:, None], new_k[None, :], k)
            v = tl.where(current_token[:, None], new_v[None, :], v)

        scores = tl.sum(k * q[None, :], axis=1) * softmax_scale
        scores = tl.where(token_mask, scores, -float("inf"))
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(token_mask, tl.exp(scores - m_new), 0.0)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(
        out_ptr + (batch_id * num_q_heads + q_head_id) * head_dim + offs_d,
        acc / l_i,
        mask=offs_d < head_dim,
    )


@triton.jit
def flash_attn_decode_paged_kernel(
    q_ptr,
    new_k_ptr,
    new_v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    out_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    slot_mapping_ptr,
    new_k_stride_t: tl.constexpr,
    new_k_stride_h: tl.constexpr,
    new_v_stride_t: tl.constexpr,
    new_v_stride_h: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    HAS_NEW_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // num_q_heads
    q_head_id = pid_bh - batch_id * num_q_heads
    kv_head_id = q_head_id // (num_q_heads // num_kv_heads)

    cache_len = tl.load(cache_seqlens_ptr + batch_id)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Triton dot operands must have M/N/K dimensions of at least 16. Decode
    # has a single query, so duplicate it across BLOCK_M rows and reduce the
    # identical output rows before storing.
    q = tl.load(
        q_ptr + (batch_id * num_q_heads + q_head_id) * head_dim + offs_d[None, :] + offs_m[:, None] * 0,
        mask=offs_d[None, :] < head_dim,
        other=0.0,
    )

    if HAS_NEW_KV:
        new_k = tl.load(
            new_k_ptr + batch_id * new_k_stride_t + kv_head_id * new_k_stride_h + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        new_v = tl.load(
            new_v_ptr + batch_id * new_v_stride_t + kv_head_id * new_v_stride_h + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        )
        slot = tl.load(slot_mapping_ptr + batch_id)
        group_size = num_q_heads // num_kv_heads
        cache_write_offsets = slot * num_kv_heads * head_dim + kv_head_id * head_dim + offs_d
        write_mask = (slot >= 0) & (q_head_id % group_size == 0) & (offs_d < head_dim)
        tl.store(k_cache_ptr + cache_write_offsets, new_k, mask=write_mask)
        tl.store(v_cache_ptr + cache_write_offsets, new_v, mask=write_mask)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, max_num_blocks * block_size, BLOCK_N):
        n_idxs = n_start + offs_n
        logical_block_ids = n_idxs // block_size
        block_offsets = n_idxs - logical_block_ids * block_size
        physical_block_ids = tl.load(
            block_table_ptr + batch_id * max_num_blocks + logical_block_ids,
            mask=n_idxs < cache_len,
            other=0,
        )

        cache_offsets = (
            ((physical_block_ids[:, None] * block_size + block_offsets[:, None]) * num_kv_heads + kv_head_id)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            k_cache_ptr + cache_offsets,
            mask=(n_idxs[:, None] < cache_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr + cache_offsets,
            mask=(n_idxs[:, None] < cache_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        if HAS_NEW_KV:
            current_token = n_idxs == cache_len - 1
            k = tl.where(current_token[:, None], new_k[None, :], k)
            v = tl.where(current_token[:, None], new_v[None, :], v)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        score_mask = n_idxs[None, :] < cache_len
        scores = tl.where(score_mask, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(score_mask, tl.exp(scores - m_new[:, None]), 0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out_offsets = (
        out_ptr
        + (batch_id * num_q_heads + q_head_id) * head_dim
        + offs_d[None, :]
        + offs_m[:, None] * 0
    )
    tl.store(
        out_offsets,
        acc / l_i[:, None],
        mask=(offs_m[:, None] == 0) & (offs_d[None, :] < head_dim),
    )


def _block_d(head_dim: int) -> int:
    return triton.next_power_of_2(head_dim)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, block_table=None):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 3
    total_q, num_q_heads, head_dim = q.shape
    block_n = 32
    block_m = 16
    block_d = _block_d(head_dim)
    out = torch.empty((total_q, num_q_heads, head_dim), device=q.device, dtype=q.dtype)
    batch = cu_seqlens_q.numel() - 1
    grid = (triton.cdiv(max_seqlen_q, block_m), batch * num_q_heads)

    if block_table is None:
        assert k.ndim == 3 and v.ndim == 3
        num_kv_heads = k.shape[1]
        flash_attn_varlen_packed_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            cu_seqlens_q, cu_seqlens_k,
            num_q_heads, num_kv_heads, max_seqlen_k, head_dim,
            softmax_scale, causal,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        )
    else:
        assert k.ndim == 4 and v.ndim == 4
        num_kv_heads = k.shape[2]
        block_size = k.shape[1]
        max_num_blocks = block_table.shape[1]
        flash_attn_varlen_paged_kernel[grid](
            q, k, v, out, cu_seqlens_q, cu_seqlens_k, block_table,
            num_q_heads, num_kv_heads, max_seqlen_k, max_num_blocks,
            block_size, head_dim, softmax_scale, causal,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        )
    return out


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    causal=True,
    new_k=None,
    new_v=None,
    slot_mapping=None,
):
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.ndim == 4 and q.shape[1] == 1
    batch, _, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    max_num_blocks = block_table.shape[1]
    block_m = 16
    block_n = 32
    vector_block_n = 64
    block_d = _block_d(head_dim)
    has_new_kv = new_k is not None
    assert has_new_kv == (new_v is not None and slot_mapping is not None)
    new_k_arg = new_k if has_new_kv else q
    new_v_arg = new_v if has_new_kv else q
    slot_mapping_arg = slot_mapping if has_new_kv else cache_seqlens
    new_k_stride_t = new_k.stride(0) if has_new_kv else 0
    new_k_stride_h = new_k.stride(1) if has_new_kv else 0
    new_v_stride_t = new_v.stride(0) if has_new_kv else 0
    new_v_stride_h = new_v.stride(1) if has_new_kv else 0

    out = torch.empty((batch, num_q_heads, head_dim), device=q.device, dtype=q.dtype)
    if batch < 16:
        flash_attn_decode_vector_kernel[(batch * num_q_heads,)](
            q[:, 0],
            new_k_arg,
            new_v_arg,
            k_cache,
            v_cache,
            out,
            cache_seqlens,
            block_table,
            slot_mapping_arg,
            new_k_stride_t,
            new_k_stride_h,
            new_v_stride_t,
            new_v_stride_h,
            num_q_heads,
            num_kv_heads,
            max_num_blocks,
            block_size,
            head_dim,
            softmax_scale,
            HAS_NEW_KV=has_new_kv,
            BLOCK_N=vector_block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
    else:
        flash_attn_decode_paged_kernel[(batch * num_q_heads,)](
            q[:, 0],
            new_k_arg,
            new_v_arg,
            k_cache,
            v_cache,
            out,
            cache_seqlens,
            block_table,
            slot_mapping_arg,
            new_k_stride_t,
            new_k_stride_h,
            new_v_stride_t,
            new_v_stride_h,
            num_q_heads,
            num_kv_heads,
            max_num_blocks,
            block_size,
            head_dim,
            softmax_scale,
            HAS_NEW_KV=has_new_kv,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
    return out.unsqueeze(1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.is_prefill and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        new_k=k, new_v=v, slot_mapping=context.slot_mapping)
        return o
