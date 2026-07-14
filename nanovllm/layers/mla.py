"""Correctness-first latent-cache attention for DeepSeek MLA.

The module intentionally owns a single paged cache.  Each cache entry contains
the normalized KV latent followed by the decoupled RoPE key.  Queries are
already projected into the latent space by the caller, so neither per-head keys
nor per-head values are materialized for cached tokens.
"""

import torch
from torch import nn

from nanovllm.utils.context import get_context


def _causal_latent_attention(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    latent: torch.Tensor,
    key_rope: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend one packed sequence, allowing a cached prefix in ``latent``."""
    query_length = query_latent.shape[0]
    key_length = latent.shape[0]
    if query_length <= 0 or key_length < query_length:
        raise ValueError(
            "MLA requires a non-empty query and key_length >= query_length, "
            f"got query_length={query_length}, key_length={key_length}"
        )

    scores = torch.einsum("qhd,kd->hqk", query_latent, latent)
    scores = scores + torch.einsum("qhr,kr->hqk", query_rope, key_rope)
    scores = scores.float() * scale

    # Chunked/prefix prefill queries correspond to the suffix of the complete
    # key sequence.  Decode (query_length == 1) therefore naturally sees every
    # cached token, including the newly inserted token.
    query_positions = (
        torch.arange(query_length, device=scores.device)
        + key_length
        - query_length
    )
    key_positions = torch.arange(key_length, device=scores.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    probabilities = torch.softmax(
        scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf")),
        dim=-1,
        dtype=torch.float32,
    ).to(latent.dtype)
    return torch.einsum("hqk,kd->qhd", probabilities, latent)


class LatentMLAAttention(nn.Module):
    """Paged latent attention used by the DeepSeek-V2 weight-absorbed path.

    ``kv_cache_count = 1`` is consumed by ``ModelRunner.allocate_kv_cache`` and
    is what removes the redundant expanded K and V allocations.
    """

    kv_cache_count = 1

    def __init__(
        self,
        num_heads: int,
        latent_dim: int,
        rope_dim: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.scale = scale

        # These names retain the common cache-module protocol.  v_cache stays
        # empty because the latent entry is sufficient to recover both K and V.
        self.num_kv_heads = 1
        self.head_dim = latent_dim + rope_dim
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def _packed_cache_entry(
        self,
        latent: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"invalid MLA latent shape {tuple(latent.shape)}, expected "
                f"[tokens, {self.latent_dim}]"
            )
        if key_rope.shape != (latent.shape[0], self.rope_dim):
            raise ValueError(
                f"invalid MLA RoPE key shape {tuple(key_rope.shape)}, expected "
                f"[{latent.shape[0]}, {self.rope_dim}]"
            )
        return torch.cat((latent, key_rope), dim=-1)

    def _store_cache(
        self,
        entry: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if (
            not self.k_cache.numel()
            or slot_mapping is None
            or not slot_mapping.numel()
        ):
            return
        if slot_mapping.numel() != entry.shape[0]:
            raise ValueError(
                "MLA cache slot count must match the number of new tokens, "
                f"got {slot_mapping.numel()} slots and {entry.shape[0]} tokens"
            )
        valid = slot_mapping >= 0
        flat_cache = self.k_cache.view(-1, self.head_dim)
        flat_cache.index_copy_(
            0,
            slot_mapping[valid].long(),
            entry[valid],
        )

    def _read_paged_sequence(
        self,
        block_table: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length <= 0:
            raise ValueError(
                f"MLA cache sequence length must be positive, got {length}"
            )
        block_size = self.k_cache.shape[1]
        num_blocks = (length + block_size - 1) // block_size
        physical_blocks = block_table[:num_blocks].long()
        if (
            physical_blocks.device.type == "cpu"
            and torch.any(physical_blocks < 0)
        ):
            raise ValueError("MLA block table contains an unallocated block")
        offsets = torch.arange(length, device=block_table.device)
        slots = physical_blocks[offsets // block_size] * block_size
        slots = slots + offsets.remainder(block_size)
        entries = self.k_cache.view(-1, self.head_dim).index_select(0, slots)
        return entries[:, : self.latent_dim], entries[:, self.latent_dim :]

    def _packed_prefill(
        self,
        query_latent: torch.Tensor,
        query_rope: torch.Tensor,
        latent: torch.Tensor,
        key_rope: torch.Tensor,
        cu_seqlens_q: torch.Tensor | list[int] | None,
    ) -> torch.Tensor:
        if cu_seqlens_q is None:
            return _causal_latent_attention(
                query_latent, query_rope, latent, key_rope, self.scale
            )

        outputs = []
        boundaries = (
            cu_seqlens_q.tolist()
            if isinstance(cu_seqlens_q, torch.Tensor)
            else cu_seqlens_q
        )
        for start, end in zip(boundaries, boundaries[1:]):
            outputs.append(
                _causal_latent_attention(
                    query_latent[start:end],
                    query_rope[start:end],
                    latent[start:end],
                    key_rope[start:end],
                    self.scale,
                )
            )
        return torch.cat(outputs, dim=0)

    def _paged_attention(
        self,
        query_latent: torch.Tensor,
        query_rope: torch.Tensor,
        query_boundaries: list[int],
        key_lengths: list[int],
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for batch_idx, (start, end, key_length) in enumerate(
            zip(query_boundaries, query_boundaries[1:], key_lengths)
        ):
            cached_latent, cached_rope = self._read_paged_sequence(
                block_tables[batch_idx], key_length
            )
            outputs.append(
                _causal_latent_attention(
                    query_latent[start:end],
                    query_rope[start:end],
                    cached_latent,
                    cached_rope,
                    self.scale,
                )
            )
        return torch.cat(outputs, dim=0)

    def _paged_decode(
        self,
        query_latent: torch.Tensor,
        query_rope: torch.Tensor,
        key_lengths: list[int],
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized single-token decode over a variable-length batch."""
        batch = query_latent.shape[0]
        if len(key_lengths) != batch:
            raise ValueError(
                f"MLA decode received {len(key_lengths)} lengths for batch {batch}"
            )
        max_length = max(key_lengths)
        block_size = self.k_cache.shape[1]
        positions = torch.arange(max_length, device=block_tables.device)
        lengths = torch.tensor(
            key_lengths, device=block_tables.device, dtype=positions.dtype
        )
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        logical_blocks = positions.div(block_size, rounding_mode="floor")
        physical_blocks = block_tables[:, logical_blocks]
        physical_blocks = torch.where(valid, physical_blocks, 0)
        slots = physical_blocks.long() * block_size + positions.remainder(block_size)
        entries = self.k_cache.view(-1, self.head_dim).index_select(
            0, slots.flatten()
        ).view(batch, max_length, self.head_dim)

        query = torch.cat((query_latent, query_rope), dim=-1)
        scores = torch.bmm(query, entries.transpose(1, 2)).float()
        scores = scores * self.scale
        probabilities = torch.softmax(
            scores.masked_fill(~valid.unsqueeze(1), float("-inf")),
            dim=-1,
            dtype=torch.float32,
        ).to(entries.dtype)
        return torch.bmm(
            probabilities,
            entries[..., : self.latent_dim],
        )

    def forward(
        self,
        query_latent: torch.Tensor,
        query_rope: torch.Tensor,
        latent: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> torch.Tensor:
        if query_latent.ndim != 3 or query_latent.shape[1:] != (
            self.num_heads,
            self.latent_dim,
        ):
            raise ValueError(
                f"invalid absorbed MLA query shape {tuple(query_latent.shape)}"
            )
        if query_rope.shape != (
            query_latent.shape[0],
            self.num_heads,
            self.rope_dim,
        ):
            raise ValueError(
                f"invalid MLA rotary query shape {tuple(query_rope.shape)}"
            )

        context = get_context()
        entry = self._packed_cache_entry(latent, key_rope)

        # A module-level call without engine context is treated as one prefill
        # sequence.  This keeps the mathematical path independently testable on
        # CPU without manufacturing scheduler state.
        has_engine_context = (
            context.cu_seqlens_q is not None
            or context.context_lens is not None
            or context.block_tables is not None
            or context.slot_mapping is not None
        )
        if not has_engine_context:
            return self._packed_prefill(
                query_latent, query_rope, latent, key_rope, None
            )

        self._store_cache(entry, context.slot_mapping)
        if context.is_prefill:
            query_boundaries = (
                list(context.host_cu_seqlens_q)
                if context.host_cu_seqlens_q is not None
                else context.cu_seqlens_q.tolist()
            )
            if context.block_tables is None:
                return self._packed_prefill(
                    query_latent,
                    query_rope,
                    latent,
                    key_rope,
                    query_boundaries,
                )
            if not self.k_cache.numel():
                raise RuntimeError("prefix prefill requires an allocated MLA cache")
            key_boundaries = (
                list(context.host_cu_seqlens_k)
                if context.host_cu_seqlens_k is not None
                else context.cu_seqlens_k.tolist()
            )
            return self._paged_attention(
                query_latent,
                query_rope,
                query_boundaries,
                [
                    end - start
                    for start, end in zip(
                        key_boundaries,
                        key_boundaries[1:],
                    )
                ],
                context.block_tables,
            )

        if context.context_lens is None or context.block_tables is None:
            raise RuntimeError(
                "MLA decode requires context lengths and block tables"
            )
        if not self.k_cache.numel():
            raise RuntimeError("MLA decode requires an allocated latent cache")
        key_lengths = (
            list(context.host_context_lens)
            if context.host_context_lens is not None
            else context.context_lens.tolist()
        )
        return self._paged_decode(
            query_latent,
            query_rope,
            key_lengths,
            context.block_tables,
        )


__all__ = ["LatentMLAAttention"]
