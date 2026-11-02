"""Inference SDPA with bounded query batches and unchanged visibility rules."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


def resolve_query_attention_config(backend=None, chunk_size=None) -> Tuple[str, int]:
    """Resolve optional constructor settings without changing existing defaults."""
    if backend is None:
        backend = os.environ.get("FLOWHMR_ATTENTION_BACKEND", "original")
    if backend not in ("original", "query_batched"):
        raise ValueError("attention_backend must be 'original' or 'query_batched'")
    if chunk_size is None:
        chunk_size = os.environ.get("FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE", "256")
    if isinstance(chunk_size, str):
        if not chunk_size.isdecimal():
            raise ValueError("attention_query_chunk_size must be a positive integer")
        chunk_size = int(chunk_size)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("attention_query_chunk_size must be a positive integer")
    return backend, chunk_size


@dataclass(frozen=True)
class LocalVisibility:
    query_start: int
    motion_length: int
    motion_window: Optional[int]
    condition_window: Optional[int]
    key_padding_mask: Optional[Tensor]
    fully_masked: Optional[Tensor]
    key_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]


@dataclass(frozen=True)
class AttentionChunk:
    query_start: int
    query_stop: int
    key_indices: Optional[Tensor]
    attn_mask: Optional[Tensor]
    local_visibility: Optional[LocalVisibility] = None


@dataclass(frozen=True)
class AttentionPlan:
    length: int
    chunks: Tuple[AttentionChunk, ...]
    # Refiner masks stay factored; caching every global chunk would retain L² masks.
    valid_tokens: Optional[Tensor] = None


def _validate_length(length: int, name: str) -> None:
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_window(window: Optional[int], name: str) -> None:
    if window is not None and (isinstance(window, bool) or not isinstance(window, int)):
        raise ValueError(f"{name} must be an integer or None")


def _prepare_mask(mask: Optional[Tensor], length: int, device) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != length:
        raise ValueError(f"token mask must be bool [B, {length}], got {mask.shape}/{mask.dtype}")
    return mask.to(device=device) if device is not None else mask


def _key_range(query_start: int, query_stop: int, key_length: int,
               window: Optional[int]) -> Tuple[int, int]:
    if window is None:
        return 0, key_length
    if window < 0:
        return 0, 0
    start = min(key_length, max(0, query_start - window))
    stop = min(key_length, query_stop + window)
    return start, max(start, stop)


def _quadrant_visible(queries: Tensor, keys: Tensor, window: Optional[int]) -> Tensor:
    if window is None:
        return torch.ones((queries.numel(), keys.numel()), dtype=torch.bool, device=queries.device)
    return (queries[:, None] - keys[None, :]).abs() <= window


def _local_mask(queries, indices, motion_length, motion_window, condition_window, padding):
    motion_keys = indices < motion_length
    local_keys = torch.where(motion_keys, indices, indices - motion_length)
    visible = torch.where(
        motion_keys[None, :],
        _quadrant_visible(queries, local_keys, motion_window),
        _quadrant_visible(queries, local_keys, condition_window),
    )[None, None, :, :]
    if padding is not None:
        visible = visible & ~padding[:, None, None, indices]
    return visible


def build_local_attention_plan(
    motion_length: int,
    condition_length: int,
    *,
    chunk_size: int = 256,
    mm_window: Optional[int],
    mc_window: Optional[int] = None,
    cm_window: Optional[int] = -1,
    cc_window: Optional[int] = None,
    key_padding_mask: Optional[Tensor] = None,
    device=None,
) -> AttentionPlan:
    """Plan the four narrowband_v2m quadrants without a full sequence mask.

    Window ``None`` means full visibility, a nonnegative integer is an inclusive
    radius in each stream's original coordinates, and a negative integer disables
    that quadrant. ``key_padding_mask`` is bool [B, M+C], True for invalid keys.
    Plans are shared by every attention layer in one model forward.
    """
    _validate_length(motion_length, "motion_length")
    _validate_length(condition_length, "condition_length")
    _, chunk_size = resolve_query_attention_config("query_batched", chunk_size)
    for name, window in (("mm_window", mm_window), ("mc_window", mc_window),
                         ("cm_window", cm_window), ("cc_window", cc_window)):
        _validate_window(window, name)
    length = motion_length + condition_length
    padding = _prepare_mask(key_padding_mask, length, device)
    if device is None:
        device = padding.device if padding is not None else torch.device("cpu")

    pending = []
    for offset, query_length, motion_window, condition_window in (
        (0, motion_length, mm_window, mc_window),
        (motion_length, condition_length, cm_window, cc_window),
    ):
        for start in range(0, query_length, chunk_size):
            stop = min(start + chunk_size, query_length)
            mr = _key_range(start, stop, motion_length, motion_window)
            cr = _key_range(start, stop, condition_length, condition_window)
            key_ranges = (mr, (cr[0] + motion_length, cr[1] + motion_length))
            indices = torch.cat([torch.arange(*bounds, device=device) for bounds in key_ranges])
            queries = torch.arange(start, stop, device=device)
            visible = _local_mask(queries, indices, motion_length, motion_window, condition_window, padding)
            fully_masked = ~visible.any(dim=-1, keepdim=True)
            # Retain only compact local masks. Full/very wide quadrants otherwise
            # accumulate quadratic mask storage even when each Q batch is small.
            cache_mask = indices.numel() < length and indices.numel() <= 4 * chunk_size
            pending.append((offset + start, offset + stop, start,
                            indices if cache_mask else None, visible if cache_mask else None,
                            fully_masked, motion_window, condition_window,
                            None if indices.numel() == length else key_ranges))

    # One device synchronization during planning, never one per attention layer.
    fallback_chunks = torch.stack([item[5].any() for item in pending]).tolist() if pending else []
    chunks = []
    for item, needs_fallback in zip(pending, fallback_chunks):
        start, stop, local_start, indices, visible, fully_masked, mw, cw, key_ranges = item
        lazy_visibility = None
        if needs_fallback:
            # The original MATH path unmasks every sequence key for an empty row,
            # including padded keys and keys outside the nominal local window.
            lazy_visibility = LocalVisibility(local_start, motion_length, mw, cw,
                                              padding, fully_masked, None)
            indices = None
            visible = None
        elif visible is None:
            lazy_visibility = LocalVisibility(local_start, motion_length, mw, cw,
                                              padding, None, key_ranges)
        chunks.append(AttentionChunk(start, stop, indices, visible, lazy_visibility))
    return AttentionPlan(length, tuple(chunks))


def build_global_attention_plan(
    length: int,
    *,
    chunk_size: int = 256,
    valid_tokens: Optional[Tensor] = None,
    device=None,
) -> AttentionPlan:
    """Plan refiner attention, retaining every valid key for every valid query.

    Invalid queries attend only their own diagonal, as in IndividualTokenRefiner.
    Keeping the validity vector factored avoids retaining quadratic masks.
    """
    _validate_length(length, "length")
    _, chunk_size = resolve_query_attention_config("query_batched", chunk_size)
    valid_tokens = _prepare_mask(valid_tokens, length, device)
    chunks = tuple(AttentionChunk(start, min(start + chunk_size, length), None, None)
                   for start in range(0, length, chunk_size))
    return AttentionPlan(length, chunks, valid_tokens)


def query_batched_attention(q: Tensor, k: Tensor, v: Tensor, plan: AttentionPlan) -> Tensor:
    """Apply inference-only MATH SDPA to [B, L, H, D], returning [B, L, H*D].

    Q/K already include positional encoding. Motion and condition keys stay in
    one softmax; only masked-out keys are omitted. Input precision is unchanged.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[1] != plan.length:
        raise ValueError("q, k and v must have matching [B, plan.length, H, D] shapes")
    if plan.valid_tokens is not None and plan.valid_tokens.shape[0] != q.shape[0]:
        raise ValueError("valid_tokens batch size does not match q")
    if not plan.chunks:
        return q.flatten(2)
    output = []
    keys = torch.arange(plan.length, device=q.device) if plan.valid_tokens is not None else None
    with sdpa_kernel(SDPBackend.MATH):
        for chunk in plan.chunks:
            q_chunk = q[:, chunk.query_start:chunk.query_stop].transpose(1, 2)
            indices = chunk.key_indices
            mask = chunk.attn_mask
            if chunk.local_visibility is not None:
                rule = chunk.local_visibility
                if rule.key_ranges is None:
                    if keys is None:
                        keys = torch.arange(plan.length, device=q.device)
                    mask_keys = keys
                else:
                    indices = torch.cat([torch.arange(*bounds, device=q.device) for bounds in rule.key_ranges])
                    mask_keys = indices
                queries = torch.arange(rule.query_start,
                                       rule.query_start + chunk.query_stop - chunk.query_start, device=q.device)
                mask = _local_mask(queries, mask_keys, rule.motion_length,
                                   rule.motion_window, rule.condition_window, rule.key_padding_mask)
                if rule.fully_masked is not None:
                    mask = mask | rule.fully_masked
            if indices is None:
                k_chunk, v_chunk = k, v
            else:
                k_chunk = k.index_select(1, indices)
                v_chunk = v.index_select(1, indices)
            if plan.valid_tokens is not None:
                valid_q = plan.valid_tokens[:, None, chunk.query_start:chunk.query_stop, None]
                queries = keys[chunk.query_start:chunk.query_stop]
                diagonal = queries[:, None] == keys[None, :]
                mask = ((valid_q & plan.valid_tokens[:, None, None, :]) |
                        (~valid_q & diagonal[None, None, :, :]))
            attended = F.scaled_dot_product_attention(
                q_chunk, k_chunk.transpose(1, 2), v_chunk.transpose(1, 2),
                attn_mask=mask, dropout_p=0.0, is_causal=False,
            )
            output.append(attended.transpose(1, 2).flatten(2))
    return torch.cat(output, dim=1)
