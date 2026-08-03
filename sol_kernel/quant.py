"""INT8 quantization for the optional SageAttention-style q/k path.

q is quantized per token (one scale per row); k is quantized per 64-token
block after centering by the block mean, so the exact-path dot decomposes as
q*k = (q8*k8)*(q_scale*k_scale) + q*kc, where the q*kc term is the routing
score the forward kernel already computes in bf16.

Loads use plain pointers with explicit strides so the kernels also run on
pre-Hopper arches without TMA.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_SIZE = 64


@triton.jit
def _quantize_q_kernel(
    q_ptr,
    q8_ptr,
    q_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        q_ptr + batch * s_b + rows[:, None] * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(tile), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    q8 = libdevice.rint(tile / scale[:, None]).to(tl.int8)
    tl.store(
        q8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        q8,
        mask=valid[:, None],
    )
    tl.store(q_scale + (batch * T + rows) * H + head, scale, mask=valid)


@triton.jit
def _quantize_k_kernel(
    k_ptr,
    kc_ptr,
    k8_ptr,
    k_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    NB,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        k_ptr + batch * s_b + rows[:, None] * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(kc_ptr + ((batch * NB + block) * H + head) * D + d).to(tl.float32)
    centered = tl.where(valid[:, None], tile - mean[None, :], 0.0)
    amax = tl.max(tl.abs(centered))
    scale = tl.maximum(amax / 127.0, 1e-8)
    k8 = libdevice.rint(centered / scale).to(tl.int8)
    tl.store(
        k8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        k8,
        mask=valid[:, None],
    )
    tl.store(k_scale + (batch * NB + block) * H + head, scale)


def quantize_qk(q, k, kc):
    """q/k [B, T, H, 128] bf16, kc [B, NB, H, 128] bf16 block means.

    Returns q_int8 [B, T, H, 128], q_scale [B, T, H] fp32,
    k_int8 [B, T, H, 128], k_scale [B, NB, H] fp32.
    """
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    k8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    q_scale = torch.empty((batch, tokens, heads), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((batch, blocks, heads), device=k.device, dtype=torch.float32)
    grid = (blocks, batch * heads)
    _quantize_q_kernel[grid](
        q,
        q8,
        q_scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    _quantize_k_kernel[grid](
        k,
        kc,
        k8,
        k_scale,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    return q8, q_scale, k8, k_scale


__all__ = ["quantize_qk"]
