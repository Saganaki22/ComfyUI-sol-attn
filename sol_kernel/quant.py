"""INT8 quantization for the optional SageAttention-style q/k path.

q is quantized per token (one scale per row); k is quantized per 64-token
block after centering by the block mean, so the exact-path dot decomposes as
q*k = (q8*k8)*(q_scale*k_scale) + q*kc, where the q*kc term is the routing
score the forward kernel already computes in bf16.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_SIZE = 64


@triton.jit
def _quantize_q_kernel(
    q_desc,
    q8_desc,
    q_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    tile = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK, D]).to(tl.float32)
    amax = tl.max(tl.abs(tile), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    q8 = libdevice.rint(tile / scale[:, None]).to(tl.int8)
    q8_desc.store([batch, q_start, head, 0], q8[None, :, None, :])
    offsets = q_start + tl.arange(0, BLOCK)
    tl.store(q_scale + (batch * T + offsets) * H + head, scale, mask=offsets < T)


@triton.jit
def _quantize_k_kernel(
    k_desc,
    kc_desc,
    k8_desc,
    k_scale,
    T,
    H: tl.constexpr,
    NB,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    kv_start = block * BLOCK
    tile = k_desc.load([batch, kv_start, head, 0]).reshape([BLOCK, D]).to(tl.float32)
    mean = kc_desc.load([batch, block, head, 0]).reshape([1, D]).to(tl.float32)
    centered = tile - mean
    valid = (kv_start + tl.arange(0, BLOCK)) < T
    amax = tl.max(tl.abs(tl.where(valid[:, None], centered, 0.0)))
    scale = tl.maximum(amax / 127.0, 1e-8)
    k8 = libdevice.rint(centered / scale).to(tl.int8)
    k8_desc.store([batch, kv_start, head, 0], k8[None, :, None, :])
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
    tile = [1, BLOCK_SIZE, 1, head_dim]
    row = [1, 1, 1, head_dim]
    grid = (blocks, batch * heads)
    _quantize_q_kernel[grid](
        TensorDescriptor.from_tensor(q, tile),
        TensorDescriptor.from_tensor(q8, tile),
        q_scale,
        tokens,
        heads,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    _quantize_k_kernel[grid](
        TensorDescriptor.from_tensor(k, tile),
        TensorDescriptor.from_tensor(kc, row),
        TensorDescriptor.from_tensor(k8, tile),
        k_scale,
        tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    return q8, q_scale, k8, k_scale


__all__ = ["quantize_qk"]
