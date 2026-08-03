"""Triton Sol-Attn reference."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .preprocess import prepare


def _tma_compatible(t):
    """What TensorDescriptor accepts: dense last dim, 16-byte alignment elsewhere."""
    if t.data_ptr() % 16 != 0 or t.stride(-1) != 1:
        return False
    return all(stride * t.element_size() % 16 == 0 for stride in t.stride()[:-1])


def _validate(q, k, v, kv_splits, thresh_type):
    """Upstream validation extended to the locally tested SM120 path.

    The public upstream dispatcher allows SM90/SM100. This standalone Triton
    reference has also been validated on SM120; other architectures remain
    disabled instead of being treated as implicitly compatible.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not all(_tma_compatible(x) for x in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous or TMA-compatible BTHD tensors")
    if kv_splits not in (1, 2, 4):
        raise ValueError("kv_splits must be 1, 2, or 4")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    route_groups = ((q.shape[1] + 63) // 64 + 63) // 64
    if kv_splits > route_groups:
        raise ValueError("each KV split must contain at least one N64 route group")
    arch = torch.cuda.get_device_capability(q.device)
    if arch not in ((9, 0), (10, 0), (12, 0)):
        raise RuntimeError("Sol-Attn supports SM90, SM100, and SM120")
    return arch


BLOCK = 64
GROUP = 32


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["T"],
)
@triton.jit
def _forward(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    threshold,
    o_desc,
    scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    v_tile, q_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    q_start = q_block * BLOCK_SIZE
    q = q_desc.load([batch, q_start, head, 0]).reshape([BLOCK_SIZE, D])
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)

    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    route_threshold = tl.load(
        threshold + (batch * NT + q_block) * H + head
    )

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([GROUP_SIZE, D])
        vc = vc_desc.load(
            [batch, group_start, head, v_tile * BV]
        ).reshape([GROUP_SIZE, BV])
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        exact = (
            (tl.sum(scores, axis=0) / q_len > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
        ) & valid

        approximate = valid & ~exact
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(approximate_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            approximate_probability.to(vc.dtype), vc
        )
        lengths = tl.where(
            block_indices == NT - 1, tail_length, BLOCK_SIZE
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            approximate_probability * lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, GROUP_SIZE, exact_offsets
            )
            kv_start = block * BLOCK_SIZE
            k = k_desc.load(
                [batch, kv_start, head, 0]
            ).reshape([BLOCK_SIZE, D])
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (kv_start + token_offsets)[None, :] < T,
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v = v_desc.load(
                [batch, kv_start, head, v_tile * BV]
            ).reshape([BLOCK_SIZE, BV])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(v.dtype), v
            )
            row_max = new_max

    o_desc.store(
        [batch, q_start, head, v_tile * BV],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
) -> torch.Tensor:
    """Run the readable Triton reference on BTHD inputs."""

    _validate(q, k, v, 1, thresh_type)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK)
    kc, vc, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    block_shape = [1, BLOCK, 1, head_dim]
    summary_shape = [1, GROUP, 1, head_dim]
    _forward[(1, blocks, batch * heads)](
        TensorDescriptor.from_tensor(q, block_shape),
        TensorDescriptor.from_tensor(k, block_shape),
        TensorDescriptor.from_tensor(v, block_shape),
        TensorDescriptor.from_tensor(kc, summary_shape),
        TensorDescriptor.from_tensor(vc, summary_shape),
        threshold,
        TensorDescriptor.from_tensor(output, block_shape),
        scale,
        tokens,
        heads,
        head_dim,
        blocks,
        head_dim,
        BLOCK,
        GROUP,
    )
    return output


__all__ = ["sol_attn"]
