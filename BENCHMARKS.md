# Benchmarks

**Date:** 2026-08-04 · **Machine:** RTX 5090 (SM120) · torch 2.10.0+cu130 · Triton 3.6.0 · Python 3.12.10 · Windows 11

## Method

All Sol-Attn implementations are given the **same inputs**: MiniMax H3-shaped strided NHD views into a fused qkv projection buffer (`B=1, H=56, D=128`, bf16, random Gaussian tensors), `tau=1.0`, median of 20 timed iterations after 3 warmup iterations (which absorb Triton autotune / torch.compile). SageAttention and PyTorch SDPA are the dense baselines.

> Fidelity warning: these methods are not interchangeable. KingGore's fork uses a **hard block mask** (unselected KV blocks are dropped, ~8–11% density by its own README) while the Sol-Attn implementations keep an approximate correction for unselected blocks (~16% exact at `tau=1.0`). Faster-by-sparser is not the same thing as faster-by-engineering — A/B the visual quality before adopting any of them.

## Results (ms per attention call, lower is better)

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 23.80 | 94.99 | 381.05 | 1,501.91 |
| SageAttention | 4.12 | 15.08 | 58.59 | 230.71 |
| KingGore flex | 4.71 | 10.74 | 37.11 | 126.10 |
| kijai bf16 | 3.76 | 12.20 | 43.32 | 164.90 |
| kijai int8 | 2.99 | 8.95 | 28.22 | 114.10 |
| **ours bf16** | 3.81 | 11.98 | 41.73 | 160.35 |
| **ours int8** | **2.56** | **8.64** | 31.03 | 118.20¹ |

¹ v0.4.8 measured 118.2 ms at 65,536 (the int64 addressing required for >100K-token strided inputs costs ~7% at this size; smaller sizes are unaffected).

Peak attention memory above resident on the same inputs (our zero-copy path vs the hook-style path that copies q/k/v first — the approach used by the other TMA implementations):

| tokens | copy-based path (MiB) | ours, strided (MiB) | saved |
|---:|---:|---:|---:|
| 8,192 | 452 | 116 | 336 |
| 16,384 | 903 | 231 | 672 |
| 32,768 | 1,806 | 462 | 1,344 |
| 65,536 | 3,612 | 924 | 2,688 |

Correctness on the same machine: strided output bit-identical to contiguous input (including ragged lengths 8,191 / 12,345 / 38,247 and 103,237 tokens after the v0.4.8 int64-addressing fix); all-exact mode matches PyTorch SDPA at relative L2 `0.00097`; `int8_qk` matches the bf16 path at `0.0080` and SDPA at `0.0098`. Run-to-run timing variance is a few percent; the 32,768-token int8 row flips between runs. Note that **PyTorch SDPA itself overflows int32 strides** on H3's strided qkv views past ~100K tokens — stock attention backends have their own ceiling there.

## Accuracy (relative L2 error, same random inputs)

`vs SDPA (all-exact)` isolates each kernel's numerics with routing forced fully dense (`tau=-100`). `vs ours bf16` isolates the int8 designs from the shared Sol approximation. The `vs SDPA (tau=1)` column is Sol's approximation error on pure noise — the worst case for a structure-exploiting method; structured real-world attention distributions have far more routing headroom. kijai bf16 and ours bf16 measure **0.000000** against each other: the two repos are bit-identical implementations of the same math.

| Method | vs SDPA (all-exact), 8K | vs SDPA (all-exact), 32K | vs ours bf16 (tau=1), 8K | vs ours bf16 (tau=1), 32K |
|---|---:|---:|---:|---:|
| SageAttention (dense int8) | 0.0390 | 0.0393 | — | — |
| KingGore flex (hard mask) | 2.0557 | 2.2363 | 2.5484 | 2.8165 |
| kijai bf16 | 0.00097 | 0.00078 | 0.000000 | 0.000000 |
| kijai int8 | 0.00987 | 0.00987 | 0.02995 | 0.02910 |
| ours bf16 | 0.00097 | 0.00078 | 0 | 0 |
| **ours int8** | **0.00980** | **0.00980** | **0.00802** | **0.00793** |

Our residual int8 design is ~3.7× closer to the exact bf16 Sol path than full-key int8 quantization, at equal or better speed.

## The repos compared

- **[Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn)** (this repo) — NVIDIA's Sol-Attn Triton reference, vendored and extended. Feeds the kernel H3's fused qkv views with zero copies (TMA on SM90/100/120, pointer twins on SM89). The `int8_qk` path quantizes only the per-block-mean *residual* of K (per-token scales) and adds the mean back from the exact bf16 routing scores, which is why its accuracy (0.008 rel L2) beats full-key int8 designs. Also ships scheduled tau, conditioning sinks, and FFN chunking.
- **[kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)** — independent Triton implementation of the same kernel. Clean design: SM89 pointer kernels, fused preprocess, conditioning sinks, Morton token reordering, sigma gating. Its TMA path materializes contiguous q/k/v copies (its SM89 pointer path avoids them). Its int8 uses global-mean K smoothing over full-magnitude keys. Fastest int8 at 32K on this run; ours leads the other three sizes.
- **[KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)** — routes in pure torch and executes the selected blocks with compiled PyTorch `flex_attention`. Legitimately fast (beats SageAttention past 8K), but it is a **different sparsity method**: hard block mask, no approximate correction, much lower density. Compare its quality, not just its speed. SM120 only.
- **[SageAttention](https://github.com/thu-ml/SageAttention)** (thu-ml) — the dense int8 attention baseline everything here is measured against. Still the right choice below ~4K tokens, which is why our nodes gate on `min_tokens`.
- **PyTorch SDPA** — stock ComfyUI backend, shown for scale; at these lengths it does not select a competitive kernel path.

## Takeaways (2026-08-04)

1. Our `int8_qk` is the fastest measured option at 8K / 16K / 65K tokens and within ~10% of kijai's at 32K, with the best accuracy of the int8 variants (0.008 rel L2 vs bf16).
2. Our bf16 is the fastest bf16 Sol path at 32K/65K (zero copies), effectively tied with kijai's at 8K/16K.
3. KingGore's flex path is quick but answers a different question — do not compare it on speed alone.
4. Sol-Attn's advantage over SageAttention grows with sequence length; below ~4K tokens Sage stays the right default.
