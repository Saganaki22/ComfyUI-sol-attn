# Benchmarks

**Date:** 2026-08-07 · **Machine:** RTX 5090 (SM120) · torch 2.10.0+cu130 · Triton 3.6.0 · Python 3.12.10 · Windows 11

## Method

All Sol-Attn implementations are given the **same inputs**: MiniMax H3-shaped strided NHD views into a fused qkv projection buffer (`B=1, H=56, D=128`, bf16, random Gaussian tensors), `tau=1.0`, median of 20 timed iterations after 3 warmup iterations (which absorb Triton autotune / torch.compile). SageAttention and PyTorch SDPA are the dense baselines. kijai's repo is benchmarked from commit `0a92202` (latest at time of writing).

> Fidelity warning: these methods are not interchangeable. KingGore's fork uses a **hard block mask** (unselected KV blocks are dropped, ~8–11% density by its own README) while the Sol-Attn implementations keep an approximate correction for unselected blocks (~16% exact at `tau=1.0`). Faster-by-sparser is not the same thing as faster-by-engineering — A/B the visual quality before adopting any of them.

## Speed (ms per attention call, lower is better)

Full cross-repo rerun 2026-08-07. All eight methods on the same inputs in one session:

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 22.49 | 96.86 | 349.03 | 1,405.18 |
| SageAttention | 3.80 | 15.15 | 54.69 | 234.51 |
| KingGore flex | 4.57 | 10.56 | 32.23 | 134.77 |
| kijai bf16 | 2.90 | 9.05 | 33.16 | 145.33 |
| kijai int8 | **2.60** | **7.78** | **27.85** | **113.88** |
| ours bf16 | 3.20 | 10.37 | 37.91 | 151.63 |
| ours int8_qk | 2.52 | 8.17 | 28.75 | 114.81 |
| ours int8_qk+pv | 3.04 | 8.91 | 29.99 | 118.11 |

**Read on the 2026-08-07 numbers:** our `int8_qk` is fastest at 8K (2.52 vs kijai's 2.60). kijai leads at 16K–65K by ~3–5% on int8. Our bf16 path is ~10–15% behind his bf16 at every size — the difference is autotune configurations and his fused-preprocess rework (commit `0a92202`, not the installed `0e334dc` which was bit-identical to ours). `int8_pv` does not beat `int8_qk` alone in this run (earlier isolated bench showed a marginal win at 32K+; autotune variance flipped it). Our int8 remains ~3.6× more accurate than his (0.008 vs 0.029 rel L2 vs the bf16 path).

<details>
<summary>Previous run (2026-08-05, installed kijai 0e334dc)</summary>

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 23.88 | 94.84 | 389.87 | 1,542.68 |
| SageAttention | 4.00 | 15.02 | 58.83 | 231.43 |
| KingGore flex | 4.69 | 10.90 | 34.89 | 126.16 |
| kijai bf16 | 3.10 | 9.77 | 35.58 | 137.49 |
| kijai int8 | **2.44** | **8.36** | **30.25** | **113.07** |
| ours bf16 | 3.33 | 11.57 | 44.95 | 162.14 |
| ours int8 | 2.72 | 8.64 | 31.30 | 118.02 |

</details>

<details>
<summary>Original run (2026-08-04, pre-update kijai repo)</summary>

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 23.80 | 94.99 | 381.05 | 1,501.91 |
| SageAttention | 4.12 | 15.08 | 58.59 | 230.71 |
| KingGore flex | 4.71 | 10.74 | 37.11 | 126.10 |
| kijai bf16 | 3.76 | 12.20 | 43.32 | 164.90 |
| kijai int8 | 2.99 | 8.95 | 28.22 | 114.10 |
| ours bf16 | 3.81 | 11.98 | 41.73 | 160.35 |
| ours int8 | 2.56 | 8.64 | 31.03 | 108.95 |

</details>

## Pipeline simulation (sage + Sol combo, 2026-08-05)

A simulated 22-step sampling run — attention calls only — shaped like the recommended workflow: first 20% of steps dense via SageAttention (`dense_percent=0.2`), the rest Sol-Attn with the cosine tau ramp (1.19 → 0.80). Median of 10 sequence iterations:

| tokens | all SDPA (ms) | all sage (ms) | all Sol bf16 | combo bf16 | combo int8 | combo vs sage | combo vs SDPA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 2,079 | 334 | 244 | 274 | 227 | 1.22× | 7.58× |
| 32,768 | 8,510 | 1,344 | 989 | 1,085 | 874 | 1.24× | 7.84× |
| 65,536 | 33,782 | 5,259 | 3,717 | 4,173 | 3,353 | 1.26× | 8.09× |

The combo with int8 beats even all-Sol bf16: the dense early steps protect quality while int8 keeps the sparse steps cheapest.

Peak attention memory above resident on the same inputs (our zero-copy path vs the hook-style path that copies q/k/v first — the approach used by the other TMA implementations):

| tokens | copy-based path (MiB) | ours, strided (MiB) | saved |
|---:|---:|---:|---:|
| 8,192 | 452 | 116 | 336 |
| 16,384 | 903 | 231 | 672 |
| 32,768 | 1,806 | 462 | 1,344 |
| 65,536 | 3,612 | 924 | 2,688 |

Correctness on the same machine: strided output bit-identical to contiguous input (including ragged lengths 8,191 / 12,345 / 38,247 and 103,237 tokens after the v0.4.8 int64-addressing fix); all-exact mode matches PyTorch SDPA at relative L2 `0.00097`; `int8_qk` matches the bf16 path at `0.0080` and SDPA at `0.0098`. Run-to-run timing variance is a few percent; the 32,768-token int8 row flips between runs. Note that **PyTorch SDPA itself overflows int32 strides** on H3's strided qkv views past ~100K tokens — stock attention backends have their own ceiling there.

## Accuracy (relative L2 error, same random inputs, 2026-08-07)

`vs SDPA (all-exact)` isolates each kernel's numerics with routing forced fully dense (`tau=-100`). `vs ours bf16` isolates the int8 designs from the shared Sol approximation.

> kijai's bf16 path diverged from ours by 0.0018 on commit `0a92202` — his fused-preprocess rework changed the kernel's numerics slightly. The installed commit `0e334dc` (the one most ComfyUI users have) was bit-identical (0.000000) in the 2026-08-05 run.

| Method | vs SDPA (all-exact), 8K | vs ours bf16 (tau=1), 8K |
|---|---:|---:|
| SageAttention (dense int8) | 0.0390 | — |
| KingGore flex (hard mask) | 2.0557 | 2.5484 |
| kijai bf16 | 0.00097 | 0.00181 |
| kijai int8 | 0.00987 | 0.02855 |
| ours bf16 | 0.00097 | 0 |
| **ours int8_qk** | **0.00980** | **0.00803** |
| ours int8_qk+pv | 0.01741 | 0.01398 |

Our residual int8 design (`int8_qk`) is ~3.6× closer to the exact bf16 Sol path than full-key int8 quantization (0.008 vs 0.029 rel L2). `int8_pv` adds V quantization on top, degrading to 0.014 rel L2 — still 2.1× more accurate than kijai's full-key int8. Default off; enable only when the extra speed at long sequences matters and the accuracy trade-off is acceptable.

## The repos compared

- **[Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn)** (this repo) — NVIDIA's Sol-Attn Triton reference, vendored and extended. Feeds the kernel H3's fused qkv views with zero copies (TMA on SM90/100/120/121, pointer twins on SM89). The `int8_qk` path quantizes only the per-block-mean *residual* of K (per-token scales) and adds the mean back from the exact bf16 routing scores, which is why its accuracy (0.008 rel L2) beats full-key int8 designs. `int8_pv` additionally quantizes P·V (per-token P, per-channel V) — opt-in. Also ships scheduled tau, conditioning sinks, FFN chunking, and SM121 (DGX Spark) support.
- **[kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)** — independent Triton implementation of the same kernel. Clean design: SM89 pointer kernels, fused preprocess, conditioning sinks, Morton token reordering, sigma gating. Its int8 uses global-mean K smoothing over full-magnitude keys. Fastest int8 at 16K–65K on this run; ours leads at 8K.
- **[KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)** — routes in pure torch and executes the selected blocks with compiled PyTorch `flex_attention`. Legitimately fast (beats SageAttention past 8K), but it is a **different sparsity method**: hard block mask, no approximate correction, much lower density. Compare its quality, not just its speed. SM120 only.
- **[SageAttention](https://github.com/thu-ml/SageAttention)** (thu-ml) — the dense int8 attention baseline everything here is measured against. Still the right choice below ~4K tokens, which is why our nodes gate on `min_tokens`.
- **PyTorch SDPA** — stock ComfyUI backend, shown for scale; at these lengths it does not select a competitive kernel path.

## Takeaways (2026-08-07)

1. **Speed:** our `int8_qk` leads at 8K (2.52 ms); kijai's int8 leads at 16K–65K by ~3–5%. The bf16 gap widened to ~10–15% after his `0a92202` fused-preprocess rework (the installed `0e334dc` was bit-identical to ours and ~equal speed). Run-to-run autotune variance can flip margins at the margins.
2. **Accuracy:** our residual `int8_qk` is ~3.6× closer to the exact bf16 path than full-key int8 (0.008 vs 0.029 rel L2). `int8_pv` trades to 0.014 for marginal speed — still 2.1× better than kijai's int8.
3. **int8_pv:** opt-in, default off. In this run it did not beat `int8_qk` alone at any size; an earlier isolated bench showed a marginal win at 32K+ that autotune variance flipped. Keep for the option; don't enable by default.
4. **Pipeline:** the sage+Sol combo (dense first 20% + ramped Sol) delivers 1.22–1.26× over pure SageAttention on the attention math, and the int8 combo beats even all-Sol bf16.
5. KingGore's flex path is quick but answers a different question (hard mask, ~8–11% density) — do not compare it on speed alone.
6. Sol-Attn's advantage over SageAttention grows with sequence length; below ~4K tokens Sage stays the right default.

## Community benchmark: DGX Spark (SM121, 2026-08-07)

Contributed by a community user on **NVIDIA DGX Spark** (SM121, aarch64, CUDA 13.0, Triton 3.6.0). `B=1, H=24, D=128`, bf16, median of 10 after autotune warmup. Speedup is `SageAttention / Sol-Attn`:

| tokens | SDPA | SageAttention | Sol-Attn bf16 | Sol-Attn int8_qk | vs Sage | vs Sage (int8) |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 9.42 | 6.53 | 4.42 | 3.87 | 1.48× | 1.69× |
| 16,384 | 38.03 | 22.78 | 14.91 | 12.52 | 1.53× | 1.82× |
| 32,768 | 152.94 | 86.89 | 55.72 | 45.36 | 1.56× | 1.92× |

These ratios are above the 1.19–1.55× this README measures on RTX 5090. That is expected rather than surprising: Sol-Attn skips loading whole K/V blocks, so it saves memory bandwidth, and the DGX Spark's GB10 is bandwidth-bound (LPDDR5X unified memory) far more than an RTX 5090 is. A bandwidth-saving method returns more on this hardware. Absolute times are much slower than a 5090; only the ratios transfer.
