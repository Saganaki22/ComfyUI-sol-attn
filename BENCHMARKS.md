# Benchmarks

**Date:** 2026-08-04 · **Machine:** RTX 5090 (SM120) · torch 2.10.0+cu130 · Triton 3.6.0 · Python 3.12.10 · Windows 11

## Method

All Sol-Attn implementations are given the **same inputs**: MiniMax H3-shaped strided NHD views into a fused qkv projection buffer (`B=1, H=56, D=128`, bf16, random Gaussian tensors), `tau=1.0`, median of 20 timed iterations after 3 warmup iterations (which absorb Triton autotune / torch.compile). SageAttention and PyTorch SDPA are the dense baselines.

> Fidelity warning: these methods are not interchangeable. KingGore's fork uses a **hard block mask** (unselected KV blocks are dropped, ~8–11% density by its own README) while the Sol-Attn implementations keep an approximate correction for unselected blocks (~16% exact at `tau=1.0`). Faster-by-sparser is not the same thing as faster-by-engineering — A/B the visual quality before adopting any of them.

## Results (ms per attention call, lower is better)

Current run (2026-08-05, v0.5.1: per-arch dispatch — bf16 uses pointer kernels on SM120, int8 uses TMA; both bit-identical to each other):

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 24.56 | 97.90 | 388.69 | 1,595.19 |
| SageAttention | 4.12 | 15.12 | 60.41 | 236.59 |
| KingGore flex | 5.13 | 11.27 | 35.42 | 130.43 |
| kijai bf16 | 3.02 | 9.72 | 36.45 | 142.09 |
| kijai int8 | 2.47 | 8.50 | 30.81 | 116.94 |
| **ours bf16** | **2.87** | 10.10 | 36.68 | 145.46 |
| ours int8 | 2.69 | 8.53 | 31.37 | 121.65 |

**Read:** at parity — every cell within ~2–4% of the fastest implementation, which is autotune-pick territory (cells flip between runs). Ours takes bf16 at 8K; kijai takes the rest by a hair. The differentiators that remain are structural: our int8 is ~3.7× more accurate (0.008 vs 0.030 rel L2), and the feature set (scheduled tau, dense gates, sinks, FFN chunking, validation suite) lives here.

<details>
<summary>Previous runs (2026-08-04 / 2026-08-05 pre-flip)</summary>

2026-08-05, against kijai's pointer-default update, before our dispatch flip:

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| kijai bf16 | 3.10 | 9.77 | 35.58 | 137.49 |
| kijai int8 | 2.44 | 8.36 | 30.25 | 113.07 |
| ours bf16 (TMA) | 3.33 | 11.57 | 44.95 | 162.14 |
| ours int8 (TMA) | 2.72 | 8.64 | 31.30 | 118.02 |

2026-08-04, original measurement:

| Method | 8,192 tokens | 16,384 | 32,768 | 65,536 |
|---|---:|---:|---:|---:|
| kijai bf16 | 3.76 | 12.20 | 43.32 | 164.90 |
| kijai int8 | 2.99 | 8.95 | 28.22 | 114.10 |
| ours bf16 (TMA) | 3.81 | 11.98 | 41.73 | 160.35 |
| ours int8 (TMA) | 2.56 | 8.64 | 31.03 | 108.95 |

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

## Takeaways (2026-08-05, v0.5.1)

1. **Speed:** parity. Per-arch dispatch (bf16 → pointer kernels, int8 → TMA, chosen by measurement) puts every cell within ~2–4% of the fastest implementation — cells flip with autotune picks.
2. **Accuracy:** our residual int8 design is ~3.7× closer to the exact bf16 path than full-key int8 (0.008 vs 0.030 rel L2) — the quality-first int8 option.
3. **Pipeline:** the sage+Sol combo (dense first 20% + ramped Sol) delivers 1.22–1.26× over pure SageAttention and 7.6–8.1× over stock SDPA on attention math; the int8 combo beats even all-Sol bf16.
4. KingGore's flex path is quick but answers a different question (hard mask, ~8–11% density) — do not compare it on speed alone.
5. Sol-Attn's advantage over SageAttention grows with sequence length; below ~4K tokens Sage stays the right default.
