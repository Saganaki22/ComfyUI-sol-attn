# ComfyUI-sol-attn

<img width="1123" height="853" alt="Screenshot 2026-08-03 200716" src="https://github.com/user-attachments/assets/43b7626f-5a9b-48d7-990a-462a5ad13a82" />


**English** | **[中文](./README_ZH.md)**

**Version: v0.4.7**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![GPU](https://img.shields.io/badge/tested-RTX%205090%20(SM120)-76b900)](https://www.nvidia.com/)
[![Triton](https://img.shields.io/badge/Triton-3.6.0-blue)](https://github.com/triton-lang/triton)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Sparse attention and memory patches for video diffusion in ComfyUI, built around NVIDIA's **Sol-Attn** Triton reference kernel — running on consumer Blackwell (SM120 / RTX 50-series), which NVIDIA's public dispatcher does not enable. Ships a generic per-model Sol-Attn patch plus three MiniMax H3-specific nodes for copy-free attention, scheduled sparsity, and feed-forward peak-memory reduction.

> Legal note: the kernel in `sol_kernel/` is vendored NVIDIA source (Apache-2.0), modified locally. The SM120 enablement is this repository's change, not NVIDIA's.

## Why this repo

Measured on an RTX 5090 against the other Sol-Attn ComfyUI implementations (full tables in [BENCHMARKS.md](BENCHMARKS.md), 2026-08-04):

- **Fastest int8 attention at 8K / 16K / 65K tokens** (2.56 / 8.64 / 108.95 ms) and within ~10% at 32K, against kijai's Triton node, KingGore's flex fork, and SageAttention.
- **Best int8 accuracy** — our kernel quantizes only K's per-block *residual* and keeps the mean term exact in bf16: 0.008 relative L2 vs the exact path, ~3.7× closer than full-key int8 designs (0.030).
- **Zero-copy by design** — the kernel reads H3's fused qkv views directly; other TMA implementations copy q/k/v first (1.3–2.7 GiB extra peak memory at long lengths, plus the copy time).
- **Cross-validated math** — our bf16 path is bit-identical to kijai's independent implementation (0.000000), and SDPA-parity in all-exact mode (0.00097).
- **More than a kernel** — scheduled tau with graph preview, conditioning exact-KV sink, feed-forward chunking (−37% MLP peak), SM89–SM120 support, and honest per-call fallback.

## Features

- **Opt-in per-model patching** — only the model you wire through a node is affected; the rest of your graph is untouched.
- **Two Sol-Attn integration paths** — a generic hook-based node for any model, and a MiniMax H3 node that feeds the kernel strided views of the fused qkv projection with zero q/k/v copies, plus an exact-KV sink for H3's packed conditioning rows.
- **SM89 through SM120** — TMA descriptor kernels on SM90/100/120, pointer kernel twins on SM89 (RTX 40-series).
- **Scheduled sparsity** — ramp `tau` across sampling (sparse early, dense late) with a plotted schedule preview.
- **Feed-forward chunking** — caps MiniMax H3's MLP peak activation memory, bit-identical output.
- **Honest fallback** — any shape or GPU the kernel can't handle uses your normal attention backend and logs why; `strict` mode raises instead while validating a new environment.
- **No new dependencies** — torch and Triton only; matplotlib is used for the schedule plot when available.

## Prerequisites

- NVIDIA GPU: **SM89, SM90, SM100, or SM120** — SM90/100/120 run the TMA kernel path; SM89 (RTX 40-series) runs pointer kernel twins. Only SM120 is tested on hardware by this repository; the SM89 path is validated by forced dispatch, not on an SM89 GPU.
- PyTorch with CUDA, **bfloat16** support
- **Triton** with `triton.tools.tensor_descriptor` (TMA) — verified on 3.6.0
- ComfyUI (developed against 0.30.0)
- For the MiniMax H3 nodes: a MiniMax H3 checkpoint, e.g. `minimax_h3_fl2va_pruned_int8_convrot.safetensors` in `ComfyUI/models/diffusion_models/`
- matplotlib (optional — only for the tau schedule preview image)

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-sol-attn.git
```

Restart ComfyUI. There is nothing to pip-install beyond the prerequisites.

## Tested on

```text
RTX 5090 (SM120)  ·  torch 2.10.0+cu130  ·  Triton 3.6.0
Python 3.12.10    ·  ComfyUI 0.30.1      ·  Windows 11
MiniMax H3 (56 heads × 128, bf16, mask=None) — satisfies all kernel constraints
```

Sol-Attn's runtime constraints: `head_dim` exactly 128, bf16, no attention mask, 4D q/k/v with contiguous or TMA-compatible strides. Anything else falls back and logs the reason once per cause.

<img width="1430" height="360" alt="Screenshot 2026-08-03 200727" src="https://github.com/user-attachments/assets/cca19a4e-043b-4b6a-b0b6-3af079e54db2" />



## Nodes

<details>
<summary><strong>1. Sol-Attn (sparse attention)</strong> — generic per-model patch, <code>model_patches/attention</code></summary>

Routes any model's self-attention through Sol-Attn via ComfyUI's `optimized_attention_override` hook. Falls back to your existing attention override or ComfyUI's selected backend for anything the kernel can't handle.

```text
UNETLoader → Sol-Attn → BasicGuider
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | The model to patch. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `tau` | FLOAT | `1.0` | Routing threshold in standard deviations above the mean block score. Higher = more KV blocks take the approximate path = faster, lower fidelity. `1.0` is the Sol-Attn default. |
| `min_tokens` | INT | `8192` | Use the normal backend below this sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. Enable while validating a new GPU or Triton version. |
| `thresh_type` | COMBO | `diag` | `diag` (evaluated default) or `exact` — second-moment statistics for more precise routing at extra precompute cost. |
| `int8_qk` | BOOLEAN | `False` | Quantize q/k to int8 for the exact attention path. Measured 1.2–1.3× faster above 16K tokens at ~1% extra numerical error; slightly slower at 8K. This is a repository addition, not part of NVIDIA's source. |

**Output:** `model` (`MODEL`)

</details>

<details>
<summary><strong>2. MiniMax H3 Memory Efficient Sol Attention Patch</strong> — zero-copy H3 attention, <code>model_patches/attention</code></summary>

H3-specific alternative to the generic node. The generic node receives BHSD q/k/v from ComfyUI's attention hook and must make contiguous copies for the kernel; this node replaces the attention module's forward directly and feeds the kernel strided NHD views of the model's fused qkv projection — no q/k/v copies, same fused in-place RMSNorm+RoPE the stock model uses.

```text
UNETLoader → MiniMax H3 Memory Efficient Sol Attention Patch → BasicGuider
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model; anything else passes through unchanged with a warning. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `tau` | FLOAT | `1.0` | Same routing threshold as the generic node. |
| `min_tokens` | INT | `8192` | Use the stock attention forward below this packed sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. |
| `thresh_type` | COMBO | `diag` | Same estimator choice as node 1. |
| `int8_qk` | BOOLEAN | `False` | Same int8 q/k toggle as node 1. |
| `sink_conditioning` | COMBO | `exact_kv` | Keep H3's packed text/conditioning/reference/audio KV blocks exact (~3% cost, protects prompt adherence and audio sync). `exact_kv_and_rows` also runs those query rows dense (~20% cost). `off` disables the sink. |

Only the 30 main DiT blocks are patched; the token refiner and short sequences behave exactly as stock. It can follow a memory-efficient sage attention patch (e.g. KJNodes' MiniMax H3 one): applied **after** it, this node adopts the sage forward as its fallback, so gated and ineligible steps run mem-efficient sage while eligible steps run Sol-Attn. Applied **before** it, the sage patch shadows this node entirely — order matters.

**Output:** `model` (`MODEL`)

</details>

<details>
<summary><strong>3. MiniMax H3 Scheduled Sol Attention Patch</strong> — tau ramp with graph preview, <code>model_patches/attention</code></summary>

Same zero-copy attention path as node 2, but `tau` ramps across sampling: sparse on the early high-noise steps where attention structure is loose, denser on the late steps where detail forms. The current step is tracked from the diffusion timestep, so the schedule adapts to any step count automatically.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `tau_start` | FLOAT | `2.0` | tau on the first, highest-noise step. |
| `tau_end` | FLOAT | `0.8` | tau on the final, low-noise steps. |
| `curve` | COMBO | `linear` | `linear`, `cosine`, `sqrt`, `smoothstep`, `exponential`, or `step` (hard switch at the midpoint) — how tau interpolates between the two ends. |
| `min_tokens` | INT | `8192` | Use the stock attention forward below this packed sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. |
| `dense_percent` | FLOAT | `0.0` | Keep the stock dense attention for this fraction of early sampling — the Sol-Attn paper's recipe is `0.2`. `0` disables the gate. |
| `thresh_type` | COMBO | `diag` | Same estimator choice as node 1. |
| `int8_qk` | BOOLEAN | `False` | Same int8 q/k toggle as node 1. |
| `sink_conditioning` | COMBO | `exact_kv` | Same conditioning-sink choice as node 2. |

**Outputs:** `model` (`MODEL`), `tau_graph` (`IMAGE`) — wire to a Preview Image node to see the schedule curve.

</details>

<details>
<summary><strong>4. MiniMax H3 Chunk FeedForward</strong> — MLP peak-memory reduction, <code>model_patches/memory</code></summary>

Splits H3's token-local feed-forward over the packed sequence dimension while preserving ComfyUI's `linear_input_act` implementation inside each chunk. Independent of Sol-Attn — it works with any attention backend. Most effective on the INT8 ConvRot checkpoint, where the swiglu activation is fused into the INT8 quantizer and the `[tokens, 28672]` first projection dominates peak MLP memory; on a plain bf16 checkpoint the eager swiglu path holds extra intermediates and the saving is small.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `chunks` | INT | `2` | More chunks reduce peak MLP activation memory further. |
| `min_tokens` | INT | `4096` | Keep the normal full-width MLP below this packed sequence length. |

Chunking is token-independent math and retains H3's INT8 ConvRot path, whose activation scales are row-wise; outputs were bit-identical in testing. Inputs requiring gradients use the original unchunked MLP.

**Output:** `model` (`MODEL`)

</details>

## Benchmarks

Cross-repo comparison table (kijai, KingGore, SageAttention, SDPA): [BENCHMARKS.md](BENCHMARKS.md).

Everything below is one machine (see "Tested on"), one kernel build. Treat it as a smoke test with real numbers attached, not a benchmark suite.

<details>
<summary><strong>Attention speed — Sol-Attn vs SageAttention vs PyTorch SDPA</strong></summary>

H3 width (B=1, H=56, D=128, bf16), random tensors, median of 20 iterations after autotune warmup. Speedup is `baseline / Sol-Attn`:

| tokens | PyTorch SDPA (ms) | SageAttention (ms) | Sol-Attn strided (ms) | vs Sage | vs SDPA |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.45 | 0.60 | 0.89 | 0.67× | 1.63× |
| 8,192 | 23.96 | 3.72 | 3.25 | 1.14× | 7.36× |
| 16,384 | 84.95 | 13.94 | 10.08 | 1.38× | 8.42× |
| 32,768 | 352.15 | 55.90 | 38.73 | 1.44× | 9.09× |
| 65,536 | 1,350.54 | 221.06 | 153.56 | 1.44× | 8.79× |

- `0.67× vs Sage` at 2,048 tokens means **Sage is faster there** — below roughly 4K tokens Sage wins outright, which is why `min_tokens` defaults to 8,192.
- SageAttention is the fair baseline. PyTorch SDPA is shown only because other Sol-Attn plugins quote it — at these sizes it does not use a competitive kernel path.
- Random Gaussian inputs are the worst case for Sol-Attn's content-dependent routing, so real prompts should meet or beat these ratios; repeated runs vary by a few percent.
- The generic node's q/k/v copies cost a further 0.2–2 ms per call depending on length ("Sol generic" in the test output).
- With `int8_qk` enabled (repository addition), the same strided path measures 0.97× at 8,192, 1.30× at 16,384, 1.21× at 32,768, and 1.18× at 65,536 relative to the bf16 numbers above, at ~1% extra numerical error.

Full-model context: one controlled pair (MiniMax H3, 15 s, 480×864, 20 steps, `res_multistep`, fixed seed, same input image) measured Sage 9.91 s/it → Sol 8.92 s/it (−10%) with the hook-based node.

</details>

<details>
<summary><strong>Peak VRAM — attention, generic copies vs strided views</strong></summary>

Peak activation memory above resident for one H3 attention call:

| tokens | generic path (MiB) | strided path (MiB) | saved (MiB) |
|---:|---:|---:|---:|
| 8,192 | 452 | 116 | 336 |
| 16,384 | 903 | 231 | 672 |
| 32,768 | 1,806 | 462 | 1,344 |
| 65,536 | 3,612 | 924 | 2,688 |

The saving matches the three `[tokens, 7168]` bf16 contiguous copies the strided path avoids.

</details>

<details>
<summary><strong>Peak VRAM — feed-forward chunking (real INT8 checkpoint)</strong></summary>

One real first-block MLP from `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, loaded through ComfyUI's normal diffusion-model loader. Same input for both paths, peak above resident after warmup, outputs verified equal with `torch.testing.assert_close`:

| Tokens | FFN full | FFN chunked ×2 | Saved MiB | Saved % |
|---:|---:|---:|---:|---:|
| 8,192 | 644 MiB | 406 MiB | 238 MiB | 37.0% |
| 16,384 | 1,288 MiB | 812 MiB | 476 MiB | 37.0% |
| 32,768 | 2,576 MiB | 1,624 MiB | 952 MiB | 37.0% |
| 65,536 | 5,152 MiB | 3,248 MiB | 1,904 MiB | 37.0% |

Throughput was neutral in isolation (within noise, ±4%). These numbers are specific to the INT8 ConvRot path; see node 4's notes for the bf16 caveat.

</details>

<details>
<summary><strong>Correctness checks</strong></summary>

On the environment under "Tested on":

- Strided-view kernel output is **bit-identical** to contiguous input (max abs diff 0), including at ragged sequence lengths (8,191 / 12,345 / 38,247 tokens).
- All-exact mode (`tau=-100`, validation only) matches PyTorch SDPA at relative L2 error `0.00097`.
- The full patched H3 attention module matches the stock forward at relative L2 error `0.00009`, consistent with bf16 accumulation differences.
- The SM89 pointer kernel twins are bit-identical to the TMA kernels (bf16 and int8_qk); sink-forced-exact mode matches dense output.
- Chunked ×2 MLP output matches the full MLP exactly (`assert_close`, rtol=atol=0).

</details>

## Pairing with EasyCache

ComfyUI ships core **EasyCache**/`LazyCache` nodes (`comfy_extras/nodes_easycache.py`) that skip whole model evaluations when the input hasn't changed much — the maintained equivalent of TeaCache, and it handles H3's dual audio/video outputs. It composes with every node here: EasyCache skips some steps entirely, while Sol-Attn and chunking make the remaining steps cheaper and smaller. Don't max both approximations at once — an aggressive `reuse_threshold` plus a high `tau` will show in the output. A/B with a fixed seed.

## Console output

```text
[Sol-Attn] patched (tau=1.00, min_tokens=8192, strict=False)
[Sol-Attn] active
[Sol-Attn] dense fallback: <reason>
[MiniMax H3 Sol] patched 30 attention blocks (tau=1.00, min_tokens=8192, strict=False)
[MiniMax H3 FFN] patched 62 MLPs (chunks=2, min_tokens=4096)
```

Each distinct fallback reason is logged once per run. Also note the compile tax: Triton autotunes with `key=["T"]`, so the **first run at any new token count pays a JIT sweep inside the sampling loop** — change resolution or duration and you pay it again. Benchmark the second run.

## Caveats

- **Sol-Attn is approximate.** Output will not be bit-identical to dense attention; whether that shows in your content is your call — A/B it with `enabled`.
- **MiniMax H3 is not evaluated in the Sol-Attn paper.** H3 uses a joint packed sequence (text, conditioning, audio, video); the `sink_conditioning` option implements the paper's exact conditioning-K/V handling, but dense first-layer scheduling and the rest of the paper's more conservative recipe are not implemented.
- **SM120 support is this repository's change**, not NVIDIA's. NVIDIA's public source gate still names SM90/SM100. Report numerics issues here, not upstream against NVlabs/Sana.
- **The H3-specific nodes bypass ComfyUI's attention hook.** Other patches attached to `optimized_attention_override` do not run on blocks where Sol is active, and attention `transformer_options` patches are not applied on the Sol path.
- **The strided-view TMA layout is relaxed in this repository's copy of the validator**, exercised on SM120 only and verified at ragged sequence lengths (e.g. 38,247 tokens, real H3 size). The SM89 pointer kernels are validated by forced dispatch on SM120, not on SM89 hardware. Run with `strict=true` once on a new environment.
- NVIDIA's published ~2.0–2.3× figures are for Sol-Engine as a whole (CuTe kernels, NVFP4, block fusion, datacenter GPUs). This is the Triton reference kernel alone — a different thing entirely.
- The [KingGore Blackwell fork](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell) was evaluated at H3 width: 4.334 ms for its `flex_attention` path vs 3.256 ms for this Triton reference at 8,192 tokens. It uses a hard block mask (unselected blocks are dropped, not approximated), so it is a different method, and its import-time repair that moves files inside the installed PyTorch package is intentionally excluded here.

## Acknowledgements

- **Sol-Attn** — Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu, Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, and Song Han (NVIDIA Research, Efficient AI Team & Singapore Lab). The kernel, the method, and the preprocessing in `sol_kernel/` are theirs.
  - Project page: https://nvlabs.github.io/Sana/Sol-Attn/
  - Source: https://github.com/NVlabs/Sana/tree/sol-engine
  - Sol-Attn paper: https://arxiv.org/abs/2607.24027
  - Sol-Engine paper: https://arxiv.org/abs/2606.23743

```bibtex
@misc{li2026solattnacceleratingvideogeneration,
      title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
      Attention Sparsification},
      author={Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and Haozhe
      Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and Zeke Xie and
      Enze Xie and Song Han},
      year={2026},
      eprint={2607.24027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.24027},
}
```

- **[FlashAttention](https://github.com/dao-ailab/flash-attention)** (Tri Dao et al.) — NVIDIA's third-party notices record that parts of the SM90/SM100 scaffold in Sol-Engine derive from FlashAttention (BSD-3-Clause). Those files are not redistributed here.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** (comfyanonymous and contributors) — the `optimized_attention_override` hook and the object-patch machinery these nodes integrate against.
- **[ComfyUI-SolAttn](https://github.com/sumeetprashant/ComfyUI-SolAttn)** ([@sumeetprashant](https://github.com/sumeetprashant)) — the original hook-based integration and SM120 validation this repository extends with the MiniMax H3 nodes.
- **[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)** ([@KingGore](https://github.com/KingGore)) — an SM120-focused alternative using compiled PyTorch `flex_attention`; evaluated above, not adopted.
- **[ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn)** ([@woct0rdho](https://github.com/woct0rdho)) — established the opt-in `MODEL → MODEL` sparse-attention patch pattern this follows, and maintains the SageAttention Windows builds used as the fallback backend and baseline.
- **[RadialAttention](https://github.com/mit-han-lab/radial-attention)** (MIT Han Lab) — the sparse attention method that port wraps.
- **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)** and **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** ([@kijai](https://github.com/kijai)) — parallel prior art for sparse-attention patch nodes; the memory-efficient attention patch pattern the H3 nodes follow comes from KJNodes' `MiniMaxH3MemoryEfficientSageAttentionPatch`.
- **[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)** ([@kijai](https://github.com/kijai)) — kijai's own Triton Sol-Attn node. v0.4.0 adopts patterns proven there: the conditioning exact-KV sink, sigma gating through `transformer_options["sigmas"]`, pointer kernel twins for SM89, and the trimmed autotune lists. Independent implementations of the same kernel; no code is shared.
- **[Triton](https://github.com/triton-lang/triton)** (OpenAI and contributors) — compiles the kernel.
- **[SageAttention](https://github.com/thu-ml/SageAttention)** (thu-ml) — the dense baseline every number above is measured against.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana.
