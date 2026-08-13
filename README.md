# ComfyUI-sol-attn

<img width="1103" height="1085" alt="image" src="https://github.com/user-attachments/assets/00612c2c-aba0-4806-adca-bd58ec15b9dc" />



**English** | **[中文](./README_ZH.md)**

**Version: v0.6.1**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![GPU](https://img.shields.io/badge/tested-RTX%205090%20(SM120)-76b900)](https://www.nvidia.com/)
[![Triton](https://img.shields.io/badge/Triton-3.6.0-blue)](https://github.com/triton-lang/triton)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Sparse attention and memory patches for video diffusion in ComfyUI, built around NVIDIA's **Sol-Attn** Triton reference kernel and tuned for native-Windows consumer Blackwell (SM120 / RTX 50-series). Ships a generic per-model Sol-Attn patch plus four MiniMax H3-specific nodes for copy-free attention, scheduled sparsity, exact modulation fusion, and feed-forward peak-memory reduction.

> Legal note: the kernel in `sol_kernel/` is vendored NVIDIA source (Apache-2.0), modified locally. NVIDIA now ships an optional SM120 CuTe backend for Linux; this repository's native-Windows Triton pointer path and residual-int8 extensions remain local changes.

## v0.6.1

- **KJNodes Low-VRAM compatibility** — both MiniMax H3 Sol nodes now support the single-item activation-list handoff used by KJNodes' `MiniMax H3 Low VRAM Attention`. Sol peeks at the tensor while applying its eligibility gates, leaves the handoff intact for dense fallback, and consumes/releases it when the Sol path runs.
- **Combination covered by regression tests** — `KJ MiniMax H3 Low VRAM Attention → MiniMax H3 Memory Efficient Sol Attention` and the scheduled Sol variant now run together without the previous `'list' object has no attribute 'shape'` error. KJ's early activation release is preserved on both sparse and dense calls.
- **No numerical or kernel changes** — attention math, model weights, SM86/SM89/SM120 pointer dispatch, SM90/SM100/SM121 TMA dispatch, and output accuracy are unchanged. All seven regression tests pass, and a real KJ low-VRAM block-forward GPU integration test also passed. The v0.6.0 benchmark matrix remains current.
- **Existing limitation remains** — KJNodes' `MiniMax H3 Low VRAM Attention` still should not be combined with `MiniMax H3 Fused Modulation`, because both patch the complete H3 block forward. This release fixes its composition with the two local H3 **Sol Attention** nodes.

## v0.6.0

- **Faster SM120 forward dispatch** — RTX 5090 now uses the pointer forward kernels, while SM89 remains pointer and SM90/100/121 remain TMA. At H3's `B=1, T=8192, H=56, D=128` shape, the SM120 pointer path measured 1.25× the TMA throughput in bf16 and produced bit-identical output; residual-int8 also remained bit-identical.
- **Inline residual-int8 Q preparation** — SM86/SM89/SM120 diagonal-threshold pointer kernels quantize Q and derive the routing threshold from the BF16 Q tile already loaded by the forward. This removes the materialized Q-int8/Q-scale/threshold producer. At 32K H3 tokens it reduced measured peak allocation by 189 MiB; output matched the former path bit-for-bit for aligned/ragged lengths, exact sinks, and `int8_pv` on/off.
- **Exact MiniMax H3 modulation fusion** — the new `MiniMax H3 Fused Modulation` node fuses segmented AdaLN scale/shift and gated residual updates across all 50 DiT blocks. It explicitly reproduces eager BF16 rounding and matched a real ComfyUI `DiTBlock` bit-for-bit. At 38,247 × 5,376, scale/shift measured 1.91× faster and gate/add 1.22× faster in isolation.
- **Attention patches still compose** — the fusion resolves each block's attention and MLP dynamically, so the recommended `global KJ Sage → H3 memory-efficient Sage → local H3 Sol` chain remains intact. It does not install an attention backend or change calls outside Sol.
- **Fresh full release matrix** — a warmed-cache rerun at 8K/16K/32K/65K puts bf16 at 1.38–1.65× SageAttention throughput, residual `int8_qk` at 1.73–1.97×, and opt-in `int8_qk+pv` at 1.98–2.33×. Accuracy stayed at relative L2 `0.00802`/`0.01396` versus the bf16 Sol path.

## v0.5.9

- **Faster residual-int8 preprocessing** — K's 64-token block-mean reduction and residual quantization now run in one Triton kernel with one read of K. On an RTX 5090, the isolated K/V-summary + K-quant preprocessing segment measured 26–36% faster at 8K, 16K, and 65K tokens (the 32K result was noisier). The residual-int8 formulation and FP32 accumulation are unchanged; validation found no routing changes.
- **All supported architectures keep their forward path** — this is a shared preprocessing optimization. SM89 still uses the pointer forward kernels; SM90/100/120/121 still use the TMA forward kernels. No architecture dispatch was changed.
- **KJNodes composition is documented explicitly** — for the three-patch MiniMax H3 stack, apply global KJ Sage first, KJ's MiniMax memory-efficient Sage patch second, and this repository's MiniMax Sol patch last. Tokens that Sol declines use the captured memory-efficient Sage forward; attention calls outside that H3 object patch continue to use the global Sage override.

## Why this repo

Measured on an RTX 5090 (current local release matrix in [BENCHMARKS.md](BENCHMARKS.md), 2026-08-09; third-party comparison retained there as historical context):

- **1.38–1.65× Sage throughput in bf16** at 8K–65K; residual int8 reaches 1.73–1.97× and opt-in P·V int8 reaches 1.98–2.33×.
- **Best int8 accuracy** — our kernel quantizes only K's per-block *residual* and keeps the mean term exact in bf16: 0.008 relative L2 vs the exact path, ~3.6× closer than full-key int8 designs (0.029).
- **Zero-copy by design** — the kernel reads H3's fused qkv views directly; no contiguous copies, 1.3–2.7 GiB lower peak at long lengths.
- **Cross-validated math** — our bf16 path is bit-identical to kijai's independent implementation (0.000000), and SDPA-parity in all-exact mode (0.00097).
- **More than a kernel** — scheduled tau with graph preview, dense-step and dense-block gates, conditioning exact-KV sink, feed-forward chunking (−37% MLP peak), int8 q/k and P·V quantization, SM86–SM121 support, and honest per-call fallback.

## Features

- **Opt-in per-model patching** — only the model you wire through a node is affected; the rest of your graph is untouched.
- **Two Sol-Attn integration paths** — a generic hook-based node for any model, and a MiniMax H3 node that feeds the kernel strided views of the fused qkv projection with zero q/k/v copies, plus an exact-KV sink for H3's packed conditioning rows.
- **SM86 through SM121** — pointer kernels on SM86, SM89, and SM120; TMA descriptor kernels on SM90, SM100, and SM121. SM121 covers the DGX Spark.
- **Scheduled sparsity** — ramp `tau` across sampling (sparse early, dense late) with a plotted schedule preview.
- **Bit-exact H3 modulation fusion** — combines segmented AdaLN and gated residual elementwise work without changing eager BF16 output or the selected attention backend.
- **Feed-forward chunking** — caps MiniMax H3's MLP peak activation memory, bit-identical output.
- **Honest fallback** — any shape or GPU the kernel can't handle uses your normal attention backend and logs why; `strict` mode raises instead while validating a new environment.
- **No new dependencies** — torch and Triton only; matplotlib is used for the schedule plot when available.

## Prerequisites

- NVIDIA GPU: **SM86, SM89, SM90, SM100, SM120, or SM121** — SM86/89/120 run the pointer forward kernels; SM90/100/121 run TMA. SM120 is tested and benchmarked locally; SM86 is community hardware smoke-tested on an RTX 3090ti, SM89 is community hardware smoke-tested on an RTX 4080 SUPER ([issue #2](https://github.com/Saganaki22/ComfyUI-sol-attn/issues/2)); and SM121 (DGX Spark) is community-tested. SM86/SM89 performance is not yet benchmarked here.
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

Community-tested on **NVIDIA DGX Spark** (SM121, aarch64, CUDA 13.0, Triton 3.6.0) — see [BENCHMARKS.md](BENCHMARKS.md) for the DGX Spark numbers. Sol-Attn's speedup over SageAttention is higher there (1.48–1.92×) because the GB10's LPDDR5X unified memory is bandwidth-bound and Sol-Attn saves bandwidth.

Sol-Attn's runtime constraints: `head_dim` exactly 128, bf16, no attention mask, 4D q/k/v with contiguous or TMA-compatible strides. Anything else falls back and logs the reason once per cause.

<img width="1713" height="448" alt="Screenshot 2026-08-05 170148" src="https://github.com/user-attachments/assets/f9accb38-211e-4c02-afda-2f4c59c08b4d" />




## Pipeline order

Where each node sits in a MiniMax H3 workflow:

```text
UNETLoader → (LoRA / other model patches)
          → Patch Sage Attention (KJNodes, optional global fallback)
          → MiniMax H3 Memory Efficient Sage Attention Patch (KJNodes, optional H3 fallback)
          → MiniMax H3 Scheduled Sol Attention Patch   (or the Memory Efficient one)
          → MiniMax H3 Fused Modulation
          → MiniMax H3 Chunk FeedForward
          → EasyCache (optional, core node)
          → guider / sampler
```

- **The two H3 attention nodes are alternatives — never both.** The scheduled node is a superset of the memory-efficient one (set `tau_start = tau_end` to make them identical).
- **For all three attention patches, order is mandatory:** `Patch Sage Attention (KJNodes) → MiniMax H3 Memory Efficient Sage Attention Patch (KJNodes) → MiniMax H3 Sol patch (this repo)`. Our H3 wrapper captures the already-patched memory-efficient Sage forward and uses it for short sequences, gated steps, unsupported shapes, and non-strict kernel failures. Other attention calls use KJNodes' global Sage override. Putting the H3 Sage patch after Sol shadows Sol entirely.
- **The generic `SolAttentionPatch` is not a substitute for the MiniMax-specific Sol node in this stack.** KJNodes' MiniMax memory-efficient Sage patch replaces each H3 attention module's `forward` directly, bypassing the global attention override that the generic node uses.
- **Fused Modulation is attention-independent.** It can sit before or after the attention patches and dynamically calls whichever H3 attention/MLP object patches are installed. The order shown above is recommended for readability. KJNodes' separate `MiniMax H3 Low VRAM Attention` replaces the whole block forward and therefore does not stack with this node; the fusion leaves an already block-patched layer unchanged.
- For non-H3 models use the generic `SolAttentionPatch` instead: `UNETLoader → Sol-Attn → guider`.

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
| `tau` | FLOAT | `1.3` | Routing threshold in standard deviations above the mean block score. Higher = more KV blocks take the approximate path = faster, lower fidelity. `1.0` is the Sol-Attn paper default; `1.3` is the tuned default here. |
| `min_tokens` | INT | `4096` | Use the normal backend below this sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. Enable while validating a new GPU or Triton version. |
| `thresh_type` | COMBO | `diag` | `diag` (evaluated default) or `exact` — second-moment statistics for more precise routing at extra precompute cost. |
| `int8_qk` | BOOLEAN | `False` | Quantize q/k to int8 for the exact attention path. The SM120 inline-Q pointer path measured faster from 8K upward and reduced peak allocation by 189 MiB at 32K, at ~1% extra numerical error. This is a repository addition. |
| `int8_pv` | BOOLEAN | `False` | Additionally quantize the P·V dot to int8 (per-token P, per-channel V). Requires `int8_qk`. Speed is within noise of int8_qk alone on current hardware; accuracy drops to 0.014 rel L2 vs bf16 (from 0.008 with int8_qk alone). Opt-in. |

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
| `tau` | FLOAT | `1.3` | Same routing threshold as the generic node. |
| `min_tokens` | INT | `4096` | Use the stock attention forward below this packed sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. |
| `thresh_type` | COMBO | `diag` | Same estimator choice as node 1. |
| `int8_qk` | BOOLEAN | `False` | Same int8 q/k toggle as node 1. |
| `int8_pv` | BOOLEAN | `False` | Same int8 P·V toggle as node 1. Requires `int8_qk`. |
| `sink_conditioning` | COMBO | `exact_kv` | Keep H3's packed text/conditioning/reference/audio KV blocks exact (~3% cost, protects prompt adherence and audio sync). `exact_kv_and_rows` also runs those query rows dense (~20% cost). `off` disables the sink. |
| `dense_blocks` | STRING | empty | Transformer blocks to keep dense, e.g. `0-2,-1` for the first three and the last (negative counts from the end). First/last blocks are the most approximation-sensitive. Empty sparsifies all. |

Only the 50 main DiT blocks are patched; the token refiner and short sequences behave exactly as stock. It can follow a memory-efficient sage attention patch (e.g. KJNodes' MiniMax H3 one): applied **after** it, this node adopts the sage forward as its fallback, so gated and ineligible steps run mem-efficient sage while eligible steps run Sol-Attn. Applied **before** it, the sage patch shadows this node entirely — order matters.

**Output:** `model` (`MODEL`)

</details>

<details>
<summary><strong>3. MiniMax H3 Scheduled Sol Attention Patch</strong> — tau ramp with graph preview, <code>model_patches/attention</code></summary>

Same zero-copy attention path as node 2, but `tau` ramps across sampling: sparse on the early high-noise steps where attention structure is loose, denser on the late steps where detail forms. The current step is tracked from the diffusion timestep, so the schedule adapts to any step count automatically.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `tau_start` | FLOAT | `1.3` | tau on the first, highest-noise step. |
| `tau_end` | FLOAT | `0.8` | tau on the final, low-noise steps. |
| `curve` | COMBO | `linear` | `linear`, `cosine`, `sqrt`, `smoothstep`, `exponential`, or `step` (hard switch at the midpoint) — how tau interpolates between the two ends. |
| `min_tokens` | INT | `4096` | Use the stock attention forward below this packed sequence length. |
| `strict` | BOOLEAN | `False` | Raise kernel errors instead of falling back. |
| `dense_percent` | FLOAT | `0.0` | Keep the stock dense attention for this fraction of early sampling — the Sol-Attn paper's recipe is `0.2`. `0` disables the gate. |
| `thresh_type` | COMBO | `diag` | Same estimator choice as node 1. |
| `int8_qk` | BOOLEAN | `False` | Same int8 q/k toggle as node 1. |
| `int8_pv` | BOOLEAN | `False` | Same int8 P·V toggle as node 1. Requires `int8_qk`. |
| `sink_conditioning` | COMBO | `exact_kv` | Same conditioning-sink choice as node 2. |
| `dense_blocks` | STRING | empty | Same dense-block spec as node 2. |

**Outputs:** `model` (`MODEL`), `tau_graph` (`IMAGE`) — wire to a Preview Image node to see the schedule curve.

</details>

<details>
<summary><strong>4. MiniMax H3 Fused Modulation</strong> — bit-exact DiT elementwise fusion, <code>model_patches/optimization</code></summary>

Replaces H3's per-segment eager AdaLN scale/shift and gated residual updates with four Triton launches per block. The token-to-AdaLN-row lookup is built once per packed layout and shared by all 50 blocks. Intermediate BF16 rounding is reproduced explicitly; random tensors and a real ComfyUI `DiTBlock` matched eager output with `torch.equal`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model; anything else passes through unchanged. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |

This node does not select or wrap attention. Global Sage, KJNodes' H3 memory-efficient Sage, local H3 Sol, and the chunked MLP are resolved dynamically and continue to compose in either patch order. An unknown whole-block `forward` patch is left untouched instead of being bypassed.

**Output:** `model` (`MODEL`)

</details>

<details>
<summary><strong>5. MiniMax H3 Chunk FeedForward</strong> — MLP peak-memory reduction, <code>model_patches/memory</code></summary>

Splits H3's token-local feed-forward over the packed sequence dimension while preserving ComfyUI's `linear_input_act` implementation inside each chunk. Independent of Sol-Attn — it works with any attention backend. Most effective on the INT8 ConvRot checkpoint, where the swiglu activation is fused into the INT8 quantizer and the `[tokens, 28672]` first projection dominates peak MLP memory; on a plain bf16 checkpoint the eager swiglu path holds extra intermediates and the saving is small.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | required | A MiniMax H3 model. |
| `enabled` | BOOLEAN | `True` | Flip to `False` to A/B without rewiring. |
| `chunks` | INT | `2` | More chunks reduce peak MLP activation memory further. |
| `min_tokens` | INT | `8192` | Keep the normal full-width MLP below this packed sequence length. |

Chunking is token-independent math and retains H3's INT8 ConvRot path, whose activation scales are row-wise; outputs were bit-identical in testing. Inputs requiring gradients use the original unchunked MLP.

The benefit scales linearly with the packed sequence: the intermediate is 56 KB per token, so the saving grows from ~238 MiB at 8K tokens to ~1.9 GiB at 65K (measured, ×2 chunks, int8 checkpoint). Below `min_tokens` the node does nothing at all — lower it if you want relief on short renders too, raise it if you only care about long ones.

**Output:** `model` (`MODEL`)

</details>

## Benchmarks

Cross-repo comparison table (kijai, KingGore, SageAttention, SDPA): [BENCHMARKS.md](BENCHMARKS.md).

Everything below is one machine (see "Tested on"), one kernel build. Treat it as a smoke test with real numbers attached, not a benchmark suite.

<details>
<summary><strong>Attention speed — Sol-Attn vs SageAttention vs PyTorch SDPA</strong></summary>

H3 width (B=1, H=56, D=128, bf16 inputs), random tensors, `tau=1.0`, median of 20 iterations after three warmups. These are the second-process numbers after autotune caching:

| tokens | PyTorch SDPA (ms) | SageAttention (ms) | Sol bf16 | residual int8_qk | int8_qk+pv |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 23.97 | 3.93 | 2.84 | 2.27 | **1.98** |
| 16,384 | 94.85 | 14.68 | 9.57 | 8.10 | **7.05** |
| 32,768 | 378.41 | 58.09 | 35.72 | 30.06 | **25.91** |
| 65,536 | 1,516.16 | 229.63 | 139.28 | 116.64 | **98.46** |

- BF16 is 1.38–1.65× faster than Sage across the table. Residual `int8_qk` reaches 1.73–1.97× Sage throughput while remaining the default INT8 quality choice.
- `int8_qk+pv` is the fastest measured local path at 1.98–2.33× Sage throughput. It remains opt-in/default-off because relative L2 error versus bf16 rises from `0.00802` to `0.01396`.
- Below roughly 4K tokens Sage wins outright. `min_tokens` defaults to 4,096; raise it toward 8,192 if you want only the measured wins.
- SageAttention is the fair baseline. PyTorch SDPA is shown only because other Sol-Attn plugins quote it — at these sizes it does not use a competitive kernel path.
- Random Gaussian inputs are the worst case for Sol-Attn's content-dependent routing, so real prompts should meet or beat these ratios; repeated runs vary by a few percent.
- The generic node's q/k/v copies cost a further 0.2–2 ms per call depending on length ("Sol generic" in the test output).

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

Throughput was neutral in isolation (within noise, ±4%). These numbers are specific to the INT8 ConvRot path; see node 5's notes for the bf16 caveat.

</details>

<details>
<summary><strong>Correctness checks</strong></summary>

On the environment under "Tested on":

- Strided-view kernel output is **bit-identical** to contiguous input (max abs diff 0), including at ragged sequence lengths (8,191 / 12,345 / 38,247 tokens).
- All-exact mode (`tau=-100`, validation only) matches PyTorch SDPA at relative L2 error `0.00097`.
- The full patched H3 attention module matches the stock forward at relative L2 error `0.00009`, consistent with bf16 accumulation differences.
- The shared SM86/SM89/SM120 pointer implementation is bit-identical to the TMA kernels when cross-checked by forced dispatch on SM120 (bf16 and int8_qk); sink-forced-exact mode matches dense output.
- Inline-Q residual-int8 matches the former materialized-Q path bit-for-bit for aligned/ragged lengths, conditioning sinks, dense query rows, and `int8_pv` on/off.
- Fused H3 modulation/gating matches eager BF16 output bit-for-bit with both BF16 and FP32 AdaLN tables, including a real ComfyUI `DiTBlock` integration test.
- Chunked ×2 MLP output matches the full MLP exactly (`assert_close`, rtol=atol=0).

</details>

## Pairing with EasyCache

ComfyUI ships core **EasyCache**/`LazyCache` nodes (`comfy_extras/nodes_easycache.py`) that skip whole model evaluations when the input hasn't changed much — the maintained equivalent of TeaCache, and it handles H3's dual audio/video outputs. It composes with every node here: EasyCache skips some steps entirely, while Sol-Attn and chunking make the remaining steps cheaper and smaller. Don't max both approximations at once — an aggressive `reuse_threshold` plus a high `tau` will show in the output. A/B with a fixed seed.

## Console output

```text
[Sol-Attn] patched (tau=1.30, min_tokens=4096, strict=False)
[Sol-Attn] active
[Sol-Attn] dense fallback: <reason>
[MiniMax H3 Sol] patched 50 attention blocks (tau=1.30, min_tokens=4096, strict=False)
[MiniMax H3 fusion] patched 50 of 50 blocks
[MiniMax H3 fusion] active (38247 tokens, 8 modulation segments)
[MiniMax H3 FFN] patched 62 MLPs (chunks=2, min_tokens=8192)
```

Each distinct fallback reason is logged once per run. Also note the compile tax: Triton autotunes with `key=["T"]`, so the **first run at any new token count pays a JIT sweep inside the sampling loop** — timings are cached to disk, so a given token count pays it only once ever, but change resolution or duration and you pay it again for the new size. Benchmark the second run.

## Caveats

- **Sol-Attn is approximate.** Output will not be bit-identical to dense attention; whether that shows in your content is your call — A/B it with `enabled`.
- **MiniMax H3 is not evaluated in the Sol-Attn paper.** H3 uses a joint packed sequence (text, conditioning, audio, video); the `sink_conditioning` option implements the paper's exact conditioning-K/V handling, but dense first-layer scheduling and the rest of the paper's more conservative recipe are not implemented.
- **The native-Windows SM120 pointer path and SM121 enablement are repository integrations.** NVIDIA's current Sol-Engine branch has an optional Linux CuTe backend for SM120 and a portable Triton fallback; this package retains its own Comfy-compatible strided, residual-int8 implementation. SM121 (DGX Spark) support originated in community PR #3.
- **The H3-specific nodes bypass ComfyUI's attention hook.** Other patches attached to `optimized_attention_override` do not run on blocks where Sol is active, and attention `transformer_options` patches are not applied on the Sol path.
- **Architecture paths are intentionally separate.** SM86, SM89, and SM120 use the pointer family; SM120 is validated and benchmarked on RTX 5090, while SM86 and SM89 are hardware smoke-tested on RTX 30- and RTX 40-series respectively but are not yet performance-benchmarked here. The shared SM86/SM89/SM120 implementation is additionally cross-checked by forced dispatch on SM120. SM90/100/121 remain TMA. Run with `strict=true` once on a new environment.
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
