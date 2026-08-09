"""MiniMax H3 memory patches."""

import logging
import math
import re

import torch

log = logging.getLogger(__name__)

try:
    import comfy.model_management
    import comfy.quant_ops

    from comfy.ldm.minimax.model import PackedLayout

    from .sol_kernel import sol_attn
except Exception:  # Triton or ComfyUI kitchen ops unavailable; FFN node still loads
    sol_attn = None
    PackedLayout = None

SOL_ARCHES = {(8, 9), (9, 0), (10, 0), (12, 0), (12, 1)}


class _ChunkLog:
    def __init__(self):
        self.active = False

    def hit(self, tokens, chunks):
        if not self.active:
            log.info("[MiniMax H3 FFN] active (%d tokens, %d chunks)", tokens, chunks)
            self.active = True


def _make_chunked_forward(original_forward, chunks, min_tokens, chunk_log):
    def forward(x):
        if x.ndim != 2 or x.shape[0] < min_tokens or x.requires_grad:
            return original_forward(x)

        chunk_log.hit(x.shape[0], chunks)
        output = torch.empty_like(x)
        offset = 0
        for part in x.chunk(chunks, dim=0):
            end = offset + part.shape[0]
            output[offset:end].copy_(original_forward(part))
            offset = end
        return output

    forward._minimax_h3_ffn_fallback = original_forward
    return forward


class MiniMaxH3ChunkFeedForward:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "chunks": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "More chunks reduce peak MLP activation memory but add overhead.",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 256,
                        "max": 131072,
                        "step": 256,
                        "tooltip": "Keep the normal full-width MLP below this packed sequence length.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/memory"
    DESCRIPTION = (
        "Chunk MiniMax H3's token-local feed-forward activations to reduce peak "
        "VRAM. More chunks may reduce throughput or produce small numerical differences."
    )

    def patch(self, model, enabled, chunks, min_tokens):
        chunks = int(chunks)
        if not enabled or chunks == 1:
            return (model,)

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        token_refiner = getattr(diffusion_model, "token_refiner", None)
        refiner_blocks = getattr(token_refiner, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None or refiner_blocks is None:
            log.warning("[MiniMax H3 FFN] expected a MiniMax H3 model; returning it unchanged")
            return (model,)

        patched = model.clone()
        paths = [f"diffusion_model.blocks.{i}.mlp.forward" for i in range(len(blocks))]
        paths.extend(f"diffusion_model.token_refiner.blocks.{i}.mlp.forward" for i in range(len(refiner_blocks)))
        chunk_log = _ChunkLog()
        for path in paths:
            original_forward = patched.get_model_object(path)
            if hasattr(original_forward, "_minimax_h3_ffn_fallback"):
                original_forward = original_forward._minimax_h3_ffn_fallback
            patched.add_object_patch(
                path,
                _make_chunked_forward(original_forward, chunks, int(min_tokens), chunk_log),
            )

        log.info(
            "[MiniMax H3 FFN] patched %d MLPs (chunks=%d, min_tokens=%d)",
            len(paths),
            chunks,
            int(min_tokens),
        )
        return (patched,)


class _FusionUnsupported(Exception):
    pass


class _FusionLog:
    def __init__(self):
        self.active = False
        self.fallbacks = set()

    def hit(self, tokens, segments):
        if not self.active:
            log.info(
                "[MiniMax H3 fusion] active (%d tokens, %d modulation segments)",
                tokens,
                segments,
            )
            self.active = True

    def miss(self, reason):
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.info("[MiniMax H3 fusion] eager fallback: %s", reason)


class _SegmentIndexCache:
    """One request-layout lookup shared by all 50 patched H3 blocks."""

    def __init__(self, make_segment_index):
        self.make_segment_index = make_segment_index
        self.key = None
        self.value = None

    def get(self, tokens, segments, device, table_rows):
        normalized = tuple((int(a), int(b), int(row)) for a, b, row in segments)
        if not normalized or max(row for _, _, row in normalized) >= int(table_rows):
            raise ValueError("modulation segment references a missing AdaLN row")
        key = (str(device), int(tokens), normalized)
        if key != self.key:
            self.value = self.make_segment_index(tokens, normalized, device)
            self.key = key
        return self.value


def _make_fused_h3_block_forward(
    block,
    fallback_forward,
    index_cache,
    fusion_log,
    fused_modulate,
    fused_gate_add,
):
    """Replace only H3's modulation/residual elementwise work.

    Attention and MLP calls are resolved from ``block`` at execution time, so
    object patches installed by KJNodes, the local Sol nodes, or the FFN chunk
    node continue to compose regardless of node order.
    """

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        try:
            if (
                x.ndim != 2
                or x.dtype != torch.bfloat16
                or x.device.type != "cuda"
                or x.requires_grad
            ):
                raise _FusionUnsupported("requires a non-autograd CUDA BF16 [tokens, hidden] activation")

            mods = block.adaln_proj(t_emb)
            if len(mods) != 6:
                raise _FusionUnsupported("expected six H3 AdaLN modulation tables")
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
            table_rows = min(int(table.shape[0]) for table in mods)
            row_index = index_cache.get(x.shape[0], mod_segments, x.device, table_rows)

            # Everything up to the first residual update leaves x untouched;
            # an eligibility/kernel failure here can safely use eager forward.
            h = fused_modulate(block.norm1(x), shift_msa, scale_msa, row_index)
        except _FusionUnsupported as exc:
            fusion_log.miss(str(exc))
            return fallback_forward(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
        except Exception as exc:
            # x is still pristine, so a backend/compiler incompatibility can
            # safely fall back. After the first gate below, exceptions must
            # propagate because eager retry would consume a modified residual.
            fusion_log.miss(f"{type(exc).__name__}: {exc}")
            return fallback_forward(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )

        # Attention is intentionally outside the fallback catch: strict errors
        # from a Sol/Sage patch must propagate, and should never be converted
        # into an eager full-block retry by this unrelated fusion.
        attention = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        x = fused_gate_add(x, gate_msa, attention, row_index)
        h = fused_modulate(block.norm2(x), shift_mlp, scale_mlp, row_index)
        x = fused_gate_add(x, gate_mlp, block.mlp(h), row_index)
        fusion_log.hit(x.shape[0], len(mod_segments))
        return x

    forward._minimax_h3_fusion_fallback = fallback_forward
    return forward


class MiniMaxH3FusedModulation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = (
        "Fuse MiniMax H3's segmented AdaLN scale/shift and gated residual "
        "updates while reproducing the eager BF16 rounding exactly. Independent "
        "of the selected attention backend."
    )

    def patch(self, model, enabled):
        if not enabled:
            return (model,)

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            log.warning("[MiniMax H3 fusion] expected a MiniMax H3 model; returning it unchanged")
            return (model,)

        try:
            from .sol_kernel.h3_fusion import (
                fused_gate_add_,
                fused_modulate_,
                make_segment_index,
            )
        except Exception as exc:
            raise RuntimeError(
                "MiniMax H3 Fused Modulation requires the installed Triton runtime"
            ) from exc

        patched = model.clone()
        fusion_log = _FusionLog()
        index_cache = _SegmentIndexCache(make_segment_index)
        installed = 0
        for i in range(len(blocks)):
            path = f"diffusion_model.blocks.{i}.forward"
            block = patched.get_model_object(f"diffusion_model.blocks.{i}")
            prior = getattr(patched, "object_patches", {}).get(path)
            if prior is not None and not hasattr(prior, "_minimax_h3_fusion_fallback"):
                log.warning(
                    "[MiniMax H3 fusion] block %d already has an unknown forward patch; leaving it unchanged",
                    i,
                )
                continue
            fallback_forward = prior if prior is not None else block.forward
            if hasattr(fallback_forward, "_minimax_h3_fusion_fallback"):
                fallback_forward = fallback_forward._minimax_h3_fusion_fallback
            patched.add_object_patch(
                path,
                _make_fused_h3_block_forward(
                    block,
                    fallback_forward,
                    index_cache,
                    fusion_log,
                    fused_modulate_,
                    fused_gate_add_,
                ),
            )
            installed += 1

        log.info("[MiniMax H3 fusion] patched %d of %d blocks", installed, len(blocks))
        return (patched,)


class _Unsupported(Exception):
    pass


class _SolLog:
    def __init__(self):
        self.active = False
        self.fallbacks = set()

    def hit(self, tokens):
        if not self.active:
            log.info("[MiniMax H3 Sol] active (%d tokens)", tokens)
            self.active = True

    def miss(self, reason):
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.info("[MiniMax H3 Sol] dense fallback: %s", reason)


def _make_sol_attention_forward(attn, fallback_forward, tau, min_tokens, strict, sol_log,
                                thresh_type="diag", dense_percent=0.0, progress_fn=None,
                                int8_qk=False, int8_pv=False, sink_conditioning="exact_kv"):
    """Sol-Attn on the packed NHD views of the fused qkv buffer, no q/k/v copies.

    `tau` may be a float or a callable of the current sigma. When `progress_fn`
    is given, calls earlier than `dense_percent` of the run use the stock dense
    forward instead. `sink_conditioning` forces H3's packed conditioning KV
    blocks exact ("exact_kv", plus dense conditioning query rows with
    "exact_kv_and_rows").
    """
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim

    def forward(x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        try:
            if x.ndim != 2 or s < min_tokens or x.requires_grad:
                raise _Unsupported("below min_tokens or autograd requested")
            if x.dtype != torch.bfloat16 or x.device.type != "cuda":
                raise _Unsupported("requires bfloat16 on CUDA")
            if head_dim != 128:
                raise _Unsupported(f"head_dim {head_dim} != 128")
            arch = torch.cuda.get_device_capability(x.device)
            if arch not in SOL_ARCHES:
                raise _Unsupported(f"unsupported SM{arch[0]}{arch[1]}")

            sigmas = (transformer_options or {}).get("sigmas")
            sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) and sigmas.numel() > 0 else None
            if dense_percent > 0.0 and progress_fn is not None and progress_fn(sigma) < dense_percent:
                raise _Unsupported(f"dense first {dense_percent:.0%} of sampling")

            sink_blocks = (0, 0)
            sink_q = (0, 0)
            if sink_conditioning != "off":
                span = (transformer_options or {}).get("sol_h3_video_span")
                if span is not None:
                    video_start, video_stop = span
                    if 0 < video_start and s >= video_stop:
                        sink_blocks = (0, (video_start + 63) // 64)
                        if sink_conditioning == "exact_kv_and_rows":
                            sink_q = sink_blocks

            q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
            q = q.view(1, s, heads, head_dim)
            k = k.view(1, s, heads, head_dim)
            v = v.view(1, s, heads, head_dim)
            if rope_freqs is not None:
                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps,
                    rot_dim=rope_freqs.shape[-3] * 2,
                )
            else:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            out = sol_attn(q, k, v, tau=tau(sigma) if callable(tau) else tau,
                           thresh_type=thresh_type, int8_qk=int8_qk, int8_pv=int8_pv,
                           sink_blocks=sink_blocks, sink_q=sink_q)
            sol_log.hit(s)
            return attn.out_proj(out.view(s, inner))
        except _Unsupported as e:
            sol_log.miss(str(e))
        except Exception as e:
            if strict:
                raise
            sol_log.miss(f"{type(e).__name__}: {e}")
        return fallback_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    forward._minimax_h3_sol_fallback = fallback_forward
    return forward


class _TauSchedule:
    """tau ramp driven by the sampler's current sigma.

    ComfyUI publishes the active timestep per model call in
    transformer_options["sigmas"]; sigma_hi/sigma_lo are the run's endpoints
    mapped at patch time, so progress is 0 on the first step and 1 on the last.
    """

    def __init__(self, tau_start, tau_end, curve, sigma_hi, sigma_lo):
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.curve = curve
        self.sigma_hi = float(sigma_hi)
        self.span = max(float(sigma_hi) - float(sigma_lo), 1e-8)

    def weight(self, f):
        if self.curve == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * f)
        if self.curve == "sqrt":
            return math.sqrt(f)
        if self.curve == "smoothstep":
            return f * f * (3.0 - 2.0 * f)
        if self.curve == "exponential":
            return math.expm1(3.0 * f) / math.expm1(3.0)
        if self.curve == "step":
            return 1.0 if f >= 0.5 else 0.0
        return f

    def progress(self, sigma):
        if sigma is None:
            return 1.0
        return min(max((self.sigma_hi - sigma) / self.span, 0.0), 1.0)

    def tau(self, sigma):
        return self.tau_end + (self.tau_start - self.tau_end) * self.weight(1.0 - self.progress(sigma))


def _make_span_injector(original_forward):
    """Publish H3's video segment span into transformer_options for the sink gate.

    Mirrors MiniMaxH3Model._forward's layout handling: reuse the payload's
    prebuilt layout when its signature matches, rebuild it the same way
    otherwise. Adds one small layout construction per model call at most.
    """

    def forward(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        if isinstance(transformer_options, dict) and PackedLayout is not None:
            payload = minimax_payload or {}
            video_x, audio_x = x[0], x[1]
            signature = (
                context.shape[1],
                video_x.shape[2],
                -(-video_x.shape[3] // 2) * 2,
                -(-video_x.shape[4] // 2) * 2,
                audio_x.shape[-1],
            )
            layout = payload.get("layout")
            try:
                if layout is None or layout.signature != signature:
                    layout = PackedLayout(
                        *signature,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                        frame_count=payload.get("frame_count"),
                    )
                span = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
            except Exception:
                span = None
            if span is not None:
                transformer_options["sol_h3_video_span"] = span
        return original_forward(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    forward._minimax_h3_span_fallback = original_forward
    return forward


def _install_span_injector(patched):
    model_forward = patched.get_model_object("diffusion_model._forward")
    if hasattr(model_forward, "_minimax_h3_span_fallback"):
        model_forward = model_forward._minimax_h3_span_fallback
    patched.add_object_patch("diffusion_model._forward", _make_span_injector(model_forward))


def _parse_dense_blocks(spec, count):
    """Parse "0-3,47,-1" into absolute block indices; negatives count from the end."""
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"cannot parse dense_blocks entry {part!r}; use indices and ranges like '0-3,47,-1'")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return out


def _plot_tau_schedule(tau_start, tau_end, curve, dense_percent=0.0, width=512, height=320):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("[MiniMax H3 Sol] matplotlib unavailable; tau graph is blank")
        return torch.zeros((1, height, width, 3))

    schedule = _TauSchedule(tau_start, tau_end, curve, 1.0, 0.0)  # weight() only
    progress = [i / 100 for i in range(101)]
    taus = [tau_end + (tau_start - tau_end) * schedule.weight(1.0 - p) for p in progress]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    if dense_percent > 0.0:
        ax.axvspan(0, dense_percent * 100, color="gray", alpha=0.25, label="dense (stock)")
        ax.legend(loc="best", fontsize=8)
    ax.plot([p * 100 for p in progress], taus)
    ax.set_xlabel("sampling progress (%)")
    ax.set_ylabel("tau")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(1.05 * max(tau_start, tau_end), 0.1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    image = torch.frombuffer(fig.canvas.buffer_rgba(), dtype=torch.uint8)
    image = image.reshape(height, width, 4)[:, :, :3].float() / 255.0
    plt.close(fig)
    return image.unsqueeze(0).clone()


class MiniMaxH3ScheduledSolAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau_start": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "tau on the first, highest-noise step. Higher = "
                        "more blocks take the approximate path = faster, lower fidelity.",
                    },
                ),
                "tau_end": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "tau on the final, low-noise steps where detail forms. "
                        "Lower = denser attention at the end of sampling.",
                    },
                ),
                "curve": (
                    ["linear", "cosine", "sqrt", "smoothstep", "exponential", "step"],
                    {
                        "default": "linear",
                        "tooltip": "How tau interpolates between tau_start and tau_end "
                        "across sampling. step switches at the midpoint.",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 131072,
                        "step": 256,
                        "tooltip": "Use the stock attention forward below this "
                        "packed sequence length.",
                    },
                ),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Raise kernel errors instead of falling back. "
                        "Enable while validating a new GPU or Triton version.",
                    },
                ),
                "dense_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "Keep the stock dense attention for this fraction "
                        "of early sampling (the Sol-Attn paper's recipe: 0.2). "
                        "0 disables the gate.",
                    },
                ),
                "thresh_type": (
                    ["diag", "exact"],
                    {
                        "default": "diag",
                        "tooltip": "Routing threshold estimator. diag is the "
                        "evaluated default; exact uses second-moment statistics "
                        "for more precise routing at extra precompute cost.",
                    },
                ),
                "int8_qk": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Quantize q/k to int8 for the exact attention "
                        "path. On SM120 the inline-Q pointer path is faster from "
                        "8K upward and reduces peak memory (189 MiB at 32K), at "
                        "~1% extra numerical error.",
                    },
                ),
                "int8_pv": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also quantize the P*V dot to int8 (per-token P, "
                        "per-channel V). Speed is within noise of int8_qk alone; "
                        "accuracy drops to 0.014 rel L2. Requires int8_qk. Opt-in.",
                    },
                ),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {
                        "default": "exact_kv",
                        "tooltip": "Keep H3's packed text/conditioning/reference/audio "
                        "KV blocks exact (~3% cost, protects prompt adherence and "
                        "audio sync). exact_kv_and_rows also runs those query rows "
                        "dense (~20% cost). off disables the sink.",
                    },
                ),
                "dense_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Transformer blocks to keep dense, e.g. '0-2,-1' "
                        "for the first three and the last. Negative indices count "
                        "from the end. First and last blocks are the most "
                        "approximation-sensitive. Empty sparsifies all.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "tau_graph")
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "MiniMax H3 memory-efficient Sol attention with tau ramped across "
        "sampling: sparse on early high-noise steps, denser on late detail "
        "steps. tau_graph previews the schedule; wire it to a Preview Image node."
    )

    def patch(self, model, enabled, tau_start, tau_end, curve, min_tokens, strict, dense_percent, thresh_type, int8_qk, int8_pv, sink_conditioning, dense_blocks):
        graph = _plot_tau_schedule(float(tau_start), float(tau_end), curve, float(dense_percent))
        if not enabled:
            return (model, graph)
        if sol_attn is None:
            raise RuntimeError(
                "MiniMax H3 Scheduled Sol Attention Patch requires Triton; "
                "the Sol Triton backend failed to import."
            )

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            log.warning("[MiniMax H3 Sol] expected a MiniMax H3 model; returning it unchanged")
            return (model, graph)

        patched = model.clone()
        model_sampling = patched.get_model_object("model_sampling")
        schedule = _TauSchedule(
            tau_start,
            tau_end,
            curve,
            float(model_sampling.percent_to_sigma(0.0)),
            float(model_sampling.percent_to_sigma(1.0)),
        )
        _install_span_injector(patched)

        sol_log = _SolLog()
        dense = _parse_dense_blocks(dense_blocks, len(blocks))
        for i in range(len(blocks)):
            if i in dense:
                continue
            attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
            # adopt an earlier node's attn.forward patch (e.g. a memory-efficient
            # sage patch) as the fallback for gated/ineligible calls
            prior = getattr(patched, "object_patches", {}).get(f"diffusion_model.blocks.{i}.attn.forward")
            fallback_forward = prior if prior is not None else attn.forward
            if hasattr(fallback_forward, "_minimax_h3_sol_fallback"):
                fallback_forward = fallback_forward._minimax_h3_sol_fallback
            patched.add_object_patch(
                f"diffusion_model.blocks.{i}.attn.forward",
                _make_sol_attention_forward(
                    attn, fallback_forward, schedule.tau, int(min_tokens), bool(strict), sol_log,
                    thresh_type, float(dense_percent), schedule.progress, bool(int8_qk),
                    bool(int8_pv), sink_conditioning,
                ),
            )

        log.info(
            "[MiniMax H3 Sol] scheduled tau %.2f -> %.2f (%s) on %d of %d blocks (min_tokens=%d, strict=%s, dense=%.0f%%, thresh=%s)",
            float(tau_start),
            float(tau_end),
            curve,
            len(blocks) - len(dense),
            len(blocks),
            int(min_tokens),
            bool(strict),
            100 * float(dense_percent),
            thresh_type,
        )
        return (patched, graph)


class MiniMaxH3MemoryEfficientSolAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Routing threshold. Higher = more blocks take "
                        "the approximate path = faster, lower fidelity. "
                        "1.0 is the Sol-Attn paper default; 1.3 is tuned here.",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 256,
                        "max": 131072,
                        "step": 256,
                        "tooltip": "Use the stock attention forward below this "
                        "packed sequence length.",
                    },
                ),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Raise kernel errors instead of falling back. "
                        "Enable while validating a new GPU or Triton version.",
                    },
                ),
                "thresh_type": (
                    ["diag", "exact"],
                    {
                        "default": "diag",
                        "tooltip": "Routing threshold estimator. diag is the "
                        "evaluated default; exact uses second-moment statistics "
                        "for more precise routing at extra precompute cost.",
                    },
                ),
                "int8_qk": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Quantize q/k to int8 for the exact attention "
                        "path. On SM120 the inline-Q pointer path is faster from "
                        "8K upward and reduces peak memory (189 MiB at 32K), at "
                        "~1% extra numerical error.",
                    },
                ),
                "int8_pv": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also quantize the P*V dot to int8 (per-token P, "
                        "per-channel V). Speed is within noise of int8_qk alone; "
                        "accuracy drops to 0.014 rel L2. Requires int8_qk. Opt-in.",
                    },
                ),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {
                        "default": "exact_kv",
                        "tooltip": "Keep H3's packed text/conditioning/reference/audio "
                        "KV blocks exact (~3% cost, protects prompt adherence and "
                        "audio sync). exact_kv_and_rows also runs those query rows "
                        "dense (~20% cost). off disables the sink.",
                    },
                ),
                "dense_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Transformer blocks to keep dense, e.g. '0-2,-1' "
                        "for the first three and the last. Negative indices count "
                        "from the end. First and last blocks are the most "
                        "approximation-sensitive. Empty sparsifies all.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Run MiniMax H3 self-attention through Sol-Attn on strided views of the "
        "fused qkv projection, avoiding the q/k/v copies the generic Sol-Attn "
        "node makes. Blocks that do not meet the kernel's constraints use the "
        "stock attention forward."
    )

    def patch(self, model, enabled, tau, min_tokens, strict, thresh_type, int8_qk, int8_pv, sink_conditioning, dense_blocks):
        if not enabled:
            return (model,)
        if sol_attn is None:
            raise RuntimeError(
                "MiniMax H3 Memory Efficient Sol Attention Patch requires Triton; "
                "the Sol Triton backend failed to import."
            )

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            log.warning("[MiniMax H3 Sol] expected a MiniMax H3 model; returning it unchanged")
            return (model,)

        patched = model.clone()
        _install_span_injector(patched)
        sol_log = _SolLog()
        dense = _parse_dense_blocks(dense_blocks, len(blocks))
        for i in range(len(blocks)):
            if i in dense:
                continue
            attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
            # adopt an earlier node's attn.forward patch (e.g. a memory-efficient
            # sage patch) as the fallback for gated/ineligible calls
            prior = getattr(patched, "object_patches", {}).get(f"diffusion_model.blocks.{i}.attn.forward")
            fallback_forward = prior if prior is not None else attn.forward
            if hasattr(fallback_forward, "_minimax_h3_sol_fallback"):
                fallback_forward = fallback_forward._minimax_h3_sol_fallback
            patched.add_object_patch(
                f"diffusion_model.blocks.{i}.attn.forward",
                _make_sol_attention_forward(
                    attn, fallback_forward, float(tau), int(min_tokens), bool(strict), sol_log,
                    thresh_type, int8_qk=bool(int8_qk), int8_pv=bool(int8_pv), sink_conditioning=sink_conditioning,
                ),
            )

        log.info(
            "[MiniMax H3 Sol] patched %d of %d attention blocks (tau=%.2f, min_tokens=%d, strict=%s)",
            len(blocks) - len(dense),
            len(blocks),
            float(tau),
            int(min_tokens),
            bool(strict),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChunkFeedForward": MiniMaxH3ChunkFeedForward,
    "MiniMaxH3FusedModulation": MiniMaxH3FusedModulation,
    "MiniMaxH3MemoryEfficientSolAttentionPatch": MiniMaxH3MemoryEfficientSolAttentionPatch,
    "MiniMaxH3ScheduledSolAttentionPatch": MiniMaxH3ScheduledSolAttentionPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChunkFeedForward": "MiniMax H3 Chunk FeedForward",
    "MiniMaxH3FusedModulation": "MiniMax H3 Fused Modulation",
    "MiniMaxH3MemoryEfficientSolAttentionPatch": "MiniMax H3 Memory Efficient Sol Attention Patch",
    "MiniMaxH3ScheduledSolAttentionPatch": "MiniMax H3 Scheduled Sol Attention Patch",
}
