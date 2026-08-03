"""MiniMax H3 memory patches."""

import logging
import math

import torch

log = logging.getLogger(__name__)

try:
    import comfy.model_management
    import comfy.quant_ops

    from .sol_kernel import sol_attn
except Exception:  # Triton or ComfyUI kitchen ops unavailable; FFN node still loads
    sol_attn = None

SOL_ARCHES = {(9, 0), (10, 0), (12, 0)}


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
                        "default": 4096,
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
                                int8_qk=False):
    """Sol-Attn on the packed NHD views of the fused qkv buffer, no q/k/v copies.

    `tau` may be a float or a zero-argument callable evaluated per call. When
    `progress_fn` is given, calls earlier than `dense_percent` of the run use
    the stock dense forward instead.
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
            if dense_percent > 0.0 and progress_fn is not None and progress_fn() < dense_percent:
                raise _Unsupported(f"dense first {dense_percent:.0%} of sampling")

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

            out = sol_attn(q, k, v, tau=tau() if callable(tau) else tau, thresh_type=thresh_type, int8_qk=int8_qk)
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
    """tau ramp driven by the diffusion timestep.

    Sigma descends monotonically within a run, so the first model call carries
    the highest timestep; any increase means a new run started. tau moves from
    tau_start at the first step to tau_end as sigma approaches zero.
    """

    def __init__(self, tau_start, tau_end, curve):
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.curve = curve
        self.t = None
        self.t_max = None

    def track(self, t):
        if self.t is None or t > self.t:
            self.t_max = t
        self.t = t

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

    def tau(self):
        if self.t is None or not self.t_max:
            return self.tau_end
        f = min(max(self.t / self.t_max, 0.0), 1.0)
        return self.tau_end + (self.tau_start - self.tau_end) * self.weight(f)

    def progress(self):
        """Fraction of the run completed: 0 on the first step, 1 at the end."""
        if self.t is None or not self.t_max:
            return 1.0
        f = min(max(self.t / self.t_max, 0.0), 1.0)
        return 1.0 - f


def _make_timestep_tracker(original_forward, schedule):
    def forward(*args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if torch.is_tensor(timestep) and timestep.numel() > 0:
            schedule.track(float(timestep.flatten()[0]))
        return original_forward(*args, **kwargs)

    forward._minimax_h3_tracker_fallback = original_forward
    return forward


def _plot_tau_schedule(tau_start, tau_end, curve, dense_percent=0.0, width=512, height=320):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("[MiniMax H3 Sol] matplotlib unavailable; tau graph is blank")
        return torch.zeros((1, height, width, 3))

    schedule = _TauSchedule(tau_start, tau_end, curve)
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
                        "default": 2.0,
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
                        "default": 8192,
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
                        "path. Faster above ~16K tokens (measured 1.2-1.3x) at "
                        "~1% extra numerical error; slightly slower at 8K.",
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

    def patch(self, model, enabled, tau_start, tau_end, curve, min_tokens, strict, dense_percent, thresh_type, int8_qk):
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
        schedule = _TauSchedule(tau_start, tau_end, curve)
        model_forward = patched.get_model_object("diffusion_model.forward")
        if hasattr(model_forward, "_minimax_h3_tracker_fallback"):
            model_forward = model_forward._minimax_h3_tracker_fallback
        patched.add_object_patch(
            "diffusion_model.forward",
            _make_timestep_tracker(model_forward, schedule),
        )

        sol_log = _SolLog()
        for i in range(len(blocks)):
            attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
            fallback_forward = attn.forward
            if hasattr(fallback_forward, "_minimax_h3_sol_fallback"):
                fallback_forward = fallback_forward._minimax_h3_sol_fallback
            patched.add_object_patch(
                f"diffusion_model.blocks.{i}.attn.forward",
                _make_sol_attention_forward(
                    attn, fallback_forward, schedule.tau, int(min_tokens), bool(strict), sol_log,
                    thresh_type, float(dense_percent), schedule.progress, bool(int8_qk),
                ),
            )

        log.info(
            "[MiniMax H3 Sol] scheduled tau %.2f -> %.2f (%s) on %d blocks (min_tokens=%d, strict=%s, dense=%.0f%%, thresh=%s)",
            float(tau_start),
            float(tau_end),
            curve,
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Routing threshold. Higher = more blocks take "
                        "the approximate path = faster, lower fidelity. "
                        "1.0 is the Sol-Attn default.",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 8192,
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
                        "path. Faster above ~16K tokens (measured 1.2-1.3x) at "
                        "~1% extra numerical error; slightly slower at 8K.",
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

    def patch(self, model, enabled, tau, min_tokens, strict, thresh_type, int8_qk):
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
        sol_log = _SolLog()
        for i in range(len(blocks)):
            attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
            fallback_forward = attn.forward
            if hasattr(fallback_forward, "_minimax_h3_sol_fallback"):
                fallback_forward = fallback_forward._minimax_h3_sol_fallback
            patched.add_object_patch(
                f"diffusion_model.blocks.{i}.attn.forward",
                _make_sol_attention_forward(
                    attn, fallback_forward, float(tau), int(min_tokens), bool(strict), sol_log,
                    thresh_type, int8_qk=bool(int8_qk),
                ),
            )

        log.info(
            "[MiniMax H3 Sol] patched %d attention blocks (tau=%.2f, min_tokens=%d, strict=%s)",
            len(blocks),
            float(tau),
            int(min_tokens),
            bool(strict),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChunkFeedForward": MiniMaxH3ChunkFeedForward,
    "MiniMaxH3MemoryEfficientSolAttentionPatch": MiniMaxH3MemoryEfficientSolAttentionPatch,
    "MiniMaxH3ScheduledSolAttentionPatch": MiniMaxH3ScheduledSolAttentionPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChunkFeedForward": "MiniMax H3 Chunk FeedForward",
    "MiniMaxH3MemoryEfficientSolAttentionPatch": "MiniMax H3 Memory Efficient Sol Attention Patch",
    "MiniMaxH3ScheduledSolAttentionPatch": "MiniMax H3 Scheduled Sol Attention Patch",
}
