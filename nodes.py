"""Sol-Attn (NVIDIA Sana / Sol-Engine) as an opt-in ComfyUI attention backend.

Uses NVIDIA's Triton reference kernel on the architectures this package has
actually validated: SM90, SM100, and SM120.

Hard requirements of the kernel (anything else falls back to your normal
backend, e.g. SageAttention):
  - head_dim == 128
  - bfloat16
  - no attention mask
  - 4D q/k/v (skip_reshape=True path)

MiniMax H3 satisfies the kernel's tensor constraints (56 heads x 128, bf16,
mask=None), but is not one of the model integrations evaluated in the paper.
"""

import logging
import torch

from .sol_kernel import sol_attn

log = logging.getLogger(__name__)

BLOCK = 64
SUPPORTED_ARCHES = {(8, 9), (9, 0), (10, 0), (12, 0), (12, 1)}


class _Unsupported(Exception):
    pass


class _DispatchLog:
    def __init__(self):
        self.active = False
        self.fallbacks = set()

    def hit(self):
        if not self.active:
            log.info("[Sol-Attn] active")
            self.active = True

    def miss(self, reason):
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.info("[Sol-Attn] dense fallback: %s", reason)


def _make_override(tau: float, min_tokens: int, strict: bool, fallback_override=None, thresh_type: str = "diag", int8_qk: bool = False, int8_pv: bool = False):
    dispatch_log = _DispatchLog()

    def override(func, q, k, v, heads, *args, **kwargs):
        mask = kwargs.get("mask", None)
        skip_reshape = kwargs.get("skip_reshape", False)
        skip_output_reshape = kwargs.get("skip_output_reshape", False)

        try:
            if mask is not None:
                raise _Unsupported("attention mask present")
            if not skip_reshape or q.dim() != 4:
                raise _Unsupported("not the 4D skip_reshape path")

            b, h, n, d = q.shape
            if heads != h:
                raise _Unsupported(f"heads argument {heads} != tensor heads {h}")
            if d != 128:
                raise _Unsupported(f"head_dim {d} != 128")
            if k.shape != q.shape or v.shape != q.shape:
                raise _Unsupported("q/k/v shape mismatch (cross-attention?)")
            if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
                raise _Unsupported("q/k/v must use bfloat16")
            if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
                raise _Unsupported("q/k/v must share a CUDA device")
            if any(x.requires_grad for x in (q, k, v)):
                raise _Unsupported("autograd requested")
            arch = torch.cuda.get_device_capability(q.device)
            if arch not in SUPPORTED_ARCHES:
                raise _Unsupported(f"unsupported SM{arch[0]}{arch[1]}")
            if n < min_tokens:
                raise _Unsupported(f"{n} tokens < minimum {min_tokens}")

            # Comfy hands us BHSD; Sol-Attn wants BTHD, contiguous.
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()

            out = sol_attn(
                qt,
                kt,
                vt,
                scale=kwargs.get("scale", None),
                tau=tau,
                thresh_type=thresh_type,
                int8_qk=int8_qk,
                int8_pv=int8_pv,
            )  # (B, T, H, D)

            dispatch_log.hit()
            if skip_output_reshape:
                return out.transpose(1, 2)  # (B, H, T, D)
            return out.reshape(b, n, h * d)  # (B, T, H*D)

        except _Unsupported as e:
            dispatch_log.miss(str(e))
        except Exception as e:
            if strict:
                raise
            dispatch_log.miss(f"{type(e).__name__}: {e}")

        if fallback_override is not None:
            return fallback_override(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    override._sol_attn_fallback = fallback_override
    return override


class SolAttentionPatch:
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
                        "default": 4096,
                        "min": BLOCK * 4,
                        "max": 131072,
                        "step": BLOCK,
                        "tooltip": "Use the normal backend below this sequence "
                        "length. The paper evaluates RTX 5090 kernels from 8K tokens.",
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
                "int8_pv": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also quantize the P*V dot to int8 (per-token P, "
                        "per-channel V). More speed on top of int8_qk at additional "
                        "accuracy cost. Requires int8_qk. Opt-in.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"

    DESCRIPTION = (
        "Route this model's self-attention through Sol-Attn (sparse, "
        "training-free). Only affects the model you wire it to. Falls back to "
        "your normal backend for any shape it can't handle."
    )

    def patch(self, model, enabled, tau, min_tokens=8192, strict=False, thresh_type="diag", int8_qk=False, int8_pv=False):
        if not enabled:
            return (model,)
        m = model.clone()
        opts = dict(m.model_options.get("transformer_options", {}))
        fallback_override = opts.get("optimized_attention_override")
        if hasattr(fallback_override, "_sol_attn_fallback"):
            fallback_override = fallback_override._sol_attn_fallback
        opts["optimized_attention_override"] = _make_override(
            float(tau),
            int(min_tokens),
            bool(strict),
            fallback_override,
            thresh_type,
            bool(int8_qk),
            bool(int8_pv),
        )
        m.model_options["transformer_options"] = opts
        log.info(
            "[Sol-Attn] patched (tau=%.2f, min_tokens=%d, strict=%s)",
            float(tau),
            int(min_tokens),
            bool(strict),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"SolAttentionPatch": SolAttentionPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"SolAttentionPatch": "Sol-Attn (sparse attention)"}
