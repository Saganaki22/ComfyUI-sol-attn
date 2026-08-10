"""GPU regression tests for the architecture, INT8, and H3 fusion paths."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
CUSTOM_NODES = REPO_ROOT.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


fwd = importlib.import_module("sol_kernel.fwd")
GPU_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() in {
    (8, 9),
    (9, 0),
    (10, 0),
    (12, 0),
    (12, 1),
}


class ArchitectureDispatchTests(unittest.TestCase):
    def test_pointer_architectures_are_explicit(self):
        self.assertTrue(fwd._use_pointer_arch((8, 9)))
        self.assertTrue(fwd._use_pointer_arch((12, 0)))
        self.assertFalse(fwd._use_pointer_arch((9, 0)))
        self.assertFalse(fwd._use_pointer_arch((10, 0)))
        self.assertFalse(fwd._use_pointer_arch((12, 1)))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "requires SM120 to compare its pointer and TMA kernels",
    )
    def test_sm120_pointer_matches_tma(self):
        torch.manual_seed(100)
        qkv = torch.randn((1, 1031, 3 * 2 * 128), device="cuda", dtype=torch.bfloat16)
        q, k, v = [part.view(1, 1031, 2, 128) for part in qkv.split(2 * 128, dim=-1)]
        original = fwd._use_pointer_arch
        try:
            fwd._use_pointer_arch = lambda arch: False
            expected = fwd.sol_attn(q, k, v, tau=1.3)
            fwd._use_pointer_arch = lambda arch: True
            actual = fwd.sol_attn(q, k, v, tau=1.3)
        finally:
            fwd._use_pointer_arch = original
        self.assertTrue(torch.equal(expected, actual))


@unittest.skipUnless(GPU_SUPPORTED, "requires a supported CUDA GPU")
class InlineQTests(unittest.TestCase):
    def test_inline_q_matches_materialized_q_with_sinks_and_ragged_tail(self):
        torch.manual_seed(101)
        tokens = 4097
        qkv = torch.randn((1, tokens, 3 * 2 * 128), device="cuda", dtype=torch.bfloat16)
        q, k, v = [part.view(1, tokens, 2, 128) for part in qkv.split(2 * 128, dim=-1)]
        original = fwd._POINTER_INLINE_Q
        try:
            fwd._POINTER_INLINE_Q = False
            expected = fwd.sol_attn(
                q,
                k,
                v,
                tau=1.3,
                int8_qk=True,
                sink_blocks=(0, 2),
                sink_q=(1, 2),
            )
            fwd._POINTER_INLINE_Q = True
            actual = fwd.sol_attn(
                q,
                k,
                v,
                tau=1.3,
                int8_qk=True,
                sink_blocks=(0, 2),
                sink_q=(1, 2),
            )
        finally:
            fwd._POINTER_INLINE_Q = original
        self.assertTrue(torch.equal(expected, actual))

    def test_inline_q_matches_materialized_q_with_int8_pv(self):
        torch.manual_seed(102)
        q = torch.randn((1, 4096, 2, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        original = fwd._POINTER_INLINE_Q
        try:
            fwd._POINTER_INLINE_Q = False
            expected = fwd.sol_attn(q, k, v, tau=1.3, int8_qk=True, int8_pv=True)
            fwd._POINTER_INLINE_Q = True
            actual = fwd.sol_attn(q, k, v, tau=1.3, int8_qk=True, int8_pv=True)
        finally:
            fwd._POINTER_INLINE_Q = original
        self.assertTrue(torch.equal(expected, actual))


@unittest.skipUnless(GPU_SUPPORTED, "requires a supported CUDA GPU")
class H3FusionTests(unittest.TestCase):
    def test_fused_ops_match_real_comfy_dit_block(self):
        from comfy.ldm.minimax.model import DiTBlock
        from sol_kernel.h3_fusion import (
            fused_gate_add_,
            fused_modulate_,
            make_segment_index,
        )

        minimax = importlib.import_module(f"{REPO_ROOT.name}.minimax")
        torch.manual_seed(103)
        tokens, hidden, heads, head_dim, ffn, temb = 259, 1024, 8, 128, 2048, 128
        block = DiTBlock(
            hidden,
            heads,
            head_dim,
            ffn,
            temb,
            1e-5,
            1e-5,
            dtype=torch.bfloat16,
            device="cuda",
            operations=torch.nn,
        ).eval()
        segments = [(0, 19, 1), (19, 100, 4), (100, 201, 8), (201, tokens, 2)]
        t_emb = torch.randn((3, temb), device="cuda", dtype=torch.bfloat16)
        x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)

        fallback = block.forward
        fusion_log = minimax._FusionLog()
        fused = minimax._make_fused_h3_block_forward(
            block,
            fallback,
            minimax._SegmentIndexCache(make_segment_index),
            fusion_log,
            fused_modulate_,
            fused_gate_add_,
        )
        with torch.inference_mode():
            expected = fallback(x.clone(), t_emb, segments, None)
            actual = fused(x.clone(), t_emb, segments, None)
        self.assertTrue(fusion_log.active)
        self.assertTrue(torch.equal(expected, actual))

        # KJNodes and the local Sol node patch attn.forward independently of
        # the block. The fusion must resolve that method at call time rather
        # than retaining the attention implementation present at construction.
        original_attention = block.attn.forward
        calls = 0

        def replacement_attention(value, **kwargs):
            nonlocal calls
            calls += 1
            return original_attention(value, **kwargs)

        block.attn.forward = replacement_attention
        with torch.inference_mode():
            expected = fallback(x.clone(), t_emb, segments, None)
            actual = fused(x.clone(), t_emb, segments, None)
        self.assertEqual(calls, 2)
        self.assertTrue(torch.equal(expected, actual))


class H3AttentionHandoffTests(unittest.TestCase):
    def test_dense_fallback_preserves_kjnodes_handoff(self):
        minimax = importlib.import_module(f"{REPO_ROOT.name}.minimax")
        activation = torch.zeros((32, 128), dtype=torch.bfloat16)
        handoff = [activation]
        fallback_calls = 0

        def fallback(value, **kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            self.assertIs(value, handoff)
            return value.pop()

        class Attention:
            heads = 1
            head_dim = 128

        wrapped = minimax._make_sol_attention_forward(
            Attention(), fallback, 1.3, 4096, False, minimax._SolLog()
        )
        actual = wrapped(handoff)
        self.assertIs(actual, activation)
        self.assertEqual(fallback_calls, 1)
        self.assertEqual(handoff, [])

    @unittest.skipUnless(GPU_SUPPORTED, "requires a supported CUDA GPU")
    def test_sol_path_consumes_kjnodes_handoff(self):
        minimax = importlib.import_module(f"{REPO_ROOT.name}.minimax")

        class Attention:
            heads = 1
            head_dim = 128
            qkv_proj = staticmethod(lambda value: torch.cat((value, value, value), dim=-1))
            q_norm = staticmethod(lambda value: value)
            k_norm = staticmethod(lambda value: value)
            out_proj = staticmethod(lambda value: value)

        activation = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16)
        handoff = [activation]
        fallback_calls = 0

        def fallback(value, **kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            return value

        original_sol_attn = minimax.sol_attn
        try:
            minimax.sol_attn = lambda q, k, v, **kwargs: v
            wrapped = minimax._make_sol_attention_forward(
                Attention(), fallback, 1.3, 1, True, minimax._SolLog()
            )
            actual = wrapped(handoff)
        finally:
            minimax.sol_attn = original_sol_attn

        self.assertEqual(handoff, [])
        self.assertEqual(fallback_calls, 0)
        self.assertTrue(torch.equal(actual, activation))


if __name__ == "__main__":
    unittest.main()
