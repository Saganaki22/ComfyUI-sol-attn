"""Sol-Attn and MiniMax H3 memory patches for ComfyUI.

The two node modules are guarded independently so the MiniMax memory patch does
not depend on Triton or the Sol-Attn kernel loading successfully.
"""

import logging

log = logging.getLogger(__name__)
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as SOL_NODE_CLASS_MAPPINGS
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as SOL_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS.update(SOL_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SOL_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:  # noqa: BLE001
    log.warning("[Sol-Attn] not loaded (%s: %s)", type(e).__name__, e)

try:
    from .minimax import NODE_CLASS_MAPPINGS as MINIMAX_NODE_CLASS_MAPPINGS
    from .minimax import NODE_DISPLAY_NAME_MAPPINGS as MINIMAX_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS.update(MINIMAX_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MINIMAX_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:  # noqa: BLE001
    log.warning("[MiniMax H3] not loaded (%s: %s)", type(e).__name__, e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
