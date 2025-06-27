from .node.lycoris_loader import (
    LycorisLoaderNode,
)
from .node.general_nodes import (
    ABS_PromptNode,
    ABS_ConcatPrompts,
)

NODE_CLASS_MAPPINGS = {
    "LycorisLoaderNode": LycorisLoaderNode,
    "ABS_PromptNode": ABS_PromptNode,
    "ABS_ConcatPrompts": ABS_ConcatPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LycorisLoaderNode": "LyCORIS Loader",
    "ABS_PromptNode": "Prompt Node",
    "ABS_ConcatPrompts": "Concatenate Prompts",
}
