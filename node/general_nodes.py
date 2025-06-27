import comfy
import comfy.utils
import importlib

import logging

logger = logging.getLogger(__name__)


class ABS_PromptNode:
    """
    A simple text node that can be used to display information or instructions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("prompt",)
    FUNCTION = "send_text"
    CATEGORY = "utils/text"

    def send_text(self, text):
        """
        Outputs the provided text as a multiline string if specified.
        """
        return (text, )

class ABS_ConcatPrompts:
    """
    A node to concatenate multiple text inputs into a single string.
    Useful for combining prompts or instructions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": ("STRING", {"default": "", "multiline": True}),
                "prompt2": ("STRING", {"default": "", "multiline": True}),
                "concatenation_style": (",", {"default": ",", "tooltip": "Choose how to concatenate the prompts."}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("concatenated_text", )
    FUNCTION = "concat_texts"
    CATEGORY = "utils/text"

    def concat_texts(self, prompt1, prompt2, concatenation_style=","):
        """
        Concatenates two strings into one.
        """
        return (prompt1 + concatenation_style + prompt2, )

class ABS_DebugNode:
    # takes in anything and outputs it as a debug message
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "*": ("ANY", {"default": ""}),
            },
        }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "debug"
    CATEGORY = "utils/debug"
    OUTPUT_NODE = True
    def debug(self, *args):
        """
        Outputs the provided arguments as a debug message.
        """
        message = "Debug: " + ", ".join(str(arg) for arg in args)
        logging.info(message)
        return ()