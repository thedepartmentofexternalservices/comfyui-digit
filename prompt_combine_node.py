"""DIGIT Prompt Combine — joins multiple text inputs into a single prompt string."""


class DigitPromptCombine:
    """Combine trigger words, generated prompts, and custom text into one prompt."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "combine"
    OUTPUT_NODE = True
    DESCRIPTION = "Combine multiple text inputs into a single prompt. Empty inputs are skipped."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Separator between non-empty inputs.",
                }),
            },
            "optional": {
                "trigger": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Trigger word from LoRA Loader.",
                }),
                "prompt_1": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Primary prompt (e.g. from Random Prompt or LLM).",
                }),
                "prompt_2": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Additional prompt text.",
                }),
                "prompt_3": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Additional prompt text.",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text prepended to the combined prompt (typed directly on node).",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text appended to the combined prompt (typed directly on node).",
                }),
            },
        }

    def combine(self, separator=", ", trigger="", prompt_1="", prompt_2="",
                prompt_3="", prefix="", suffix=""):
        parts = [
            trigger.strip(),
            prefix.strip(),
            prompt_1.strip(),
            prompt_2.strip(),
            prompt_3.strip(),
            suffix.strip(),
        ]
        # Drop empty parts
        parts = [p for p in parts if p]
        combined = separator.join(parts)

        return {"ui": {"combined_text": [combined or "(empty)"]},
                "result": (combined,)}
