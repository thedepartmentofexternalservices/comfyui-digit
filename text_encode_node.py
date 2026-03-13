"""DIGIT Text Encode — CLIP text encode with text as a connectable input."""


class DigitTextEncode:
    """CLIP Text Encode with text as a connectable input (no widget conversion needed)."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    DESCRIPTION = "Encode text into CLIP conditioning. Text input is connectable by default."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model to encode text with."}),
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Text to encode (connect from Prompt Combine, Random Prompt, etc.).",
                }),
            },
        }

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]],)
