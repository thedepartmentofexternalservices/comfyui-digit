"""DIGIT Frame Size Generator — resolution presets organized by size and aspect ratio."""


class DigitFrameSize:
    """Pick a resolution from Large / Medium / Small presets across common aspect ratios.
    Outputs width, height, and megapixel count."""

    CATEGORY = "DIGIT"

    # fmt: off
    RESOLUTIONS = {
        # ── Large (above 1920×1080) ──────────────────────────────────
        "— Large —":                          None,
        "Square 1:1  —  2048×2048":           (2048, 2048),
        "Square 1:1  —  2560×2560":           (2560, 2560),
        "Photo 4:5  —  2048×2560":            (2048, 2560),
        "Photo 5:4  —  2560×2048":            (2560, 2048),
        "Standard 4:3  —  2560×1920":         (2560, 1920),
        "Standard 3:4  —  1920×2560":         (1920, 2560),
        "Classic 3:2  —  2880×1920":          (2880, 1920),
        "Classic 2:3  —  1920×2880":          (1920, 2880),
        "HD 16:9  —  2560×1440":              (2560, 1440),
        "HD 9:16  —  1440×2560":              (1440, 2560),
        "HD 16:9  —  3840×2160":              (3840, 2160),
        "HD 9:16  —  2160×3840":              (2160, 3840),
        "Ultrawide 21:9  —  2560×1080":       (2560, 1080),
        "Ultrawide 21:9  —  3440×1440":       (3440, 1440),
        "Anamorphic 2.39:1  —  2880×1206":    (2880, 1206),
        "IMAX 1.43:1  —  2880×2016":          (2880, 2016),

        # ── Medium (1080–1920 on long edge) ──────────────────────────
        "— Medium —":                         None,
        "Square 1:1  —  1280×1280":           (1280, 1280),
        "Square 1:1  —  1536×1536":           (1536, 1536),
        "Photo 4:5  —  1080×1350":            (1080, 1350),
        "Photo 5:4  —  1350×1080":            (1350, 1080),
        "Standard 4:3  —  1440×1080":         (1440, 1080),
        "Standard 3:4  —  1080×1440":         (1080, 1440),
        "Classic 3:2  —  1620×1080":          (1620, 1080),
        "Classic 2:3  —  1080×1620":          (1080, 1620),
        "HD 16:9  —  1920×1080":              (1920, 1080),
        "HD 9:16  —  1080×1920":              (1080, 1920),
        "Ultrawide 21:9  —  1920×822":        (1920, 822),
        "Anamorphic 2.39:1  —  1920×804":     (1920, 804),
        "IMAX 1.43:1  —  1536×1074":          (1536, 1074),

        # ── Small (below 1080) ───────────────────────────────────────
        "— Small —":                          None,
        "Square 1:1  —  512×512":             (512, 512),
        "Square 1:1  —  768×768":             (768, 768),
        "Square 1:1  —  1024×1024":           (1024, 1024),
        "Photo 4:5  —  640×800":              (640, 800),
        "Photo 5:4  —  800×640":              (800, 640),
        "Standard 4:3  —  1024×768":          (1024, 768),
        "Standard 3:4  —  768×1024":          (768, 1024),
        "Classic 3:2  —  1024×682":           (1024, 682),
        "Classic 2:3  —  682×1024":           (682, 1024),
        "HD 16:9  —  1024×576":               (1024, 576),
        "HD 9:16  —  576×1024":               (576, 1024),
        "SD 16:9  —  854×480":                (854, 480),
        "SD 9:16  —  480×854":                (480, 854),
        "Ultrawide 21:9  —  1024×438":        (1024, 438),
    }
    # fmt: on

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(cls.RESOLUTIONS.keys()),
                    {"default": "HD 16:9  —  1920×1080"},
                ),
            },
            "optional": {
                "width_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 8,
                     "tooltip": "If non-zero, overrides the preset width."},
                ),
                "height_override": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 8,
                     "tooltip": "If non-zero, overrides the preset height."},
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "megapixels", "size_string")
    OUTPUT_TOOLTIPS = (
        "Frame width in pixels.",
        "Frame height in pixels.",
        "Total megapixels (width × height / 1,000,000).",
        "WIDTHxHEIGHT string for display or piping.",
    )
    FUNCTION = "get_size"

    def get_size(self, preset, width_override=0, height_override=0):
        size = self.RESOLUTIONS.get(preset)
        if size is None:
            raise ValueError(f"'{preset}' is a section header, not a resolution.")

        width, height = size

        if width_override > 0:
            width = width_override
        if height_override > 0:
            height = height_override

        megapixels = round((width * height) / 1_000_000, 2)
        size_string = f"{width}x{height}"

        return (width, height, megapixels, size_string)
