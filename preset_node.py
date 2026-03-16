"""DIGIT Preset Manager — ComfyUI nodes for naming, trigger, and sample prompt presets."""

import json


class DigitNamingPreset:
    """Manage naming convention presets for LoRA output files and directories."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "save", "load", "list", "delete", "preview",
                ], {"default": "preview"}),
            },
            "optional": {
                "preset_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Preset name",
                }),
                "description": ("STRING", {"default": "", "multiline": False}),
                "output_dir_template": ("STRING", {
                    "default": "{name}_{model}_r{rank}_{date}",
                    "multiline": False,
                    "placeholder": "Template for output directory name",
                }),
                "lora_name_template": ("STRING", {
                    "default": "{name}_{model}_r{rank}_s{steps}",
                    "multiline": False,
                    "placeholder": "Template for LoRA weight folder name",
                }),
                "checkpoint_template": ("STRING", {
                    "default": "step_{step:06d}",
                    "multiline": False,
                }),
                "sample_template": ("STRING", {
                    "default": "sample_{prompt_index:02d}_{step:06d}",
                    "multiline": False,
                }),
                # Preview variables
                "preview_name": ("STRING", {"default": "my_lora"}),
                "preview_model": ("STRING", {"default": "flux1-dev"}),
                "preview_trigger": ("STRING", {"default": "ohwx"}),
                "preview_rank": ("INT", {"default": 16, "min": 1, "max": 256}),
                "preview_steps": ("INT", {"default": 1000, "min": 10, "max": 100000}),
                "preview_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("result", "output_dir_template", "lora_name_template",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, action, preset_name="", description="",
                output_dir_template="{name}_{model}_r{rank}_{date}",
                lora_name_template="{name}_{model}_r{rank}_s{steps}",
                checkpoint_template="step_{step:06d}",
                sample_template="sample_{prompt_index:02d}_{step:06d}",
                preview_name="my_lora", preview_model="flux1-dev",
                preview_trigger="ohwx", preview_rank=16,
                preview_steps=1000, preview_resolution=1024):

        from .training.presets_db import PresetsDB
        db = PresetsDB()

        if action == "preview":
            return self._preview(
                output_dir_template, lora_name_template,
                preview_name, preview_model, preview_trigger,
                preview_rank, preview_steps, preview_resolution,
            )
        elif action == "save":
            if not preset_name:
                return ("Error: preset_name required", "", "")
            db.save_naming_preset(
                name=preset_name,
                output_dir_template=output_dir_template,
                lora_name_template=lora_name_template,
                checkpoint_template=checkpoint_template,
                sample_template=sample_template,
                description=description,
            )
            return (f"Saved naming preset: {preset_name}",
                    output_dir_template, lora_name_template)
        elif action == "load":
            preset = db.get_naming_preset(preset_name)
            if not preset:
                return (f"Not found: {preset_name}", "", "")
            return (
                json.dumps(preset, indent=2, default=str),
                preset["output_dir_template"],
                preset["lora_name_template"],
            )
        elif action == "list":
            presets = db.list_naming_presets()
            if not presets:
                return ("No naming presets saved.", "", "")
            lines = ["Naming Presets:"]
            for p in presets:
                lines.append(f"  {p['name']} — {p['description']}")
                lines.append(f"    dir:  {p['output_dir_template']}")
                lines.append(f"    lora: {p['lora_name_template']}")
            return ("\n".join(lines), "", "")
        elif action == "delete":
            if db.delete_naming_preset(preset_name):
                return (f"Deleted: {preset_name}", "", "")
            return (f"Not found: {preset_name}", "", "")

        return ("Unknown action", "", "")

    def _preview(self, output_dir_template, lora_name_template,
                 name, model, trigger, rank, steps, resolution):
        """Preview what the naming templates resolve to with sample values."""
        from datetime import datetime
        now = datetime.now()
        v = {
            "name": name,
            "model": model,
            "trigger": trigger or "notrigger",
            "trigger_class": "person",
            "rank": rank,
            "alpha": rank,
            "steps": steps,
            "res": resolution,
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "seed": 42,
            "optimizer": "adamw8bit",
            "lr": "1e-4",
        }

        try:
            resolved_dir = output_dir_template.format(**v)
            resolved_lora = lora_name_template.format(**v)
        except KeyError as e:
            return (f"Error: unknown variable {e}", output_dir_template, lora_name_template)

        lines = [
            "Naming Preview:",
            f"  Variables available: {', '.join(f'{{{k}}}' for k in sorted(v.keys()))}",
            f"",
            f"  output_dir_template: {output_dir_template}",
            f"    → {resolved_dir}",
            f"",
            f"  lora_name_template: {lora_name_template}",
            f"    → {resolved_lora}",
            f"",
            f"  Full output path example:",
            f"    ./output/{resolved_dir}/{resolved_lora}/",
        ]
        return ("\n".join(lines), output_dir_template, lora_name_template)


class DigitTriggerPreset:
    """Manage trigger word presets with class/instance/phrase support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "save", "load", "list", "delete", "preview_injection",
                ], {"default": "preview_injection"}),
            },
            "optional": {
                "preset_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "description": ("STRING", {"default": "", "multiline": False}),
                "trigger_word": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., ohwx",
                }),
                "trigger_class": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., person, man, woman, dog, style",
                }),
                "trigger_phrase": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Override: e.g., 'ohwx person' (auto if empty)",
                }),
                "test_text": ("STRING", {
                    "default": "a photo of [trigger_phrase] standing in a field\n[trigger] in cinematic lighting\na [trigger_class] portrait with dramatic shadows",
                    "multiline": True,
                    "placeholder": "Text with [trigger], [trigger_class], [trigger_phrase] placeholders",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("result", "trigger_word", "trigger_class", "trigger_phrase",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, action, preset_name="", description="",
                trigger_word="", trigger_class="", trigger_phrase="",
                test_text=""):

        from .training.presets_db import PresetsDB
        db = PresetsDB()

        # Auto-build phrase
        effective_phrase = trigger_phrase
        if not effective_phrase and trigger_word:
            effective_phrase = f"{trigger_word} {trigger_class}".strip() if trigger_class else trigger_word

        if action == "preview_injection":
            return self._preview_injection(
                trigger_word, trigger_class, effective_phrase, test_text,
            )
        elif action == "save":
            if not preset_name:
                return ("Error: preset_name required", "", "", "")
            db.save_trigger_preset(
                name=preset_name,
                trigger_word=trigger_word,
                trigger_class=trigger_class,
                trigger_phrase=effective_phrase,
                description=description,
            )
            return (f"Saved trigger preset: {preset_name}",
                    trigger_word, trigger_class, effective_phrase)
        elif action == "load":
            preset = db.get_trigger_preset(preset_name)
            if not preset:
                return (f"Not found: {preset_name}", "", "", "")
            return (
                json.dumps(preset, indent=2, default=str),
                preset["trigger_word"],
                preset["trigger_class"],
                preset["trigger_phrase"],
            )
        elif action == "list":
            presets = db.list_trigger_presets()
            if not presets:
                return ("No trigger presets saved.", "", "", "")
            lines = ["Trigger Presets:"]
            for p in presets:
                phrase = p.get("trigger_class", "")
                phrase_str = f" ({phrase})" if phrase else ""
                lines.append(f"  {p['name']}: \"{p['trigger_word']}\"{phrase_str} — {p['description']}")
            return ("\n".join(lines), "", "", "")
        elif action == "delete":
            if db.delete_trigger_preset(preset_name):
                return (f"Deleted: {preset_name}", "", "", "")
            return (f"Not found: {preset_name}", "", "", "")

        return ("Unknown action", "", "", "")

    def _preview_injection(self, trigger_word, trigger_class, trigger_phrase, test_text):
        """Show how trigger placeholders resolve in sample text."""
        if not test_text:
            test_text = (
                "a photo of [trigger_phrase] standing in a field\n"
                "[trigger] in cinematic lighting\n"
                "a [trigger_class] portrait with dramatic shadows"
            )

        lines = [
            "Trigger Injection Preview:",
            f"  [trigger]        → \"{trigger_word}\"",
            f"  [trigger_class]  → \"{trigger_class}\"",
            f"  [trigger_phrase] → \"{trigger_phrase}\"",
            f"",
            "Input:",
        ]

        resolved_lines = []
        for line in test_text.strip().split("\n"):
            original = line.strip()
            resolved = original.replace("[trigger_phrase]", trigger_phrase)
            resolved = resolved.replace("[trigger_class]", trigger_class)
            resolved = resolved.replace("[trigger]", trigger_word)
            lines.append(f"  {original}")
            resolved_lines.append(resolved)

        lines.append("")
        lines.append("Resolved:")
        for rl in resolved_lines:
            lines.append(f"  {rl}")

        return ("\n".join(lines), trigger_word, trigger_class, trigger_phrase)


class DigitSamplePromptPreset:
    """Manage sample prompt presets for training-time image generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "save", "load", "list", "delete", "preview",
                ], {"default": "preview"}),
            },
            "optional": {
                "preset_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "description": ("STRING", {"default": "", "multiline": False}),
                "prompts": ("STRING", {
                    "default": (
                        "a portrait photo of [trigger_phrase]\n"
                        "a photo of [trigger_phrase] standing outdoors in natural light\n"
                        "[trigger_phrase] in a cinematic scene, dramatic lighting\n"
                        "a close-up photo of [trigger], sharp focus, studio lighting\n"
                        "a full body shot of [trigger_phrase] walking down a city street"
                    ),
                    "multiline": True,
                    "placeholder": "One prompt per line — use [trigger], [trigger_class], [trigger_phrase]",
                }),
                # For preview: resolve placeholders
                "trigger_word": ("STRING", {"default": "", "multiline": False}),
                "trigger_class": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("result", "resolved_prompts",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, action, preset_name="", description="",
                prompts="", trigger_word="", trigger_class=""):

        from .training.presets_db import PresetsDB
        db = PresetsDB()

        prompt_list = [p.strip() for p in prompts.strip().split("\n") if p.strip()]

        if action == "preview":
            return self._preview(prompt_list, trigger_word, trigger_class)
        elif action == "save":
            if not preset_name:
                return ("Error: preset_name required", "")
            db.save_sample_prompt_preset(
                name=preset_name,
                prompts=prompt_list,
                description=description,
            )
            return (f"Saved sample prompt preset: {preset_name} ({len(prompt_list)} prompts)", "")
        elif action == "load":
            preset = db.get_sample_prompt_preset(preset_name)
            if not preset:
                return (f"Not found: {preset_name}", "")
            loaded_prompts = "\n".join(preset["prompts"])
            return (
                f"Loaded: {preset_name} ({len(preset['prompts'])} prompts)\n\n{loaded_prompts}",
                loaded_prompts,
            )
        elif action == "list":
            presets = db.list_sample_prompt_presets()
            if not presets:
                return ("No sample prompt presets saved.", "")
            lines = ["Sample Prompt Presets:"]
            for p in presets:
                lines.append(f"  {p['name']} — {p['description']}")
            return ("\n".join(lines), "")
        elif action == "delete":
            if db.delete_sample_prompt_preset(preset_name):
                return (f"Deleted: {preset_name}", "")
            return (f"Not found: {preset_name}", "")

        return ("Unknown action", "")

    def _preview(self, prompt_list, trigger_word, trigger_class):
        """Preview prompts with trigger placeholders resolved."""
        trigger_phrase = f"{trigger_word} {trigger_class}".strip() if trigger_class else trigger_word

        lines = ["Sample Prompt Preview:"]
        resolved_all = []

        for i, prompt in enumerate(prompt_list):
            resolved = prompt.replace("[trigger_phrase]", trigger_phrase)
            resolved = resolved.replace("[trigger_class]", trigger_class)
            resolved = resolved.replace("[trigger]", trigger_word)
            resolved_all.append(resolved)
            lines.append(f"  [{i+1}] {prompt}")
            if resolved != prompt:
                lines.append(f"    → {resolved}")

        return ("\n".join(lines), "\n".join(resolved_all))
