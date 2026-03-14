import torch
import numpy as np
from PIL import Image
import os
from folder_paths import get_temp_directory
import json

FIT_MODES = ["none", "crop", "fit_h", "fit_v"]


class DigitDragCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "drawing_version": ("STRING", {"default": "init"}),
                "image": ("IMAGE",),
                "crop_left": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_right": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_top": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_bottom": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "crop_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "last_width": ("INT", {"default": 0}),
                "last_height": ("INT", {"default": 0}),
                "dest_width": ("INT", {"default": 0, "min": 0, "max": 8192, "tooltip": "Destination width. 0 = use crop size as-is."}),
                "dest_height": ("INT", {"default": 0, "min": 0, "max": 8192, "tooltip": "Destination height. 0 = use crop size as-is."}),
                "fit_mode": (FIT_MODES, {"default": "none", "tooltip": "none=no resize, crop=fill and crop, fit_h=match height, fit_v=match width"}),
            },
            "optional": {
                "mask": ("MASK",)
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "CROP_JSON")
    FUNCTION = "crop"
    CATEGORY = "DIGIT"

    def crop(
        self,
        drawing_version,
        image: torch.Tensor,
        crop_left: int,
        crop_right: int,
        crop_top: int,
        crop_bottom: int,
        crop_width: int,
        crop_height: int,
        last_width: int,
        last_height: int,
        dest_width: int = 0,
        dest_height: int = 0,
        fit_mode: str = "none",
        node_id=None,
        mask=None,
    ):
        print(f"[DigitDragCrop] Node {node_id} executed")

        batch_size, current_height, current_width, channels = image.shape

        resolution_changed = (current_width != last_width or current_height != last_height)
        reset_frontend_crop = False

        if resolution_changed:
            crop_left = 0
            crop_top = 0
            crop_right = 0
            crop_bottom = 0
            crop_width = current_width
            crop_height = current_height
            reset_frontend_crop = True

        computed_crop_right = crop_left + crop_width
        computed_crop_bottom = crop_top + crop_height

        if (crop_left < 0 or crop_top < 0 or
            computed_crop_right > current_width or computed_crop_bottom > current_height or
            crop_width <= 0 or crop_height <= 0):
            print("[DigitDragCrop] Invalid crop area, resetting to full image.")
            crop_left = 0
            crop_top = 0
            crop_right = 0
            crop_bottom = 0
            crop_width = current_width
            crop_height = current_height
            computed_crop_right = crop_left + crop_width
            computed_crop_bottom = crop_top + crop_height
            reset_frontend_crop = True

        cropped_image = image[:, crop_top:computed_crop_bottom, crop_left:computed_crop_right, :]

        def _make_zero_mask(bs, h, w, device):
            return torch.zeros((bs, h, w), dtype=torch.float32, device=device)

        cropped_mask = None
        if mask is None or not torch.is_tensor(mask) or mask.numel() == 0:
            cropped_mask = _make_zero_mask(batch_size, crop_height, crop_width, image.device)
        else:
            m = mask
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.squeeze(1)
            elif m.dim() == 2:
                m = m.unsqueeze(0)

            if m.dim() != 3:
                cropped_mask = _make_zero_mask(batch_size, crop_height, crop_width, image.device)
            else:
                if m.shape[0] != batch_size:
                    if m.shape[0] == 1 and batch_size > 1:
                        m = m.repeat(batch_size, 1, 1)
                    else:
                        if m.shape[0] > batch_size:
                            m = m[:batch_size]
                        else:
                            m = m.repeat(int(np.ceil(batch_size / m.shape[0])), 1, 1)[:batch_size]

                mh, mw = m.shape[1], m.shape[2]
                cl = max(0, min(crop_left, mw))
                cr = max(0, min(computed_crop_right, mw))
                ct = max(0, min(crop_top, mh))
                cb = max(0, min(computed_crop_bottom, mh))

                if cr <= cl or cb <= ct:
                    cropped_mask = _make_zero_mask(batch_size, crop_height, crop_width, image.device)
                else:
                    region = m[:, ct:cb, cl:cr]
                    cropped_mask = _make_zero_mask(batch_size, crop_height, crop_width, image.device)
                    rh, rw = region.shape[1], region.shape[2]
                    cropped_mask[:, :rh, :rw] = region.to(torch.float32)

        # Apply destination resize if requested
        if fit_mode != "none" and dest_width > 0 and dest_height > 0:
            cropped_image, cropped_mask = self._reformat(
                cropped_image, cropped_mask, dest_width, dest_height, fit_mode
            )

        original_filename = None
        if batch_size > 0:
            img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            temp_dir = get_temp_directory()
            filename_hash = hash(f"{node_id}_{current_width}x{current_height}")
            original_filename = f"digit_dragcrop_{filename_hash}.png"
            filepath = os.path.join(temp_dir, original_filename)
            os.makedirs(temp_dir, exist_ok=True)
            try:
                pil_image.save(filepath)
            except Exception as e:
                print(f"[DigitDragCrop] Error saving preview image: {e}")
                original_filename = None

        # Get actual output dimensions
        out_h, out_w = cropped_image.shape[1], cropped_image.shape[2]

        crop_payload = {
            "left": crop_left,
            "top": crop_top,
            "right": crop_right,
            "bottom": crop_bottom,
            "width": crop_width,
            "height": crop_height,
            "original_size": [current_width, current_height],
            "cropped_size": [crop_width, crop_height],
            "output_size": [out_w, out_h],
            "fit_mode": fit_mode,
            "reset_crop_ui": reset_frontend_crop
        }

        crop_json = json.dumps(crop_payload)

        return {
            "ui": {
                "images_custom": [{
                    "filename": original_filename,
                    "subfolder": "",
                    "type": "temp"
                }] if original_filename else [],
                "crop_info": [crop_payload]
            },
            "result": (cropped_image, cropped_mask, crop_json),
        }

    def _reformat(self, image, mask, dest_w, dest_h, mode):
        """Resize cropped image/mask to destination size.

        Modes:
          crop  — scale to fill dest, center-crop the excess (no black bars)
          fit_h — scale to match dest height, center-crop or pad width
          fit_v — scale to match dest width, center-crop or pad height
        """
        bs, src_h, src_w, ch = image.shape

        if mode == "crop":
            scale = max(dest_w / src_w, dest_h / src_h)
        elif mode == "fit_h":
            scale = dest_h / src_h
        else:  # fit_v
            scale = dest_w / src_w

        scaled_w = max(1, round(src_w * scale))
        scaled_h = max(1, round(src_h * scale))

        # Resize via PIL for quality
        out_images = []
        out_masks = []
        for i in range(bs):
            # Resize image
            img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).resize((scaled_w, scaled_h), Image.LANCZOS)

            # Place into destination canvas (black padding if needed)
            canvas = Image.new("RGB", (dest_w, dest_h), (0, 0, 0))
            offset_x = (dest_w - scaled_w) // 2
            offset_y = (dest_h - scaled_h) // 2
            # If scaled > dest, offset will be negative = center-crop
            canvas.paste(pil_img, (offset_x, offset_y))
            out_images.append(np.array(canvas).astype(np.float32) / 255.0)

            # Resize mask
            if mask is not None and mask.numel() > 0:
                m_np = (mask[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_m = Image.fromarray(m_np, mode="L").resize((scaled_w, scaled_h), Image.LANCZOS)
                m_canvas = Image.new("L", (dest_w, dest_h), 0)
                m_canvas.paste(pil_m, (offset_x, offset_y))
                out_masks.append(np.array(m_canvas).astype(np.float32) / 255.0)
            else:
                out_masks.append(np.zeros((dest_h, dest_w), dtype=np.float32))

        result_image = torch.from_numpy(np.stack(out_images)).to(image.device)
        result_mask = torch.from_numpy(np.stack(out_masks)).to(image.device)
        return result_image, result_mask


class DigitCropInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("left", "top", "right", "bottom", "width", "height", "csv", "pretty")
    FUNCTION = "interpret"
    CATEGORY = "DIGIT"

    def interpret(self, crop_json: str):
        try:
            data = json.loads(crop_json) if crop_json else {}
        except Exception:
            data = {}

        left = int(data.get("left", 0))
        top = int(data.get("top", 0))
        right = int(data.get("right", left))
        bottom = int(data.get("bottom", top))
        width = int(data.get("width", max(0, right - left)))
        height = int(data.get("height", max(0, bottom - top)))

        csv = f"{left},{top},{right},{bottom},{width},{height}"
        pretty = f"left={left}, top={top}, right={right}, bottom={bottom}, width={width}, height={height}"

        return (left, top, right, bottom, width, height, csv, pretty)
