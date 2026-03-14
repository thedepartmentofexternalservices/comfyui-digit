from .llm_node import LLMQueryNode
from .image_saver_node import DigitImageSaver
from .image_loader_node import DigitImageLoader
from .gemini_image_node import DigitGeminiImage
from .veo_video_node import DigitVeoVideo
from .video_saver_node import DigitVideoSaver
from .drag_crop_node import DigitDragCrop, DigitCropInfo
from .srt_maker_node import DigitSRTMaker
from .random_prompt_node import DigitRandomPrompt
from .lora_loader_node import DigitLoraLoader, DigitLoraLoaderModelOnly
from .prompt_combine_node import DigitPromptCombine
from .batch_caption_node import DigitBatchCaption
from .text_encode_node import DigitTextEncode
from .dataset_prep_node import DigitDatasetPrep
from .caption_viewer_node import DigitCaptionViewer
from .console_doctor_node import DigitConsoleDoctor

NODE_CLASS_MAPPINGS = {
    "DigitLLMQuery": LLMQueryNode,
    "DigitImageSaver": DigitImageSaver,
    "DigitImageLoader": DigitImageLoader,
    "DigitGeminiImage": DigitGeminiImage,
    "DigitVeoVideo": DigitVeoVideo,
    "DigitVideoSaver": DigitVideoSaver,
    "DigitDragCrop": DigitDragCrop,
    "DigitCropInfo": DigitCropInfo,
    "DigitSRTMaker": DigitSRTMaker,
    "DigitRandomPrompt": DigitRandomPrompt,
    "DigitLoraLoader": DigitLoraLoader,
    "DigitLoraLoaderModelOnly": DigitLoraLoaderModelOnly,
    "DigitPromptCombine": DigitPromptCombine,
    "DigitBatchCaption": DigitBatchCaption,
    "DigitTextEncode": DigitTextEncode,
    "DigitDatasetPrep": DigitDatasetPrep,
    "DigitCaptionViewer": DigitCaptionViewer,
    "DigitConsoleDoctor": DigitConsoleDoctor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DigitLLMQuery": "DIGIT LLM Query",
    "DigitImageSaver": "DIGIT Image Saver",
    "DigitImageLoader": "DIGIT Image Loader",
    "DigitGeminiImage": "DIGIT Gemini Image",
    "DigitVeoVideo": "DIGIT Veo Video",
    "DigitVideoSaver": "DIGIT Video Saver",
    "DigitDragCrop": "DIGIT Drag Crop",
    "DigitCropInfo": "DIGIT Crop Info",
    "DigitSRTMaker": "DIGIT SRT Maker",
    "DigitRandomPrompt": "DIGIT Random Prompt",
    "DigitLoraLoader": "DIGIT LoRA Loader",
    "DigitLoraLoaderModelOnly": "DIGIT LoRA Loader (Model Only)",
    "DigitPromptCombine": "DIGIT Prompt Combine",
    "DigitBatchCaption": "DIGIT Batch Caption",
    "DigitTextEncode": "DIGIT Text Encode",
    "DigitDatasetPrep": "DIGIT Dataset Prep",
    "DigitCaptionViewer": "DIGIT Caption Viewer",
    "DigitConsoleDoctor": "DIGIT Console Doctor",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
