import { app } from "../../scripts/app.js";
import { setupGeminiImageResolutionFilter } from "./digit_gemini_image_utils.js";

app.registerExtension({
    name: "DIGIT.GeminiImage",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitGeminiImage") return;
        setupGeminiImageResolutionFilter(node, "model");
    },
});
