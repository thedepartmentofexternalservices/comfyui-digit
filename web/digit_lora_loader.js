import { app } from "../../scripts/app.js";
import { openFileBrowserDialog } from "./digit_browse_utils.js";

function setupLoraLoaderNode(node) {
    // Add Browse button for lora_path_override
    const overrideWidget = node.widgets.find(w => w.name === "lora_path_override");
    if (overrideWidget) {
        const browseBtn = node.addWidget("button", "browse_lora_btn", "Browse LoRA File", () => {
            const startPath = overrideWidget.value || "/";
            // Navigate to parent directory if a file path is set
            const dir = startPath.includes("/") ? startPath.replace(/\/[^/]+$/, "") || "/" : "/";
            openFileBrowserDialog(dir, (selectedPath) => {
                overrideWidget.value = selectedPath;
                node.setDirtyCanvas(true);
            }, "loras");
        });
        const oIdx = node.widgets.indexOf(overrideWidget);
        const bIdx = node.widgets.indexOf(browseBtn);
        if (oIdx >= 0 && bIdx >= 0 && bIdx !== oIdx + 1) {
            node.widgets.splice(bIdx, 1);
            node.widgets.splice(oIdx + 1, 0, browseBtn);
        }
    }

    // Add read-only display widget for trigger word
    const triggerWidget = node.addWidget("text", "trigger_display", "", () => {}, {
        serialize: false,
    });
    if (triggerWidget.inputEl) {
        triggerWidget.inputEl.readOnly = true;
        triggerWidget.inputEl.style.opacity = "0.8";
        triggerWidget.inputEl.style.fontWeight = "bold";
    }

    // Add read-only display widget for info line
    const infoWidget = node.addWidget("text", "info_display", "", () => {}, {
        serialize: false,
    });
    if (infoWidget.inputEl) {
        infoWidget.inputEl.readOnly = true;
        infoWidget.inputEl.style.opacity = "0.6";
        infoWidget.inputEl.style.fontSize = "90%";
    }

    // Update display widgets when node executes
    const onExecuted = node.onExecuted;
    node.onExecuted = function (data) {
        if (onExecuted) onExecuted.call(this, data);
        if (data?.trigger_text?.length > 0) {
            triggerWidget.value = "trigger: " + data.trigger_text[0];
        }
        if (data?.info_text?.length > 0) {
            infoWidget.value = data.info_text[0];
        }
    };
}

app.registerExtension({
    name: "DIGIT.LoraLoader",

    async nodeCreated(node) {
        if (node.comfyClass === "DigitLoraLoader" || node.comfyClass === "DigitLoraLoaderModelOnly") {
            setupLoraLoaderNode(node);
        }
    },
});
