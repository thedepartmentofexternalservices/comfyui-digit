import { app } from "../../scripts/app.js";
import { openFolderBrowserDialog } from "./digit_browse_utils.js";

app.registerExtension({
    name: "DIGIT.BatchCaption",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitBatchCaption") return;

        // Add Browse button for image_folder
        const folderWidget = node.widgets.find(w => w.name === "image_folder");
        if (folderWidget) {
            const browseBtn = node.addWidget("button", "browse_folder_btn", "Browse Image Folder", () => {
                const startPath = folderWidget.value || "/";
                openFolderBrowserDialog(startPath, (selectedPath) => {
                    folderWidget.value = selectedPath;
                    node.setDirtyCanvas(true);
                });
            });
            // Move button right after the folder widget
            const fIdx = node.widgets.indexOf(folderWidget);
            const bIdx = node.widgets.indexOf(browseBtn);
            if (fIdx >= 0 && bIdx >= 0 && bIdx !== fIdx + 1) {
                node.widgets.splice(bIdx, 1);
                node.widgets.splice(fIdx + 1, 0, browseBtn);
            }
        }

        // Add read-only display widget for progress/status
        const logWidget = node.addWidget("text", "log_display", "", () => {}, {
            serialize: false,
        });
        if (logWidget.inputEl) {
            logWidget.inputEl.readOnly = true;
            logWidget.inputEl.style.opacity = "0.8";
            logWidget.inputEl.style.fontFamily = "monospace";
            logWidget.inputEl.style.fontSize = "90%";
        }

        // Update display when node executes
        const onExecuted = node.onExecuted;
        node.onExecuted = function (data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.log_text?.length > 0) {
                logWidget.value = data.log_text[0];
            }
        };
    },
});
