import { app } from "../../scripts/app.js";
import { openFolderBrowserDialog } from "./digit_browse_utils.js";

app.registerExtension({
    name: "DIGIT.CaptionViewer",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitCaptionViewer") return;

        // Add Browse button for dataset_folder
        const folderWidget = node.widgets.find(w => w.name === "dataset_folder");
        if (folderWidget) {
            const browseBtn = node.addWidget("button", "browse_dataset_btn", "Browse Dataset Folder", () => {
                const startPath = folderWidget.value || "/";
                openFolderBrowserDialog(startPath, (selectedPath) => {
                    folderWidget.value = selectedPath;
                    node.setDirtyCanvas(true);
                });
            });
            const fIdx = node.widgets.indexOf(folderWidget);
            const bIdx = node.widgets.indexOf(browseBtn);
            if (fIdx >= 0 && bIdx >= 0 && bIdx !== fIdx + 1) {
                node.widgets.splice(bIdx, 1);
                node.widgets.splice(fIdx + 1, 0, browseBtn);
            }
        }

        // Add Prev / Next buttons for navigating index
        const indexWidget = node.widgets.find(w => w.name === "index");
        if (indexWidget) {
            const navContainer = node.addWidget("button", "prev_btn", "\u25C0 Prev", () => {
                const cur = indexWidget.value || 0;
                indexWidget.value = Math.max(0, cur - 1);
                node.setDirtyCanvas(true);
                // Trigger queue to re-execute the node
                app.queuePrompt(0, 1);
            });

            const nextBtn = node.addWidget("button", "next_btn", "Next \u25B6", () => {
                const cur = indexWidget.value || 0;
                indexWidget.value = cur + 1;
                node.setDirtyCanvas(true);
                app.queuePrompt(0, 1);
            });

            // Position buttons right after the index widget
            const iIdx = node.widgets.indexOf(indexWidget);
            const pIdx = node.widgets.indexOf(navContainer);
            const nIdx = node.widgets.indexOf(nextBtn);
            // Move prev after index, next after prev
            if (iIdx >= 0 && pIdx >= 0) {
                node.widgets.splice(pIdx, 1);
                node.widgets.splice(iIdx + 1, 0, navContainer);
            }
            const nIdx2 = node.widgets.indexOf(nextBtn);
            const pIdx2 = node.widgets.indexOf(navContainer);
            if (pIdx2 >= 0 && nIdx2 >= 0) {
                node.widgets.splice(nIdx2, 1);
                node.widgets.splice(pIdx2 + 1, 0, nextBtn);
            }
        }

        const displayWidget = node.addWidget("text", "viewer_display", "", () => {}, {
            serialize: false,
        });
        if (displayWidget.inputEl) {
            displayWidget.inputEl.readOnly = true;
            displayWidget.inputEl.style.opacity = "0.8";
            displayWidget.inputEl.style.fontFamily = "monospace";
            displayWidget.inputEl.style.fontSize = "90%";
        }

        const onExecuted = node.onExecuted;
        node.onExecuted = function (data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.viewer_text?.length > 0) {
                displayWidget.value = data.viewer_text[0];
            }
        };
    },
});
