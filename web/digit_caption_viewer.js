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

        // Create a DOM container for status, filename, and caption
        const container = document.createElement("div");
        container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;padding:4px;box-sizing:border-box;";

        const statusEl = document.createElement("div");
        statusEl.style.cssText = "font-family:monospace;font-size:11px;opacity:0.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        statusEl.textContent = "—";
        container.appendChild(statusEl);

        const filenameEl = document.createElement("div");
        filenameEl.style.cssText = "font-family:monospace;font-size:12px;font-weight:bold;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        filenameEl.textContent = "";
        container.appendChild(filenameEl);

        const captionEl = document.createElement("textarea");
        captionEl.readOnly = true;
        captionEl.style.cssText = "font-family:monospace;font-size:11px;line-height:1.4;width:100%;min-height:140px;max-height:300px;resize:vertical;background:rgba(0,0,0,0.3);color:#ddd;border:1px solid rgba(255,255,255,0.15);border-radius:4px;padding:6px;box-sizing:border-box;white-space:pre-wrap;";
        captionEl.placeholder = "Caption will appear here after execution...";
        container.appendChild(captionEl);

        const captionWidget = node.addDOMWidget("caption_area", "customtext", container, {
            serialize: false,
            getMinHeight() { return 200; },
        });

        const onExecuted = node.onExecuted;
        node.onExecuted = function (data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.viewer_text?.length > 0) {
                statusEl.textContent = data.viewer_text[0];
            }
            if (data?.filename_text?.length > 0) {
                filenameEl.textContent = data.filename_text[0];
            }
            if (data?.caption_text?.length > 0) {
                captionEl.value = data.caption_text[0];
            }
        };
    },
});
