import { app } from "../../scripts/app.js";
import { openFolderBrowserDialog, openFileBrowserDialog } from "./digit_browse_utils.js";

app.registerExtension({
    name: "DIGIT.SRTFromVideo",

    async nodeCreated(node) {
        // ── Single file node: video_path file browser ──────────────────────
        if (node.comfyClass === "DigitSRTFromVideo") {
            const pathWidget = node.widgets.find(w => w.name === "video_path");
            if (pathWidget) {
                const browseBtn = node.addWidget("button", "browse_video_btn", "Browse Video File", () => {
                    const startPath = pathWidget.value
                        ? pathWidget.value.replace(/\/[^/]+$/, "")
                        : "/";
                    openFileBrowserDialog(startPath, (selectedPath) => {
                        pathWidget.value = selectedPath;
                        node.setDirtyCanvas(true);
                    }, "videos");
                });
                const pIdx = node.widgets.indexOf(pathWidget);
                const bIdx = node.widgets.indexOf(browseBtn);
                if (pIdx >= 0 && bIdx >= 0 && bIdx !== pIdx + 1) {
                    node.widgets.splice(bIdx, 1);
                    node.widgets.splice(pIdx + 1, 0, browseBtn);
                }
            }

            // Status display
            const logWidget = node.addWidget("text", "log_display", "", () => {}, {
                serialize: false,
            });
            if (logWidget.inputEl) {
                logWidget.inputEl.readOnly = true;
                logWidget.inputEl.style.opacity = "0.8";
                logWidget.inputEl.style.fontFamily = "monospace";
                logWidget.inputEl.style.fontSize = "90%";
            }
            const onExecuted = node.onExecuted;
            node.onExecuted = function (data) {
                if (onExecuted) onExecuted.call(this, data);
                if (data?.filepath_text?.length > 0) {
                    logWidget.value = data.filepath_text[0];
                }
            };
        }

        // ── Batch node: video_folder browser ───────────────────────────────
        if (node.comfyClass === "DigitBatchSRTFromVideo") {
            const folderWidget = node.widgets.find(w => w.name === "video_folder");
            if (folderWidget) {
                const browseBtn = node.addWidget("button", "browse_folder_btn", "Browse Video Folder", () => {
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

            // Log display
            const logWidget = node.addWidget("text", "log_display", "", () => {}, {
                serialize: false,
            });
            if (logWidget.inputEl) {
                logWidget.inputEl.readOnly = true;
                logWidget.inputEl.style.opacity = "0.8";
                logWidget.inputEl.style.fontFamily = "monospace";
                logWidget.inputEl.style.fontSize = "90%";
            }
            const onExecuted = node.onExecuted;
            node.onExecuted = function (data) {
                if (onExecuted) onExecuted.call(this, data);
                if (data?.log_text?.length > 0) {
                    logWidget.value = data.log_text[0];
                }
            };
        }

        // ── SRT Tools: srt_filepath file browser ───────────────────────────
        if (node.comfyClass === "DigitSRTTools") {
            const fileWidget = node.widgets.find(w => w.name === "srt_filepath");
            if (fileWidget) {
                const browseBtn = node.addWidget("button", "browse_srt_btn", "Browse SRT File", () => {
                    const startPath = fileWidget.value
                        ? fileWidget.value.replace(/\/[^/]+$/, "")
                        : "/";
                    openFileBrowserDialog(startPath, (selectedPath) => {
                        fileWidget.value = selectedPath;
                        node.setDirtyCanvas(true);
                    }, "all");
                });
                const fIdx = node.widgets.indexOf(fileWidget);
                const bIdx = node.widgets.indexOf(browseBtn);
                if (fIdx >= 0 && bIdx >= 0 && bIdx !== fIdx + 1) {
                    node.widgets.splice(bIdx, 1);
                    node.widgets.splice(fIdx + 1, 0, browseBtn);
                }
            }

            // Log display
            const logWidget = node.addWidget("text", "log_display", "", () => {}, {
                serialize: false,
            });
            if (logWidget.inputEl) {
                logWidget.inputEl.readOnly = true;
                logWidget.inputEl.style.opacity = "0.8";
                logWidget.inputEl.style.fontFamily = "monospace";
                logWidget.inputEl.style.fontSize = "90%";
            }
            const onExecuted = node.onExecuted;
            node.onExecuted = function (data) {
                if (onExecuted) onExecuted.call(this, data);
                if (data?.log_text?.length > 0) {
                    logWidget.value = data.log_text[0];
                }
            };
        }

        // ── SRT Preview: srt_filepath file browser + display ───────────────
        if (node.comfyClass === "DigitSRTPreview") {
            const fileWidget = node.widgets.find(w => w.name === "srt_filepath");
            if (fileWidget) {
                const browseBtn = node.addWidget("button", "browse_srt_btn", "Browse SRT File", () => {
                    const startPath = fileWidget.value
                        ? fileWidget.value.replace(/\/[^/]+$/, "")
                        : "/";
                    openFileBrowserDialog(startPath, (selectedPath) => {
                        fileWidget.value = selectedPath;
                        node.setDirtyCanvas(true);
                    }, "all");
                });
                const fIdx = node.widgets.indexOf(fileWidget);
                const bIdx = node.widgets.indexOf(browseBtn);
                if (fIdx >= 0 && bIdx >= 0 && bIdx !== fIdx + 1) {
                    node.widgets.splice(bIdx, 1);
                    node.widgets.splice(fIdx + 1, 0, browseBtn);
                }
            }

            // Preview display
            const logWidget = node.addWidget("text", "preview_display", "", () => {}, {
                serialize: false,
            });
            if (logWidget.inputEl) {
                logWidget.inputEl.readOnly = true;
                logWidget.inputEl.style.opacity = "0.8";
                logWidget.inputEl.style.fontFamily = "monospace";
                logWidget.inputEl.style.fontSize = "90%";
                logWidget.inputEl.style.minHeight = "100px";
            }
            const onExecuted = node.onExecuted;
            node.onExecuted = function (data) {
                if (onExecuted) onExecuted.call(this, data);
                if (data?.log_text?.length > 0) {
                    logWidget.value = data.log_text[0];
                }
            };
        }
    },
});
