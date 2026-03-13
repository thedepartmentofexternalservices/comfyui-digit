import { app } from "../../scripts/app.js";
import { openFolderBrowserDialog } from "./digit_browse_utils.js";

function addFolderBrowseButton(node, widgetName, buttonLabel) {
    const folderWidget = node.widgets.find(w => w.name === widgetName);
    if (!folderWidget) return;

    const browseBtn = node.addWidget("button", `browse_${widgetName}_btn`, buttonLabel, () => {
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

app.registerExtension({
    name: "DIGIT.DatasetPrep",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitDatasetPrep") return;

        addFolderBrowseButton(node, "source_folder", "Browse Source Folder");
        addFolderBrowseButton(node, "output_folder", "Browse Output Folder");

        const displayWidget = node.addWidget("text", "log_display", "", () => {}, {
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
            if (data?.log_text?.length > 0) {
                displayWidget.value = data.log_text[0];
            }
        };
    },
});
