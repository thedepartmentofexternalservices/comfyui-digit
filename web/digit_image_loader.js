import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openFileBrowserDialog } from "./digit_browse_utils.js";

app.registerExtension({
    name: "DIGIT.ImageLoaderBrowser",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitImageLoader") return;

        const browsePathWidget = node.widgets.find(w => w.name === "browse_path");
        if (!browsePathWidget) return;

        // Add a Browse button widget
        const browseBtn = node.addWidget("button", "browse_btn", "Browse Filesystem", async () => {
            // Build starting path from the node's current widget values
            const rootWidget = node.widgets.find(w => w.name === "projekts_root");
            const projectWidget = node.widgets.find(w => w.name === "project");
            const shotWidget = node.widgets.find(w => w.name === "shot");
            const subfolderWidget = node.widgets.find(w => w.name === "subfolder");
            const taskWidget = node.widgets.find(w => w.name === "task");

            let startPath = rootWidget ? rootWidget.value : "/";

            // Try to build the deepest valid path from the node's fields
            if (rootWidget && projectWidget && shotWidget && subfolderWidget && taskWidget) {
                const candidates = [
                    [rootWidget.value, projectWidget.value, "shots", shotWidget.value, subfolderWidget.value, taskWidget.value].join("/"),
                    [rootWidget.value, projectWidget.value, "shots", shotWidget.value, subfolderWidget.value].join("/"),
                    [rootWidget.value, projectWidget.value, "shots", shotWidget.value].join("/"),
                    [rootWidget.value, projectWidget.value, "shots"].join("/"),
                    [rootWidget.value, projectWidget.value].join("/"),
                    rootWidget.value,
                ];
                for (const candidate of candidates) {
                    const resp = await api.fetchApi(`/digit/browse?path=${encodeURIComponent(candidate)}`);
                    if (resp.status === 200) {
                        startPath = candidate;
                        break;
                    }
                }
            }

            openFileBrowserDialog(startPath, (selectedPath) => {
                browsePathWidget.value = selectedPath;
                node.setDirtyCanvas(true);
            });
        });

        // Move browse button right after browse_path
        const bpIdx = node.widgets.indexOf(browsePathWidget);
        const btnIdx = node.widgets.indexOf(browseBtn);
        if (bpIdx >= 0 && btnIdx >= 0 && btnIdx !== bpIdx + 1) {
            node.widgets.splice(btnIdx, 1);
            node.widgets.splice(bpIdx + 1, 0, browseBtn);
        }
    }
});
