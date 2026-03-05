import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "DIGIT.ImageSaver",

    async nodeCreated(node) {
        const hasShotNodes = ["DigitImageSaver", "DigitImageLoader", "DigitVideoSaver"];
        const projectOnlyNodes = ["DigitSRTMaker"];

        const isHasShotNode = hasShotNodes.includes(node.comfyClass);
        const isProjectOnlyNode = projectOnlyNodes.includes(node.comfyClass);

        if (!isHasShotNode && !isProjectOnlyNode) return;

        const rootWidget = node.widgets.find(w => w.name === "projekts_root");
        const projectWidget = node.widgets.find(w => w.name === "project");
        const shotWidget = node.widgets.find(w => w.name === "shot");

        if (!rootWidget || !projectWidget) return;
        if (isHasShotNode && !shotWidget) return;

        // Add a read-only text widget to show the filepath after execution
        const filepathWidget = node.addWidget("text", "filepath_display", "", () => {}, {
            serialize: false,
        });
        filepathWidget.inputEl && (filepathWidget.inputEl.readOnly = true);

        // Listen for execution results to update the filepath display
        const onExecuted = node.onExecuted;
        node.onExecuted = function(data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data && data.filepath_text && data.filepath_text.length > 0) {
                filepathWidget.value = data.filepath_text[0];
            }
        };

        async function refreshProjects() {
            const root = rootWidget.value;
            const resp = await api.fetchApi(`/digit/projects?root=${encodeURIComponent(root)}`);
            if (resp.status === 200) {
                const projects = await resp.json();
                projectWidget.options.values = projects;
                if (!projects.includes(projectWidget.value)) {
                    projectWidget.value = projects[0] || "";
                }
                if (isHasShotNode) {
                    await refreshShots();
                }
            }
        }

        async function refreshShots() {
            if (!shotWidget) return;
            const root = rootWidget.value;
            const project = projectWidget.value;
            const resp = await api.fetchApi(
                `/digit/shots?root=${encodeURIComponent(root)}&project=${encodeURIComponent(project)}`
            );
            if (resp.status === 200) {
                const shots = await resp.json();
                shotWidget.options.values = shots;
                if (!shots.includes(shotWidget.value)) {
                    shotWidget.value = shots[0] || "";
                }
            }
        }

        const origRootCallback = rootWidget.callback;
        rootWidget.callback = async function(value) {
            if (origRootCallback) origRootCallback.call(this, value);
            await refreshProjects();
        };

        const origProjectCallback = projectWidget.callback;
        projectWidget.callback = async function(value) {
            if (origProjectCallback) origProjectCallback.call(this, value);
            if (isHasShotNode) {
                await refreshShots();
            }
        };

        // Initial load
        if (isHasShotNode) {
            await refreshShots();
        }
    }
});
