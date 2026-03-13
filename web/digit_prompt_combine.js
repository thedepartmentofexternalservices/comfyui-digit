import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DIGIT.PromptCombine",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitPromptCombine") return;

        // Add read-only display widget showing the combined prompt
        const displayWidget = node.addWidget("text", "combined_display", "", () => {}, {
            serialize: false,
        });
        if (displayWidget.inputEl) {
            displayWidget.inputEl.readOnly = true;
            displayWidget.inputEl.style.opacity = "0.8";
            displayWidget.inputEl.style.fontStyle = "italic";
        }

        // Update display when node executes
        const onExecuted = node.onExecuted;
        node.onExecuted = function (data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.combined_text?.length > 0) {
                displayWidget.value = data.combined_text[0];
            }
        };
    },
});
