import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DIGIT.LoraLoader",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitLoraLoader") return;

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
    },
});
