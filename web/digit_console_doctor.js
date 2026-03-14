import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DIGIT.ConsoleDoctor",

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitConsoleDoctor") return;

        // Add read-only display widget for diagnosis
        const displayWidget = node.addWidget("text", "doctor_display", "", () => {}, {
            serialize: false,
        });
        if (displayWidget.inputEl) {
            displayWidget.inputEl.readOnly = true;
            displayWidget.inputEl.style.opacity = "0.8";
            displayWidget.inputEl.style.fontFamily = "monospace";
            displayWidget.inputEl.style.fontSize = "90%";
            displayWidget.inputEl.style.minHeight = "120px";
        }

        const onExecuted = node.onExecuted;
        node.onExecuted = function (data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.doctor_text?.length > 0) {
                displayWidget.value = data.doctor_text[0];
            }
        };
    },
});
