import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DIGIT.DriftGate",

    async nodeCreated(node) {
        const driftNodes = ["DigitDriftGate", "DigitDriftGateFromPaths"];
        if (!driftNodes.includes(node.comfyClass)) return;

        const reportWidget = node.addWidget("text", "drift_report_display", "", () => {}, {
            multiline: true,
            serialize: false,
        });
        if (reportWidget.inputEl) {
            reportWidget.inputEl.readOnly = true;
            reportWidget.inputEl.rows = 18;
            reportWidget.inputEl.style.fontFamily = "monospace";
            reportWidget.inputEl.style.fontSize = "11px";
        }

        const verdictWidget = node.addWidget("text", "verdict_display", "", () => {}, {
            serialize: false,
        });
        if (verdictWidget.inputEl) {
            verdictWidget.inputEl.readOnly = true;
            verdictWidget.inputEl.style.fontWeight = "bold";
        }

        const qcPathWidget = node.addWidget("text", "qc_path_display", "", () => {}, {
            serialize: false,
        });
        if (qcPathWidget.inputEl) {
            qcPathWidget.inputEl.readOnly = true;
        }

        const onExecuted = node.onExecuted;
        node.onExecuted = function(data) {
            if (onExecuted) onExecuted.call(this, data);
            if (data?.drift_report_text?.length) {
                reportWidget.value = data.drift_report_text[0];
            }
            if (data?.confidence_text?.length && data?.verdict_text?.length) {
                verdictWidget.value = `${data.verdict_text[0]} — ${data.confidence_text[0]} confidence`;
                if (verdictWidget.inputEl) {
                    verdictWidget.inputEl.style.color =
                        data.verdict_text[0] === "PASS" ? "#4ade80" : "#f87171";
                }
            }
            if (data?.qc_filepath_text?.length) {
                qcPathWidget.value = data.qc_filepath_text[0];
            }
        };
    },
});
