import { app } from "../../scripts/app.js";

const LAYER_ORDER = ["reference", "generated", "diff", "edge", "annotated"];
const LAYER_LABELS = {
    reference: "Reference",
    generated: "Generated",
    diff: "Pixel Diff",
    edge: "Edge Diff",
    annotated: "Annotated",
};
const WIDGET_ROW_H = 20;
const PLAYER_MIN_H = 420;

function stopTimer(node) {
    if (node._qcTimer) {
        clearInterval(node._qcTimer);
        node._qcTimer = null;
    }
}

function getIntervalMs(node) {
    return Math.max(250, Math.round((node._qc?.intervalSec ?? 1.0) * 1000));
}

function currentLayerId(node) {
    const mode = node._qc?.viewMode || "ab";

    if (mode === "single") {
        return node._qc?.singleLayer || "reference";
    }
    if (mode === "cycle") {
        const idx = node.properties.cycleIndex || 0;
        return LAYER_ORDER[idx % LAYER_ORDER.length];
    }
    return node.properties.abShowingRef ? "reference" : "generated";
}

function imageDataToUrl(data) {
    const fmt = app.getPreviewFormatParam?.() ?? "";
    const rand = app.getRandParam?.() ?? `&rand=${Date.now()}`;
    return app.api.apiURL(
        `/view?filename=${encodeURIComponent(data.filename)}&type=${encodeURIComponent(data.type || "temp")}&subfolder=${encodeURIComponent(data.subfolder || "")}${fmt}${rand}`
    );
}

function updateBanner(node) {
    if (!node._qc?.bannerEl) return;
    const layerId = currentLayerId(node);
    const mode = node._qc?.viewMode || "ab";
    const label = LAYER_LABELS[layerId] || layerId;
    let text = label;

    if (mode === "ab") text += node.properties.abShowingRef ? " (A)" : " (B)";
    if (mode === "cycle") text += ` [${(node.properties.cycleIndex || 0) + 1}/${LAYER_ORDER.length}]`;

    const verdict = node._qc.manifest?.verdict;
    const conf = node._qc.manifest?.confidence;
    if (verdict) text += `  |  ${verdict}`;
    if (typeof conf === "number") text += ` ${conf.toFixed(1)}%`;

    node._qc.bannerEl.textContent = text;
}

function showCurrentLayer(node) {
    if (!node._qc?.imgEl) return;
    const layerId = currentLayerId(node);
    const url = node._qc.layerUrls?.[layerId];
    updateBanner(node);

    if (!url) {
        node._qc.imgEl.removeAttribute("src");
        node._qc.imgEl.alt = "Waiting for QC layers…";
        return;
    }
    node._qc.imgEl.alt = LAYER_LABELS[layerId] || layerId;
    if (node._qc.imgEl.src !== url) {
        node._qc.imgEl.src = url;
    }
}

function updatePlayButton(node) {
    const btn = node._qc?.playBtn;
    if (btn) btn.textContent = node.properties.qcPlaying ? "⏸" : "▶";
}

function startTimer(node) {
    stopTimer(node);
    if (!node.properties.qcPlaying) return;

    node._qcTimer = setInterval(() => {
        const mode = node._qc?.viewMode || "ab";
        if (mode === "ab") {
            node.properties.abShowingRef = !node.properties.abShowingRef;
        } else if (mode === "cycle") {
            node.properties.cycleIndex = ((node.properties.cycleIndex || 0) + 1) % LAYER_ORDER.length;
        } else {
            return;
        }
        showCurrentLayer(node);
    }, getIntervalMs(node));
}

function normalizeOutput(data) {
    if (!data) return null;
    if (data.qc_layers || data.layer_manifest_text || data.images) return data;
    if (data.output) return data.output;
    return data;
}

function parseManifest(data) {
    const output = normalizeOutput(data);
    const raw = output?.qc_layers?.[0] || output?.layer_manifest_text?.[0] || null;
    if (!raw) return null;
    try {
        return typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch (e) {
        console.warn("[DIGIT QC Preview] Could not parse manifest", e);
        return null;
    }
}

function bindLayerUrls(node, output, manifest) {
    node._qc.layerUrls = {};

    if (node.imgs?.length >= LAYER_ORDER.length) {
        LAYER_ORDER.forEach((id, i) => {
            const img = node.imgs[i];
            if (img?.src) node._qc.layerUrls[id] = img.src;
        });
        return true;
    }

    const images = output?.images || manifest?.layers || [];
    if (images.length >= LAYER_ORDER.length) {
        LAYER_ORDER.forEach((id, i) => {
            const entry = images[i];
            if (entry?.filename) node._qc.layerUrls[id] = imageDataToUrl(entry);
        });
        return Object.keys(node._qc.layerUrls).length > 0;
    }

    return false;
}

async function waitForNodeImages(node, maxMs = 3000) {
    const steps = Math.ceil(maxMs / 100);
    for (let i = 0; i < steps; i++) {
        if (node.imgs?.length >= LAYER_ORDER.length) return true;
        await new Promise((r) => setTimeout(r, 100));
    }
    return (node.imgs?.length ?? 0) > 0;
}

async function handleExecuted(node, data) {
    const output = normalizeOutput(data);
    const manifest = parseManifest(output);

    if (!manifest && !output?.images?.length) {
        if (node._qc?.bannerEl) {
            node._qc.bannerEl.textContent = "No QC data — wire Drift Gate and re-queue";
        }
        return;
    }

    node._qc = node._qc || {};
    node._qc.manifest = manifest || node._qc.manifest;

    await waitForNodeImages(node);

    const ok = bindLayerUrls(node, output, manifest);
    if (!ok) {
        if (node._qc.bannerEl) {
            node._qc.bannerEl.textContent = "Layers failed to load — check terminal";
        }
        return;
    }

    node.properties.cycleIndex = 0;
    node.properties.abShowingRef = true;
    node.properties.qcPlaying = true;
    updatePlayButton(node);
    showCurrentLayer(node);
    startTimer(node);
    node.setDirtyCanvas(true, true);
}

function mkBtn(label, title) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title || label;
    b.style.cssText =
        "height:20px;min-width:22px;padding:0 5px;font:11px/1 system-ui,sans-serif;" +
        "background:#2a2a32;color:#e8e8ee;border:1px solid #444;border-radius:3px;cursor:pointer;";
    b.onmouseenter = () => { b.style.background = "#3a3a44"; };
    b.onmouseleave = () => { b.style.background = "#2a2a32"; };
    return b;
}

function mkSelect(options, title) {
    const s = document.createElement("select");
    s.title = title || "";
    s.style.cssText =
        "height:20px;font:11px/1 system-ui,sans-serif;background:#2a2a32;color:#e8e8ee;" +
        "border:1px solid #444;border-radius:3px;padding:0 4px;max-width:110px;";
    for (const [val, label] of options) {
        const o = document.createElement("option");
        o.value = val;
        o.textContent = label;
        s.appendChild(o);
    }
    return s;
}

function setupDomPreview(node) {
    node.properties = node.properties || {};
    node.properties.cycleIndex = 0;
    node.properties.abShowingRef = true;
    node.properties.qcPlaying = false;

    const root = document.createElement("div");
    root.style.cssText =
        "display:flex;flex-direction:column;gap:3px;width:100%;height:100%;padding:2px;box-sizing:border-box;background:#111116;border-radius:4px;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText =
        "display:flex;flex-wrap:wrap;align-items:center;gap:3px;padding:2px 0;";

    const modeSelect = mkSelect(
        [["ab", "A/B"], ["cycle", "Cycle"], ["single", "Single"]],
        "View mode"
    );
    const layerSelect = mkSelect(
        LAYER_ORDER.map((id) => [id, LAYER_LABELS[id]]),
        "Layer (single mode)"
    );
    const playBtn = mkBtn("▶", "Play / Pause");
    const abBtn = mkBtn("A/B", "Toggle reference / generated");
    const prevBtn = mkBtn("◀", "Previous layer");
    const nextBtn = mkBtn("▶▶", "Next layer");

    const intervalLabel = document.createElement("span");
    intervalLabel.style.cssText = "font:10px monospace;color:#999;margin-left:2px;";
    intervalLabel.textContent = "1.0s";

    const intervalRange = document.createElement("input");
    intervalRange.type = "range";
    intervalRange.min = "0.25";
    intervalRange.max = "5";
    intervalRange.step = "0.25";
    intervalRange.value = "1";
    intervalRange.title = "Blink interval (seconds)";
    intervalRange.style.cssText = "width:72px;height:14px;margin:0;padding:0;vertical-align:middle;";

    toolbar.append(modeSelect, layerSelect, playBtn, abBtn, prevBtn, nextBtn, intervalRange, intervalLabel);
    root.appendChild(toolbar);

    const bannerEl = document.createElement("div");
    bannerEl.style.cssText =
        "font:10px monospace;color:#c8c8d0;padding:2px 4px;background:rgba(0,0,0,0.35);border-radius:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    bannerEl.textContent = "Queue workflow to load QC preview";
    root.appendChild(bannerEl);

    const imgEl = document.createElement("img");
    imgEl.style.cssText =
        "display:block;flex:1 1 auto;width:100%;min-height:320px;object-fit:contain;background:#0a0a0e;border-radius:3px;";
    imgEl.alt = "QC preview";
    imgEl.onerror = () => {
        bannerEl.textContent = "Image load failed — Cmd+Shift+R and re-queue";
    };
    root.appendChild(imgEl);

    node._qc = {
        imgEl,
        bannerEl,
        layerUrls: {},
        manifest: null,
        viewMode: "ab",
        singleLayer: "reference",
        intervalSec: 1.0,
        playBtn,
    };

    modeSelect.onchange = () => {
        node._qc.viewMode = modeSelect.value;
        showCurrentLayer(node);
        startTimer(node);
    };

    layerSelect.onchange = () => {
        node._qc.singleLayer = layerSelect.value;
        node._qc.viewMode = "single";
        modeSelect.value = "single";
        showCurrentLayer(node);
    };

    intervalRange.oninput = () => {
        node._qc.intervalSec = parseFloat(intervalRange.value);
        intervalLabel.textContent = `${node._qc.intervalSec.toFixed(2)}s`;
        startTimer(node);
    };

    playBtn.onclick = () => {
        node.properties.qcPlaying = !node.properties.qcPlaying;
        updatePlayButton(node);
        if (node.properties.qcPlaying) startTimer(node);
        else stopTimer(node);
    };

    abBtn.onclick = () => {
        node._qc.viewMode = "ab";
        modeSelect.value = "ab";
        node.properties.abShowingRef = !node.properties.abShowingRef;
        showCurrentLayer(node);
    };

    prevBtn.onclick = () => {
        node._qc.viewMode = "cycle";
        modeSelect.value = "cycle";
        node.properties.cycleIndex = ((node.properties.cycleIndex || 0) - 1 + LAYER_ORDER.length) % LAYER_ORDER.length;
        showCurrentLayer(node);
    };

    nextBtn.onclick = () => {
        node._qc.viewMode = "cycle";
        modeSelect.value = "cycle";
        node.properties.cycleIndex = ((node.properties.cycleIndex || 0) + 1) % LAYER_ORDER.length;
        showCurrentLayer(node);
    };

    node.addDOMWidget("qc_preview_panel", "qcpreview", root, {
        serialize: false,
        getMinHeight() {
            return PLAYER_MIN_H;
        },
        getHeight() {
            return Math.max(PLAYER_MIN_H, (node.size?.[1] || 0) - countInputRows(node) * WIDGET_ROW_H - 40);
        },
    });

    const inputRows = countInputRows(node);
    node.setSize([Math.max(node.size?.[0] || 0, 480), Math.max(node.size?.[1] || 0, inputRows * WIDGET_ROW_H + PLAYER_MIN_H + 48)]);
}

function countInputRows(node) {
    return (node.inputs?.length || 0) + (node.outputs?.length || 0) > 6 ? 3 : 2;
}

app.registerExtension({
    name: "DIGIT.DriftQCPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DigitDriftQCPreview") return;

        nodeType.prototype.computeSize = function () {
            const inputH = countInputRows(this) * WIDGET_ROW_H;
            return [480, inputH + PLAYER_MIN_H + 40];
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (this.comfyClass === "DigitDriftQCPreview") {
                const nodeRef = this;
                const msg = message;
                setTimeout(() => {
                    handleExecuted(nodeRef, msg).catch((e) => {
                        console.error("[DIGIT QC Preview] handleExecuted failed", e);
                    });
                }, 150);
            }
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            stopTimer(this);
            onRemoved?.apply(this, arguments);
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "DigitDriftQCPreview") return;
        setupDomPreview(node);
    },
});
