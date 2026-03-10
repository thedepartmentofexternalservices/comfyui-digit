import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

            openBrowserDialog(startPath, (selectedPath) => {
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


async function fetchDir(path) {
    const resp = await api.fetchApi(`/digit/browse?path=${encodeURIComponent(path)}`);
    if (resp.status !== 200) return null;
    return await resp.json();
}


function openBrowserDialog(startPath, onSelect) {
    // Create overlay
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "rgba(0,0,0,0.6)", zIndex: "10000",
        display: "flex", alignItems: "center", justifyContent: "center",
    });

    // Dialog container
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        backgroundColor: "#1e1e1e", border: "1px solid #555", borderRadius: "8px",
        width: "550px", maxHeight: "70vh", display: "flex", flexDirection: "column",
        color: "#ddd", fontFamily: "monospace", fontSize: "13px",
    });

    // Header with current path and up button
    const header = document.createElement("div");
    Object.assign(header.style, {
        padding: "10px 14px", borderBottom: "1px solid #444",
        display: "flex", alignItems: "center", gap: "8px", flexShrink: "0",
    });

    const upBtn = document.createElement("button");
    upBtn.textContent = "\u2191 Up";
    Object.assign(upBtn.style, {
        padding: "4px 10px", backgroundColor: "#333", color: "#ddd",
        border: "1px solid #555", borderRadius: "4px", cursor: "pointer", flexShrink: "0",
    });

    const pathLabel = document.createElement("div");
    Object.assign(pathLabel.style, {
        flex: "1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        color: "#aaa",
    });

    header.appendChild(upBtn);
    header.appendChild(pathLabel);

    // File list
    const listContainer = document.createElement("div");
    Object.assign(listContainer.style, {
        flex: "1", overflowY: "auto", padding: "6px 0",
    });

    // Footer with cancel
    const footer = document.createElement("div");
    Object.assign(footer.style, {
        padding: "10px 14px", borderTop: "1px solid #444",
        display: "flex", justifyContent: "flex-end", gap: "8px", flexShrink: "0",
    });

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    Object.assign(cancelBtn.style, {
        padding: "6px 16px", backgroundColor: "#333", color: "#ddd",
        border: "1px solid #555", borderRadius: "4px", cursor: "pointer",
    });
    cancelBtn.onclick = () => document.body.removeChild(overlay);

    footer.appendChild(cancelBtn);

    dialog.appendChild(header);
    dialog.appendChild(listContainer);
    dialog.appendChild(footer);
    overlay.appendChild(dialog);

    // Close on overlay click (not dialog)
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) document.body.removeChild(overlay);
    });

    let currentPath = startPath;

    async function navigateTo(path) {
        const data = await fetchDir(path);
        if (!data || data.error) return;

        currentPath = data.path;
        pathLabel.textContent = currentPath;
        listContainer.innerHTML = "";

        // Directories
        for (const dir of data.dirs) {
            const row = createRow("\uD83D\uDCC1 " + dir, true);
            row.addEventListener("dblclick", () => navigateTo(currentPath + "/" + dir));
            row.addEventListener("click", () => highlightRow(row));
            listContainer.appendChild(row);
        }

        // Files
        for (const file of data.files) {
            const row = createRow("\uD83D\uDDBC\uFE0F " + file, false);
            row.dataset.filepath = currentPath + "/" + file;
            row.addEventListener("click", () => highlightRow(row));
            row.addEventListener("dblclick", () => {
                onSelect(row.dataset.filepath);
                document.body.removeChild(overlay);
            });
            listContainer.appendChild(row);
        }

        if (data.dirs.length === 0 && data.files.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "(empty directory)";
            Object.assign(empty.style, { padding: "12px 14px", color: "#666" });
            listContainer.appendChild(empty);
        }
    }

    function createRow(text, isDir) {
        const row = document.createElement("div");
        row.textContent = text;
        row.dataset.isDir = isDir;
        Object.assign(row.style, {
            padding: "5px 14px", cursor: "pointer", userSelect: "none",
        });
        row.addEventListener("mouseenter", () => {
            if (!row.classList.contains("selected")) row.style.backgroundColor = "#2a2a2a";
        });
        row.addEventListener("mouseleave", () => {
            if (!row.classList.contains("selected")) row.style.backgroundColor = "";
        });
        return row;
    }

    function highlightRow(row) {
        listContainer.querySelectorAll("div").forEach(r => {
            r.classList.remove("selected");
            r.style.backgroundColor = "";
        });
        row.classList.add("selected");
        row.style.backgroundColor = "#3a5a8a";
    }

    upBtn.onclick = () => {
        const parent = currentPath.replace(/\/[^/]+\/?$/, "") || "/";
        if (parent !== currentPath) navigateTo(parent);
    };

    document.body.appendChild(overlay);
    navigateTo(startPath);
}
