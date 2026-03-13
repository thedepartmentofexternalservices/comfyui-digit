/**
 * Shared filesystem browser dialog for DIGIT nodes.
 * Provides openFolderBrowserDialog() for selecting folders
 * and openFileBrowserDialog() for selecting files.
 */
import { api } from "../../scripts/api.js";

async function fetchDir(path) {
    const resp = await api.fetchApi(`/digit/browse?path=${encodeURIComponent(path)}`);
    if (resp.status !== 200) return null;
    return await resp.json();
}

/**
 * Open a dialog to browse and select a folder.
 * @param {string} startPath - Starting directory path
 * @param {function} onSelect - Called with the selected folder path
 */
export function openFolderBrowserDialog(startPath, onSelect) {
    _openBrowserDialog(startPath, onSelect, "folder");
}

/**
 * Open a dialog to browse and select a file.
 * @param {string} startPath - Starting directory path
 * @param {function} onSelect - Called with the selected file path
 */
export function openFileBrowserDialog(startPath, onSelect) {
    _openBrowserDialog(startPath, onSelect, "file");
}

function _openBrowserDialog(startPath, onSelect, mode) {
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

    // Header
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

    // Footer
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

    if (mode === "folder") {
        const selectFolderBtn = document.createElement("button");
        selectFolderBtn.textContent = "Select This Folder";
        Object.assign(selectFolderBtn.style, {
            padding: "6px 16px", backgroundColor: "#2a5a3a", color: "#ddd",
            border: "1px solid #4a8a5a", borderRadius: "4px", cursor: "pointer",
        });
        selectFolderBtn.onclick = () => {
            onSelect(currentPath);
            document.body.removeChild(overlay);
        };
        footer.appendChild(selectFolderBtn);
    }

    footer.appendChild(cancelBtn);

    dialog.appendChild(header);
    dialog.appendChild(listContainer);
    dialog.appendChild(footer);
    overlay.appendChild(dialog);

    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) document.body.removeChild(overlay);
    });

    let currentPath = startPath || "/";

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

        // Files (only show in file mode)
        if (mode === "file") {
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
        }

        if (listContainer.children.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = mode === "folder" ? "(empty directory — use Select This Folder)" : "(empty directory)";
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
    navigateTo(currentPath);
}
