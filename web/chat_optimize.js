import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const CHAT_CLASSES = ["ChatNode", "Chat"];
const VIEWER_CLASS = "ChatHistoryViewer";
const ROUTE_CHAT = "/chat_optimize/chat";
let historyModal = null;

function ensureHistoryModal() {
    if (historyModal) return historyModal;

    const overlay = document.createElement("div");
    overlay.id = "chat-optimize-history-modal";
    overlay.style.position = "fixed";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.display = "none";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";
    overlay.style.background = "rgba(0, 0, 0, 0.6)";
    overlay.style.zIndex = "9999";

    const panel = document.createElement("div");
    panel.style.width = "min(900px, 92%)";
    panel.style.height = "80%";
    panel.style.background = "#111";
    panel.style.color = "#ddd";
    panel.style.border = "1px solid #333";
    panel.style.borderRadius = "8px";
    panel.style.display = "flex";
    panel.style.flexDirection = "column";
    panel.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.5)";

    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.padding = "10px 12px";
    header.style.borderBottom = "1px solid #333";
    header.style.background = "#151515";

    const title = document.createElement("div");
    title.textContent = "Chat History";
    title.style.fontSize = "14px";
    title.style.fontWeight = "600";

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.style.background = "#222";
    closeBtn.style.color = "#ddd";
    closeBtn.style.border = "1px solid #444";
    closeBtn.style.borderRadius = "6px";
    closeBtn.style.padding = "6px 10px";
    closeBtn.style.cursor = "pointer";
    closeBtn.addEventListener("click", () => closeHistoryModal());

    header.appendChild(title);
    header.appendChild(closeBtn);

    const content = document.createElement("pre");
    content.style.flex = "1";
    content.style.margin = "0";
    content.style.padding = "12px";
    content.style.overflow = "auto";
    content.style.whiteSpace = "pre-wrap";
    content.style.fontFamily = "monospace";
    content.style.fontSize = "13px";

    panel.appendChild(header);
    panel.appendChild(content);
    overlay.appendChild(panel);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeHistoryModal();
    });

    document.body.appendChild(overlay);
    const onKey = (e) => {
        if (e.key === "Escape") closeHistoryModal();
    };
    document.addEventListener("keydown", onKey);

    historyModal = { overlay, content, onKey };
    return historyModal;
}

function openHistoryModal(text) {
    const modal = ensureHistoryModal();
    modal.content.textContent = text || "";
    modal.overlay.style.display = "flex";
    setTimeout(() => {
        modal.content.scrollTop = modal.content.scrollHeight;
    }, 0);
}

function closeHistoryModal() {
    if (!historyModal) return;
    historyModal.overlay.style.display = "none";
}

function isHistoryModalOpen() {
    return historyModal && historyModal.overlay.style.display !== "none";
}

function setWidgetValue(node, name, value) {
    if (!node.widgets) return;
    const w = node.widgets.find((w) => w.name === name);
    if (w) {
        w.value = value ?? "";
        if (w.onChange) w.onChange(value);
        node.setDirtyCanvas(true, true);
    }
}

function addOrGetWidget(node, type, name, opts = {}) {
    if (!node.widgets) node.widgets = [];
    let w = node.widgets.find((w) => w.name === name);
    if (w) return w;
    w = node.addWidget(type, name, opts.default ?? "", () => { }, opts);
    return w;
}

async function postJSON(path, body) {
    const res = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

function propagateOutput(node, slotIndex, value) {
    if (!node.outputs || !node.outputs[slotIndex] || !node.outputs[slotIndex].links) return;

    node.outputs[slotIndex].links.forEach((linkId) => {
        const link = app.graph.links[linkId];
        if (!link) return;
        const targetNode = app.graph.getNodeById(link.target_id);
        if (!targetNode) return;

        if (targetNode.inputs && targetNode.inputs[link.target_slot] && targetNode.inputs[link.target_slot].widget) {
            const widgetName = targetNode.inputs[link.target_slot].widget.name;
            const w = targetNode.widgets?.find(w => w.name === widgetName);
            if (w) {
                w.value = value;
                if (w.onChange) w.onChange(value);
                targetNode.setDirtyCanvas(true, true);
                return;
            }
        }
    });
}

function getLLMConfig(node) {
    const inputIndex = node.inputs?.findIndex(i => i.name === "llm_config");
    if (inputIndex === undefined || inputIndex === -1) return null;

    const input = node.inputs[inputIndex];
    if (!input.link) return null;

    const linkId = input.link;
    const link = app.graph.links[linkId];
    if (!link) return null;

    const sourceNode = app.graph.getNodeById(link.origin_id);
    if (!sourceNode) return null;

    const config = {};
    const providerW = sourceNode.widgets?.find(w => w.name === "provider");
    if (providerW) config.provider = providerW.value;

    const baseUrlW = sourceNode.widgets?.find(w => w.name === "base_url");
    if (baseUrlW) config.base_url = baseUrlW.value;

    const modelW = sourceNode.widgets?.find(w => w.name === "model_name");
    if (modelW) config.model_name = modelW.value;

    const tempW = sourceNode.widgets?.find(w => w.name === "temperature");
    if (tempW) config.temperature = tempW.value;

    const topPW = sourceNode.widgets?.find(w => w.name === "top_p");
    if (topPW) config.top_p = topPW.value;

    const maxTokensW = sourceNode.widgets?.find(w => w.name === "max_new_tokens");
    if (maxTokensW) config.max_new_tokens = maxTokensW.value;

    const tokenW = sourceNode.widgets?.find(w => w.name === "hf_token");
    if (tokenW) config.hf_token = tokenW.value;

    return config;
}

function updateWidgetLock(node) {
    const input = node.inputs?.find(i => i.name === "llm_config");
    const isConnected = input && input.link !== null;

    ["base_url", "model_name"].forEach(name => {
        const w = node.widgets?.find(w => w.name === name);
        if (w) {
            w.disabled = isConnected;
        }
    });
}

function setupChatNode(node) {
    console.log("[ChatNode] Setting up");
    
    let statusWidget = addOrGetWidget(node, "text", "status", { default: "idle" });
    if (statusWidget.inputEl) {
        statusWidget.inputEl.readOnly = true;
        statusWidget.inputEl.style.fontSize = "11px";
        statusWidget.inputEl.style.color = "#888";
        statusWidget.inputEl.style.backgroundColor = "#1a1a1a";
        statusWidget.inputEl.style.border = "1px solid #333";
    }

    let historyStore = addOrGetWidget(node, "hidden", "history_preview", { default: "" });
    let responseStore = addOrGetWidget(node, "hidden", "assistant_response_preview", { default: "" });

    const sendAction = async () => {
        const actionWidget = node.widgets?.find((w) => w.name === "action");
        const action = actionWidget ? actionWidget.value : "send";
        
        try {
            statusWidget.value = `working: ${action}...`;
            node.setDirtyCanvas(true, true);
            
            const payload = {};
            const excludeWidgets = ["status", "history_preview", "assistant_response_preview", "auto_clear_input"];
            
            (node.widgets || []).forEach((w) => {
                if (excludeWidgets.includes(w.name)) return;
                payload[w.name] = w.value;
            });
            payload["action"] = action;

            const llmConfig = getLLMConfig(node);
            if (llmConfig) payload["llm_config"] = llmConfig;

            const data = await postJSON(ROUTE_CHAT, payload);
            
            responseStore.value = data.assistant_response || "";
            historyStore.value = data.readable_history || "";

            propagateOutput(node, 0, data.assistant_response || "");
            propagateOutput(node, 1, data.readable_history || "");

            const autoClear = node.widgets?.find((w) => w.name === "auto_clear_input")?.value;
            if (action === "send" && autoClear) {
                setWidgetValue(node, "user_message", "");
            }

            statusWidget.value = "done";
            node.setDirtyCanvas(true, true);
            
            console.log("[ChatNode] Response received");
        } catch (err) {
            console.error("[ChatNode] Error:", err);
            statusWidget.value = `error: ${err.message}`;
            node.setDirtyCanvas(true, true);
        }
    };

    node.onConnectionsChange = () => setTimeout(() => updateWidgetLock(node), 100);
    setTimeout(() => updateWidgetLock(node), 100);

    const existingButton = node.widgets?.find(w => w.name === "Execute");
    if (!existingButton) {
        node.addWidget("button", "Execute", null, () => sendAction());
    }
}

function setupHistoryViewer(node) {
    console.log("[ChatHistoryViewer] Setting up stable scroll version");
    
    // 1. Find or create the history widget
    let historyWidget = node.widgets?.find(w => w.name === "history");
    if (!historyWidget) {
        historyWidget = node.addWidget("text", "history", "", () => {}, { multiline: true });
    }
    const getHistoryText = () => historyWidget.value ?? historyWidget.inputEl?.value ?? "";

    const existingButton = node.widgets?.find(w => w.name === "Open History");
    if (!existingButton) {
        node.addWidget("button", "Open History", null, () => openHistoryModal(getHistoryText()));
    }

    // 2. Clear default drawing of text on canvas
    node.onDrawForeground = function(ctx) {
        // Stop default text preview if any
    };

    // 3. Configure the textarea
    if (historyWidget.inputEl) {
        const el = historyWidget.inputEl;
        el.readOnly = true;
        el.style.display = "block"; // Force visible
        el.style.visibility = "visible";
        el.style.pointerEvents = "auto"; // Ensure scrollable
        el.style.height = "100%";
        el.style.width = "100%";
        el.style.backgroundColor = "#121212";
        el.style.color = "#ddd";
        el.style.fontFamily = "monospace";
        el.style.fontSize = "13px";
        el.style.padding = "10px";
        el.style.border = "none";
        el.style.boxSizing = "border-box";
        el.style.resize = "none";
        el.style.overflowY = "scroll";
        el.style.whiteSpace = "pre-wrap";
        
        // Custom scrollbar
        el.parentElement.style.overflow = "hidden"; // Container shouldn't scroll
    }

    // 4. Override widget visibility when connected
    // ComfyUI usually hides widgets linked to inputs. We want this one visible.
    const originalUpdate = node.onNodeCreated;
    historyWidget.type = "text"; // Keep it as text so it renders the textarea

    // Prevent ComfyUI from hiding it
    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function() {
        if (originalOnConnectionsChange) originalOnConnectionsChange.apply(this, arguments);
        if (historyWidget.inputEl) {
           historyWidget.inputEl.style.display = "block";
           historyWidget.inputEl.style.visibility = "visible";
        }
    };

    // 5. High-frequency sync and auto-scroll
    const originalOnChange = historyWidget.onChange;
    historyWidget.onChange = function(v) {
        if (originalOnChange) originalOnChange.call(this, v);
        // Sync DOM explicitly if needed
        if (historyWidget.inputEl && historyWidget.inputEl.value !== v) {
            historyWidget.inputEl.value = v || "";
        }
        if (isHistoryModalOpen()) {
            openHistoryModal(v || "");
        }
        // Auto scroll to bottom
        setTimeout(() => {
            if (historyWidget.inputEl) {
                historyWidget.inputEl.scrollTop = historyWidget.inputEl.scrollHeight;
            }
        }, 10);
    };

    // 6. Set initial size
    node.setSize([600, 500]);
}

app.registerExtension({
    name: "chat_optimize_ui",
    priority: 10,
    async nodeCreated(node) {
        if (CHAT_CLASSES.includes(node.comfyClass)) {
            setupChatNode(node);
        } else if (node.comfyClass === VIEWER_CLASS) {
            setupHistoryViewer(node);
        }
    },
});
