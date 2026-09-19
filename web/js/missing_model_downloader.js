import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load CSS — co-located in same WEB_DIRECTORY (web/js/) so it's always served
const cssLink = document.createElement("link");
cssLink.rel = "stylesheet";
cssLink.type = "text/css";
cssLink.href = new URL("style.css", import.meta.url).href;
document.head.appendChild(cssLink);

// Sorted folder list for dropdowns
const SORTED_FOLDERS = [
    "checkpoints", "clip", "clip_vision", "controlnet",
    "diffusion_models", "embeddings", "loras",
    "upscale_models", "vae"
];

// Escapes text before it's interpolated into an innerHTML template. Model filenames, node
// types, and search-result metadata all originate from workflow files or third-party APIs,
// so they must never be trusted as raw HTML.
function escapeHtml(value) {
    const str = value === null || value === undefined ? "" : String(value);
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

class MissingModelDownloaderUI {
    constructor() {
        this.missingModels = [];
        this.activeDownloads = {};
        this.searchCache = {};
        this.config = {
            has_hf_token: false,
            hf_token_masked: "",
            has_civitai_token: false,
            civitai_token_masked: "",
            default_provider: "huggingface",
            auto_detect_on_load: true
        };
        this.modal = null;
        this.nativeBtn = null;
        this.isScanning = false;
        this.activeSearches = new Set();
    }

    async init() {
        await this.loadConfig();
        this.createTopbarButton();
        this.createModal();
        this.setupWebSocketListeners();
        this.setupWorkflowHooks();
        this.setupKeyboardShortcut();

        // Initial scan after UI is ready
        setTimeout(() => this.scanWorkflow(false), 2000);
    }

    async loadConfig() {
        try {
            const resp = await api.fetchApi("/model_downloader/config");
            if (resp.ok) {
                this.config = await resp.json();
            }
        } catch (e) {
            console.error("[ModelDownloader] Failed to load config:", e);
        }
    }

    async createTopbarButton() {
        // ComfyUI V2 Topbar Integration only — no floating button
        try {
            let ComfyBtn = null;
            let ComfyBtnGroup = null;

            if (window.comfyAPI) {
                ComfyBtn = window.comfyAPI.button?.ComfyButton;
                ComfyBtnGroup = window.comfyAPI.buttonGroup?.ComfyButtonGroup;
            }

            if (!ComfyBtn) {
                try {
                    const mod = await import("../../scripts/ui/components/button.js");
                    ComfyBtn = mod.ComfyButton;
                } catch (e) { /* Not available */ }
            }
            if (!ComfyBtnGroup) {
                try {
                    const mod = await import("../../scripts/ui/components/buttonGroup.js");
                    ComfyBtnGroup = mod.ComfyButtonGroup;
                } catch (e) { /* Not available */ }
            }

            if (ComfyBtn && ComfyBtnGroup) {
                const nativeButton = new ComfyBtn({
                    icon: "download",
                    action: () => this.openModal(),
                    tooltip: "Missing Model Downloader (Ctrl+Shift+M)",
                    content: "Missing Models",
                    classList: "comfyui-button comfyui-menu-mobile-collapse primary"
                });
                this.nativeBtn = nativeButton;
                const group = new ComfyBtnGroup(nativeButton.element);

                const mountNative = () => {
                    if (app.menu?.settingsGroup?.element) {
                        app.menu.settingsGroup.element.before(group.element);
                        return true;
                    } else if (app.menu?.actionsGroup?.element) {
                        app.menu.actionsGroup.element.before(group.element);
                        return true;
                    }
                    return false;
                };

                if (!mountNative()) {
                    let attempts = 0;
                    const poll = setInterval(() => {
                        attempts++;
                        if (mountNative() || attempts > 30) {
                            clearInterval(poll);
                            if (!mountNative()) this.createFloatingButton();
                        }
                    }, 300);
                }
            } else {
                this.createFloatingButton();
            }
        } catch (e) {
            console.warn("[ModelDownloader] ComfyButton integration error:", e);
            this.createFloatingButton();
        }
    }

    createFloatingButton() {
        if (this.floatingBtn) return;
        const btn = document.createElement("button");
        btn.className = "mmd-btn mmd-btn-primary";
        btn.style.cssText = `
            position: fixed; bottom: 20px; right: 20px; z-index: 10000;
            padding: 10px 16px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            display: flex; align-items: center; gap: 8px; font-weight: 600;
        `;
        btn.innerHTML = `<span>📦</span> <span class="mmd-fb-text">Missing Models</span>`;
        btn.onclick = () => this.openModal();
        document.body.appendChild(btn);
        this.floatingBtn = btn;
    }

    setupKeyboardShortcut() {
        window.addEventListener("keydown", (e) => {
            const isCtrlShiftM = (e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "m" || e.key === "M");
            if (isCtrlShiftM) {
                e.preventDefault();
                e.stopPropagation();
                if (this.modal && this.modal.classList.contains("active")) {
                    this.closeModal();
                } else {
                    this.openModal();
                }
            }
        }, true);
    }

    updateBadge(count) {
        if (this.nativeBtn && this.nativeBtn.element) {
            if (count > 0) {
                this.nativeBtn.element.textContent = `Missing Models (${count})`;
                this.nativeBtn.element.style.color = "#cd5c5c";
            } else {
                this.nativeBtn.element.textContent = "Missing Models";
                this.nativeBtn.element.style.color = "";
            }
        }
        if (this.floatingBtn) {
            const textSpan = this.floatingBtn.querySelector(".mmd-fb-text");
            if (count > 0) {
                textSpan.textContent = `Missing Models (${count})`;
                this.floatingBtn.style.backgroundColor = "#cd5c5c";
            } else {
                textSpan.textContent = "Missing Models";
                this.floatingBtn.style.backgroundColor = "";
            }
        }
    }

    setupWebSocketListeners() {
        api.addEventListener("model_downloader_progress", (event) => {
            const task = event.detail;
            if (!task || !task.id) return;
            this.activeDownloads[task.id] = task;
            this.updateDownloadsTab();
        });

        api.addEventListener("model_downloader_completed", async (event) => {
            const task = event.detail;
            if (task) {
                this.showToast(`Downloaded: ${task.filename}`);
                try { await api.fetchApi("/model_downloader/refresh_cache", {method: "POST"}); } catch (e) {}
                if (app.refreshObjectInfo) {
                    await app.refreshObjectInfo();
                }
                setTimeout(() => this.scanWorkflow(false), 500);
            }
        });
    }

    setupWorkflowHooks() {
        const origLoadGraphData = app.loadGraphData;
        if (origLoadGraphData) {
            app.loadGraphData = (...args) => {
                const res = origLoadGraphData.apply(app, args);
                if (this.config.auto_detect_on_load) {
                    setTimeout(() => this.scanWorkflow(true), 800);
                }
                return res;
            };
        }
    }

    async scanWorkflow(notifyUser = false) {
        if (this.isScanning) return;
        this.isScanning = true;

        try {
            const candidates = [];
            if (app.graph && app.graph._nodes) {
                for (const node of app.graph._nodes) {
                    if (node.widgets) {
                        for (const w of node.widgets) {
                            if (w && typeof w.value === "string" && w.value.trim().length > 0) {
                                const val = w.value.trim();
                                const isModelExt = /\.(safetensors|gguf|ckpt|pt|bin|pth|onnx)$/i.test(val);
                                const isModelWidget = /(ckpt|lora|vae|controlnet|unet|clip|model)/i.test(w.name || "");
                                if (isModelExt || isModelWidget) {
                                    candidates.push({
                                        node_id: node.id,
                                        node_type: node.type || node.title,
                                        widget_name: w.name,
                                        value: val
                                    });
                                }
                            }
                        }
                    }
                }
            }

            let promptData = null;
            try {
                if (app.graphToPrompt) {
                    const p = await app.graphToPrompt();
                    promptData = p.output || p.prompt; // Support both just in case
                }
            } catch (err) {
                console.warn("[ModelDownloader] Failed to get graphToPrompt:", err);
            }
            
            let workflowData = null;
            try {
                if (app.graph) {
                    workflowData = app.graph.serialize();
                }
            } catch (err) {
                console.warn("[ModelDownloader] Failed to serialize graph:", err);
            }

            const resp = await api.fetchApi("/model_downloader/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ candidates, prompt: promptData, workflow: workflowData })
            });

            if (resp.ok) {
                const data = await resp.json();
                this.missingModels = data.missing_models || [];
                this.updateBadge(this.missingModels.length);
                this.renderMissingModels();
                this.checkExactMatches();

                if (notifyUser && this.missingModels.length > 0) {
                    this.showBanner(`${this.missingModels.length} missing model(s) detected.`);
                }

                this.missingModels.forEach((m, index) => {
                    if (!this.searchCache[m.filename]) {
                        this.prefetchSearch(m.filename, m.folder_type, index);
                    }
                });
            }
        } catch (e) {
            console.error("[ModelDownloader] Error scanning workflow:", e);
        } finally {
            this.isScanning = false;
        }
    }

    async prefetchSearch(filename, folderType, index) {
        if (!this.activeSearches) this.activeSearches = new Set();
        this.activeSearches.add(filename);

        const card = this.modal ? this.modal.querySelector(`#mmd-missing-card-${index}`) : null;
        let searchIndicator = null;
        if (card) {
            const metaDiv = card.querySelector('.mmd-model-meta');
            if (metaDiv && !metaDiv.querySelector('.mmd-card-searching')) {
                searchIndicator = document.createElement("span");
                searchIndicator.className = "mmd-card-searching mmd-searching-anim";
                searchIndicator.innerHTML = `<div class="mmd-loader" style="width: 10px; height: 10px; border-width: 2px; border-top-color: inherit; margin: 0; display: inline-block;"></div> Searching...`;
                metaDiv.appendChild(searchIndicator);
            }
        }

        try {
            const resp = await api.fetchApi("/model_downloader/search", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: filename, provider: "all", limit: 10 })
            });
            if (resp.ok) {
                const data = await resp.json();
                this.searchCache[filename] = { results: data.results || [], time: Date.now(), folderType };
                this.checkExactMatches();
            }
        } catch (e) {
        } finally {
            this.activeSearches.delete(filename);
            const activeCard = this.modal ? this.modal.querySelector(`#mmd-missing-card-${index}`) : null;
            if (activeCard) {
                const indicator = activeCard.querySelector('.mmd-card-searching');
                if (indicator) indicator.remove();
            }
            if (searchIndicator && searchIndicator.parentNode) {
                searchIndicator.remove();
            }
        }
    }

    checkExactMatches() {
        const exactMatches = [];
        this.missingModels.forEach((m, index) => {
            const cache = this.searchCache[m.filename];
            
            const card = this.modal ? this.modal.querySelector(`#mmd-missing-card-${index}`) : null;
            const metaDiv = card ? card.querySelector('.mmd-model-meta') : null;
            const quickDlBtn = card ? card.querySelector('.mmd-quick-dl-btn') : null;
            let exactText = metaDiv ? metaDiv.querySelector('.mmd-exact-match-text') : null;

            if (cache && Date.now() - cache.time < 300000) {
                const exact = cache.results.find(r => r.exact_match);
                if (exact) {
                    exactMatches.push({ model: m, exactResult: exact, folderType: cache.folderType });
                    if (metaDiv && !exactText) {
                        exactText = document.createElement("span");
                        exactText.className = "mmd-exact-match-text";
                        exactText.style.cssText = "color: var(--mmd-success); font-weight: 600; font-size: 11px; margin-left: 6px;";
                        exactText.textContent = "Exact match found ✓";
                        metaDiv.appendChild(exactText);
                    }
                    if (quickDlBtn) {
                        quickDlBtn.style.display = "inline-block";
                        quickDlBtn.onclick = () => {
                            this.startDownload(exact.download_url, m.filename, cache.folderType, "", false, exact.sha256);
                        };
                    }
                } else {
                    if (exactText) exactText.remove();
                    if (quickDlBtn) quickDlBtn.style.display = "none";
                }
            } else {
                if (exactText) exactText.remove();
                if (quickDlBtn) quickDlBtn.style.display = "none";
            }
        });

        const list = this.modal ? this.modal.querySelector("#mmd-missing-list") : null;
        if (!list) return;
        
        let headerAction = list.querySelector("#mmd-exact-matches-btn");
        if (exactMatches.length > 0) {
            if (!headerAction) {
                const btnDiv = document.createElement("div");
                btnDiv.innerHTML = `<button id="mmd-exact-matches-btn" class="mmd-btn mmd-btn-primary" style="width: 100%; margin-bottom: 8px;">Download ${exactMatches.length} Exact Matches</button>`;
                list.insertBefore(btnDiv.firstChild, list.firstChild);
                headerAction = list.querySelector("#mmd-exact-matches-btn");
            } else {
                headerAction.textContent = `Download ${exactMatches.length} Exact Matches`;
            }
            headerAction.onclick = () => {
                exactMatches.forEach(match => {
                    this.startDownload(match.exactResult.download_url, match.model.filename, match.folderType, "", false, match.exactResult.sha256);
                });
            };
        } else if (headerAction) {
            if (headerAction.parentElement === list) {
                list.removeChild(headerAction);
            }
        }
    }

    /** Build sorted folder <option> HTML */
    buildFolderOptions(folders, selectedFolder) {
        const sorted = [...folders].sort((a, b) => a.localeCompare(b));
        return sorted.map(f =>
            `<option value="${f}" ${f === selectedFolder ? 'selected' : ''}>${f}</option>`
        ).join("");
    }

    createModal() {
        const folderOptionsHTML = this.buildFolderOptions(SORTED_FOLDERS, "checkpoints");

        const backdrop = document.createElement("div");
        backdrop.className = "mmd-modal-backdrop";
        backdrop.id = "mmd-modal-backdrop";

        backdrop.innerHTML = `
            <div class="mmd-modal">
                <div class="mmd-header">
                    <div class="mmd-header-left">
                        <div class="mmd-header-title">
                            <span class="accent">Model Downloader</span>
                        </div>
                    </div>
                    <button class="mmd-close-btn" id="mmd-close-btn">&times;</button>
                </div>

                <div class="mmd-tabs">
                    <button class="mmd-tab active" data-tab="missing">Missing Models</button>
                    <button class="mmd-tab" data-tab="downloads">Active Downloads</button>
                    <button class="mmd-tab" data-tab="direct">Direct Download</button>
                    <button class="mmd-tab" data-tab="settings">Settings</button>
                </div>

                <div class="mmd-body">
                    <!-- Tab 1: Missing Models -->
                    <div class="mmd-panel active" id="mmd-panel-missing">
                        <div class="mmd-toolbar">
                            <span style="font-size: 12px; color: var(--mmd-text-muted);">
                                Models referenced in the workflow but not found locally.
                            </span>
                            <button class="mmd-btn mmd-btn-outline" id="mmd-rescan-btn">Rescan</button>
                        </div>
                        <div id="mmd-missing-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
                    </div>

                    <!-- Tab 2: Active Downloads -->
                    <div class="mmd-panel" id="mmd-panel-downloads">
                        <div class="mmd-toolbar" style="justify-content: flex-end; margin-bottom: 8px;">
                            <button class="mmd-btn mmd-btn-outline" id="mmd-clear-history-btn">Clear History</button>
                        </div>
                        <div id="mmd-downloads-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        
                        <div style="margin-top: 16px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 13px; color: var(--mmd-text-main); border-bottom: 1px solid var(--mmd-border); padding-bottom: 4px;">History</h4>
                            <div id="mmd-history-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        </div>
                    </div>

                    <!-- Tab 3: Direct Download -->
                    <div class="mmd-panel" id="mmd-panel-direct">
                        <div class="mmd-card">
                            <div class="mmd-form-group">
                                <label class="mmd-form-label">URL</label>
                                <input type="text" id="mmd-direct-url" class="mmd-input" 
                                       placeholder="https://huggingface.co/org/model/blob/main/file.safetensors" />
                                <span class="mmd-form-hint">Hugging Face or Civitai file URL.</span>
                            </div>
                            <div class="mmd-form-group">
                                <label class="mmd-form-label">Target Folder</label>
                                <select id="mmd-direct-folder" class="mmd-select">
                                    ${folderOptionsHTML}
                                </select>
                            </div>
                            <div class="mmd-form-group">
                                <label class="mmd-form-label">Filename (optional)</label>
                                <input type="text" id="mmd-direct-filename" class="mmd-input" placeholder="Auto-detected from URL" />
                            </div>
                            <button class="mmd-btn mmd-btn-primary" id="mmd-direct-start-btn" style="align-self: flex-start;">
                                Download
                            </button>
                        </div>
                    </div>

                    <!-- Tab 4: Settings -->
                    <div class="mmd-panel" id="mmd-panel-settings">
                        <div class="mmd-card">
                            <h3 style="margin: 0 0 4px 0; font-size: 13px; font-weight: 600; color: var(--mmd-text-main);">API Tokens</h3>
                            <div class="mmd-form-group">
                                <label class="mmd-form-label">
                                    <span>Hugging Face Token</span>
                                    <span id="mmd-hf-status" style="font-size: 10px;"></span>
                                </label>
                                <div class="mmd-input-group">
                                    <input type="password" id="mmd-hf-token" class="mmd-input" 
                                           placeholder="hf_..." autocomplete="off" />
                                    <button class="mmd-btn mmd-btn-outline" id="mmd-verify-hf-btn">Verify</button>
                                </div>
                                <span class="mmd-form-hint">
                                    Required for gated models (FLUX, SD3). 
                                    <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Get token</a>
                                </span>
                            </div>

                            <div class="mmd-form-group">
                                <label class="mmd-form-label">
                                    <span>Civitai API Key</span>
                                    <span id="mmd-civitai-status" style="font-size: 10px;"></span>
                                </label>
                                <div class="mmd-input-group">
                                    <input type="password" id="mmd-civitai-token" class="mmd-input" 
                                           placeholder="Civitai API Key" autocomplete="off" />
                                    <button class="mmd-btn mmd-btn-outline" id="mmd-verify-civitai-btn">Verify</button>
                                </div>
                                <span class="mmd-form-hint">
                                    For authenticated models.
                                    <a href="https://civitai.com/user/account" target="_blank" rel="noreferrer">Get key</a>
                                </span>
                            </div>

                            <div class="mmd-form-group">
                                <label class="mmd-form-label">Default Provider</label>
                                <select id="mmd-default-provider" class="mmd-select">
                                    <option value="huggingface">Hugging Face</option>
                                    <option value="civitai">Civitai</option>
                                </select>
                            </div>

                            <div class="mmd-form-group" style="flex-direction: row; align-items: center; gap: 8px;">
                                <input type="checkbox" id="mmd-auto-detect" style="cursor: pointer; accent-color: #888;" />
                                <label for="mmd-auto-detect" style="font-size: 12px; color: var(--mmd-text-secondary); cursor: pointer;">
                                    Auto-scan on workflow load
                                </label>
                            </div>

                            <button class="mmd-btn mmd-btn-primary" id="mmd-save-settings-btn" style="align-self: flex-start;">
                                Save Settings
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(backdrop);
        this.modal = backdrop;

        // Events
        backdrop.querySelector("#mmd-close-btn").onclick = () => this.closeModal();
        backdrop.onclick = (e) => {
            if (e.target === backdrop) this.closeModal();
        };

        // Escape key to close
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && this.modal.classList.contains("active")) {
                this.closeModal();
            }
        });

        // Tab switching
        const tabs = backdrop.querySelectorAll(".mmd-tab");
        tabs.forEach(tab => {
            tab.onclick = () => {
                tabs.forEach(t => t.classList.remove("active"));
                backdrop.querySelectorAll(".mmd-panel").forEach(p => p.classList.remove("active"));
                tab.classList.add("active");
                const targetPanel = backdrop.querySelector(`#mmd-panel-${tab.dataset.tab}`);
                if (targetPanel) targetPanel.classList.add("active");
                if (tab.dataset.tab === "downloads") this.fetchActiveDownloads();
            };
        });

        backdrop.querySelector("#mmd-rescan-btn").onclick = () => this.scanWorkflow(true);
        backdrop.querySelector("#mmd-clear-history-btn").onclick = async () => {
            try {
                await api.fetchApi("/model_downloader/clear_history", { method: "POST" });
                this.fetchActiveDownloads();
            } catch (e) {}
        };
        backdrop.querySelector("#mmd-direct-start-btn").onclick = () => this.handleDirectDownload();
        backdrop.querySelector("#mmd-save-settings-btn").onclick = () => this.handleSaveSettings();
        backdrop.querySelector("#mmd-verify-hf-btn").onclick = () => this.handleVerifyHfToken();
        backdrop.querySelector("#mmd-verify-civitai-btn").onclick = () => this.handleVerifyCivitaiToken();
    }

    openModal() {
        if (!this.modal) return;
        this.modal.classList.add("active");
        this.populateSettingsFields();
        this.fetchActiveDownloads();
    }

    closeModal() {
        if (!this.modal) return;
        this.modal.classList.remove("active");
    }

    populateSettingsFields() {
        const hfInput = this.modal.querySelector("#mmd-hf-token");
        const civitaiInput = this.modal.querySelector("#mmd-civitai-token");
        const providerSelect = this.modal.querySelector("#mmd-default-provider");
        const autoDetect = this.modal.querySelector("#mmd-auto-detect");

        if (this.config.has_hf_token) {
            hfInput.value = this.config.hf_token_masked || "hf_••••••••";
            this.modal.querySelector("#mmd-hf-status").innerHTML = `<span style="color: var(--mmd-success);">Saved</span>`;
        } else {
            hfInput.value = "";
            this.modal.querySelector("#mmd-hf-status").innerHTML = `<span style="color: var(--mmd-text-muted);">—</span>`;
        }

        if (this.config.has_civitai_token) {
            civitaiInput.value = this.config.civitai_token_masked || "••••••••";
            this.modal.querySelector("#mmd-civitai-status").innerHTML = `<span style="color: var(--mmd-success);">Saved</span>`;
        } else {
            civitaiInput.value = "";
            this.modal.querySelector("#mmd-civitai-status").innerHTML = `<span style="color: var(--mmd-text-muted);">—</span>`;
        }

        providerSelect.value = this.config.default_provider || "huggingface";
        autoDetect.checked = this.config.auto_detect_on_load !== false;
    }

    async handleVerifyHfToken() {
        const tokenVal = this.modal.querySelector("#mmd-hf-token").value.trim();
        const statusSpan = this.modal.querySelector("#mmd-hf-status");
        statusSpan.innerHTML = `<span style="color: var(--mmd-text-muted);">Checking...</span>`;

        try {
            const resp = await api.fetchApi("/model_downloader/test_token", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ provider: "huggingface", token: tokenVal })
            });
            const data = await resp.json();
            if (data.valid) {
                statusSpan.innerHTML = `<span style="color: var(--mmd-success);">Valid · ${data.username}</span>`;
            } else {
                statusSpan.innerHTML = `<span style="color: var(--mmd-danger);">Invalid</span>`;
            }
        } catch (e) {
            statusSpan.innerHTML = `<span style="color: var(--mmd-danger);">Error</span>`;
        }
    }

    async handleVerifyCivitaiToken() {
        const tokenVal = this.modal.querySelector("#mmd-civitai-token").value.trim();
        const statusSpan = this.modal.querySelector("#mmd-civitai-status");
        statusSpan.innerHTML = `<span style="color: var(--mmd-text-muted);">Checking...</span>`;

        try {
            const resp = await api.fetchApi("/model_downloader/test_token", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ provider: "civitai", token: tokenVal })
            });
            const data = await resp.json();
            if (data.valid) {
                statusSpan.innerHTML = `<span style="color: var(--mmd-success);">Valid</span>`;
            } else {
                statusSpan.innerHTML = `<span style="color: var(--mmd-danger);">Invalid</span>`;
            }
        } catch (e) {
            statusSpan.innerHTML = `<span style="color: var(--mmd-danger);">Error</span>`;
        }
    }

    async handleSaveSettings() {
        const hfToken = this.modal.querySelector("#mmd-hf-token").value.trim();
        const civitaiToken = this.modal.querySelector("#mmd-civitai-token").value.trim();
        const defaultProvider = this.modal.querySelector("#mmd-default-provider").value;
        const autoDetect = this.modal.querySelector("#mmd-auto-detect").checked;

        const payload = {
            default_provider: defaultProvider,
            auto_detect_on_load: autoDetect
        };

        if (hfToken && !hfToken.includes("••••")) payload.hf_token = hfToken;
        if (civitaiToken && !civitaiToken.includes("••••")) payload.civitai_token = civitaiToken;

        try {
            const resp = await api.fetchApi("/model_downloader/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (resp.ok) {
                const res = await resp.json();
                this.config = res.config;
                this.populateSettingsFields();
                this.showToast("Settings saved.");
            } else {
                this.showToast("Failed to save.", true);
            }
        } catch (e) {
            this.showToast("Failed to save.", true);
        }
    }

    renderMissingModels() {
        const list = this.modal ? this.modal.querySelector("#mmd-missing-list") : null;
        if (!list) return;

        if (this.missingModels.length === 0) {
            list.innerHTML = `
                <div class="mmd-empty-state">
                    <div class="icon">—</div>
                    <div style="font-size: 13px; font-weight: 500; color: var(--mmd-text-secondary);">No missing models</div>
                    <div style="font-size: 12px;">All referenced models were found locally.</div>
                </div>
            `;
            return;
        }

        list.innerHTML = "";
        this.missingModels.forEach((model, index) => {
            const card = document.createElement("div");
            card.className = "mmd-card";
            card.id = `mmd-missing-card-${index}`;

            const availableFolders = model.available_folders || SORTED_FOLDERS;
            const folderOptions = this.buildFolderOptions(availableFolders, model.folder_type);

            card.innerHTML = `
                <div class="mmd-card-header">
                    <div>
                        <div class="mmd-model-title">${escapeHtml(model.filename)}</div>
                        <div class="mmd-model-meta" style="margin-top: 4px;">
                            <span class="mmd-tag mmd-tag-node" style="cursor: pointer;" title="Click to locate node">${escapeHtml(model.node_type)}</span>
                            <span>${escapeHtml(model.widget_name)}</span>
                            ${this.activeSearches && this.activeSearches.has(model.filename) ? '<span class="mmd-card-searching mmd-searching-anim"><div class="mmd-loader" style="width: 10px; height: 10px; border-width: 2px; border-top-color: inherit; margin: 0; display: inline-block;"></div> Searching...</span>' : ''}
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <select class="mmd-select mmd-folder-select" style="min-width: 120px;">
                            ${folderOptions}
                        </select>
                        <button class="mmd-btn mmd-btn-outline mmd-collapse-btn" style="display: none; padding: 4px 8px;" title="Toggle results">▼</button>
                        <button class="mmd-btn mmd-quick-dl-btn" style="display: none; background: var(--mmd-success); color: white; border: none; font-weight: 500;">Download</button>
                        <button class="mmd-btn mmd-btn-primary mmd-search-btn">Search</button>
                    </div>
                </div>
                <div class="mmd-results-container" style="display: none;" id="mmd-results-${index}"></div>
            `;

            const searchBtn = card.querySelector(".mmd-search-btn");
            const collapseBtn = card.querySelector(".mmd-collapse-btn");
            const resultsContainer = card.querySelector(`#mmd-results-${index}`);
            const folderSelect = card.querySelector(".mmd-folder-select");
            const nodeTag = card.querySelector(".mmd-tag-node");
            
            nodeTag.onclick = () => {
                this.closeModal();
                if (app.canvas && app.graph) {
                    const n = app.graph.getNodeById(model.node_id);
                    if (n) {
                        app.canvas.centerOnNode(n);
                        app.canvas.selectNode(n);
                    }
                }
            };

            collapseBtn.onclick = () => {
                if (resultsContainer.style.display === "none") {
                    resultsContainer.style.display = "flex";
                    collapseBtn.textContent = "▼";
                } else {
                    resultsContainer.style.display = "none";
                    collapseBtn.textContent = "▶";
                }
            };

            searchBtn.onclick = () => this.searchModel(model.filename, folderSelect.value, resultsContainer, searchBtn, collapseBtn);

            list.appendChild(card);
        });
        
        this.checkExactMatches();
    }

    async searchModel(filename, folderType, container, btn, collapseBtn) {
        btn.disabled = true;
        btn.textContent = "Searching...";
        collapseBtn.style.display = "none";
        container.style.display = "flex";
        container.innerHTML = `<div style="font-size: 11px; color: var(--mmd-text-muted);">Querying sources...</div>`;
        
        try {
            let results = [];
            const cache = this.searchCache[filename];
            if (cache && Date.now() - cache.time < 300000) {
                results = cache.results;
            } else {
                const resp = await api.fetchApi("/model_downloader/search", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: filename, provider: "all", limit: 10 })
                });
                if (!resp.ok) throw new Error("Search failed.");
                const data = await resp.json();
                results = data.results || [];
                this.searchCache[filename] = { results, time: Date.now(), folderType };
                this.checkExactMatches();
            }

            if (results.length === 0) {
                container.innerHTML = `
                    <div style="font-size: 11px; color: var(--mmd-text-muted); padding: 6px 0;">
                        No results found. Try the Direct Download tab with a URL.
                    </div>
                `;
                return;
            }

            container.innerHTML = "";
            results.forEach(res => {
                const item = document.createElement("div");
                item.className = "mmd-result-item";

                const isHF = res.source === "huggingface";
                const sourceBadge = `<span class="mmd-tag mmd-tag-hf">${isHF ? "HF" : "Civitai"}</span>`;

                const exactBadge = res.exact_match
                    ? `<span class="mmd-tag" style="background: var(--mmd-success-bg); color: var(--mmd-success);">Exact</span>`
                    : "";

                const sizeDisplay = res.size_bytes > 0
                    ? `${(res.size_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
                    : "—";

                const gatedBadge = res.is_gated
                    ? `<span class="mmd-tag" style="background: var(--mmd-danger-bg); color: var(--mmd-danger);">Auth</span>`
                    : "";

                const thumbHtml = res.thumbnail
                    ? `<img src="${escapeHtml(res.thumbnail)}" alt="" class="mmd-result-thumb" loading="lazy" />`
                    : "";

                item.innerHTML = `
                    ${thumbHtml}
                    <div class="mmd-result-info">
                        <div style="display: flex; align-items: center; gap: 5px; flex-wrap: wrap;">
                            ${sourceBadge}${exactBadge}${gatedBadge}
                            <span class="mmd-result-name">${escapeHtml(res.name || filename)}</span>
                        </div>
                        <div class="mmd-result-details">
                            <span>${escapeHtml(res.repo_id || res.creator || res.model_name || "—")}</span>
                            <span>· ${sizeDisplay}</span>
                            ${res.downloads ? `<span>· ${Number(res.downloads).toLocaleString()} dl</span>` : ""}
                        </div>
                    </div>
                    <button class="mmd-btn mmd-btn-primary mmd-dl-btn">Download</button>
                `;

                item.querySelector(".mmd-dl-btn").onclick = () => {
                    this.startDownload(res.download_url, filename, folderType, "", false, res.sha256);
                };

                container.appendChild(item);
            });
            
            collapseBtn.style.display = "inline-block";
            collapseBtn.textContent = "▼";

        } catch (e) {
            container.innerHTML = `<div style="color: var(--mmd-danger); font-size: 11px;">${e.message}</div>`;
        } finally {
            btn.disabled = false;
            btn.textContent = "Search";
        }
    }

    async startDownload(url, filename, folderType, targetDir = "", overwrite = false, sha256 = "") {
        try {
            const resp = await api.fetchApi("/model_downloader/start_download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url, filename, folder_type: folderType, target_dir: targetDir, overwrite, sha256 })
            });

            if (resp.ok) {
                this.showToast(`Starting: ${filename}`);
                const dlTab = this.modal.querySelector('.mmd-tab[data-tab="downloads"]');
                if (dlTab) dlTab.click();
                return;
            }

            const err = await resp.json();
            if (resp.status === 409) {
                const confirmed = window.confirm(`"${filename}" already exists in the target folder. Overwrite it?`);
                if (confirmed) {
                    await this.startDownload(url, filename, folderType, targetDir, true);
                }
                return;
            }
            this.showToast(err.message || "Download failed.", true);
        } catch (e) {
            this.showToast(e.message, true);
        }
    }

    async handleDirectDownload() {
        const urlInput = this.modal.querySelector("#mmd-direct-url");
        const folderSelect = this.modal.querySelector("#mmd-direct-folder");
        const filenameInput = this.modal.querySelector("#mmd-direct-filename");

        const url = urlInput.value.trim();
        if (!url) { this.showToast("Enter a URL.", true); return; }

        let filename = filenameInput.value.trim();
        if (!filename) {
            filename = url.split("?")[0].split("/").pop();
            if (!filename || !filename.includes(".")) filename = "model.safetensors";
        }

        await this.startDownload(url, filename, folderSelect.value);
        urlInput.value = "";
    }

    async fetchActiveDownloads() {
        try {
            const resp = await api.fetchApi("/model_downloader/downloads");
            if (resp.ok) {
                const data = await resp.json();
                (data.downloads || []).forEach(task => {
                    this.activeDownloads[task.id] = task;
                });
                this.updateDownloadsTab();
            }
        } catch (e) { /* silent */ }
    }

    updateDownloadsTab() {
        const list = this.modal ? this.modal.querySelector("#mmd-downloads-list") : null;
        const historyList = this.modal ? this.modal.querySelector("#mmd-history-list") : null;
        if (!list || !historyList) return;

        const tasks = Object.values(this.activeDownloads);
        
        const activeTasks = tasks.filter(t => ["downloading", "queued", "pending", "verifying"].includes(t.status) || (t.status === "paused"));
        const historyTasks = tasks.filter(t => ["completed", "failed", "cancelled"].includes(t.status) && t.status !== "paused");

        list.innerHTML = "";
        historyList.innerHTML = "";

        if (activeTasks.length === 0) {
            list.innerHTML = `<div style="font-size: 12px; color: var(--mmd-text-muted);">No active downloads.</div>`;
        }
        if (historyTasks.length === 0) {
            historyList.innerHTML = `<div style="font-size: 12px; color: var(--mmd-text-muted);">No history.</div>`;
        }

        const renderCard = (task, container) => {
            const card = document.createElement("div");
            card.className = "mmd-card";

            const isDone = task.status === "completed";
            const isFailed = task.status === "failed";
            const isCancelled = task.status === "cancelled";
            const isPaused = task.status === "paused";
            const isQueued = task.status === "queued" || task.status === "pending";
            const isDownloading = task.status === "downloading";
            const isVerifying = task.status === "verifying";

            let statusText = "";
            let statusColor = "var(--mmd-text-muted)";
            if (isDone) { statusText = "Completed"; statusColor = "var(--mmd-success)"; }
            else if (isFailed) { statusText = "Failed"; statusColor = "var(--mmd-danger)"; }
            else if (isCancelled) { statusText = "Cancelled"; statusColor = "var(--mmd-text-muted)"; }
            else if (isPaused) { statusText = "Paused"; statusColor = "var(--mmd-warn)"; }
            else if (isQueued) { statusText = "Queued"; statusColor = "var(--mmd-warn)"; }
            else if (isVerifying) { statusText = "Verifying SHA256"; statusColor = "var(--mmd-warn)"; }
            else { statusText = "Downloading"; statusColor = "var(--mmd-text-secondary)"; }

            const dlMB = ((task.downloaded_bytes || 0) / (1024 * 1024)).toFixed(1);
            const totalMB = task.total_bytes > 0 ? (task.total_bytes / (1024 * 1024)).toFixed(1) : "?";

            let statsText = "";
            if (isDownloading) {
                const speed = task.speed_mb || 0;
                const etaMin = Math.floor((task.eta_seconds || 0) / 60);
                const etaSec = Math.floor((task.eta_seconds || 0) % 60);
                statsText = `${speed} MB/s · ${dlMB}/${totalMB} MB (${task.percentage || 0}%) · ETA ${etaMin}m ${etaSec}s`;
            } else if (isDone) {
                statsText = `${dlMB} MB → ${task.folder_type}`;
            } else if (isFailed) {
                statsText = task.error || "Error";
            } else if (isPaused) {
                statsText = `${dlMB}/${totalMB} MB (${task.percentage || 0}%)`;
            }

            card.innerHTML = `
                <div class="mmd-card-header">
                    <div>
                        <div class="mmd-model-title">${escapeHtml(task.filename)}</div>
                        <div class="mmd-model-meta" style="margin-top: 3px;">
                            <span style="color: ${statusColor}; font-size: 10px; font-weight: 600; text-transform: uppercase;">${statusText}</span>
                            <span class="mmd-tag mmd-tag-folder">${escapeHtml(task.folder_type)}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 4px;">
                        ${isDownloading ? `<button class="mmd-btn mmd-btn-outline mmd-pause-btn">Pause</button>` : ""}
                        ${isPaused ? `<button class="mmd-btn mmd-btn-outline mmd-resume-btn">Resume</button>` : ""}
                        ${(isDownloading || isQueued || isPaused || isVerifying) ? `<button class="mmd-btn mmd-btn-danger mmd-cancel-btn">Cancel</button>` : ""}
                        ${isFailed ? `<button class="mmd-btn mmd-btn-outline mmd-retry-btn">Retry</button>` : ""}
                    </div>
                </div>
                <div class="mmd-progress-wrap">
                    <div class="mmd-progress-bar" style="width: ${task.percentage || 0}%; ${isPaused ? "background: var(--mmd-warn);" : ""}"></div>
                </div>
                <div style="font-size: 11px; color: var(--mmd-text-muted);">${escapeHtml(statsText)}</div>
            `;

            if (isDownloading) {
                card.querySelector(".mmd-pause-btn").onclick = async () => {
                    await api.fetchApi("/model_downloader/pause_download", {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ task_id: task.id })
                    });
                };
            }
            if (isPaused) {
                card.querySelector(".mmd-resume-btn").onclick = async () => {
                    await api.fetchApi("/model_downloader/resume_download", {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ task_id: task.id })
                    });
                };
            }
            if (isDownloading || isQueued || isPaused || isVerifying) {
                card.querySelector(".mmd-cancel-btn").onclick = async () => {
                    await api.fetchApi("/model_downloader/cancel_download", {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ task_id: task.id })
                    });
                    this.showToast(`Cancelled: ${task.filename}`);
                };
            }
            if (isFailed) {
                card.querySelector(".mmd-retry-btn").onclick = () => {
                    this.startDownload(task.url, task.filename, task.folder_type, "", true, task.expected_sha256);
                };
            }

            container.appendChild(card);
        };

        activeTasks.slice().reverse().forEach(task => renderCard(task, list));
        historyTasks.slice().reverse().forEach(task => renderCard(task, historyList));
    }

    showBanner(msg) {
        const existing = document.querySelector(".mmd-banner");
        if (existing) existing.remove();

        const banner = document.createElement("div");
        banner.className = "mmd-banner";
        banner.innerHTML = `
            <span>${msg}</span>
            <button class="mmd-btn mmd-btn-primary" style="padding: 3px 8px; font-size: 10px;" id="mmd-banner-btn">Resolve</button>
            <button style="background: none; border: none; color: var(--mmd-text-muted); cursor: pointer; font-size: 14px;" id="mmd-banner-close">&times;</button>
        `;

        banner.querySelector("#mmd-banner-btn").onclick = () => { banner.remove(); this.openModal(); };
        banner.querySelector("#mmd-banner-close").onclick = () => banner.remove();

        document.body.appendChild(banner);
        setTimeout(() => { if (banner.parentNode) banner.remove(); }, 6000);
    }

    showToast(msg, isError = false) {
        const toast = document.createElement("div");
        toast.style.cssText = `
            position: fixed; bottom: 24px; left: 24px; z-index: 100001;
            padding: 8px 14px; border-radius: 4px;
            background: ${isError ? "rgba(205, 92, 92, 0.9)" : "rgba(50, 50, 52, 0.95)"};
            color: ${isError ? "#fff" : "var(--mmd-text-main)"};
            font-family: var(--mmd-font); font-size: 12px; font-weight: 500;
            border: 1px solid ${isError ? "rgba(205, 92, 92, 0.3)" : "rgba(255,255,255,0.08)"};
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        `;
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transition = "opacity 0.2s ease";
            setTimeout(() => toast.remove(), 200);
        }, 3000);
    }
}

let mmdUIInstance = null;

app.registerExtension({
    name: "ComfyUI.MissingModelDownloader",
    commands: [
        {
            id: "ComfyUI.ModelDownloader.Open",
            label: "Open Model Downloader",
            icon: "pi pi-download",
            function: () => {
                if (mmdUIInstance) mmdUIInstance.openModal();
            }
        }
    ],
    keybindings: [
        {
            commandId: "ComfyUI.ModelDownloader.Open",
            combo: { ctrl: true, shift: true, key: "m" }
        }
    ],
    async setup() {
        mmdUIInstance = new MissingModelDownloaderUI();
        window.mmdUI = mmdUIInstance;
        await mmdUIInstance.init();
    },
    nodeCreated(node) {
        if (node && (node.comfyClass === "ModelDownloaderNode" || node.type === "ModelDownloaderNode")) {
            node.addWidget("button", "Open Model Downloader", null, () => {
                if (mmdUIInstance) mmdUIInstance.openModal();
            });
        }
    }
});
