# 📦 ComfyUI-ModelDownloader_SG

A ComfyUI custom node extension that automatically detects missing models in your workflows and lets you download them directly from **Hugging Face** (priority) and **Civitai** — with API token support for gated/private models.

## ✨ Features

- **Automatic Missing Model Detection** — Scans your workflow graph on load and identifies models not found in your ComfyUI `models/` directories
- **Hugging Face Integration (Priority)** — Search and download models directly from Hugging Face, including gated models (FLUX, SD3, etc.) with token authentication
- **Civitai Integration** — Search and download from Civitai as a fallback
- **Resumable Downloads** — Background threaded downloads with HTTP range request support for resume capability
- **Real-time Progress** — Live download speed, progress bar, percentage, and ETA via WebSocket
- **Direct URL Downloads** — Paste any Hugging Face or Civitai URL to download directly
- **Auto-Refresh** — Clears ComfyUI's model cache after download so models appear immediately without restart
- **Modern UI** — Sleek monochrome modal with tabs for missing models, downloads, direct URL, and settings

## 📸 How It Works

Once installed, the addon adds:

1. **A "Missing Models" button in the top menu bar** (ComfyUI V2 topbar, if available)
2. **Keyboard shortcut `Ctrl+Shift+M`** to toggle the modal

### The modal has 4 tabs:
| Tab | Description |
|-----|-------------|
| **Missing Models** | Shows models referenced in your workflow but not found locally. Click "Search Model" to find them on HF/Civitai |
| **Active Downloads** | Real-time progress of ongoing downloads with speed, ETA, and cancel support |
| **Direct Download** | Paste a Hugging Face or Civitai URL to download any model directly |
| **Settings** | Configure HF/Civitai API tokens, preferred provider, and auto-scan behavior |

## 🔧 Installation

### Method 1: Manual Installation

```bash
cd <your-comfyui-path>/custom_nodes/
git clone https://github.com/BoredYama/ComfyUI-ModelDownloader_SG.git
cd ComfyUI-ModelDownloader_SG
pip install -r requirements.txt
```

### Method 2: Copy Files

Copy the `ComfyUI-ModelDownloader_SG` folder into your ComfyUI `custom_nodes/` directory:

```
<ComfyUI>/
  custom_nodes/
    ComfyUI-ModelDownloader_SG/
      __init__.py
      config_manager.py
      detector.py
      hf_client.py
      civitai_client.py
      downloader.py
      pyproject.toml
      requirements.txt
      web/
        js/
          missing_model_downloader.js
          style.css
```

Then restart ComfyUI.

## ⚙️ Configuration

### API Tokens

Open the Model Downloader modal (click the "Missing Models" button in the top menu or press `Ctrl+Shift+M`), go to the **Settings** tab:

1. **Hugging Face Token** — Required for gated models like FLUX.1, SD3, etc. Get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Civitai API Key** — Enables downloading authenticated models. Get yours at [civitai.com/user/account](https://civitai.com/user/account)

Tokens are stored locally in `config.json` within the addon directory and are never transmitted anywhere except to the respective API endpoints.

## 🖥️ Usage

1. **Load a workflow** that references models you don't have locally
2. The addon automatically scans and shows a badge count on the top menu button
3. Click the button (or `Ctrl+Shift+M`) to open the modal
4. Click **"🔎 Search Model"** next to any missing model
5. Browse results from Hugging Face and Civitai
6. Click **"⬇️ Download"** to start a background download
7. Monitor progress in the **Active Downloads** tab
8. The model is automatically available in ComfyUI after download — no restart needed!

## 🧩 Custom Node

The addon also registers a passthrough node called **"📦 Model Downloader (SG)"** in the `model_downloader` category. You can add it to any workflow as a convenient button to open the downloader dialog.

## 📋 Requirements

- ComfyUI (any recent version)
- Python 3.9+
- `aiohttp >= 3.8.0`
- `requests >= 2.28.0`

## ✨ v1.2.1 Changes

- **Hugging Face downloads fixed.** They failed every time with huggingface_hub 1.x. HF files now download over plain HTTP like everything else, so pause, resume, cancel and auto-retry work for them too. `huggingface-hub` and `tqdm` are no longer required.
- **HF downloads are hash-checked.** The SHA256 (and exact size) for Hugging Face files was always coming back empty, so they were never verified. Search results and pasted HF links now carry it.
- **Hugging Face search fixed.** Repos found by the main search were silently skipped (HF's API changed what it returns), so many files were never found. Official publishers (Comfy-Org, Kijai, Lightricks…) now rank above re-uploads, and names with a `comfy_` prefix also match the file without it (e.g. `comfy_gemma_3_12B_it` → `Comfy-Org/ltx-2`'s `gemma_3_12B_it`).
- **Clear message for gated repos.** HTTP 401/403 from Hugging Face now explains how to get access.

## ✨ v1.2.0 Changes

- **Downloads straight from workflow URLs.** Official ComfyUI templates record each model's download link and folder. Those models now show "URL from workflow" and download with one click, without a search.
- **Any model folder.** Every folder ComfyUI has registered (including ones added by other custom nodes) can be picked, plus a custom folder under `models/`.
- **Subfolders kept.** A workflow that references `flux/model.safetensors` gets the file saved into `<folder>/flux/`.
- **Update the node.** After downloading a file with a different name than the workflow expects, a banner offers to point the node at it.
- **Pausing Hugging Face downloads keeps progress.** Resume continues where it left off, including after a ComfyUI restart.
- **Civitai downloads are verified** against the SHA256 that Civitai publishes. Direct Civitai links now get the real filename.
- **Auto-retry.** Civitai and direct-URL downloads retry dropped connections up to 3 times and resume from where they stopped.
- **Disk-space check.** Downloads that won't fit (plus 1 GB to spare) fail straight away with a clear message.
- **Exact matches stay visible** after other downloads finish, and show the file size.
- The download endpoint no longer accepts an arbitrary target directory from the browser.

## 🔒 v1.1.0 Changes

Security and reliability pass:

- Fixed a token-leak bug where a look-alike domain (e.g. `huggingface.co.evil.com`) could receive your HF/Civitai bearer token due to a substring host check.
- Fixed a path-traversal bug in the download endpoint — filenames are now sanitized and the resolved path is verified to stay inside the target folder.
- Blocked non-`http(s)` URL schemes (previously a `file://` URL could be used to read local files off disk).
- Civitai API tokens are no longer embedded in download URLs (sent only via the `Authorization` header), and any token accidentally left in a URL is redacted before it's ever sent back to the browser or over the WebSocket.
- `max_concurrent_downloads` (already in Settings) is now actually enforced — extra downloads queue instead of all firing at once.
- Downloads to a filename that already exists now prompt for confirmation instead of silently overwriting.
- Cancelling a download now cleans up its partial `.downloading` file; failed downloads got a **Retry** button.
- Escaped all workflow/API-derived text before rendering it in the UI (was vulnerable to injected HTML via a crafted workflow file or search result).
- Task history is capped so long sessions don't grow the download list forever.
- Civitai search results now show a thumbnail image.

## 📄 License

MIT License
