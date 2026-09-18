"""
ComfyUI Missing Model Detector & Downloader Addon
Detects missing models in workflows and downloads them from Hugging Face (priority) or Civitai using API tokens.
"""

import os
import sys

# Ensure local custom node directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config_manager import config_manager
from detector import detector
from hf_client import hf_client
from civitai_client import civitai_client
from downloader import download_manager

# Register Web Directory for ComfyUI Frontend extensions
# Points to the directory containing .js files that ComfyUI will auto-load
WEB_DIRECTORY = "./web/js"

class ModelDownloaderNode:
    """
    Missing Model Downloader Helper Node.
    Can be placed anywhere in a workflow to monitor missing models and open the downloader dialog.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*", {}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "passthrough"
    CATEGORY = "model_downloader"

    def passthrough(self, any_input=None):
        return (any_input,)

NODE_CLASS_MAPPINGS = {
    "ModelDownloaderNode": ModelDownloaderNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDownloaderNode": "📦 Model Downloader (SG)"
}

# Required for ComfyUI to discover WEB_DIRECTORY
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Register API routes with ComfyUI PromptServer if running inside ComfyUI
try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.get("/model_downloader/config")
    async def get_config(request):
        """Returns sanitized configuration (tokens masked)."""
        return web.json_response(config_manager.get_sanitized())

    @routes.post("/model_downloader/config")
    async def update_config(request):
        """Updates configuration and tokens."""
        try:
            data = await request.json()
            updated = config_manager.update(data)
            return web.json_response({"status": "success", "config": updated})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    @routes.post("/model_downloader/test_token")
    async def test_token(request):
        """Tests validity of Hugging Face or Civitai API tokens."""
        try:
            data = await request.json()
            provider = data.get("provider", "huggingface")
            token = data.get("token") or (
                config_manager.get_hf_token() if provider == "huggingface" else config_manager.get_civitai_token()
            )

            if provider == "huggingface":
                result = config_manager.test_hf_token(token)
            else:
                result = config_manager.test_civitai_token(token)

            return web.json_response(result)
        except Exception as e:
            return web.json_response({"valid": False, "error": str(e)}, status=400)

    @routes.post("/model_downloader/detect")
    async def detect_models(request):
        """Scans workflow JSON or candidates and returns missing models."""
        try:
            data = await request.json()
            missing = detector.detect_missing_models(data)
            return web.json_response({
                "status": "success",
                "missing_models": missing,
                "count": len(missing)
            })
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    @routes.post("/model_downloader/search")
    async def search_models(request):
        """
        Searches Hugging Face (priority) and Civitai for missing model files.
        """
        try:
            data = await request.json()
            query = data.get("query", "").strip()
            provider = data.get("provider", "all")  # "all", "huggingface", or "civitai"
            limit = int(data.get("limit", 10))

            if not query:
                return web.json_response({"results": []})

            results = []

            # Hugging Face Search (Priority)
            if provider in ("all", "huggingface"):
                hf_results = hf_client.search_for_model(query, limit=limit)
                results.extend(hf_results)

            # Civitai Search
            if provider in ("all", "civitai"):
                civitai_results = civitai_client.search_for_model(query, limit=limit)
                results.extend(civitai_results)

            # Ensure exact matches and Hugging Face results are appropriately prioritized
            results.sort(key=lambda x: (
                x.get("exact_match", False),
                1 if x.get("source") == "huggingface" else 0,
                x.get("score", 0)
            ), reverse=True)

            return web.json_response({"results": results})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    @routes.post("/model_downloader/parse_url")
    async def parse_url(request):
        """Parses a direct Hugging Face or Civitai URL into download information."""
        try:
            data = await request.json()
            url = data.get("url", "").strip()
            if not url:
                return web.json_response({"valid": False, "error": "URL is empty"}, status=400)

            if "huggingface.co" in url:
                parsed = hf_client.parse_direct_url(url)
            elif "civitai.com" in url:
                parsed = civitai_client.parse_direct_url(url)
            else:
                # Generic direct URL
                filename = url.split("?")[0].split("/")[-1] or "model.safetensors"
                parsed = {
                    "source": "direct",
                    "valid": True,
                    "filename": filename,
                    "download_url": url
                }

            return web.json_response(parsed)
        except Exception as e:
            return web.json_response({"valid": False, "error": str(e)}, status=400)

    @routes.post("/model_downloader/start_download")
    async def start_download(request):
        """Starts a background download task."""
        try:
            data = await request.json()
            url = data.get("url", "").strip()
            filename = data.get("filename", "").strip()
            folder_type = data.get("folder_type", "checkpoints").strip()
            target_dir = data.get("target_dir", "").strip()
            overwrite = bool(data.get("overwrite", False))

            if not url or not filename:
                return web.json_response({"status": "error", "message": "Missing url or filename"}, status=400)

            if not target_dir:
                target_dir = detector.get_target_directory(folder_type)

            task_id = download_manager.start_download(url, filename, target_dir, folder_type, overwrite=overwrite)
            return web.json_response({
                "status": "success",
                "task_id": task_id,
                "filename": filename,
                "folder_type": folder_type,
                "target_dir": target_dir
            })
        except FileExistsError as e:
            return web.json_response({"status": "exists", "message": str(e)}, status=409)
        except ValueError as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post("/model_downloader/cancel_download")
    async def cancel_download(request):
        """Cancels an ongoing download."""
        try:
            data = await request.json()
            task_id = data.get("task_id", "").strip()
            success = download_manager.cancel_task(task_id)
            return web.json_response({"status": "success" if success else "not_found"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    @routes.get("/model_downloader/downloads")
    async def get_downloads(request):
        """Returns list of active and recent download tasks."""
        return web.json_response({"downloads": download_manager.get_all_tasks()})

    @routes.get("/model_downloader/folders")
    async def get_folders(request):
        """Returns available ComfyUI folders."""
        return web.json_response({"folders": detector.get_registered_folders()})

    @routes.post("/model_downloader/refresh_cache")
    async def refresh_cache(request):
        """Clears ComfyUI model filename cache."""
        try:
            import folder_paths
            if hasattr(folder_paths, "filename_list_cache"):
                folder_paths.filename_list_cache.clear()
            return web.json_response({"status": "success"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    print("[ComfyUI-ModelDownloader_SG] ✅ API routes registered successfully.")
    print("[ComfyUI-ModelDownloader_SG] 📦 Floating button & Ctrl+Shift+M shortcut available in the UI.")

except ImportError:
    print("[ComfyUI-ModelDownloader_SG] Running outside ComfyUI server environment.")
