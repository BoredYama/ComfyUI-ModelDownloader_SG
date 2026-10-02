import json
import re
import urllib.request
import urllib.parse
import urllib.error
from config_manager import config_manager

CIVITAI_API_BASE = "https://civitai.com/api/v1"
MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth", ".onnx")

def file_sha256(file_info) -> str:
    """Civitai lists per-file hashes; SHA256 lets the downloader verify the result."""
    if not file_info:
        return ""
    return ((file_info.get("hashes") or {}).get("SHA256") or "").lower()


def primary_file(files: list):
    """The version's main model file (Civitai flags it), else the first listed file."""
    for f in files or []:
        if f.get("primary"):
            return f
    return files[0] if files else None


class CivitaiClient:
    def __init__(self):
        pass

    def _get_headers(self, token=None):
        if not token:
            token = config_manager.get_civitai_token()
        headers = {
            "User-Agent": "ComfyUI-MissingModelDownloader"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def search_for_model(self, search_query: str, limit: int = 10) -> list:
        """
        Searches Civitai API for models and matching file versions.
        """
        raw_query = search_query.strip()
        if not raw_query:
            return []

        # Strip common extensions for search
        clean_query = raw_query
        for ext in MODEL_EXTENSIONS:
            if clean_query.lower().endswith(ext):
                clean_query = clean_query[:-len(ext)]
                break

        # Remove quantization and precision suffixes (like -Q4_K_M, -Q8_0, -FP16, etc.) 
        clean_query = re.sub(r'(?i)[-_]?(?:q[1-8]_[a-z0-9_]+|q[1-8]_[0-9]|fp16|fp32|bf16|int8)$', '', clean_query)
        
        is_gguf = raw_query.lower().endswith(".gguf")
        repo_search_query = clean_query + (" gguf" if is_gguf and 'gguf' not in clean_query.lower() else "")

        token = config_manager.get_civitai_token()
        headers = self._get_headers(token)
        encoded_query = urllib.parse.quote(repo_search_query)
        api_url = f"{CIVITAI_API_BASE}/models?query={encoded_query}&limit={limit}"
        if token:
            api_url += f"&token={token}"

        results = []
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
        except Exception as e:
            print(f"[ModelDownloader] Civitai search error for '{clean_query}': {e}")
            return []

        target_lower = raw_query.lower()

        for item in items:
            model_id = item.get("id")
            model_name = item.get("name", "Unknown Model")
            model_type = item.get("type", "")
            creator = item.get("creator", {}).get("username", "Unknown")
            stats = item.get("stats", {})
            downloads = stats.get("downloadCount", 0)
            favorites = stats.get("favoriteCount", 0)

            versions = item.get("modelVersions", [])
            for version in versions:
                version_id = version.get("id")
                version_name = version.get("name", "Default")
                files = version.get("files", [])

                # Get preview thumbnail if available
                images = version.get("images", [])
                thumbnail = images[0].get("url") if images else None

                # No token appended here: the downloader attaches it as an Authorization
                # header at fetch time (downloader.py), so it never has to sit in a URL
                # that gets echoed back to the client, logged, or stored in task history.
                final_download_url = version.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}"

                # Check if specific files match
                matched_file = None
                for f in files:
                    fname = f.get("name", "")
                    if fname.lower() == target_lower or clean_query.lower() in fname.lower():
                        matched_file = f
                        break

                chosen_file = matched_file or (files[0] if files else None)
                display_file_name = chosen_file.get("name") if chosen_file else f"{clean_query}.safetensors"
                size_kb = chosen_file.get("sizeKB") if chosen_file else 0
                size_bytes = int(size_kb * 1024) if size_kb else 0

                is_exact_match = (display_file_name.lower() == target_lower)
                score = 0
                if is_exact_match:
                    score += 100
                if clean_query.lower() in model_name.lower():
                    score += 30
                if downloads:
                    score += min(downloads // 500, 30)

                results.append({
                    "source": "civitai",
                    "model_id": model_id,
                    "version_id": version_id,
                    "name": display_file_name,
                    "model_name": model_name,
                    "version_name": version_name,
                    "creator": creator,
                    "type": model_type,
                    "download_url": final_download_url,
                    "thumbnail": thumbnail,
                    "size_bytes": size_bytes,
                    "sha256": file_sha256(chosen_file),
                    "downloads": downloads,
                    "favorites": favorites,
                    "score": score,
                    "exact_match": is_exact_match
                })

        results.sort(key=lambda x: (x.get("exact_match", False), x.get("score", 0)), reverse=True)
        return results

    def parse_direct_url(self, url: str) -> dict:
        """
        Parses direct Civitai model or version URLs.
        Examples:
        - https://civitai.com/models/12345
        - https://civitai.com/models/12345?modelVersionId=67890
        - https://civitai.com/api/download/models/67890
        """
        url = url.strip()
        token = config_manager.get_civitai_token()

        # Note: download_url is intentionally returned WITHOUT a token query param.
        # downloader.py attaches the token as an Authorization header at fetch time,
        # so it never needs to sit in a URL that's echoed to the client or logged.

        # Direct download endpoint, or model page with a version parameter
        match_version = re.search(r"/api/download/models/(\d+)", url) or \
            re.search(r"civitai\.com/models/\d+.*?[?&]modelVersionId=(\d+)", url)
        if match_version:
            version_id = match_version.group(1)
            dl_url = f"https://civitai.com/api/download/models/{version_id}"
            result = {
                "source": "civitai",
                "valid": True,
                "version_id": int(version_id),
                "download_url": dl_url,
                "filename": f"civitai_model_{version_id}.safetensors",
                "sha256": "",
                "size_bytes": 0
            }
            # Look up the real filename and hash; fall back to the placeholder name if the API is unreachable
            try:
                api_url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
                req = urllib.request.Request(api_url, headers=self._get_headers(token))
                with urllib.request.urlopen(req, timeout=10) as resp:
                    version = json.loads(resp.read().decode("utf-8"))
                f = primary_file(version.get("files", []))
                if f:
                    result["filename"] = f.get("name") or result["filename"]
                    result["sha256"] = file_sha256(f)
                    result["size_bytes"] = int((f.get("sizeKB") or 0) * 1024)
            except Exception as e:
                print(f"[ModelDownloader] Civitai version lookup failed for {version_id}: {e}")
            return result

        # Model page base
        match_model = re.search(r"civitai\.com/models/(\d+)", url)
        if match_model:
            model_id = match_model.group(1)
            # Query API to get primary version ID (token sent via header only)
            try:
                api_url = f"{CIVITAI_API_BASE}/models/{model_id}"
                req = urllib.request.Request(api_url, headers=self._get_headers(token))
                with urllib.request.urlopen(req, timeout=10) as resp:
                    info = json.loads(resp.read().decode("utf-8"))
                    versions = info.get("modelVersions", [])
                    if versions:
                        prim_ver = versions[0]
                        v_id = prim_ver.get("id")
                        f = primary_file(prim_ver.get("files", []))
                        fname = f.get("name") if f else f"{info.get('name')}.safetensors"
                        dl_url = prim_ver.get("downloadUrl") or f"https://civitai.com/api/download/models/{v_id}"
                        return {
                            "source": "civitai",
                            "valid": True,
                            "model_id": int(model_id),
                            "version_id": v_id,
                            "filename": fname,
                            "download_url": dl_url,
                            "sha256": file_sha256(f),
                            "size_bytes": int((f.get("sizeKB") or 0) * 1024) if f else 0
                        }
            except Exception as e:
                return {"valid": False, "error": f"Failed to retrieve Civitai model details: {e}"}

        return {"valid": False, "error": "Not a recognized Civitai URL"}

civitai_client = CivitaiClient()
