import json
import re
import urllib.request
import urllib.parse
import urllib.error
import concurrent.futures
from config_manager import config_manager

HF_API_BASE = "https://huggingface.co/api"
MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth", ".onnx")

class HuggingFaceClient:
    def __init__(self):
        pass

    def _get_headers(self, token=None):
        if not token:
            token = config_manager.get_hf_token()
        headers = {
            "User-Agent": "ComfyUI-MissingModelDownloader"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_file_info(self, repo_id: str, filename: str, revision: str = "main", token: str = None) -> dict:
        """Sends a HEAD request to get exact file size and accessibility."""
        resolve_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
        headers = self._get_headers(token)
        req = urllib.request.Request(resolve_url, headers=headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                size_str = resp.headers.get("Content-Length")
                size_bytes = int(size_str) if size_str and size_str.isdigit() else 0
                return {
                    "accessible": True,
                    "size_bytes": size_bytes,
                    "url": resolve_url,
                    "status_code": resp.status
                }
        except urllib.error.HTTPError as e:
            return {
                "accessible": False,
                "size_bytes": 0,
                "url": resolve_url,
                "status_code": e.code,
                "error": f"HTTP {e.code}: {e.reason}"
            }
        except Exception as e:
            return {
                "accessible": False,
                "size_bytes": 0,
                "url": resolve_url,
                "error": str(e)
            }

    def _parse_repo_detail(self, repo_detail, raw_query, clean_query, results):
        repo_id = repo_detail.get("id")
        if not repo_id:
            return
            
        siblings = repo_detail.get("siblings", [])
        is_gated = repo_detail.get("gated", False)
        downloads = repo_detail.get("downloads", 0)
        likes = repo_detail.get("likes", 0)
        target_filename_lower = raw_query.lower()

        for sib in siblings:
            rfilename = sib.get("rfilename", "")
            if not rfilename:
                continue

            # Filter for model extensions
            if not any(rfilename.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                continue

            base_sibling_name = rfilename.split("/")[-1]
            base_sibling_lower = base_sibling_name.lower()

            # Check match relevance
            is_exact_match = (base_sibling_lower == target_filename_lower)
            is_partial_match = (
                clean_query.lower() in base_sibling_lower or 
                base_sibling_lower in target_filename_lower
            )

            if is_exact_match or is_partial_match:
                resolve_url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(rfilename)}"
                
                # Deduplicate
                if any(r["download_url"] == resolve_url for r in results):
                    continue

                # Calculate match score (higher is better)
                score = 0
                if is_exact_match:
                    score += 100
                if clean_query.lower() in base_sibling_lower:
                    score += 20
                if downloads:
                    score += min(downloads // 1000, 30)
                if likes:
                    score += min(likes, 20)

                results.append({
                    "source": "huggingface",
                    "name": base_sibling_name,
                    "relative_path": rfilename,
                    "repo_id": repo_id,
                    "download_url": resolve_url,
                    "is_gated": bool(is_gated),
                    "likes": likes,
                    "downloads": downloads,
                    "size_bytes": 0,
                    "score": score,
                    "exact_match": is_exact_match
                })

    def search_for_model(self, search_query: str, limit: int = 15) -> list:
        """
        Searches Hugging Face for matching models and file siblings.
        search_query can be a filename (e.g., 'v1-5-pruned-emaonly.safetensors') or model name.
        """
        raw_query = search_query.strip()
        if not raw_query:
            return []

        # Strip common model extensions for the repo search query to broaden matches
        clean_query = raw_query
        for ext in MODEL_EXTENSIONS:
            if clean_query.lower().endswith(ext):
                clean_query = clean_query[:-len(ext)]
                break
                
        # Remove quantization and precision suffixes (like -Q4_K_M, -Q8_0, -FP16, etc.) 
        # which ruin repo search since repos usually just contain the base model name
        clean_query = re.sub(r'(?i)[-_]?(?:q[1-8]_[a-z0-9_]+|q[1-8]_[0-9]|fp16|fp32|bf16|int8)$', '', clean_query)
        
        is_gguf = raw_query.lower().endswith(".gguf")
        repo_search_query = clean_query + (" gguf" if is_gguf and 'gguf' not in clean_query.lower() else "")

        # Also remove common suffixes like _fp8, _fp16, -pruned for search flexibility
        keywords = re.split(r'[-_.\s]+', clean_query)
        primary_keyword = keywords[0] if keywords else clean_query

        results = []
        token = config_manager.get_hf_token()
        headers = self._get_headers(token)

        # 1. First, search Hugging Face models API
        encoded_query = urllib.parse.quote(repo_search_query)
        api_url = f"{HF_API_BASE}/models?search={encoded_query}&limit={limit}&full=false"

        matched_repos = []
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                matched_repos = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[ModelDownloader] HF search error for '{clean_query}': {e}")

        # If repo search with full name had few results, try primary keyword
        if len(matched_repos) < 3 and primary_keyword != repo_search_query and len(primary_keyword) >= 3:
            try:
                kw_query = primary_keyword + (" gguf" if is_gguf else "")
                kw_url = f"{HF_API_BASE}/models?search={urllib.parse.quote(kw_query)}&limit=10&full=false"
                req = urllib.request.Request(kw_url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    more_repos = json.loads(resp.read().decode("utf-8"))
                    existing_ids = {r.get("id") for r in matched_repos}
                    for mr in more_repos:
                        if mr.get("id") not in existing_ids:
                            matched_repos.append(mr)
            except Exception:
                pass

        # 2. For the top candidate repositories, fetch repo details to inspect siblings (files)
        for repo_info in matched_repos[:8]:
            repo_id = repo_info.get("id")
            if not repo_id:
                continue
            try:
                detail_url = f"{HF_API_BASE}/models/{repo_id}"
                req = urllib.request.Request(detail_url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    repo_detail = json.loads(resp.read().decode("utf-8"))
                self._parse_repo_detail(repo_detail, raw_query, clean_query, results)
            except Exception as e:
                if "401" in str(e) or "403" in str(e):
                    results.append({
                        "source": "huggingface",
                        "name": raw_query,
                        "relative_path": raw_query,
                        "repo_id": repo_id,
                        "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(raw_query)}",
                        "is_gated": True,
                        "likes": repo_info.get("likes", 0),
                        "downloads": repo_info.get("downloads", 0),
                        "size_bytes": 0,
                        "score": 50,
                        "exact_match": False
                    })

        # 3. Fallback for ComfyUI orgs if no exact match is found and few results
        if not any(r["exact_match"] for r in results) and len(results) < 8:
            KNOWN_ORGS = [
                "Comfy-Org", "Kijai", "city96", "lllyasviel", 
                "black-forest-labs", "stabilityai", "mcmonkey", 
                "RunDiffusion", "cocktailpeanut", "ByteDance",
                "lightx2v"
            ]
            
            def fetch_org(org):
                url1 = f"{HF_API_BASE}/models?author={org}&search={urllib.parse.quote(primary_keyword)}&limit=3&full=true"
                url2 = f"{HF_API_BASE}/models?author={org}&sort=downloads&limit=5&full=true"
                repos = []
                for url in [url1, url2]:
                    try:
                        req = urllib.request.Request(url, headers=headers)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            repos.extend(json.loads(resp.read().decode("utf-8")))
                    except Exception:
                        pass
                return repos
                    
            org_repos = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(fetch_org, org) for org in KNOWN_ORGS]
                for future in concurrent.futures.as_completed(futures):
                    org_repos.extend(future.result())
                    
            for repo_detail in org_repos:
                self._parse_repo_detail(repo_detail, raw_query, clean_query, results)

        # Sort results: exact matches first, then highest score
        results.sort(key=lambda x: (x.get("exact_match", False), x.get("score", 0)), reverse=True)
        return results

    def parse_direct_url(self, url: str) -> dict:
        """
        Parses a direct Hugging Face URL.
        Formats:
        - https://huggingface.co/user/repo/blob/main/path/to/model.safetensors
        - https://huggingface.co/user/repo/resolve/main/path/to/model.safetensors
        - https://huggingface.co/user/repo
        """
        url = url.strip()
        # Pattern 1: Direct file link
        match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)", url)
        if match:
            repo_id = match.group(1)
            revision = match.group(2)
            file_path = match.group(3)
            filename = file_path.split("/")[-1]
            download_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{urllib.parse.quote(file_path)}"
            return {
                "source": "huggingface",
                "valid": True,
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "file_path": file_path,
                "download_url": download_url
            }

        # Pattern 2: Model repo link without file
        match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/?$", url)
        if match:
            repo_id = match.group(1)
            return {
                "source": "huggingface",
                "valid": True,
                "repo_id": repo_id,
                "is_repo_only": True
            }

        return {"valid": False, "error": "Not a recognized Hugging Face URL"}

hf_client = HuggingFaceClient()
