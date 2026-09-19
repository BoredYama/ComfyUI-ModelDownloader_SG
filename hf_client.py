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
                etag = resp.headers.get("X-Linked-Etag", "").strip('"')
                sha256 = ""
                if etag and len(etag) == 64:
                    sha256 = etag
                return {
                    "accessible": True,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
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
        # Clean up known user-added suffixes like _KJ which aren't in the actual repo filenames
        target_filename_lower = re.sub(r'(?i)[-_]kj(\.[a-z0-9]+)$', r'\1', raw_query.lower())
        
        # Strip extension from target for matching
        target_stem = target_filename_lower
        for ext in MODEL_EXTENSIONS:
            if target_stem.endswith(ext):
                target_stem = target_stem[:-len(ext)]
                break
        
        # Build keyword set from the target filename for fuzzy matching
        target_keywords = set(re.split(r'[-_.\s]+', target_stem))
        target_keywords = {k for k in target_keywords if len(k) >= 2}

        for sib in siblings:
            rfilename = sib.get("rfilename", "")
            if not rfilename:
                continue

            # Filter for model extensions
            if not any(rfilename.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                continue

            base_sibling_name = rfilename.split("/")[-1]
            base_sibling_lower = base_sibling_name.lower()
            
            # Strip extension from sibling for matching
            sibling_stem = base_sibling_lower
            for ext in MODEL_EXTENSIONS:
                if sibling_stem.endswith(ext):
                    sibling_stem = sibling_stem[:-len(ext)]
                    break

            # Check match relevance
            is_exact_match = (base_sibling_lower == target_filename_lower)
            
            # Substring match (on stems, not full filenames with extensions)
            is_partial_match = (
                clean_query.lower() in sibling_stem or
                sibling_stem in target_stem or
                target_stem in sibling_stem
            )
            
            # Keyword-based fuzzy match: check how many keywords overlap
            if not is_exact_match and not is_partial_match and target_keywords:
                sibling_keywords = set(re.split(r'[-_.\s]+', sibling_stem))
                sibling_keywords = {k for k in sibling_keywords if len(k) >= 2}
                if sibling_keywords and target_keywords:
                    overlap = target_keywords & sibling_keywords
                    # Require at least 60% keyword overlap AND minimum 3 matching keywords
                    overlap_ratio = len(overlap) / min(len(target_keywords), len(sibling_keywords))
                    if overlap_ratio >= 0.6 and len(overlap) >= 3:
                        is_partial_match = True

            if is_exact_match or is_partial_match:
                resolve_url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(rfilename)}"
                
                # Deduplicate
                if any(r["download_url"] == resolve_url for r in results):
                    continue

                # Calculate match score (higher is better)
                score = 0
                if is_exact_match:
                    score += 100
                if clean_query.lower() in sibling_stem:
                    score += 20
                elif sibling_stem in target_stem or target_stem in sibling_stem:
                    score += 15
                else:
                    # Fuzzy match - score based on overlap
                    sibling_keywords = set(re.split(r'[-_.\s]+', sibling_stem))
                    overlap = target_keywords & sibling_keywords
                    score += min(len(overlap) * 3, 12)
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
                    "sha256": "",
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
        clean_query = re.sub(r'(?i)[-_. ]?(?:q[1-8]_[a-z0-9_]+|q[1-8]_[0-9]|fp16|fp32|bf16|int8)$', '', clean_query)
        
        is_gguf = raw_query.lower().endswith(".gguf")
        repo_search_query = clean_query + (" gguf" if is_gguf and 'gguf' not in clean_query.lower() else "")

        # Also remove common suffixes like _fp8, _fp16, -pruned for search flexibility
        keywords = re.split(r'[-_.\s]+', clean_query)
        
        # Add common ecosystem synonyms to keywords to boost scoring and fallback search
        SYNONYMS = {
            "umt5": "wan",
            "ltx23": "ltx",
            "ltx25": "ltx",
            "ltx_video": "ltx",
            "clip_g": "sdxl",
            "clip_l": "sdxl",
            "t5xxl": "flux",
            "wan2": "wan",
            "gemma": "ltx",
            "svi": "wanvideo"
        }
        
        fallback_kw = keywords[0] if keywords else clean_query
        for k in list(keywords):
            for syn_k, syn_v in SYNONYMS.items():
                if syn_k in k.lower():
                    if syn_v not in [x.lower() for x in keywords]:
                        keywords.append(syn_v)
                    if fallback_kw.lower() == k.lower():
                        fallback_kw = syn_v

        primary_keyword = keywords[0] if keywords else clean_query
        
        # Build a version-expanded search query for cases like LTX25 -> LTX-2.5, wan22 -> wan-2.2
        # This helps find repos named "LTX-2.5-Quantized" when the filename says "LTX25"
        def expand_versions(text):
            """Split concatenated name+version like LTX25 into LTX 2.5, wan22 into wan 2.2"""
            # Pattern: letters followed by digits, where digits look like a version (2-4 chars)
            expanded = re.sub(r'([a-zA-Z]+)(\d)(\d)(?=[^0-9]|$)', 
                            lambda m: f"{m.group(1)}-{m.group(2)}.{m.group(3)}", text)
            return expanded if expanded != text else None
        
        version_expanded_query = expand_versions(clean_query)

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
                kw_url = f"{HF_API_BASE}/models?search={urllib.parse.quote(kw_query)}&limit=50&full=false"
                req = urllib.request.Request(kw_url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    more_repos = json.loads(resp.read().decode("utf-8"))
                    existing_ids = {r.get("id") for r in matched_repos}
                    for mr in more_repos:
                        if mr.get("id") not in existing_ids:
                            matched_repos.append(mr)
            except Exception:
                pass

        # 1b. If still few results, try version-expanded and spaced-keyword searches WITH full=true
        # This finds repos where the file is inside but the repo name doesn't match the filename
        if len(matched_repos) < 5:
            extra_queries = []
            
            # Extract meaningful keywords (skip noise like fp16, mix, w4a8, 17GB, etc.)
            meaningful_kws = [k for k in keywords if len(k) >= 3 and not re.match(
                r'^(?:fp\d+|bf\d+|int\d+|q\d|mix\d*|w\d+a?\d*|x\d+|\d+GB?|scaled|pruned|ema|only)$', k, re.I)]
            
            # Version-expanded first keyword: LTX25 -> LTX-2.5
            expanded_first = expand_versions(keywords[0]) if keywords else None
            if expanded_first:
                expanded_first = re.sub(r'[-_]', ' ', expanded_first)
            
            # Build candidate queries (order matters - best first)
            if expanded_first:
                # "LTX 2.5 comfy" - version expanded + comfy keyword if present
                if 'comfy' in [k.lower() for k in meaningful_kws]:
                    extra_queries.append(f"{expanded_first} comfy")
                # "LTX 2.5" alone
                extra_queries.append(expanded_first)
                # "LTX 2.5 quantized/distilled" - version expanded + second meaningful keyword
                for mk in meaningful_kws[1:3]:
                    if mk.lower() != 'comfy':
                        extra_queries.append(f"{expanded_first} {mk}")
            
            # Also try original first keyword + comfy (e.g. "gemma4 comfy ltx25")
            if len(meaningful_kws) >= 2:
                extra_queries.append(' '.join(meaningful_kws[:3]))
            
            # Deduplicate while preserving order
            seen = set()
            unique_queries = []
            for q in extra_queries:
                ql = q.lower()
                if ql not in seen:
                    seen.add(ql)
                    unique_queries.append(q)
            
            existing_ids = {r.get("id") for r in matched_repos}
            for eq in unique_queries[:3]:  # Limit to 3 extra searches to avoid slowdowns
                try:
                    eq_url = f"{HF_API_BASE}/models?search={urllib.parse.quote(eq)}&limit=10&full=true"
                    req = urllib.request.Request(eq_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        extra_repos = json.loads(resp.read().decode("utf-8"))
                        for er in extra_repos:
                            if er.get("id") not in existing_ids:
                                existing_ids.add(er.get("id"))
                                matched_repos.append(er)
                                # These repos came with full=true so they have siblings already
                                # Parse them immediately for file matches
                                self._parse_repo_detail(er, raw_query, clean_query, results)
                except Exception:
                    pass

        # Sort matched_repos to prioritize ones matching more keywords from the filename
        if matched_repos and len(matched_repos) > 1:
            def score_repo(r):
                rid = (r.get("id") or "").lower()
                return sum(1 for k in keywords if len(k) > 2 and k.lower() in rid)
            matched_repos.sort(key=score_repo, reverse=True)


        # 2. For the top candidate repositories, fetch repo details to inspect siblings (files)
        for repo_info in matched_repos[:8]:
            repo_id = repo_info.get("id")
            if not repo_id:
                continue
            # Skip repos already parsed with full=true (they have siblings data)
            if repo_info.get("siblings"):
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

        # 3. Fallback for ComfyUI orgs if no exact match is found
        if not any(r["exact_match"] for r in results):
            KNOWN_ORGS = [
                "Comfy-Org", "Kijai", "city96", "lllyasviel", 
                "black-forest-labs", "stabilityai", "mcmonkey", 
                "RunDiffusion", "cocktailpeanut", "ByteDance",
                "lightx2v", "bartowski", "mradermacher", "Lightricks", "joeygambino", "LootingGod"
            ]
            
            # Better fallback keyword that doesn't split version numbers (e.g. wan2.2)
            fallback_kw = re.split(r'[-_\s]+', clean_query)[0] if clean_query else ""
            
            def fetch_org(org):
                org_results = []
                url1 = f"{HF_API_BASE}/models?author={org}&search={urllib.parse.quote(fallback_kw)}&limit=10&full=true"
                url2 = f"{HF_API_BASE}/models?author={org}&sort=downloads&limit=10&full=true"
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

        # 4. Concurrently fetch exact file sizes for Hugging Face results
        def fetch_size(r):
            if r.get("source") == "huggingface" and r.get("size_bytes", 0) == 0:
                try:
                    info = self.get_file_info(r["repo_id"], r["relative_path"])
                    r["size_bytes"] = info.get("size_bytes", 0)
                    r["sha256"] = info.get("sha256", "")
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(fetch_size, results))

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
                "download_url": download_url,
                "sha256": ""
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
