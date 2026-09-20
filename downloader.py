import os
import time
import uuid
import threading
import urllib.request
import urllib.parse
import urllib.error
from config_manager import config_manager

# Handle ComfyUI imports if running inside ComfyUI
try:
    import server
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

def redact_url_secrets(url: str) -> str:
    """Masks sensitive query params (e.g. Civitai's ?token=) before a URL is ever sent to the client."""
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.query:
            return url
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted = [(k, "***" if k.lower() == "token" else v) for k, v in params]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(redacted, safe="*")))
    except Exception:
        return url


class DownloadTask:
    def __init__(self, task_id: str, url: str, target_path: str, filename: str, folder_type: str = "", expected_sha256: str = ""):
        self.id = task_id
        self.url = url
        self.target_path = target_path
        self.temp_path = f"{target_path}.downloading"
        self.filename = filename
        self.folder_type = folder_type
        self.expected_sha256 = expected_sha256
        
        self.status = "pending"  # pending, downloading, completed, failed, cancelled
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.speed_bytes_per_sec = 0
        self.eta_seconds = 0
        self.percentage = 0.0
        self.error_message = ""
        
        self.cancel_requested = False
        self.is_paused = False
        self.start_time = 0
        self.last_update_time = 0
        self.last_downloaded_bytes = 0

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "folder_type": self.folder_type,
            "target_path": self.target_path,
            "url": redact_url_secrets(self.url),
            "status": "paused" if self.is_paused else self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "percentage": round(self.percentage, 1),
            "speed_mb": round(self.speed_bytes_per_sec / (1024 * 1024), 2),
            "eta_seconds": int(self.eta_seconds),
            "error": self.error_message
        }


def host_matches(netloc: str, domain: str) -> bool:
    """
    Exact/subdomain host match (e.g. 'huggingface.co' matches 'huggingface.co' and
    'files.huggingface.co', but NOT 'huggingface.co.evil.com' or 'nothuggingface.co').
    Guards against substring-match token leaks to look-alike hosts.
    """
    host = (netloc or "").split("@")[-1].split(":")[0].lower()
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    Custom redirect handler that strips Authorization headers when redirected
    to third-party domains (such as AWS S3 / CloudFront for Hugging Face or Cloudflare for Civitai).
    """
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req:
            orig_host = urllib.parse.urlparse(req.full_url).netloc
            new_host = urllib.parse.urlparse(newurl).netloc
            # If redirected to a different domain, strip Authorization to avoid AWS S3 signature rejection
            if orig_host.lower() != new_host.lower():
                if "Authorization" in new_req.headers:
                    del new_req.headers["Authorization"]
                if "authorization" in new_req.headers:
                    del new_req.headers["authorization"]
        return new_req


MAX_FINISHED_TASKS = 50  # cap retained completed/failed/cancelled tasks so history doesn't grow forever


class DownloadManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DownloadManager, cls).__new__(cls)
            cls._instance.tasks = {}
            cls._instance.lock = threading.Lock()
            cls._instance.queue = []  # task_ids waiting for a concurrency slot
            cls._instance._active_count = 0
            cls._instance.load_state()
        return cls._instance

    def load_state(self):
        tasks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads.json")
        if os.path.exists(tasks_file):
            try:
                import json
                with open(tasks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for task_dict in data:
                    t = DownloadTask(
                        task_id=task_dict["id"],
                        url=task_dict["url"],
                        target_path=task_dict["target_path"],
                        filename=task_dict["filename"],
                        folder_type=task_dict.get("folder_type", ""),
                        expected_sha256=task_dict.get("expected_sha256", "")
                    )
                    t.status = task_dict["status"]
                    if t.status in ("downloading", "queued", "pending"):
                        t.status = "paused"
                        t.is_paused = True
                    t.total_bytes = task_dict.get("total_bytes", 0)
                    t.downloaded_bytes = task_dict.get("downloaded_bytes", 0)
                    t.percentage = task_dict.get("percentage", 0.0)
                    t.error_message = task_dict.get("error_message", "")
                    self.tasks[t.id] = t
            except Exception as e:
                print(f"[ModelDownloader] Failed to load downloads.json: {e}")

    def save_state(self):
        tasks_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads.json")
        try:
            import json
            with self.lock:
                save_data = []
                for t in self.tasks.values():
                    save_data.append({
                        "id": t.id,
                        "url": t.url,
                        "target_path": t.target_path,
                        "filename": t.filename,
                        "folder_type": t.folder_type,
                        "expected_sha256": getattr(t, 'expected_sha256', ""),
                        "status": "paused" if getattr(t, 'is_paused', False) else t.status,
                        "total_bytes": getattr(t, 'total_bytes', 0),
                        "downloaded_bytes": getattr(t, 'downloaded_bytes', 0),
                        "percentage": getattr(t, 'percentage', 0.0),
                        "error_message": getattr(t, 'error_message', "")
                    })
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            print(f"[ModelDownloader] Failed to save downloads.json: {e}")

    def get_all_tasks(self):
        with self.lock:
            return [task.to_dict() for task in self.tasks.values()]

    def get_task(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
            return task.to_dict() if task else None

    def cancel_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            task.cancel_requested = True
            if task.status in ("downloading", "queued", "pending"):
                task.status = "cancelled"
            if task_id in self.queue:
                self.queue.remove(task_id)
            self._notify_progress(task)
        self.save_state()
        return True

    def pause_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task.status not in ("downloading", "queued", "pending"):
                return False
            task.is_paused = True
            if task_id in self.queue:
                self.queue.remove(task_id)
            self._notify_progress(task)
        self.save_state()
        return True

    def resume_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or not task.is_paused:
                return False
            task.is_paused = False
            task.status = "queued"
            self.queue.append(task_id)
            self._notify_progress(task)
        
        self.save_state()
        self._start_next_queued()
        return True

    def clear_history(self) -> int:
        cleared = 0
        with self.lock:
            active_ids = [tid for tid, t in self.tasks.items() if t.status in ("downloading", "queued", "pending") and not t.is_paused]
            to_remove = [tid for tid in self.tasks.keys() if tid not in active_ids]
            for tid in to_remove:
                del self.tasks[tid]
                cleared += 1
        self.save_state()
        return cleared

    def _prune_finished_locked(self):
        """Keeps only the most recent MAX_FINISHED_TASKS non-active tasks. Caller holds self.lock."""
        finished_ids = [
            tid for tid, t in self.tasks.items()
            if t.status in ("completed", "failed", "cancelled")
        ]
        if len(finished_ids) <= MAX_FINISHED_TASKS:
            return
        # Oldest tasks were inserted first (dict preserves insertion order)
        excess = len(finished_ids) - MAX_FINISHED_TASKS
        for tid in finished_ids[:excess]:
            self.tasks.pop(tid, None)

    def start_download(self, url: str, filename: str, target_dir: str, folder_type: str = "", overwrite: bool = False, expected_sha256: str = "") -> str:
        """
        Validates the destination is contained within target_dir (no path traversal),
        registers a task, and either starts it immediately or queues it if the
        configured concurrency limit is already reached.
        """
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme or '(none)'}")

        # Sanitize filename: strip any directory components so a crafted
        # "../../foo" (or an absolute path) can't escape target_dir.
        safe_filename = os.path.basename(filename.replace("\\", "/")).strip()
        if not safe_filename or safe_filename in (".", ".."):
            raise ValueError("Invalid filename")

        target_dir = os.path.abspath(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.abspath(os.path.join(target_dir, safe_filename))

        # Belt-and-braces: confirm the resolved path is still inside target_dir
        if os.path.commonpath([target_dir, target_path]) != target_dir:
            raise ValueError("Resolved download path escapes the target directory")

        if os.path.exists(target_path) and not overwrite:
            raise FileExistsError(f"'{safe_filename}' already exists in the target folder")

        task_id = str(uuid.uuid4())[:8]
        task = DownloadTask(task_id, url, target_path, safe_filename, folder_type, expected_sha256)

        with self.lock:
            self._prune_finished_locked()
            self.tasks[task_id] = task
            max_concurrent = max(1, int(config_manager.get("max_concurrent_downloads", 2) or 2))
            if self._active_count < max_concurrent:
                self._active_count += 1
                start_now = True
            else:
                task.status = "queued"
                self.queue.append(task_id)
                start_now = False

        self.save_state()

        if start_now:
            thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
            thread.start()
        else:
            self._notify_progress(task)

        return task_id

    def _run_task(self, task: DownloadTask):
        """Runs a single download then releases its concurrency slot and starts the next queued task."""
        try:
            self._download_worker(task)
        finally:
            self.save_state()
            self._start_next_queued()

    def _start_next_queued(self):
        next_task = None
        with self.lock:
            while self.queue:
                tid = self.queue.pop(0)
                candidate = self.tasks.get(tid)
                if candidate and candidate.status == "queued":
                    next_task = candidate
                    break
            if next_task is None:
                self._active_count = max(0, self._active_count - 1)

        if next_task:
            thread = threading.Thread(target=self._run_task, args=(next_task,), daemon=True)
            thread.start()

    def _notify_progress(self, task: DownloadTask):
        """Broadcasts download progress via ComfyUI WebSocket."""
        data = task.to_dict()
        if HAS_SERVER:
            try:
                server.PromptServer.instance.send_sync("model_downloader_progress", data)
            except Exception:
                pass

    def _notify_completion(self, task: DownloadTask):
        """Clears ComfyUI cache and broadcasts completion event."""
        if HAS_FOLDER_PATHS:
            try:
                if hasattr(folder_paths, "filename_list_cache"):
                    folder_paths.filename_list_cache.clear()
            except Exception as e:
                print(f"[ModelDownloader] Cache clear error: {e}")

        if HAS_SERVER:
            try:
                server.PromptServer.instance.send_sync("model_downloader_completed", task.to_dict())
            except Exception:
                pass

    def _finish_cancelled(self, task: DownloadTask):
        """Marks a task cancelled and removes its partial .downloading file (explicit cancel = discard)."""
        task.status = "cancelled"
        try:
            if os.path.exists(task.temp_path):
                os.remove(task.temp_path)
        except OSError as e:
            print(f"[ModelDownloader] Failed to remove temp file after cancel: {e}")
        self._notify_progress(task)
        self.save_state()

    def _create_tqdm_class(self, task):
        import huggingface_hub.utils
        import time
        parent = self
        
        shared_state = {
            "instances": [],
            "last_calc_time": time.time(),
            "last_total_bytes": 0,
            "speed_window": []
        }
        
        class CustomTqdm(huggingface_hub.utils.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                shared_state["instances"].append(self)
                if not hasattr(self, 'n'):
                    self.n = 0

            def update(self, n=1):
                if not hasattr(self, 'n'):
                    self.n = 0
                self.n += n
                super().update(n)
                
                current_total_n = max((getattr(inst, 'n', 0) for inst in shared_state["instances"]), default=0)
                current_total_bytes = max((getattr(inst, 'total', 0) or 0 for inst in shared_state["instances"]), default=0)
                
                task.downloaded_bytes = current_total_n
                task.total_bytes = current_total_bytes
                if task.total_bytes > 0:
                    task.percentage = (task.downloaded_bytes / task.total_bytes) * 100
                
                now = time.time()
                time_delta = now - shared_state["last_calc_time"]
                
                if time_delta >= 0.5:
                    bytes_delta = current_total_n - shared_state["last_total_bytes"]
                    instant_speed = bytes_delta / time_delta
                    
                    shared_state["speed_window"].append(instant_speed)
                    if len(shared_state["speed_window"]) > 5:
                        shared_state["speed_window"].pop(0)

                    task.speed_bytes_per_sec = sum(shared_state["speed_window"]) / len(shared_state["speed_window"])
                    
                    if task.total_bytes > 0:
                        remaining_bytes = max(0, task.total_bytes - task.downloaded_bytes)
                        if task.speed_bytes_per_sec > 0:
                            task.eta_seconds = remaining_bytes / task.speed_bytes_per_sec
                        else:
                            task.eta_seconds = 0

                    shared_state["last_calc_time"] = now
                    shared_state["last_total_bytes"] = current_total_n
                    
                    parent._notify_progress(task)

                if task.cancel_requested:
                    raise KeyboardInterrupt("Cancelled by user")
                if task.is_paused:
                    raise KeyboardInterrupt("Paused by user")
                    
        return CustomTqdm

    def _download_worker(self, task: DownloadTask):
        import traceback
        task.status = "downloading"
        task.start_time = time.time()
        task.last_update_time = time.time()
        self._notify_progress(task)
        
        parsed_url = urllib.parse.urlparse(task.url)
        
        # 1. Check if Hugging Face URL
        if "huggingface.co" in parsed_url.netloc:
            import re
            m = re.search(r"huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.*)", task.url)
            if m:
                repo_id = m.group(1)
                revision = m.group(2)
                filename_in_repo = urllib.parse.unquote(m.group(3))
                hf_token = config_manager.get_hf_token()
                
                import huggingface_hub
                from huggingface_hub import hf_hub_download
                from unittest.mock import patch
                import shutil
                
                CustomTqdm = self._create_tqdm_class(task)
                
                try:
                    with patch('huggingface_hub.utils._tqdm.tqdm', CustomTqdm):
                        target_dir = os.path.dirname(task.temp_path)
                        hf_temp_dir = os.path.join(target_dir, f".hf_tmp_{task.id}")
                        os.makedirs(hf_temp_dir, exist_ok=True)

                        try:
                            cached_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=filename_in_repo,
                                revision=revision,
                                local_dir=hf_temp_dir,
                                token=hf_token if hf_token else None
                            )
                            # Hub download completed successfully
                            # Move from the temp dir to the actual temp path (instant on same drive)
                            shutil.move(cached_path, task.temp_path)
                            task.downloaded_bytes = os.path.getsize(task.temp_path)
                            task.total_bytes = task.downloaded_bytes
                            task.percentage = 100.0
                            
                            # Use the same verification and rename logic below
                            self._finish_success(task)
                            return
                        finally:
                            if os.path.exists(hf_temp_dir):
                                shutil.rmtree(hf_temp_dir, ignore_errors=True)
                except ModuleNotFoundError:
                    task.status = "failed"
                    task.error_message = "huggingface_hub is not installed. Please restart ComfyUI or run: pip install huggingface-hub"
                    self._notify_progress(task)
                    print("[ModelDownloader] Missing huggingface_hub dependency.")
                    return
                except KeyboardInterrupt as e:
                    if task.cancel_requested:
                        self._finish_cancelled(task)
                    elif task.is_paused:
                        task.status = "paused"
                        self._notify_progress(task)
                    return
                except Exception as e:
                    task.status = "failed"
                    task.error_message = f"HuggingFace Hub Error: {e}"
                    self._notify_progress(task)
                    print(f"[ModelDownloader] HF Download failed: {e}")
                    return

        # 2. Fallback for Civitai / Others using requests
        existing_size = 0
        if os.path.exists(task.temp_path):
            existing_size = os.path.getsize(task.temp_path)
        task.downloaded_bytes = existing_size

        headers = {
            "User-Agent": "ComfyUI-MissingModelDownloader"
        }

        if host_matches(parsed_url.netloc, "civitai.com"):
            civitai_token = config_manager.get_civitai_token()
            if civitai_token and "token=" not in task.url:
                headers["Authorization"] = f"Bearer {civitai_token}"

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        import requests
        try:
            with requests.get(task.url, headers=headers, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                
                status_code = resp.status_code
                content_length = resp.headers.get("Content-Length")
                
                if status_code == 206:
                    task.total_bytes = existing_size + (int(content_length) if content_length else 0)
                    write_mode = "ab"
                else:
                    task.total_bytes = int(content_length) if content_length else 0
                    task.downloaded_bytes = 0
                    write_mode = "wb"

                chunk_size = 1024 * 1024
                speed_window = []
                last_calc_time = time.time()
                last_bytes_recorded = task.downloaded_bytes

                with open(task.temp_path, write_mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if task.cancel_requested:
                            f.close()
                            self._finish_cancelled(task)
                            return
                        if task.is_paused:
                            f.close()
                            task.status = "paused"
                            self._notify_progress(task)
                            return

                        if chunk:
                            f.write(chunk)
                            task.downloaded_bytes += len(chunk)

                            now = time.time()
                            time_delta = now - last_calc_time
                            if time_delta >= 0.5:
                                bytes_delta = task.downloaded_bytes - last_bytes_recorded
                                instant_speed = bytes_delta / time_delta
                                speed_window.append(instant_speed)
                                if len(speed_window) > 5:
                                    speed_window.pop(0)

                                task.speed_bytes_per_sec = sum(speed_window) / len(speed_window)
                                if task.total_bytes > 0:
                                    task.percentage = (task.downloaded_bytes / task.total_bytes) * 100
                                    remaining_bytes = max(0, task.total_bytes - task.downloaded_bytes)
                                    if task.speed_bytes_per_sec > 0:
                                        task.eta_seconds = remaining_bytes / task.speed_bytes_per_sec
                                    else:
                                        task.eta_seconds = 0

                                last_calc_time = now
                                last_bytes_recorded = task.downloaded_bytes
                                self._notify_progress(task)
            
            # Successfully downloaded all chunks
            self._finish_success(task)

        except requests.exceptions.RequestException as e:
            task.status = "failed"
            task.error_message = f"Network Error: {e}"
            self._notify_progress(task)
            print(f"[ModelDownloader] Request failed: {e}")
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            self._notify_progress(task)
            print(f"[ModelDownloader] Download exception: {e}")

    def _finish_success(self, task: DownloadTask):
        # Verification
        if task.expected_sha256:
            import hashlib
            task.status = "verifying"
            self._notify_progress(task)
            sha256_hash = hashlib.sha256()
            try:
                with open(task.temp_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                actual_sha256 = sha256_hash.hexdigest().lower()
                if actual_sha256 != task.expected_sha256.lower():
                    task.status = "failed"
                    task.error_message = f"SHA256 mismatch. Expected: {task.expected_sha256}, got: {actual_sha256}"
                    self._notify_progress(task)
                    print(f"[ModelDownloader] Download failed: {task.error_message}")
                    os.remove(task.temp_path)
                    return
            except Exception as e:
                task.status = "failed"
                task.error_message = f"Failed to verify SHA256: {e}"
                self._notify_progress(task)
                print(f"[ModelDownloader] Hash verification failed: {e}")
                return

        # Rename temp file to destination
        if os.path.exists(task.target_path):
            os.remove(task.target_path)
        os.rename(task.temp_path, task.target_path)

        task.status = "completed"
        task.percentage = 100.0
        task.speed_bytes_per_sec = 0
        task.eta_seconds = 0
        self._notify_progress(task)
        self._notify_completion(task)
        print(f"[ModelDownloader] Successfully downloaded: {task.filename} to {task.target_path}")

download_manager = DownloadManager()
