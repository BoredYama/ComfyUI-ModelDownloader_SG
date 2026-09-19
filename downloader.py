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
        return cls._instance

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

    def _download_worker(self, task: DownloadTask):
        task.status = "downloading"
        task.start_time = time.time()
        task.last_update_time = time.time()
        self._notify_progress(task)

        # Determine existing size for resuming
        existing_size = 0
        if os.path.exists(task.temp_path):
            existing_size = os.path.getsize(task.temp_path)
        task.downloaded_bytes = existing_size

        headers = {
            "User-Agent": "ComfyUI-MissingModelDownloader"
        }

        # Attach token if appropriate (exact-host match only, never a substring match,
        # so a look-alike domain can't trick us into leaking the bearer token)
        parsed_url = urllib.parse.urlparse(task.url)
        if host_matches(parsed_url.netloc, "huggingface.co"):
            hf_token = config_manager.get_hf_token()
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
        elif host_matches(parsed_url.netloc, "civitai.com"):
            civitai_token = config_manager.get_civitai_token()
            if civitai_token and "token=" not in task.url:
                headers["Authorization"] = f"Bearer {civitai_token}"

        # Resume support
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        opener = urllib.request.build_opener(SafeRedirectHandler())
        req = urllib.request.Request(task.url, headers=headers)

        try:
            with opener.open(req, timeout=30) as resp:
                status_code = getattr(resp, "status", 200)
                content_length = resp.headers.get("Content-Length")
                
                if status_code == 206:
                    # Partial content (resumed)
                    task.total_bytes = existing_size + (int(content_length) if content_length else 0)
                    write_mode = "ab"
                else:
                    # Full content
                    task.total_bytes = int(content_length) if content_length else 0
                    task.downloaded_bytes = 0
                    write_mode = "wb"

                chunk_size = 1024 * 1024  # 1MB chunks
                speed_window = []
                last_calc_time = time.time()
                last_bytes_recorded = task.downloaded_bytes

                with open(task.temp_path, write_mode) as f:
                    while True:
                        if task.cancel_requested:
                            f.close()
                            self._finish_cancelled(task)
                            return
                        if task.is_paused:
                            f.close()
                            task.status = "paused"
                            self._notify_progress(task)
                            return

                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break

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

            if task.cancel_requested:
                self._finish_cancelled(task)
                return
            if task.is_paused:
                task.status = "paused"
                self._notify_progress(task)
                return

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

        except urllib.error.HTTPError as e:
            task.status = "failed"
            task.error_message = f"HTTP Error {e.code}: {e.reason}"
            if e.code in (401, 403):
                task.error_message += " (Authentication or gated access required. Check your API token)."
            self._notify_progress(task)
            print(f"[ModelDownloader] Download failed: {task.error_message}")
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            self._notify_progress(task)
            print(f"[ModelDownloader] Download exception: {e}")

download_manager = DownloadManager()
