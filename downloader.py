import os
import re
import shutil
import time
import uuid
import threading
import urllib.parse
from config_manager import config_manager
from detector import is_safe_folder_name
from hf_client import hf_client

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
    def __init__(self, task_id: str, url: str, target_path: str, filename: str, folder_type: str = "", expected_sha256: str = "", subfolder: str = ""):
        self.id = task_id
        self.url = url
        self.target_path = target_path
        self.temp_path = f"{target_path}.downloading"
        self.filename = filename
        self.folder_type = folder_type
        self.expected_sha256 = expected_sha256
        self.subfolder = subfolder
        
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
            "subfolder": self.subfolder,
            "expected_sha256": self.expected_sha256,
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


# Statuses of tasks that hold (or wait for) a download slot
ACTIVE_STATUSES = ("pending", "queued", "downloading", "retrying")

# Free space kept on the drive beyond the file itself
DISK_SPACE_MARGIN = 1024 ** 3

# Waits between HTTP retries; the count resets whenever an attempt makes progress
HTTP_RETRY_DELAYS = (2, 5, 10)

HF_URL_RE = re.compile(r"huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/([^?#]*)")


class InsufficientDiskSpace(Exception):
    pass


class RetryableHTTPError(Exception):
    """A server-side (5xx) response worth retrying."""


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.1f} GB"
    return f"{num_bytes / (1024 ** 2):.1f} MB"


def check_disk_space(directory: str, needed_bytes: int):
    """Raises InsufficientDiskSpace if needed_bytes (plus a safety margin) won't fit in directory's drive."""
    if needed_bytes <= 0:
        return
    free = shutil.disk_usage(directory).free
    if needed_bytes + DISK_SPACE_MARGIN > free:
        raise InsufficientDiskSpace(
            f"Not enough disk space: needs {format_size(needed_bytes)} (+{format_size(DISK_SPACE_MARGIN)} margin), {format_size(free)} free"
        )


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


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
                        expected_sha256=task_dict.get("expected_sha256", ""),
                        subfolder=task_dict.get("subfolder", "")
                    )
                    t.status = task_dict["status"]
                    if t.status in ACTIVE_STATUSES:
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
                        "subfolder": getattr(t, 'subfolder', ""),
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
            if task.status in ACTIVE_STATUSES:
                task.status = "cancelled"
            if task_id in self.queue:
                self.queue.remove(task_id)
            self._notify_progress(task)
        self.save_state()
        return True

    def pause_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task.status not in ACTIVE_STATUSES:
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
            active_ids = [tid for tid, t in self.tasks.items() if t.status in ACTIVE_STATUSES and not t.is_paused]
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

    def start_download(self, url: str, filename: str, target_dir: str, folder_type: str = "", overwrite: bool = False, expected_sha256: str = "", subfolder: str = "") -> str:
        """
        Validates the destination is contained within target_dir (no path traversal),
        registers a task, and either starts it immediately or queues it if the
        configured concurrency limit is already reached. `subfolder` (e.g. "flux/dev")
        is created under target_dir.
        """
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme or '(none)'}")

        # Sanitize filename: strip any directory components so a crafted
        # "../../foo" (or an absolute path) can't escape target_dir.
        safe_filename = os.path.basename(filename.replace("\\", "/")).strip()
        if not safe_filename or safe_filename in (".", ".."):
            raise ValueError("Invalid filename")

        segments = [seg for seg in (subfolder or "").replace("\\", "/").split("/")] if subfolder else []
        if any(not is_safe_folder_name(seg) for seg in segments):
            raise ValueError(f"Invalid subfolder: '{subfolder}'")

        base_dir = os.path.abspath(target_dir)
        dest_dir = os.path.abspath(os.path.join(base_dir, *segments))
        target_path = os.path.abspath(os.path.join(dest_dir, safe_filename))

        # Belt-and-braces: confirm the resolved path is still inside target_dir
        if os.path.commonpath([base_dir, target_path]) != base_dir:
            raise ValueError("Resolved download path escapes the target directory")
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(target_path) and not overwrite:
            raise FileExistsError(f"'{safe_filename}' already exists in the target folder")

        task_id = str(uuid.uuid4())[:8]
        task = DownloadTask(task_id, url, target_path, safe_filename, folder_type, expected_sha256, "/".join(segments))

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

    def _hf_temp_dir(self, task: DownloadTask) -> str:
        """Per-task local_dir for hf_hub_download. Kept while paused so the partial download resumes."""
        return os.path.join(os.path.dirname(task.temp_path), f".hf_tmp_{task.id}")

    def _remove_partial_files(self, task: DownloadTask):
        try:
            if os.path.exists(task.temp_path):
                os.remove(task.temp_path)
        except OSError as e:
            print(f"[ModelDownloader] Failed to remove temp file: {e}")
        shutil.rmtree(self._hf_temp_dir(task), ignore_errors=True)

    def _finish_cancelled(self, task: DownloadTask):
        """Marks a task cancelled and removes its partial files (explicit cancel = discard)."""
        task.status = "cancelled"
        self._remove_partial_files(task)
        self._notify_progress(task)
        self.save_state()

    def _finish_paused(self, task: DownloadTask):
        task.status = "paused"
        self._notify_progress(task)

    def _fail(self, task: DownloadTask, message: str):
        task.status = "failed"
        task.error_message = message
        task.speed_bytes_per_sec = 0
        self._notify_progress(task)
        print(f"[ModelDownloader] Download failed ({task.filename}): {message}")

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
        task.status = "downloading"
        task.error_message = ""
        task.start_time = time.time()
        task.last_update_time = time.time()
        self._notify_progress(task)

        m = HF_URL_RE.search(task.url) if host_matches(urllib.parse.urlparse(task.url).netloc, "huggingface.co") else None
        if m:
            self._download_hf(task, m.group(1), m.group(2), urllib.parse.unquote(m.group(3)))
        else:
            self._download_http(task)

    def _download_hf(self, task: DownloadTask, repo_id: str, revision: str, path_in_repo: str):
        """Downloads through huggingface_hub (handles auth, xet/hf_transfer, and resume)."""
        try:
            from huggingface_hub import hf_hub_download
            from unittest.mock import patch
            CustomTqdm = self._create_tqdm_class(task)
        except ModuleNotFoundError:
            self._fail(task, "huggingface_hub is not installed. Please restart ComfyUI or run: pip install huggingface-hub")
            return

        hf_token = config_manager.get_hf_token() or None
        temp_dir = self._hf_temp_dir(task)
        try:
            info = hf_client.get_file_info(repo_id, path_in_repo, revision, hf_token)
            check_disk_space(os.path.dirname(task.temp_path), info.get("size_bytes", 0) - dir_size(temp_dir))

            os.makedirs(temp_dir, exist_ok=True)
            with patch('huggingface_hub.utils._tqdm.tqdm', CustomTqdm):
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    revision=revision,
                    local_dir=temp_dir,
                    token=hf_token
                )
            # Move out of the temp dir (instant on the same drive), then verify and rename as usual
            shutil.move(downloaded_path, task.temp_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            task.downloaded_bytes = os.path.getsize(task.temp_path)
            task.total_bytes = task.downloaded_bytes
            task.percentage = 100.0
            self._finish_success(task)
        except KeyboardInterrupt:
            # Raised from CustomTqdm.update on pause/cancel
            if task.cancel_requested:
                self._finish_cancelled(task)
            elif task.is_paused:
                self._finish_paused(task)
        except InsufficientDiskSpace as e:
            self._fail(task, str(e))
        except Exception as e:
            self._remove_partial_files(task)
            self._fail(task, f"HuggingFace Hub Error: {e}")

    def _download_http(self, task: DownloadTask):
        """Streams Civitai / direct URLs with Range resume, retrying dropped connections."""
        import requests

        headers = {"User-Agent": "ComfyUI-MissingModelDownloader"}
        if host_matches(urllib.parse.urlparse(task.url).netloc, "civitai.com"):
            civitai_token = config_manager.get_civitai_token()
            if civitai_token and "token=" not in task.url:
                headers["Authorization"] = f"Bearer {civitai_token}"

        retryable = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            RetryableHTTPError,
        )
        attempt = 0
        while True:
            bytes_before = task.downloaded_bytes
            try:
                outcome = self._http_attempt(task, headers)
                if outcome == "done":
                    self._finish_success(task)
                elif outcome == "cancelled":
                    self._finish_cancelled(task)
                elif outcome == "paused":
                    self._finish_paused(task)
                return
            except InsufficientDiskSpace as e:
                self._fail(task, str(e))
                return
            except retryable as e:
                if task.downloaded_bytes > bytes_before:
                    attempt = 0
                if attempt >= len(HTTP_RETRY_DELAYS):
                    self._fail(task, f"Network Error: {e}")
                    return
                delay = HTTP_RETRY_DELAYS[attempt]
                attempt += 1
                print(f"[ModelDownloader] {task.filename}: {e}; retry {attempt}/{len(HTTP_RETRY_DELAYS)} in {delay}s")
                task.status = "retrying"
                task.speed_bytes_per_sec = 0
                task.error_message = f"Connection lost, retrying ({attempt}/{len(HTTP_RETRY_DELAYS)})"
                self._notify_progress(task)
                if not self._wait_unless_stopped(task, delay):
                    return
                task.status = "downloading"
                task.error_message = ""
                self._notify_progress(task)
            except requests.exceptions.RequestException as e:
                self._fail(task, f"Network Error: {e}")
                return
            except Exception as e:
                self._fail(task, str(e))
                return

    def _wait_unless_stopped(self, task: DownloadTask, seconds: float) -> bool:
        """Sleeps between retries; returns False (after finishing the task) if it was paused or cancelled meanwhile."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if task.cancel_requested:
                self._finish_cancelled(task)
                return False
            if task.is_paused:
                self._finish_paused(task)
                return False
            time.sleep(0.25)
        return True

    def _http_attempt(self, task: DownloadTask, base_headers: dict) -> str:
        """One request, resuming from the partial file. Returns "done", "paused" or "cancelled"."""
        import requests

        existing_size = os.path.getsize(task.temp_path) if os.path.exists(task.temp_path) else 0
        task.downloaded_bytes = existing_size
        headers = dict(base_headers)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        with requests.get(task.url, headers=headers, stream=True, timeout=30) as resp:
            if resp.status_code == 416 and existing_size > 0:
                # Range starts at/after the end: the partial file already holds the whole download
                task.total_bytes = existing_size
                return "done"
            if resp.status_code >= 500:
                raise RetryableHTTPError(f"HTTP {resp.status_code} from server")
            resp.raise_for_status()

            content_length = resp.headers.get("Content-Length")
            remaining = int(content_length) if content_length and content_length.isdigit() else 0
            if resp.status_code == 206:
                task.total_bytes = existing_size + remaining
                write_mode = "ab"
            else:
                # Server ignored the Range header: start over
                task.total_bytes = remaining
                task.downloaded_bytes = 0
                write_mode = "wb"
            check_disk_space(os.path.dirname(task.temp_path), remaining)

            speed_window = []
            last_calc_time = time.time()
            last_bytes_recorded = task.downloaded_bytes

            with open(task.temp_path, write_mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if task.cancel_requested:
                        return "cancelled"
                    if task.is_paused:
                        return "paused"
                    if not chunk:
                        continue

                    f.write(chunk)
                    task.downloaded_bytes += len(chunk)

                    now = time.time()
                    time_delta = now - last_calc_time
                    if time_delta >= 0.5:
                        bytes_delta = task.downloaded_bytes - last_bytes_recorded
                        speed_window.append(bytes_delta / time_delta)
                        if len(speed_window) > 5:
                            speed_window.pop(0)

                        task.speed_bytes_per_sec = sum(speed_window) / len(speed_window)
                        if task.total_bytes > 0:
                            task.percentage = (task.downloaded_bytes / task.total_bytes) * 100
                            remaining_bytes = max(0, task.total_bytes - task.downloaded_bytes)
                            task.eta_seconds = remaining_bytes / task.speed_bytes_per_sec if task.speed_bytes_per_sec > 0 else 0

                        last_calc_time = now
                        last_bytes_recorded = task.downloaded_bytes
                        self._notify_progress(task)

        # A clean end of stream that's shorter than advertised is a dropped connection, not success
        if task.total_bytes > 0 and task.downloaded_bytes < task.total_bytes:
            raise requests.exceptions.ChunkedEncodingError(
                f"Connection closed early ({task.downloaded_bytes} of {task.total_bytes} bytes)"
            )
        return "done"

    def _finish_success(self, task: DownloadTask):
        # Verification
        if task.expected_sha256:
            import hashlib
            task.status = "verifying"
            self._notify_progress(task)
            sha256_hash = hashlib.sha256()
            try:
                with open(task.temp_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(1024 * 1024), b""):
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
