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

class DownloadTask:
    def __init__(self, task_id: str, url: str, target_path: str, filename: str, folder_type: str = ""):
        self.id = task_id
        self.url = url
        self.target_path = target_path
        self.temp_path = f"{target_path}.downloading"
        self.filename = filename
        self.folder_type = folder_type
        
        self.status = "pending"  # pending, downloading, completed, failed, cancelled
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.speed_bytes_per_sec = 0
        self.eta_seconds = 0
        self.percentage = 0.0
        self.error_message = ""
        
        self.cancel_requested = False
        self.start_time = 0
        self.last_update_time = 0
        self.last_downloaded_bytes = 0

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "folder_type": self.folder_type,
            "target_path": self.target_path,
            "url": self.url,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "percentage": round(self.percentage, 1),
            "speed_mb": round(self.speed_bytes_per_sec / (1024 * 1024), 2),
            "eta_seconds": int(self.eta_seconds),
            "error": self.error_message
        }


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


class DownloadManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DownloadManager, cls).__new__(cls)
            cls._instance.tasks = {}
            cls._instance.lock = threading.Lock()
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
            if task.status == "downloading":
                task.status = "cancelled"
            self._notify_progress(task)
            return True

    def start_download(self, url: str, filename: str, target_dir: str, folder_type: str = "") -> str:
        """Initiates a background download task and returns the task ID."""
        task_id = str(uuid.uuid4())[:8]
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)

        task = DownloadTask(task_id, url, target_path, filename, folder_type)
        with self.lock:
            self.tasks[task_id] = task

        thread = threading.Thread(target=self._download_worker, args=(task,), daemon=True)
        thread.start()
        return task_id

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

        # Attach token if appropriate
        parsed_url = urllib.parse.urlparse(task.url)
        if "huggingface.co" in parsed_url.netloc:
            hf_token = config_manager.get_hf_token()
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
        elif "civitai.com" in parsed_url.netloc:
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
                            task.status = "cancelled"
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

            # Finished streaming
            if task.cancel_requested:
                task.status = "cancelled"
                self._notify_progress(task)
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
