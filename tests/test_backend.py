"""
Unit test and verification suite for ComfyUI Missing Model Downloader backend modules.
"""

import os
import sys
import tempfile
import threading
import time
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from config_manager import config_manager
from hf_client import hf_client
from civitai_client import civitai_client
from detector import detector
import downloader
from downloader import download_manager, host_matches, redact_url_secrets


def wait_for_task(task_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = download_manager.get_task(task_id)
        if task and task["status"] in ("completed", "failed", "cancelled", "paused"):
            return task
        time.sleep(0.1)
    return download_manager.get_task(task_id)


class FlakyFileServer:
    """
    Local HTTP server for one file with Range support. The first `drops` full requests
    send the headers for the whole file but close the connection halfway through.
    """
    def __init__(self, payload: bytes, drops: int = 0):
        self.payload = payload
        self.drops_left = drops
        self.requests = []
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                range_header = self.headers.get("Range")
                server_ref.requests.append(range_header)
                start = int(range_header.split("=")[1].rstrip("-")) if range_header else 0
                body = server_ref.payload[start:]
                self.send_response(206 if range_header else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if server_ref.drops_left > 0 and not range_header:
                    server_ref.drops_left -= 1
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/model.safetensors"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

def test_config_manager():
    print("--- Testing ConfigManager ---")
    # Redirect saves to a temp file and snapshot in-memory config so the real config.json is untouched
    original_config = dict(config_manager._config)
    with tempfile.TemporaryDirectory() as temp_dir, \
            patch("config_manager.CONFIG_PATH", os.path.join(temp_dir, "config.json")):
        try:
            config_manager.update({
                "hf_token": "hf_testdummytoken1234567890",
                "default_provider": "huggingface",
                "auto_detect_on_load": True
            })

            sanitized = config_manager.get_sanitized()
            assert sanitized["has_hf_token"] is True, "Expected has_hf_token to be True"
            assert "hf_testdummytoken1234567890" not in str(sanitized), "Raw token should not be present in sanitized output"
            assert "hf_testdummytoken1234567890" == config_manager.get_hf_token(), "Raw token should match stored token"
            print("✓ ConfigManager token storage and sanitization passed.")

            config_manager.update({"hf_token": ""})
            sanitized = config_manager.get_sanitized()
            assert sanitized["has_hf_token"] is False
            print("✓ ConfigManager reset passed.")
        finally:
            config_manager._config = original_config

def test_hf_client_url_parsing():
    print("\n--- Testing HuggingFace URL Parsing ---")
    url = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
    parsed = hf_client.parse_direct_url(url)
    assert parsed["valid"] is True
    assert parsed["repo_id"] == "black-forest-labs/FLUX.1-dev"
    assert parsed["filename"] == "flux1-dev.safetensors"
    assert parsed["download_url"] == "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
    print(f"✓ HF URL parsed: {parsed['filename']} -> {parsed['download_url']}")

def test_hf_client_search():
    print("\n--- Testing HuggingFace Search (Live Network Query) ---")
    results = hf_client.search_for_model("v1-5-pruned-emaonly.safetensors", limit=3)
    print(f"Received {len(results)} Hugging Face results.")
    if results:
        top = results[0]
        print(f"Top result: [{top['source']}] {top['name']} (repo: {top.get('repo_id')}, exact: {top.get('exact_match')})")
        assert top["source"] == "huggingface"
        assert top["download_url"].startswith("https://huggingface.co/")
        print("✓ HF Search API successfully found and resolved model.")
    else:
        print("ℹ Note: No results or network timeout on HF search.")

def test_hf_file_info_reads_hash_before_cdn_redirect():
    print("\n--- Testing HF File Info SHA256 (Live Network Query) ---")
    # HF reports an LFS/Xet file's SHA256 and size only on its own 302, not on the CDN response
    info = hf_client.get_file_info("sentence-transformers/all-MiniLM-L6-v2", "model.safetensors")
    if not info["accessible"]:
        print(f"ℹ Note: HF unreachable ({info.get('error')}), skipped.")
        return
    assert info["sha256"] == "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", info
    assert info["size_bytes"] == 90868376, info
    # Small git-stored files only have a SHA1, which must not be passed off as a SHA256
    small = hf_client.get_file_info("openai-community/gpt2", "config.json")
    assert small["accessible"] and small["sha256"] == "" and small["size_bytes"] > 0, small
    print("✓ HF SHA256 and size read from the resolve redirect.")

def test_hf_search_scans_files_listed_by_search_api():
    print("\n--- Testing HF Search Uses Listed Siblings ---")
    # The HF search API lists files (siblings) even with full=false; those repos must still be scanned
    import io, json as _json
    listing = [{"id": "someone/gemma-repo", "downloads": 0, "likes": 0,
                "siblings": [{"rfilename": "text_encoders/gemma_3_12B_it.safetensors"}]}]
    requested = []

    def fake_urlopen(req, *args, **kwargs):
        url = req.full_url
        requested.append(url)
        body = _json.dumps(listing if "/api/models?search=" in url else []).encode()
        return io.BytesIO(body)

    with patch("hf_client.urllib.request.urlopen", fake_urlopen), \
            patch.object(hf_client, "get_file_info", lambda *a, **k: {"size_bytes": 1, "sha256": ""}):
        results = hf_client.search_for_model("gemma_3_12B_it.safetensors")
    exact = [r for r in results if r["exact_match"]]
    assert exact and exact[0]["repo_id"] == "someone/gemma-repo", results
    assert not any("/api/models/someone/gemma-repo" in u for u in requested), "Listed siblings should be used without a detail fetch"
    print("✓ Repos returned with siblings are scanned for files.")

def test_civitai_client_url_parsing():
    print("\n--- Testing Civitai URL Parsing ---")
    url = "https://civitai.com/models/12345?modelVersionId=67890"
    parsed = civitai_client.parse_direct_url(url)
    assert parsed["valid"] is True
    assert parsed["version_id"] == 67890
    assert "download/models/67890" in parsed["download_url"]
    print(f"✓ Civitai URL parsed: {parsed['download_url']}")

def test_civitai_search():
    print("\n--- Testing Civitai Search (Live Network Query) ---")
    results = civitai_client.search_for_model("epicrealism", limit=2)
    print(f"Received {len(results)} Civitai results.")
    if results:
        top = results[0]
        print(f"Top result: [{top['source']}] {top['name']} (model: {top.get('model_name')})")
        assert top["source"] == "civitai"
        assert "civitai.com" in top["download_url"]
        print("✓ Civitai Search API successfully found model.")
    else:
        print("ℹ Note: No results or network timeout on Civitai search.")

def test_detector_graph_scanning():
    print("\n--- Testing Workflow Model Detector ---")
    mock_workflow = {
        "nodes": [
            {
                "id": 4,
                "type": "CheckpointLoaderSimple",
                "widgets_values": ["test_checkpoint_nonexistent_xyz.safetensors"]
            },
            {
                "id": 10,
                "type": "LoraLoader",
                "widgets_values": ["test_lora_nonexistent_abc.safetensors", 1.0, 1.0]
            },
            {
                "id": 15,
                "type": "VAELoader",
                "widgets_values": ["test_vae_nonexistent_123.safetensors"]
            }
        ]
    }

    missing = detector.detect_missing_models(mock_workflow)
    print(f"Detected {len(missing)} missing models.")
    assert len(missing) == 3, f"Expected 3 missing models, got {len(missing)}"

    types = {m["folder_type"]: m["filename"] for m in missing}
    print("Detected missing models:", types)
    assert "checkpoints" in types
    assert types["checkpoints"] == "test_checkpoint_nonexistent_xyz.safetensors"
    assert "loras" in types
    assert types["loras"] == "test_lora_nonexistent_abc.safetensors"
    assert "vae" in types
    assert types["vae"] == "test_vae_nonexistent_123.safetensors"
    print("✓ Model detection and folder inference passed flawlessly.")

def test_detector_uses_workflow_model_urls_and_subfolders():
    print("\n--- Testing Workflow-Embedded Model URLs ---")
    url = "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors?download=true"
    workflow = {
        "nodes": [{
            "id": 7,
            "type": "UNETLoader",
            "widgets_values": ["flux/flux1-dev-fp8.safetensors", "default"],
            "properties": {"models": [{
                "name": "flux1-dev-fp8.safetensors", "url": url, "directory": "diffusion_models",
                "hash": "ABC123", "hash_type": "SHA256"
            }]}
        }, {
            "id": 8,
            "type": "LoraLoader",
            "widgets_values": ["../../evil/lora_xyz.safetensors", 1.0, 1.0],
            "properties": {"models": [{"name": "lora_xyz.safetensors", "url": "file:///etc/passwd", "directory": "../x"}]}
        }]
    }
    missing = {m["filename"]: m for m in detector.detect_missing_models(workflow)}

    flux = missing["flux1-dev-fp8.safetensors"]
    assert flux["known_url"] == url
    assert flux["known_sha256"] == "abc123"
    assert flux["folder_type"] == "diffusion_models"
    assert flux["subfolder"] == "flux"

    lora = missing["lora_xyz.safetensors"]
    assert lora["known_url"] == "", "Non-http(s) workflow URLs must be ignored"
    assert lora["folder_type"] == "loras", "Unsafe workflow directories must be ignored"
    assert lora["subfolder"] == "", "Traversal subfolders must be dropped"
    print("✓ Workflow URLs, folders and subfolders are used; unsafe ones are ignored.")

def test_host_matches_rejects_lookalike_domains():
    print("\n--- Testing Exact-Host Token Matching (security) ---")
    assert host_matches("huggingface.co", "huggingface.co") is True
    assert host_matches("files.huggingface.co", "huggingface.co") is True
    assert host_matches("huggingface.co.evil.com", "huggingface.co") is False, \
        "Look-alike domain must NOT match — would leak the bearer token"
    assert host_matches("nothuggingface.co", "huggingface.co") is False
    assert host_matches("evil.com", "huggingface.co") is False
    print("✓ host_matches correctly rejects look-alike / substring domains.")

def test_redact_url_secrets():
    print("\n--- Testing URL Token Redaction ---")
    url = "https://civitai.com/api/download/models/123?type=Model&token=supersecrettoken"
    redacted = redact_url_secrets(url)
    assert "supersecrettoken" not in redacted, "Raw token must never be echoed back to the client"
    assert "token=***" in redacted
    assert "type=Model" in redacted, "Non-sensitive params must be preserved"
    print(f"✓ Token redacted: {redacted}")

def test_path_traversal_is_blocked():
    print("\n--- Testing Path Traversal Protection ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        # A "../../../evil.safetensors" filename must be normalized down to just
        # "evil.safetensors" and stay inside temp_dir — never escape it.
        task_id = download_manager.start_download(
            url="https://huggingface.co/x/y/resolve/main/z.safetensors",
            filename="../../../evil.safetensors",
            target_dir=temp_dir,
            folder_type="checkpoints"
        )
        task = download_manager.get_task(task_id)
        assert task["filename"] == "evil.safetensors", f"Filename must be sanitized, got {task['filename']!r}"
        assert task["target_path"].startswith(os.path.abspath(temp_dir)), \
            f"Target path escaped temp_dir: {task['target_path']}"
        download_manager.cancel_task(task_id)
        print(f"✓ Traversal filename normalized and contained: {task['target_path']}")

        try:
            download_manager.start_download(
                url="file:///etc/passwd",
                filename="passwd.txt",
                target_dir=temp_dir,
                folder_type="checkpoints"
            )
            assert False, "Expected a ValueError for a non-http(s) scheme"
        except ValueError as e:
            print(f"✓ Non-http(s) scheme rejected: {e}")

def test_overwrite_protection():
    print("\n--- Testing Overwrite Protection ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        existing = os.path.join(temp_dir, "already_here.safetensors")
        with open(existing, "w") as f:
            f.write("existing content")

        try:
            download_manager.start_download(
                url="https://huggingface.co/x/y/resolve/main/already_here.safetensors",
                filename="already_here.safetensors",
                target_dir=temp_dir,
                folder_type="checkpoints"
            )
            assert False, "Expected a FileExistsError without overwrite=True"
        except FileExistsError as e:
            print(f"✓ Existing file protected without overwrite flag: {e}")

def test_custom_folder_validation():
    print("\n--- Testing Custom Folder Validation ---")
    target = detector.get_target_directory("my_custom_models")
    assert target.endswith(os.path.join("models", "my_custom_models")), target
    for bad in ("../escape", "a/b", "..", "", "custom_nodes"):
        try:
            detector.get_target_directory(bad)
            raise AssertionError(f"Expected ValueError for folder '{bad}'")
        except ValueError:
            pass
    print("✓ Custom folders resolve under models/ and unsafe names are rejected.")

def test_subfolder_downloads_and_traversal():
    print("\n--- Testing Subfolder Downloads ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        for bad in ("../escape", "/abs", "a/../../b", "a//b"):
            try:
                download_manager.start_download("https://example.com/m.safetensors", "m.safetensors", temp_dir, subfolder=bad)
                raise AssertionError(f"Expected ValueError for subfolder '{bad}'")
            except ValueError:
                pass
        print("✓ Unsafe subfolders rejected.")

        server = FlakyFileServer(b"x" * 1000)
        try:
            task_id = download_manager.start_download(server.url, "m.safetensors", temp_dir, "checkpoints", subfolder="flux/dev")
            task = wait_for_task(task_id)
            assert task["status"] == "completed", task
            assert task["subfolder"] == "flux/dev"
            assert os.path.getsize(os.path.join(temp_dir, "flux", "dev", "m.safetensors")) == 1000
        finally:
            server.close()
        print("✓ Download saved into its subfolder.")

def test_disk_space_check():
    print("\n--- Testing Disk Space Check ---")
    Usage = namedtuple("Usage", "total used free")
    with tempfile.TemporaryDirectory() as temp_dir:
        server = FlakyFileServer(b"x" * 5000)
        try:
            with patch("downloader.shutil.disk_usage", return_value=Usage(10, 0, 10)):
                task_id = download_manager.start_download(server.url, "big.safetensors", temp_dir)
                task = wait_for_task(task_id)
        finally:
            server.close()
        assert task["status"] == "failed", task
        assert "Not enough disk space" in task["error"], task["error"]
        assert not os.path.exists(os.path.join(temp_dir, "big.safetensors"))
    print(f"✓ Download refused: {task['error']}")

def test_http_retry_resumes_after_drop():
    print("\n--- Testing HTTP Auto-Retry ---")
    payload = os.urandom(3 * 1024 * 1024)
    with tempfile.TemporaryDirectory() as temp_dir:
        server = FlakyFileServer(payload, drops=1)
        try:
            with patch("downloader.HTTP_RETRY_DELAYS", (0.2, 0.2, 0.2)):
                task_id = download_manager.start_download(server.url, "flaky.safetensors", temp_dir)
                task = wait_for_task(task_id)
        finally:
            server.close()
        assert task["status"] == "completed", task
        with open(os.path.join(temp_dir, "flaky.safetensors"), "rb") as f:
            assert f.read() == payload, "Resumed file must match the original byte for byte"
        assert server.requests[0] is None and server.requests[-1] and server.requests[-1].startswith("bytes="), server.requests
    print(f"✓ Connection drop retried and resumed (requests: {server.requests}).")

def test_downloader_lifecycle():
    print("\n--- Testing Downloader Manager Lifecycle ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        test_url = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt"
        task_id = download_manager.start_download(
            url=test_url,
            filename="test_requirements.txt",
            target_dir=temp_dir,
            folder_type="checkpoints"
        )
        print(f"Started download task: {task_id}")

        for _ in range(20):
            time.sleep(0.5)
            task = download_manager.get_task(task_id)
            if task and task["status"] in ("completed", "failed", "cancelled"):
                break

        final_task = download_manager.get_task(task_id)
        print(f"Final task status: {final_task['status']}, size: {final_task['downloaded_bytes']} bytes")
        assert final_task["status"] == "completed"
        dest_file = os.path.join(temp_dir, "test_requirements.txt")
        assert os.path.exists(dest_file), "Destination file must exist after completion"
        assert os.path.getsize(dest_file) > 0, "Destination file must have content"
        print("✓ Downloader completed and verified downloaded file.")

if __name__ == "__main__":
    print("Starting ComfyUI Model Downloader Verification Suite...")
    test_config_manager()
    test_hf_client_url_parsing()
    test_hf_client_search()
    test_hf_file_info_reads_hash_before_cdn_redirect()
    test_hf_search_scans_files_listed_by_search_api()
    test_civitai_client_url_parsing()
    test_civitai_search()
    test_detector_graph_scanning()
    test_host_matches_rejects_lookalike_domains()
    test_redact_url_secrets()
    test_path_traversal_is_blocked()
    test_overwrite_protection()
    test_custom_folder_validation()
    test_detector_uses_workflow_model_urls_and_subfolders()
    test_subfolder_downloads_and_traversal()
    test_disk_space_check()
    test_http_retry_resumes_after_drop()
    test_downloader_lifecycle()
    print("\n========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")
