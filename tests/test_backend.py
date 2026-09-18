"""
Unit test and verification suite for ComfyUI Missing Model Downloader backend modules.
"""

import os
import sys
import tempfile
import time

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
from downloader import download_manager, host_matches, redact_url_secrets

def test_config_manager():
    print("--- Testing ConfigManager ---")
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
    test_civitai_client_url_parsing()
    test_civitai_search()
    test_detector_graph_scanning()
    test_host_matches_rejects_lookalike_domains()
    test_redact_url_secrets()
    test_path_traversal_is_blocked()
    test_overwrite_protection()
    test_downloader_lifecycle()
    print("\n========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")
