import json
import os
import urllib.request
import urllib.error

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "hf_token": "",
    "civitai_token": "",
    "default_provider": "huggingface",
    "auto_detect_on_load": True,
    "show_notification": True,
    "max_concurrent_downloads": 2,
}

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance.load()
        return cls._instance

    def load(self):
        """Loads configuration from config.json with fallback to defaults."""
        config = dict(DEFAULT_CONFIG)
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        config.update(loaded)
            except Exception as e:
                print(f"[ModelDownloader] Warning: Failed to read config.json: {e}")
        self._config = config
        return self._config

    def save(self):
        """Saves current configuration to config.json."""
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4)
            return True
        except Exception as e:
            print(f"[ModelDownloader] Error: Failed to save config.json: {e}")
            return False

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value
        self.save()

    def update(self, data: dict):
        """Updates multiple config fields. If token values are empty or unchanged placeholders, ignore."""
        for key, val in data.items():
            if key in ("hf_token", "civitai_token"):
                # If the incoming token is masked (e.g. contains '****'), skip updating it
                if isinstance(val, str) and "****" in val:
                    continue
                self._config[key] = (val or "").strip()
            elif key in DEFAULT_CONFIG:
                self._config[key] = val
        self.save()
        return self.get_sanitized()

    def get_sanitized(self):
        """Returns config with masked sensitive tokens for safe client-side display."""
        cfg = dict(self._config)
        hf = cfg.get("hf_token", "")
        if hf:
            if len(hf) > 8:
                cfg["hf_token_masked"] = f"{hf[:4]}...{hf[-4:]}"
            else:
                cfg["hf_token_masked"] = "****"
            cfg["has_hf_token"] = True
        else:
            cfg["hf_token_masked"] = ""
            cfg["has_hf_token"] = False

        civitai = cfg.get("civitai_token", "")
        if civitai:
            if len(civitai) > 8:
                cfg["civitai_token_masked"] = f"{civitai[:4]}...{civitai[-4:]}"
            else:
                cfg["civitai_token_masked"] = "****"
            cfg["has_civitai_token"] = True
        else:
            cfg["civitai_token_masked"] = ""
            cfg["has_civitai_token"] = False

        # Exclude raw tokens from the returned dict for security
        cfg.pop("hf_token", None)
        cfg.pop("civitai_token", None)
        return cfg

    def get_hf_token(self) -> str:
        return (self._config.get("hf_token") or "").strip()

    def get_civitai_token(self) -> str:
        return (self._config.get("civitai_token") or "").strip()

    @staticmethod
    def test_hf_token(token: str) -> dict:
        """Tests validity of Hugging Face access token using whoami API."""
        token = (token or "").strip()
        if not token:
            return {"valid": False, "error": "Token is empty."}

        url = "https://huggingface.co/api/whoami-v2"
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "ComfyUI-MissingModelDownloader"
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                username = data.get("name", "User")
                orgs = [o.get("name") for o in data.get("orgs", []) if isinstance(o, dict)]
                return {
                    "valid": True,
                    "username": username,
                    "type": data.get("type", "user"),
                    "email": data.get("email", ""),
                    "orgs": orgs
                }
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {"valid": False, "error": "Invalid token (401 Unauthorized)."}
            return {"valid": False, "error": f"HTTP Error {e.code}: {e.reason}"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @staticmethod
    def test_civitai_token(token: str) -> dict:
        """Tests validity of Civitai API key."""
        token = (token or "").strip()
        if not token:
            return {"valid": False, "error": "Token is empty."}

        url = f"https://civitai.com/api/v1/models?limit=1&token={token}"
        headers = {
            "User-Agent": "ComfyUI-MissingModelDownloader",
            "Authorization": f"Bearer {token}"
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return {"valid": True, "message": "Civitai token is valid and active."}
                return {"valid": False, "error": f"Civitai responded with status {resp.status}."}
        except urllib.error.HTTPError as e:
            if e.code == 401 or e.code == 403:
                return {"valid": False, "error": f"Invalid Civitai token ({e.code} Unauthorized)."}
            return {"valid": False, "error": f"HTTP Error {e.code}: {e.reason}"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

config_manager = ConfigManager()
