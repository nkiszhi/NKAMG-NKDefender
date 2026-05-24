import configparser
import os
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.ini"


def _load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")
    return config


def _as_path(value: str, default: str) -> Path:
    raw = (value or default).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class Settings:
    def __init__(self) -> None:
        self._config = _load_config()

        self.host = self._get("server", "host", "127.0.0.1")
        self.port = self._getint("server", "port", 5005)
        self.cors_origins = self._get_list("server", "cors_origins", ["*"])
        self.max_upload_size = self._getint("server", "max_upload_size", 100 * 1024 * 1024)

        self.upload_dir = _as_path(self._get("paths", "upload_dir", "backend/uploads"), "backend/uploads")

        self.codefender_api_base = self._get("codefender", "api_base", "http://127.0.0.1:8001").rstrip("/")
        self.codefender_scan_endpoint = self._get("codefender", "scan_endpoint", "/scan/file")
        self.codefender_predict_endpoint = self._get("codefender", "predict_endpoint", "/predict")
        self.codefender_timeout = self._getint("codefender", "timeout", 300)
        self.threshold = self._getfloat("codefender", "threshold", 0.5)
        self.host_path_prefix = self._get("codefender", "host_path_prefix", "").strip()
        self.container_path_prefix = self._get("codefender", "container_path_prefix", "").strip()
        self.docker_container_name = self._get("docker", "container_name", "").strip()
        self.docker_container_upload_dir = self._get("docker", "container_upload_dir", "/codefender_uploads").strip()

    def _get(self, section: str, option: str, default: str) -> str:
        env_name = f"CODEFENDER_{section}_{option}".upper()
        return os.getenv(env_name, self._config.get(section, option, fallback=default))

    def _getint(self, section: str, option: str, default: int) -> int:
        value = self._get(section, option, str(default))
        try:
            return int(value)
        except ValueError:
            return default

    def _getfloat(self, section: str, option: str, default: float) -> float:
        value = self._get(section, option, str(default))
        try:
            return float(value)
        except ValueError:
            return default

    def _get_list(self, section: str, option: str, default: List[str]) -> List[str]:
        value = self._get(section, option, ",".join(default))
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or default


settings = Settings()
