import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
APP_MODEL_NAME = "local-qwen2.5-0.5b-cpu"
DEFAULT_HOST = os.environ.get("LLM_API_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("LLM_API_PORT", "8000"))


def get_base_dir() -> Path:
    return Path(os.environ.get("LLM_CODE_BASE_DIR", Path(__file__).resolve().parents[1]))


def get_models_dir() -> Path:
    return get_base_dir() / "models"


def resolve_device() -> str:
    return "cuda" if os.environ.get("LLM_DEVICE", "cpu").lower() == "cuda" else "cpu"
