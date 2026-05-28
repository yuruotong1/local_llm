import os
import sys
from pathlib import Path

import modelscope  # noqa: F401
import streamlit  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401
from streamlit.web import cli as stcli


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> None:
    app_dir = get_app_dir()
    app_path = app_dir / "ui" / "chat_app.py"
    torch_extensions_dir = app_dir / "torch_extensions"
    torch_extensions_dir.mkdir(exist_ok=True)

    os.environ["LLM_CODE_BASE_DIR"] = str(app_dir)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--browser.serverAddress=localhost",
        "--browser.serverPort=8501",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
