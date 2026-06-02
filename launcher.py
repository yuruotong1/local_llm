import os
import sys
from pathlib import Path

import modelscope  # noqa: F401
import gradio  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> None:
    app_dir = get_app_dir()
    sys.path.insert(0, str(app_dir))

    torch_extensions_dir = app_dir / "torch_extensions"
    torch_extensions_dir.mkdir(exist_ok=True)

    os.environ["LLM_CODE_BASE_DIR"] = str(app_dir)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
    os.environ.setdefault("GRADIO_SERVER_PORT", "8501")

    from ui.chat_app import build_demo

    build_demo().launch(
        server_name="127.0.0.1",
        server_port=8501,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
