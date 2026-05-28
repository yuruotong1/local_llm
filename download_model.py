import argparse
from pathlib import Path

from modelscope import snapshot_download

from app.config import MODEL_ID, get_models_dir


def find_model_dir(models_dir: Path) -> Path | None:
    for config_file in models_dir.rglob("config.json"):
        if (config_file.parent / "tokenizer_config.json").exists():
            return config_file.parent
    return None


def ensure_model() -> Path:
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    existing = find_model_dir(models_dir)
    if existing:
        return existing

    snapshot_download(MODEL_ID, cache_dir=str(models_dir))
    resolved = find_model_dir(models_dir)
    if not resolved:
        raise FileNotFoundError(f"模型下载完成，但未找到可用目录: {models_dir}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="下载或检查本地模型")
    parser.add_argument("--quiet", action="store_true", help="仅输出模型路径")
    args = parser.parse_args()

    model_dir = ensure_model()
    print(model_dir if args.quiet else f"模型已就绪: {model_dir}")


if __name__ == "__main__":
    main()
