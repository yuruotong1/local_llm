# Local Qwen CPU

一个 **CPU 可跑的独立大模型**：基于 `Qwen/Qwen2.5-0.5B-Instruct`，本地启动，对外暴露 OpenAI 兼容的 OpenAPI 接口，也带一个 Gradio 聊天 UI（多轮、固定高度可滚动、流式输出）。

仓库本身不带模型文件，第一次运行或打包时会自动从 ModelScope 下载。

## 特点

- 默认走 CPU，没显卡也能跑
- 自带 FastAPI 服务，路径完全兼容 OpenAI `/v1/chat/completions`
- 自动生成 OpenAPI 文档：`http://localhost:8000/docs`
- 模型不入库，缺模型时自动下载
- 一键打包成 Windows 可分发文件夹

## 安装

```bat
pip install -r requirements.txt
```

## 启动 API

```bat
run_api.bat
```

地址：

- 服务根：`http://localhost:8000`
- OpenAPI 文档：`http://localhost:8000/docs`
- OpenAPI JSON：`http://localhost:8000/openapi.json`

首次启动如果检测不到 `models/` 下的模型，会自动调用 `download_model.py` 下载。

## 启动 UI（可选）

```bat
run_ui.bat
```

地址：`http://localhost:8501`

UI 基于 Gradio `ChatInterface`：消息区固定高度、可滚动，输入框始终钉在底部，支持多轮和流式输出，长对话不再被顶出可视区。

## 调用示例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-qwen2.5-0.5b-cpu",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好，介绍一下你自己"}
    ]
  }'
```

接口列表：

- `GET  /health`
- `GET  /v1/models`
- `POST /v1/chat/completions`

## 手动下载模型

```bat
python download_model.py
```

## 一键打包

```bat
build_package.bat        :: 快速打包，复用缓存
build_package.bat full   :: 完整重建
```

打包流程：

1. （full 模式）安装依赖
2. 检查模型，缺则自动下载
3. PyInstaller 构建可执行文件
4. 把 `app/`、`ui/`、`download_model.py`、`models/` 复制到 `dist\QwenChatUI\`

产物在 `dist\QwenChatUI\`，发布请保留整个文件夹。

## 切换到 GPU

```bat
set LLM_DEVICE=cuda
```

之后再启动 API 或 UI。

## 目录

```
app/
  api.py            FastAPI 服务，提供 OpenAPI
  model_runtime.py  模型加载与推理
  config.py         公共配置
ui/chat_app.py      Gradio UI
api_server.py       API 启动入口
launcher.py         打包后 UI 启动入口
download_model.py   模型下载 / 检查
build_package.bat   打包脚本
```

## 为什么仓库里没有模型

`.gitignore` 已排除：

```
models/
dist/
build/
build_work/
```

GitHub 上只放代码，模型在第一次运行或打包时自动补齐。
