import sys
from pathlib import Path

# 允许直接 `python ui/chat_app.py` 运行：把项目根目录加入导入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from app.config import DEFAULT_PORT
from app.model_runtime import get_runtime_info, split_thinking_partial, stream_reply

SYSTEM_PROMPT = "You are a helpful assistant."


def _api_usage_markdown(model_name: str) -> str:
    base_url = f"http://localhost:{DEFAULT_PORT}/v1"
    return f"""\
本服务对外暴露 **完全兼容 OpenAI 格式** 的接口，先用 `run_api.bat` 启动 API（默认端口 {DEFAULT_PORT}）。

- Base URL：`{base_url}`
- 模型名：`{model_name}`
- API Key：任意非空字符串即可（本地服务不校验）
- 在线文档：`http://localhost:{DEFAULT_PORT}/docs`

**curl**
```bash
curl {base_url}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_name}",
    "messages": [{{"role": "user", "content": "你好"}}],
    "stream": false
  }}'
```

**OpenAI Python SDK**（把 base_url 指过来即可，其它代码不用改）
```python
from openai import OpenAI

client = OpenAI(base_url="{base_url}", api_key="not-needed")
resp = client.chat.completions.create(
    model="{model_name}",
    messages=[{{"role": "user", "content": "你好"}}],
    stream=True,  # 支持流式
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```
"""


def _content_to_text(content) -> str:
    """把 Gradio 各种形态的 content 统一成纯文本。

    Gradio v6 的历史里，content 可能是 str、list（多段/多条消息）或 dict
    （带 type/text 或 metadata 的结构）。直接丢给 apply_chat_template 会因为
    str 拼 list 而报 TypeError，这里递归拍平成字符串。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # 思考过程等带 metadata 的块不计入上下文
        if content.get("metadata"):
            return ""
        return _content_to_text(content.get("content") or content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return "".join(_content_to_text(part) for part in content)
    return str(content)


def _to_model_messages(history: list[dict]) -> list[dict]:
    """把 Gradio 的对话历史转换成模型输入；跳过思考过程折叠块。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        # 思考过程块带 metadata（如 {"title": ...}），不计入上下文
        if item.get("metadata"):
            continue
        content = _content_to_text(item.get("content"))
        if not content:
            continue
        messages.append({"role": item["role"], "content": content})
    return messages


def respond(message, history: list[dict]):
    messages = _to_model_messages(history)
    messages.append({"role": "user", "content": _content_to_text(message)})

    for raw in stream_reply(messages, max_new_tokens=512):
        thinking, answer = split_thinking_partial(raw)
        blocks = []
        if thinking:
            blocks.append(
                gr.ChatMessage(
                    role="assistant",
                    content=thinking,
                    metadata={"title": "💭 思考过程 (Thinking Process)"},
                )
            )
        blocks.append(gr.ChatMessage(role="assistant", content=answer))
        yield blocks


def build_demo() -> gr.Blocks:
    info = get_runtime_info()
    with gr.Blocks(title="Qwen Chat UI", fill_height=True) as demo:
        gr.Markdown(f"# Qwen Chat UI\n模型: {info['model']} | 设备: {info['device']}")
        with gr.Accordion("📡 API 调用说明（OpenAI 兼容）", open=False):
            gr.Markdown(_api_usage_markdown(info["model"]))
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=600, resizable=True, autoscroll=True),
            textbox=gr.Textbox(placeholder="请输入问题", scale=7),
        )
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=8501, inbrowser=True)


if __name__ == "__main__":
    main()
