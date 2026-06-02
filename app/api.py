import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import APP_MODEL_NAME
from app.model_runtime import (
    generate_reply,
    get_runtime_info,
    split_thinking_partial,
    stream_reply,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=APP_MODEL_NAME)
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


app = FastAPI(
    title="Local Qwen CPU API",
    version="1.0.0",
    description="本地 Qwen2.5-0.5B CPU 推理服务，提供 OpenAI 兼容聊天接口与 OpenAPI 文档。",
)


@app.get("/health")
def health():
    return {"status": "ok", **get_runtime_info()}


@app.get("/v1/models")
def list_models():
    info = get_runtime_info()
    return {
        "object": "list",
        "data": [{"id": info["model"], "object": "model", "owned_by": "local"}],
    }


def _stream_chat_completion(request: ChatCompletionRequest) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": APP_MODEL_NAME,
    }

    def event_stream():
        # 首个 chunk 带 role，符合 OpenAI 流式格式
        first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

        sent = ""
        for raw in stream_reply(
            messages=[m.model_dump() for m in request.messages],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            _, answer = split_thinking_partial(raw)
            if len(answer) > len(sent):
                delta = answer[len(sent):]
                sent = answer
                chunk = {**base, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return _stream_chat_completion(request)

    result = generate_reply(
        messages=[m.model_dump() for m in request.messages],
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": result["answer"]},
            }
        ],
        "usage": result["usage"],
    }
