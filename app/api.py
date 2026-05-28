import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model_runtime import generate_reply, get_runtime_info


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="local-qwen3-1.7b-cpu")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


app = FastAPI(
    title="Local Qwen CPU API",
    version="1.0.0",
    description="本地 Qwen3-1.7B CPU 推理服务，提供 OpenAI 兼容聊天接口与 OpenAPI 文档。",
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


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
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
