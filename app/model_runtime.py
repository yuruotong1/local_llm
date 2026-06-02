import re
import time
from functools import lru_cache
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.config import APP_MODEL_NAME, MODEL_ID, resolve_device
from download_model import ensure_model


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking_content(text: str) -> tuple[str | None, str]:
    match = THINK_PATTERN.search(text)
    if not match:
        return None, text.strip()
    thinking = match.group(1).strip() or None
    answer = THINK_PATTERN.sub("", text).strip()
    return thinking, answer


def split_thinking_partial(text: str) -> tuple[str | None, str]:
    """流式场景下解析可能未闭合的 <think> 块。

    返回 (thinking, answer)。当 <think> 尚未闭合时，后续文本全部算作思考过程，
    answer 为空字符串。
    """
    if "<think>" not in text:
        return None, text.strip()
    after = text.split("<think>", 1)[1]
    if "</think>" in after:
        thinking, answer = after.split("</think>", 1)
        return thinking.strip() or None, answer.strip()
    return after.strip() or None, ""


@lru_cache(maxsize=1)
def get_runtime():
    device = resolve_device()
    model_dir = ensure_model()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float32
        ).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", device_map="auto"
        )

    return {"model": model, "tokenizer": tokenizer, "device": device, "model_dir": str(model_dir)}


def generate_reply(
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    runtime = get_runtime()
    model, tokenizer, device = runtime["model"], runtime["tokenizer"], runtime["device"]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    started_at = time.time()
    generated_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.time() - started_at

    completion_ids = generated_ids[0][len(inputs.input_ids[0]):]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    thinking, answer = split_thinking_content(raw_text)

    return {
        "model": APP_MODEL_NAME,
        "source_model": MODEL_ID,
        "device": device,
        "thinking": thinking,
        "answer": answer,
        "usage": {
            "prompt_tokens": int(inputs.input_ids.shape[-1]),
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": int(inputs.input_ids.shape[-1] + completion_ids.shape[-1]),
        },
        "elapsed_seconds": round(elapsed, 3),
    }


def stream_reply(
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """流式生成：逐步 yield 已累计的原始文本（含未闭合的 <think> 块）。"""
    runtime = get_runtime()
    model, tokenizer, device = runtime["model"], runtime["tokenizer"], runtime["device"]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    accumulated = ""
    for token in streamer:
        accumulated += token
        yield accumulated
    thread.join()


def get_runtime_info() -> dict:
    runtime = get_runtime()
    return {
        "model": APP_MODEL_NAME,
        "source_model": MODEL_ID,
        "device": runtime["device"],
        "model_dir": runtime["model_dir"],
    }
