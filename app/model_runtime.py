import re
import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_runtime_info() -> dict:
    runtime = get_runtime()
    return {
        "model": APP_MODEL_NAME,
        "source_model": MODEL_ID,
        "device": runtime["device"],
        "model_dir": runtime["model_dir"],
    }
