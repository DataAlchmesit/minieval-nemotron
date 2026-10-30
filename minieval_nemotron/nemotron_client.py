"""The real model connection: Nemotron on Nebius Token Factory.

decide() takes any function shaped like call_model(model_id, system, user) -> str.
make_call_model() builds that function for Token Factory. Tests use a fake
instead, so nothing here runs without an API key.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from .policy import CascadePolicy
from .prompts import VERDICT_SCHEMA

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def make_call_model(
    policy: Optional[CascadePolicy] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: float = 30.0,
    on_usage: Optional[Callable[[dict], None]] = None,
) -> Callable[[str, str, str], str]:
    """on_usage, if given, receives {model, prompt_tokens, completion_tokens, latency_s} per call."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from openai import OpenAI

    policy = policy or CascadePolicy()
    key = api_key or os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set. Put it in your .env file.")
    client = OpenAI(
        base_url=base_url or os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
        api_key=key,
        timeout=timeout_s,
        max_retries=0,  # the cascade handles retries and fails closed itself
    )

    def call_model(model_id: str, system: str, user: str) -> str:
        kwargs = dict(
            model=model_id,
            temperature=policy.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": VERDICT_SCHEMA},
        )
        if not policy.judge_reasoning:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        if on_usage is not None:
            usage = getattr(resp, "usage", None)
            on_usage({
                "model": model_id,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "latency_s": round(time.perf_counter() - t0, 3),
            })
        return resp.choices[0].message.content or ""

    return call_model