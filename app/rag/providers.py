"""
Multi-provider LLM factory. Every provider exposes the same tiny interface:

    complete(system_prompt: str, user_prompt: str) -> str

so `agent.py` never needs to know which provider is behind it. Switch
providers by changing LLM_PROVIDER in .env — no other code changes needed.
Also used to benchmark cost/latency/quality across providers (see
scripts/benchmark_providers.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import settings


@dataclass
class CompletionResult:
    text: str
    latency_seconds: float
    provider: str
    model: str


def _complete_openai(system_prompt: str, user_prompt: str) -> CompletionResult:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    model = "gpt-4o-mini"
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return CompletionResult(
        text=resp.choices[0].message.content,
        latency_seconds=time.time() - start,
        provider="openai",
        model=model,
    )


def _complete_claude(system_prompt: str, user_prompt: str) -> CompletionResult:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = "claude-sonnet-4-6"
    start = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return CompletionResult(
        text=text,
        latency_seconds=time.time() - start,
        provider="claude",
        model=model,
    )


def _complete_gemini(system_prompt: str, user_prompt: str) -> CompletionResult:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=settings.google_api_key)

    # Try a few current model names in order. If one is temporarily
    # overloaded (503), fall back to the next before giving up.
    model_candidates = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash"]

    max_retries_per_model = 3
    last_error: Exception | None = None

    for model_name in model_candidates:
        for attempt in range(max_retries_per_model):
            try:
                start = time.time()
                resp = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                )
                return CompletionResult(
                    text=resp.text,
                    latency_seconds=time.time() - start,
                    provider="gemini",
                    model=model_name,
                )
            except genai_errors.ServerError as e:
                # 503 = temporarily overloaded on Google's side. Wait a bit
                # and retry the same model before moving to the next one.
                last_error = e
                if attempt < max_retries_per_model - 1:
                    time.sleep(2 * (attempt + 1))  # 2s, then 4s
                continue

    raise RuntimeError(
        f"All Gemini model candidates were unavailable after retries. Last error: {last_error}"
    )


_PROVIDERS = {
    "openai": _complete_openai,
    "claude": _complete_claude,
    "gemini": _complete_gemini,
}


def complete(system_prompt: str, user_prompt: str, provider: str | None = None) -> CompletionResult:
    provider = provider or settings.llm_provider
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](system_prompt, user_prompt)