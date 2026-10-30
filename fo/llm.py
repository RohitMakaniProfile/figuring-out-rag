"""Thin async wrapper over Azure OpenAI: JSON-mode calls, streaming, batched embeddings."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from openai import AsyncAzureOpenAI, BadRequestError

from .config import settings

log = logging.getLogger(__name__)

EMBED_BATCH = 64
CHAT_TIMEOUT_S = 30   # a person is waiting: a stalled Azure call must fail fast, not sit for the 180 s ingestion timeout x retries
CHAT_RETRIES = 1


@lru_cache
def client() -> AsyncAzureOpenAI:
    s = settings()
    return AsyncAzureOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_key=s.azure_api_key,
        api_version=s.azure_api_version,
        max_retries=5,
        timeout=180,
    )


def _sampling_kwargs() -> dict[str, Any]:
    t = settings().llm_temperature
    return {} if t is None else {"temperature": t}


class LLMJSONError(RuntimeError):
    pass


async def chat_json(
    system: str,
    user: str,
    *,
    deployment: str | None = None,
    max_tokens: int = 4000,
    attempts: int = 2,
) -> dict[str, Any]:
    """Call the model in JSON mode and return the parsed object. Retries once on invalid JSON."""
    dep = deployment or settings().mini_deployment
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for i in range(attempts):
        resp = await client().chat.completions.create(
            model=dep,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
            **_sampling_kwargs(),
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length":
            last_err = LLMJSONError(f"output truncated at max_tokens={max_tokens}")
            max_tokens = int(max_tokens * 1.6)
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_err = e
            log.warning("invalid JSON from %s (attempt %d): %s", dep, i + 1, e)
    raise LLMJSONError(f"model {dep} did not return valid JSON: {last_err}")


async def chat_stream(messages: list[dict[str, str]], *, deployment: str | None = None,
                      max_tokens: int = 1800) -> AsyncIterator[str]:
    dep = deployment or settings().chat_deployment
    stream = await client().with_options(timeout=CHAT_TIMEOUT_S, max_retries=CHAT_RETRIES).chat.completions.create(
        model=dep,
        messages=messages,
        stream=True,
        max_completion_tokens=max_tokens,
        **_sampling_kwargs(),
    )
    async for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches. Raises if the deployment can't produce the configured dimension."""
    s = settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [t if t.strip() else " " for t in texts[i : i + EMBED_BATCH]]
        try:
            resp = await client().embeddings.create(
                model=s.embedding_deployment, input=batch, dimensions=s.embedding_dim
            )
        except BadRequestError as e:
            raise RuntimeError(
                f"Embedding deployment '{s.embedding_deployment}' rejected dimensions={s.embedding_dim}. "
                "Use a text-embedding-3-* deployment (ada-002 does not support custom dimensions)."
            ) from e
        vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if vecs and len(vecs[0]) != s.embedding_dim:
            raise RuntimeError(f"Embedding dim {len(vecs[0])} != EMBEDDING_DIM {s.embedding_dim}")
        out.extend(vecs)
    return out


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]


def to_pgvector(vec: list[float]) -> str:
    """pgvector text literal; PostgREST passes it straight into halfvec input."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


async def gather_limited(coros, limit: int):
    """Run coroutines with a concurrency cap, preserving order."""
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
