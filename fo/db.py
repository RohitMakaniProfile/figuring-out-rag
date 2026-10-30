"""Supabase access. The sync client is wrapped with asyncio.to_thread so ingestion and the API stay async."""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any

import httpx
from supabase import Client, create_client

from .config import settings


log = logging.getLogger(__name__)
READ_ATTEMPTS = 3


@lru_cache
def sb() -> Client:
    s = settings()
    return create_client(s.supabase_url, s.supabase_key)


async def run(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def run_read(fn, *args, **kwargs):
    """Like run() for READS: a dropped HTTP/2 stream or connection (httpx.TransportError, seen as
    RemoteProtocolError StreamReset) is retried. Never use for writes: a reset after the server applied an insert would duplicate it."""
    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except httpx.TransportError as e:
            if attempt == READ_ATTEMPTS:
                raise
            log.warning("supabase read failed (%s), retry %d/%d", type(e).__name__, attempt, READ_ATTEMPTS - 1)
            await asyncio.sleep(0.4 * attempt)


async def rpc(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    res = await run_read(lambda: sb().rpc(name, params).execute())
    return res.data or []


PAGE = 1000  # PostgREST silently truncates a response at max-rows (1000 on Supabase)


async def select(table: str, columns: str = "*", **eq) -> list[dict[str, Any]]:
    """All matching rows, paged. Ordered so pages never overlap: by `id` when selected, else by every
    selected column (the link tables have composite keys)."""
    cols = [c.strip() for c in columns.split(",")]
    order = ["id"] if columns.strip() == "*" or "id" in cols else cols

    def _page(lo: int):
        q = sb().table(table).select(columns)
        for k, v in eq.items():
            q = q.eq(k, v)
        for c in order:
            q = q.order(c)
        return q.range(lo, lo + PAGE - 1).execute()

    out: list[dict[str, Any]] = []
    lo = 0
    while True:
        rows = (await run_read(_page, lo)).data or []
        out.extend(rows)
        if len(rows) < PAGE:
            return out
        lo += PAGE


async def select_in(table: str, columns: str, key: str, values: list) -> list[dict[str, Any]]:
    """Rows whose `key` is in `values`. Paged per chunk of values: one chunk can match more than PostgREST's 1000-row cap."""
    if not values:
        return []
    cols = [c.strip() for c in columns.split(",")]
    order = ["id"] if columns.strip() == "*" or "id" in cols else cols
    out: list[dict[str, Any]] = []
    for i in range(0, len(values), 200):  # keep URLs short
        part = values[i : i + 200]

        def _page(lo: int, part=part):
            q = sb().table(table).select(columns).in_(key, part)
            for c in order:
                q = q.order(c)
            return q.range(lo, lo + PAGE - 1).execute()

        lo = 0
        while True:
            rows = (await run_read(_page, lo)).data or []
            out.extend(rows)
            if len(rows) < PAGE:
                break
            lo += PAGE
    return out


async def update(table: str, values: dict[str, Any], **eq) -> None:
    def _q():
        q = sb().table(table).update(values)
        for k, v in eq.items():
            q = q.eq(k, v)
        return q.execute()

    await run(_q)


async def upsert(table: str, rows: list[dict[str, Any]], on_conflict: str | None = None,
                 batch: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        res = await run(lambda p=part: sb().table(table).upsert(p, on_conflict=on_conflict).execute())
        out.extend(res.data or [])
    return out


async def insert(table: str, rows: list[dict[str, Any]], batch: int = 200) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        res = await run(lambda p=part: sb().table(table).insert(p).execute())
        out.extend(res.data or [])
    return out


async def delete(table: str, **eq) -> None:
    def _q():
        q = sb().table(table).delete()
        for k, v in eq.items():
            q = q.eq(k, v)
        return q.execute()

    await run(_q)


# ---- Storage: raw transcripts live here, not in the 500 MB database ----

async def put_json(path: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    bucket = settings().storage_bucket
    await run(lambda: sb().storage.from_(bucket).upload(
        path, data, {"content-type": "application/json", "upsert": "true"}))


async def get_json(path: str) -> Any | None:
    bucket = settings().storage_bucket
    try:
        data = await run(lambda: sb().storage.from_(bucket).download(path))
    except Exception:
        return None
    return json.loads(data.decode("utf-8"))
