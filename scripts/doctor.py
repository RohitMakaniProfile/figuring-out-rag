"""Pre-flight check. Run this first:  python scripts/doctor.py

Verifies every external dependency with one tiny call each and prints what to fix.
Never prints secret values."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

REQUIRED = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "ELEVENLABS_API_KEY"]
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


async def check(name: str, coro, timeout: float = 45):
    try:
        detail = await asyncio.wait_for(coro, timeout)
        return record(name, True, detail or "")
    except asyncio.TimeoutError:
        return record(name, False, f"timed out after {timeout:.0f}s (network blocked or very slow)")
    except Exception as e:  # noqa: BLE001 - doctor reports everything
        msg = f"{type(e).__name__}: {str(e)[:220] or '(no message)'}"
        return record(name, False, msg)


async def azure_chat():
    from fo import llm
    from fo.config import settings
    r = await llm.client().chat.completions.create(
        model=settings().chat_deployment, messages=[{"role": "user", "content": "Reply with: ok"}],
        max_completion_tokens=20)
    return f"deployment '{settings().chat_deployment}' answered"


async def azure_embeddings():
    from fo import llm
    from fo.config import settings
    v = await llm.embed_one("Khan Sir ne China ke baare mein kya kaha?")
    return f"deployment '{settings().embedding_deployment}' returned {len(v)} dims (DB expects {settings().embedding_dim})"


async def supabase_db():
    from fo import db
    rows = await db.select("videos", "id")
    size = await db.rpc("db_size_report", {})
    total = next((r["total_mb"] for r in size if r["relation"] == "DATABASE TOTAL"), "?")
    return f"{len(rows)} episodes in videos, database {total} MB of 500 MB"


async def supabase_storage():
    from fo import db
    await db.put_json("doctor/ping.json", {"ok": True})
    got = await db.get_json("doctor/ping.json")
    assert got == {"ok": True}, "storage round trip failed (does bucket 'transcripts' exist?)"
    return "bucket 'transcripts' read/write works"


async def eleven():
    # models.list() needs no special scope, so it works for restricted keys (user.get() needs user_read).
    from elevenlabs.client import AsyncElevenLabs
    c = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    models = await c.models.list()
    n = len(list(models))
    return f"key accepted ({n} models listed). Speech-to-text permission is only proven by a real scribe run"


async def youtube():
    from fo.ingest.captions import _fetch
    from fo.ingest.captions import Blocked
    try:
        data = await asyncio.to_thread(_fetch, "Y566_T-YlNQ")  # FO556, has English captions
    except Blocked as e:
        raise RuntimeError("YouTube is blocking this IP (IpBlocked)") from e
    if not data or not data["lines"]:
        raise RuntimeError("reached YouTube but no usable en/hi captions")
    return f"{len(data['lines'])} caption lines, language {data['language']}"


async def main() -> int:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        record("env vars", False, "missing " + ", ".join(missing) + "  (copy .env.example to .env)")
        print("\nFix the .env first, then rerun.")
        return 1
    record("env vars", True, "all required variables set")
    await check("azure chat", azure_chat())
    await check("azure embeddings", azure_embeddings())
    await check("supabase database", supabase_db())
    await check("supabase storage", supabase_storage())
    await check("elevenlabs key", eleven())
    await check("youtube captions", youtube(), timeout=150)  # the library retries ~65 s before giving up
    bad = [n for n, ok, _ in results if not ok]
    print("\n" + ("All good. Run: bash scripts/pilot.sh" if not bad else "Fix these first: " + ", ".join(bad)))
    hints = {
        "azure embeddings": "needs a text-embedding-3-* deployment; set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to its deployment name",
        "youtube captions": "YouTube blocks cloud IPs: run from your own machine/home network or set YOUTUBE_PROXY_URL",
        "supabase storage": "create a private bucket named 'transcripts' (migration 001 does this)",
    }
    for n in bad:
        if n in hints:
            print(f"  - {n}: {hints[n]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
