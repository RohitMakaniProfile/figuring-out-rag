"""Stage 3 — profile: one cheap LLM call per episode on title + description + chapters + a transcript sample.

Output feeds guest dedup, clustering, depth scoring and Scribe keyterms."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .. import db, llm, prompts
from ..config import settings
from ..textutil import clean_description, fmt_ts

log = logging.getLogger(__name__)
_guest_lock = asyncio.Lock()
GUEST_MATCH_MIN = 0.6  # find_guests score above which two names are treated as the same person


def sample_transcript(lines: list[dict], n_excerpts: int = 6, words_each: int = 450) -> str:
    """Evenly spaced excerpts so the profile sees the whole conversation, not just the intro."""
    if not lines:
        return ""
    excerpts = []
    step = max(1, len(lines) // n_excerpts)
    for i in range(0, len(lines), step):
        words, j, start = [], i, lines[i]["start"]
        while j < len(lines) and len(words) < words_each:
            words += lines[j]["text"].split()
            j += 1
        excerpts.append(f"[{fmt_ts(start)}] " + " ".join(words))
        if len(excerpts) >= n_excerpts:
            break
    return "\n...\n".join(excerpts)


async def upsert_guest(name: str, role: str | None, aliases: list[str]) -> int:
    async with _guest_lock:
        matches = await db.rpc("find_guests", {"q": name, "max_results": 1, "min_score": GUEST_MATCH_MIN})
        if matches:
            gid = matches[0]["id"]
            existing = (await db.select("guests", "aliases,name", id=gid))[0]
            merged = sorted({*existing["aliases"], *aliases, name} - {existing["name"]})
            await db.update("guests", {"aliases": merged}, id=gid)
            return gid
        row = await db.insert("guests", [{"name": name, "role": role, "aliases": sorted(set(aliases) - {name})}])
        return row[0]["id"]


async def _one(video: dict) -> None:
    s = settings()
    vid = video["id"]
    caps = await db.get_json(f"captions/{vid}.json") or {}
    chapters = "\n".join(f"{fmt_ts(c['start_s'])} {c['title']}" for c in video.get("chapters") or [])
    user = (
        f"TITLE: {video['title']}\n\nDESCRIPTION:\n{clean_description(video.get('description'))}\n\n"
        f"CHAPTERS:\n{chapters or '(none)'}\n\n"
        f"TRANSCRIPT SAMPLE ({caps.get('language') or 'no captions'}):\n{sample_transcript(caps.get('lines') or [])}"
    )
    try:
        prof = await llm.chat_json(prompts.PROFILE_SYSTEM.format(host=s.host_name), user, max_tokens=1500)
    except Exception as e:
        await db.update("videos", {"last_error": f"profile: {e}"[:500], "attempts": video["attempts"] + 1}, id=vid)
        log.warning("profile failed %s: %s", vid, e)
        return

    guest_ids = []
    for g in prof.get("guests") or []:
        name = (g.get("name") or "").strip()
        if not name or name.lower() == s.host_name.lower():
            continue
        guest_ids.append(await upsert_guest(name, g.get("role"), [a for a in g.get("aliases") or [] if a]))
    await db.delete("video_guests", video_id=vid)
    if guest_ids:
        await db.upsert("video_guests", [{"video_id": vid, "guest_id": gid} for gid in set(guest_ids)],
                        on_conflict="video_id,guest_id")
    lang = prof.get("primary_language")
    await db.update("videos", {
        "profile": prof,
        "primary_language": lang if lang in ("hi", "en", "mixed") else None,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "last_error": None,
    }, id=vid)
    log.info("profiled %s: %s | density %s", vid, [g.get("name") for g in prof.get("guests") or []],
             prof.get("insight_density"))


def pick_todo(rows: list[dict], video_id: str | None, force: bool, uncaptioned: bool) -> list[dict]:
    if video_id:
        return [r for r in rows if r["id"] == video_id]
    # by default only episodes whose captions are already stored; `uncaptioned` also takes the rest
    # (title + description + chapters are enough to profile and cluster)
    return [r for r in rows if (r["captions_at"] or uncaptioned) and (force or not r["profiled_at"])]


async def run(limit: int | None = None, video_id: str | None = None, force: bool = False,
              uncaptioned: bool = False) -> None:
    rows = await db.select("videos", "id,title,description,chapters,attempts,captions_at,profiled_at")
    todo = pick_todo(rows, video_id, force, uncaptioned)
    if limit:
        todo = todo[:limit]
    log.info("profiling %d episodes", len(todo))
    await llm.gather_limited([_one(v) for v in todo], settings().concurrency)
