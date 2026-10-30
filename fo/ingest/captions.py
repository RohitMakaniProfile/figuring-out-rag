"""Stage 2 — captions: fetch YouTube captions for every episode (free) and keep them in Storage.

Every episode gets captions first because the cheap profile pass and the tier-B corpus both use them.
YouTube blocks most cloud/datacenter IPs: run this from your own machine or set YOUTUBE_PROXY_URL."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from youtube_transcript_api import (
    IpBlocked, NoTranscriptFound, RequestBlocked, TranscriptsDisabled, YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from .. import db
from ..config import settings
from ..textutil import parse_vtt

log = logging.getLogger(__name__)

# Manual tracks first (better quality), then auto-generated. Order inside a list = preference.
PREFERRED = ["en", "en-IN", "hi", "hi-IN"]


class Blocked(RuntimeError):
    pass


def _api() -> YouTubeTranscriptApi:
    proxy = settings().youtube_proxy_url
    if proxy:
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy))
    return YouTubeTranscriptApi()


# Once the transcript API says "IP blocked" every further call waits ~65 s on retries before failing,
# so remember it and go straight to yt-dlp for the rest of the run.
_api_blocked = False
YTDLP_RETRIES = 3


def choose_track(subtitles: dict, automatic: dict) -> tuple[str, bool, str] | None:
    """Pick (yt-dlp track key, is_generated, language) from yt-dlp's caption tables.

    Manual tracks win. For auto captions yt-dlp lists ~150 machine *translations* next to the spoken
    language, so only the `<lang>-orig` track (what was actually said) is accepted; a Hindi episode must
    never end up with an English auto-translation standing in for the transcript."""
    for lang in PREFERRED:
        if subtitles.get(lang):
            return lang, False, lang
    for lang in PREFERRED:
        if automatic.get(f"{lang}-orig"):
            return f"{lang}-orig", True, lang
    return None


def _fetch_ytdlp(video_id: str) -> dict | None:
    import time

    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    if settings().youtube_proxy_url:
        opts["proxy"] = settings().youtube_proxy_url
    last: Exception | None = None
    for attempt in range(YTDLP_RETRIES):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                subs, auto = info.get("subtitles") or {}, info.get("automatic_captions") or {}
                available = sorted({*subs, *(k[:-5] for k in auto if k.endswith("-orig"))})
                pick = choose_track(subs, auto)
                if pick is None:
                    return {"available": available, "language": None, "is_generated": None, "lines": []}
                key, generated, lang = pick
                fmt = next((f for f in (subs if not generated else auto)[key] if f.get("ext") == "vtt"), None)
                if fmt is None:
                    return {"available": available, "language": None, "is_generated": None, "lines": []}
                vtt = ydl.urlopen(fmt["url"]).read().decode("utf-8", "replace")
            return {"available": available, "language": lang, "is_generated": generated,
                    "lines": parse_vtt(vtt), "via": "yt-dlp"}
        except Exception as e:  # 429s are transient: back off and retry
            last = e
            if attempt < YTDLP_RETRIES - 1:
                time.sleep(5 * 3 ** attempt)
    raise Blocked(f"yt-dlp failed too: {last}")


def _fetch(video_id: str) -> dict | None:
    global _api_blocked
    if _api_blocked:
        return _fetch_ytdlp(video_id)
    api = _api()
    try:
        tl = api.list(video_id)
        available = [t.language_code for t in tl]
        chosen = None
        for finder in (tl.find_manually_created_transcript, tl.find_generated_transcript):
            try:
                chosen = finder(PREFERRED)
                break
            except NoTranscriptFound:
                continue
        if chosen is None:
            return {"available": available, "language": None, "is_generated": None, "lines": []}
        fetched = chosen.fetch()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except (IpBlocked, RequestBlocked) as e:
        _api_blocked = True
        log.warning("transcript API blocked (%s); using yt-dlp for the rest of this run", type(e).__name__)
        return _fetch_ytdlp(video_id)
    return {
        "available": available,
        "language": chosen.language_code,
        "is_generated": chosen.is_generated,
        "lines": fetched.to_raw_data(),
    }


async def _one(video: dict) -> None:
    vid = video["id"]
    try:
        data = await asyncio.to_thread(_fetch, vid)
    except Blocked:
        raise
    except Exception as e:
        await db.update("videos", {"last_error": f"captions: {e}"[:500], "attempts": video["attempts"] + 1}, id=vid)
        log.warning("captions failed %s: %s", vid, e)
        return
    now = datetime.now(timezone.utc).isoformat()
    if not data or not data["lines"]:
        await db.update("videos", {"captions_at": now, "caption_langs": (data or {}).get("available", [])}, id=vid)
        log.info("no usable captions: %s", vid)
        return
    await db.put_json(f"captions/{vid}.json", data)
    values = {"captions_at": now, "caption_langs": data["available"], "last_error": None}
    if not video.get("transcript_source"):
        values.update(transcript_source="youtube_captions", speaker_quality="inferred")
    await db.update("videos", values, id=vid)
    log.info("captions ok %s (%s, %d lines)", vid, data["language"], len(data["lines"]))


async def run(limit: int | None = None, video_id: str | None = None) -> None:
    rows = await db.select("videos", "id,transcript_source,attempts,captions_at")
    todo = [r for r in rows if (video_id and r["id"] == video_id) or (not video_id and not r["captions_at"])]
    if limit:
        todo = todo[:limit]
    log.info("fetching captions for %d episodes", len(todo))
    sem = asyncio.Semaphore(2)  # be gentle; YouTube rate-limits aggressively
    blocked = asyncio.Event()

    async def guarded(v):
        async with sem:
            if blocked.is_set():
                return
            try:
                await _one(v)
            except Blocked as e:
                if not blocked.is_set():
                    blocked.set()
                    log.error("YouTube is blocking this IP (%s). Run from a home network or set "
                              "YOUTUBE_PROXY_URL. Fetched episodes are saved; rerun to resume.", e)
                return
            await asyncio.sleep(1.0)

    await asyncio.gather(*(guarded(v) for v in todo))
