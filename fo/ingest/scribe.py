"""Stage 5 — Scribe: diarized, higher-quality transcripts for Tier A episodes.

ElevenLabs fetches the audio itself from the YouTube URL (`source_url`), so no audio is downloaded here.
Keyterms (guest names, companies, jargon from the profile pass) fix the names ASR usually gets wrong."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from elevenlabs.client import AsyncElevenLabs

from .. import db
from ..config import settings

log = logging.getLogger(__name__)

USE_KEYTERMS = os.getenv("SCRIBE_KEYTERMS", "true").lower() == "true"  # +20% cost, worth it for names
MAX_KEYTERMS = 100
_BAD_CHARS = re.compile(r"[<>{}\[\]\\]")


def describe_error(e: Exception) -> str:
    """ElevenLabs' ApiError str() starts with ~500 chars of response headers, which pushes the actual reason
    out of videos.last_error (truncated at 500). Lead with status and body instead."""
    status, body = getattr(e, "status_code", None), getattr(e, "body", None)
    if status is None and body is None:
        return str(e)
    return f"HTTP {status}: {body}"


def clean_keyterms(terms: list[str]) -> list[str]:
    """Scribe constraints: < 50 chars, <= 5 words, none of <>{}[]\\ ."""
    out, seen = [], set()
    for t in terms:
        t = _BAD_CHARS.sub("", (t or "")).strip()
        if not t or len(t) >= 50 or len(t.split()) > 5 or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
        if len(out) >= MAX_KEYTERMS:
            break
    return out


async def _guest_names(video_id: str) -> list[str]:
    links = await db.select("video_guests", "guest_id", video_id=video_id)
    guests = await db.select_in("guests", "name,aliases", "id", [l["guest_id"] for l in links])
    names = []
    for g in guests:
        names.append(g["name"])
        names += g.get("aliases") or []
    return names


async def _one(client: AsyncElevenLabs, video: dict) -> None:
    s = settings()
    vid = video["id"]
    profile = video.get("profile") or {}
    guests = await _guest_names(vid)
    kwargs = dict(
        model_id=s.scribe_model,
        source_url=f"https://www.youtube.com/watch?v={vid}",
        diarize=True,
        timestamps_granularity="word",
        tag_audio_events=False,
        no_verbatim=True,
    )
    n_guests = len(profile.get("guests") or [])
    if n_guests:
        kwargs["num_speakers"] = n_guests + 1
    if video.get("primary_language") == "hi":
        kwargs["language_code"] = "hi"
    elif video.get("primary_language") == "en":
        kwargs["language_code"] = "en"
    if USE_KEYTERMS:
        terms = clean_keyterms([s.host_name, *guests, *(profile.get("keyterms") or [])])
        if terms:
            kwargs["keyterms"] = terms

    try:
        res = await client.speech_to_text.convert(**kwargs, request_options={"timeout_in_seconds": 3600})
    except Exception as e:
        await db.update("videos", {"last_error": f"scribe: {describe_error(e)}"[:500], "attempts": video["attempts"] + 1}, id=vid)
        log.warning("scribe failed %s: %s", vid, describe_error(e))
        return

    data = res.model_dump()
    words = data.get("words") or []
    if not words:
        await db.update("videos", {"last_error": "scribe: empty transcript"}, id=vid)
        log.warning("scribe returned no words for %s", vid)
        return
    await db.put_json(f"scribe/{vid}.json", {
        "language_code": data.get("language_code"),
        "audio_duration_secs": data.get("audio_duration_secs"),
        "keyterms": kwargs.get("keyterms", []),
        "words": [{k: w.get(k) for k in ("text", "start", "end", "type", "speaker_id")} for w in words],
    })
    await db.update("videos", {
        "transcript_source": "scribe_v2",
        "speaker_quality": "diarized",
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
        "processed_at": None,  # re-process with the better transcript
        "last_error": None,
    }, id=vid)
    speakers = {w.get("speaker_id") for w in words if w.get("type") == "word"}
    log.info("scribe ok %s: %d words, %d speakers, %.0f min", vid, len(words), len(speakers),
             (data.get("audio_duration_secs") or 0) / 60)


async def run(limit: int | None = None, video_id: str | None = None) -> None:
    s = settings()
    if not s.elevenlabs_api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    rows = await db.select("videos", "id,tier,profile,primary_language,attempts,transcript_source,duration_s")
    if video_id:
        todo = [r for r in rows if r["id"] == video_id]
    else:
        todo = [r for r in rows if r.get("tier") == "A" and r.get("transcript_source") != "scribe_v2"]
    if limit:
        todo = todo[:limit]
    hours = sum((r.get("duration_s") or 0) for r in todo) / 3600
    log.info("transcribing %d episodes (~%.0f h of audio)", len(todo), hours)
    client = AsyncElevenLabs(api_key=s.elevenlabs_api_key)
    from ..llm import gather_limited
    await gather_limited([_one(client, v) for v in todo], 2)
