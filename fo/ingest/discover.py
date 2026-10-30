"""Stage 1 — discover: list every episode in the Figuring Out playlist and upsert its metadata.

Uses the official YouTube Data API v3 when YOUTUBE_API_KEY is set (cheap: ~1 quota unit per 50 videos),
otherwise falls back to yt-dlp metadata extraction (no key, slower)."""
from __future__ import annotations

import asyncio
import logging

import httpx

from .. import db
from ..config import settings
from ..textutil import parse_chapters, parse_episode_no, parse_iso8601_duration

log = logging.getLogger(__name__)
YT = "https://www.googleapis.com/youtube/v3"


def _row(video_id: str, title: str, description: str, published_at: str | None, duration_s: int | None,
         views, likes, comments) -> dict:
    return {
        "id": video_id,
        "episode_no": parse_episode_no(title),
        "title": title,
        "description": description,
        "published_at": published_at,
        "duration_s": duration_s,
        "view_count": int(views) if views is not None else None,
        "like_count": int(likes) if likes is not None else None,
        "comment_count": int(comments) if comments is not None else None,
        "chapters": parse_chapters(description, duration_s),
    }


async def _via_data_api(key: str, playlist_id: str) -> list[dict]:
    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as http:
        ids: list[str] = []
        token = None
        while True:
            params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50, "key": key}
            if token:
                params["pageToken"] = token
            r = await http.get(f"{YT}/playlistItems", params=params)
            r.raise_for_status()
            data = r.json()
            ids += [it["contentDetails"]["videoId"] for it in data.get("items", [])]
            token = data.get("nextPageToken")
            if not token:
                break
        log.info("playlist has %d items", len(ids))
        for i in range(0, len(ids), 50):
            r = await http.get(f"{YT}/videos", params={
                "part": "snippet,contentDetails,statistics", "id": ",".join(ids[i:i + 50]), "key": key})
            r.raise_for_status()
            for v in r.json().get("items", []):
                sn, st = v["snippet"], v.get("statistics", {})
                rows.append(_row(
                    v["id"], sn["title"], sn.get("description", ""), sn.get("publishedAt"),
                    parse_iso8601_duration(v["contentDetails"].get("duration")),
                    st.get("viewCount"), st.get("likeCount"), st.get("commentCount"),
                ))
    return rows


def _via_ytdlp(playlist_id: str) -> list[dict]:
    import yt_dlp

    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
        entries = (ydl.extract_info(url, download=False) or {}).get("entries") or []
    rows: list[dict] = []
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        for n, e in enumerate(entries, 1):
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={e['id']}", download=False)
            except Exception as ex:  # private/removed videos
                log.warning("skip %s: %s", e.get("id"), ex)
                continue
            ts = info.get("timestamp")
            published = None
            if ts:
                from datetime import datetime, timezone
                published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            rows.append(_row(info["id"], info.get("title", ""), info.get("description", ""), published,
                             info.get("duration"), info.get("view_count"), info.get("like_count"),
                             info.get("comment_count")))
            if n % 25 == 0:
                log.info("yt-dlp metadata %d/%d", n, len(entries))
    return rows


async def run() -> None:
    s = settings()
    if s.youtube_api_key:
        rows = await _via_data_api(s.youtube_api_key, s.playlist_id)
    else:
        log.info("YOUTUBE_API_KEY not set; using yt-dlp metadata extraction (slower)")
        rows = await asyncio.to_thread(_via_ytdlp, s.playlist_id)

    episodes = [r for r in rows if (r["duration_s"] or 0) >= s.min_episode_seconds]
    log.info("%d videos, %d full episodes (>= %ds)", len(rows), len(episodes), s.min_episode_seconds)
    await db.upsert("videos", episodes, on_conflict="id")
    with_chapters = sum(1 for r in episodes if r["chapters"])
    log.info("upserted %d episodes; %d have description chapters", len(episodes), with_chapters)
