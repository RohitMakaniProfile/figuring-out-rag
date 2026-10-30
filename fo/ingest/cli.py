"""Ingestion CLI.

  python -m fo.ingest.cli discover
  python -m fo.ingest.cli captions [--limit N] [--video ID]
  python -m fo.ingest.cli profile  [--limit N] [--video ID] [--force] [--uncaptioned]
  python -m fo.ingest.cli taxonomy           # clusters + tags (needs profiled episodes)
  python -m fo.ingest.cli taxonomy --searchable   # same, but only for episodes the chatbot can answer from
  python -m fo.ingest.cli score
  python -m fo.ingest.cli select [--all]     # Tier A (or everything with --all)
  python -m fo.ingest.cli scribe   [--limit N] [--video ID]
  python -m fo.ingest.cli process  [--limit N] [--video ID] [--force]
  python -m fo.ingest.cli status
Every stage is idempotent and resumable: rerun it and it picks up where it stopped."""
from __future__ import annotations

import argparse
import asyncio
import logging

from .. import db


async def status() -> None:
    v = await db.select("videos", "id,tier,selected,captions_at,profiled_at,transcript_source,processed_at,last_error")
    n = len(v)

    def c(pred):
        return sum(1 for r in v if pred(r))

    print(f"episodes discovered : {n}")
    print(f"captions fetched    : {c(lambda r: r['captions_at'])}")
    print(f"profiled            : {c(lambda r: r['profiled_at'])}")
    print(f"tier A / selected   : {c(lambda r: r['tier'] == 'A')} / {c(lambda r: r['selected'])}")
    print(f"scribe transcripts  : {c(lambda r: r['transcript_source'] == 'scribe_v2')}")
    print(f"processed (in RAG)  : {c(lambda r: r['processed_at'])}")
    errs = [r for r in v if r.get("last_error")]
    print(f"with errors         : {len(errs)}")
    for r in errs[:10]:
        print(f"   {r['id']}: {r['last_error'][:120]}")
    print("\nstorage (free tier limit 500 MB):")
    for row in await db.rpc("db_size_report", {}):
        if row["relation"] == "DATABASE TOTAL" or float(row["total_mb"]) >= 1:
            print(f"   {row['relation']:<16} {row['total_mb']} MB")


async def publish(video_id: str | None) -> None:
    """Make processed episodes searchable by the chatbot (videos.selected = true)."""
    if not video_id:
        raise SystemExit("publish needs --video ID")
    rows = await db.select("videos", "id,processed_at", id=video_id)
    if not rows or not rows[0]["processed_at"]:
        raise SystemExit(f"{video_id} is not processed yet; run the process stage first")
    await db.update("videos", {"selected": True}, id=video_id)
    print(f"{video_id} is now searchable")


def main() -> None:
    p = argparse.ArgumentParser(description="Figuring Out RAG ingestion")
    p.add_argument("stage", choices=["discover", "captions", "profile", "taxonomy", "tag", "score",
                                     "select", "scribe", "process", "publish", "status"])
    p.add_argument("--limit", type=int)
    p.add_argument("--video")
    p.add_argument("--force", action="store_true")
    p.add_argument("--uncaptioned", action="store_true", help="profile: also episodes without stored captions")
    p.add_argument("--searchable", action="store_true", help="taxonomy/tag: only episodes the chatbot can answer from")
    p.add_argument("--all", action="store_true", help="select: include every episode (tier B uses captions)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from . import captions, discover, process, profile, scribe, taxonomy

    jobs = {
        "discover": lambda: discover.run(),
        "captions": lambda: captions.run(a.limit, a.video),
        "profile": lambda: profile.run(a.limit, a.video, a.force, a.uncaptioned),
        "taxonomy": lambda: taxonomy.build_taxonomy(a.searchable),
        "tag": lambda: taxonomy.tag_all(searchable_only=a.searchable),
        "score": lambda: taxonomy.score_all(),
        "select": lambda: taxonomy.select_corpus(a.all),
        "scribe": lambda: scribe.run(a.limit, a.video),
        "process": lambda: process.run(a.limit, a.video, a.force),
        "publish": lambda: publish(a.video),
        "status": status,
    }
    asyncio.run(jobs[a.stage]())


if __name__ == "__main__":
    main()
