"""Stage 6 — process a selected episode into the retrieval indexes.

transcript (Scribe or captions)
  -> windows (description chapters, long ones split)
  -> normalise each window (LLM: faithful English, speakers, ad flags)       [cached in Storage]
  -> enrich each window (LLM: segment summary, host Q&A pairs, insights)
  -> speaker-aware chunks with a contextual header
  -> embeddings -> segments / chunks / qa_pairs / insights / episode summary"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from openai import BadRequestError

from .. import db, llm, prompts
from ..config import settings
from ..textutil import (
    Turn, Window, build_windows, captions_to_lines, chunk_turns, fmt_ts, hms_to_seconds,
    lines_in_range, render_lines_for_llm, scribe_words_to_lines,
)

log = logging.getLogger(__name__)
LOW_CONF = 0.5  # below this we don't attribute a turn to a named person
INSIGHT_KINDS = {"advice", "story", "stat", "book", "framework", "prediction"}


def parse_ts(value, lo: float, hi: float) -> float | None:
    """'06:16' / '1:02:03' / 376 -> seconds, clamped to the window; None if unusable."""
    try:
        # the prompt shows markers as [mm:ss]; models often copy the brackets back
        sec = float(value) if isinstance(value, (int, float)) else float(hms_to_seconds(str(value).strip().strip("[]() ")))
    except (ValueError, TypeError):
        return None
    if sec < lo - 5 or sec > hi + 5:
        return None
    return min(max(sec, lo), hi)


def is_content_filter(e: Exception) -> bool:
    return isinstance(e, BadRequestError) and (
        getattr(e, "code", None) == "content_filter" or "content management policy" in str(e))


async def tolerate_filter(coro, default, what: str):
    """Azure's content filter rejects some windows of sensitive talk (crime, war, politics). One rejected
    window must not fail the whole episode: log it, keep `default` for that window, carry on."""
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        if is_content_filter(e):
            log.warning("content filter rejected %s; skipping that window", what)
            return default
        raise


# ---------------------------------------------------------------- normalise

async def _load_lines(video: dict):
    if video["transcript_source"] == "scribe_v2":
        data = await db.get_json(f"scribe/{video['id']}.json")
        return scribe_words_to_lines((data or {}).get("words") or []), True
    data = await db.get_json(f"captions/{video['id']}.json")
    return captions_to_lines((data or {}).get("lines") or []), False


def _turns_from_llm(out: dict, win: Window, host: str) -> list[Turn]:
    turns: list[Turn] = []
    last_start = float(win.start_s)
    for t in out.get("turns") or []:
        if not isinstance(t, dict):
            continue
        text = (t.get("text") or "").strip()
        if not text:
            continue
        start = parse_ts(t.get("start"), win.start_s, win.end_s)
        start = last_start if start is None or start < last_start else start
        role = t.get("role") if t.get("role") in ("host", "guest", "unknown") else "unknown"
        try:
            conf = float(t.get("confidence") or 0)
        except (TypeError, ValueError):  # "high", "n/a": unusable means unattributed, not a failed episode
            conf = 0.0
        name = (t.get("speaker") or "").strip() or None
        if role == "host":
            name = host
        if conf < LOW_CONF or (name and name.lower() == "unknown"):
            role, name = ("unknown", None) if conf < LOW_CONF else (role, None)
        turns.append(Turn(start, start, role, name, text, bool(t.get("is_ad"))))
        last_start = start
    for i, t in enumerate(turns):  # end = next start
        t.end_s = turns[i + 1].start_s if i + 1 < len(turns) else float(win.end_s)
    return turns


async def _normalise_window(win: Window, lines, diarized: bool, ctx: dict) -> list[Turn]:
    s = settings()
    seg_lines = lines_in_range(lines, win.start_s, win.end_s)
    if not seg_lines:
        return []
    system = prompts.NORMALIZE_SYSTEM.format(
        host=s.host_name,
        guests=", ".join(ctx["guests"]) or "unknown",
        source_note="diarized speech-to-text output" if diarized else
                    "auto-generated YouTube captions (no punctuation, no speaker labels, may contain errors)",
        keyterms=", ".join(ctx["keyterms"][:40]) or "-",
        speaker_rule=prompts.NORMALIZE_RULE_DIARIZED if diarized else prompts.NORMALIZE_RULE_CAPTIONS,
        win_start=fmt_ts(win.start_s), win_end=fmt_ts(win.end_s),
    )
    header = f"EPISODE: {ctx['title']}\nCHAPTER: {win.title or '(none)'}\n\n"
    # captions carry no speaker labels, so telling host from guest needs the stronger model (mini labelled everything "guest")
    out = await llm.chat_json(system, header + render_lines_for_llm(seg_lines), max_tokens=8000,
                              deployment=None if diarized else s.chat_deployment)
    return _turns_from_llm(out, win, s.host_name)


# ---------------------------------------------------------------- enrich

NORMALISE_VERSION = "v2"  # bump when the normalise prompt/model changes so stale cached turns are not reused


def turns_cache_key(vid: str, source: str) -> str:
    return f"turns/{vid}.{source}.{NORMALISE_VERSION}.json"


def name_sole_guest(per_window: list[list[Turn]], guests: list[str]) -> None:
    """With exactly one guest in the episode, every guest-role turn is that guest; several guests stay unnamed."""
    if len(guests) != 1:
        return
    for turns in per_window:
        for t in turns:
            if t.speaker_role == "guest":
                t.speaker_name = guests[0]


def _render_turns(turns: list[Turn]) -> str:
    return "\n".join(
        f"[{fmt_ts(t.start_s)}] {t.speaker_name or t.speaker_role}: {t.text}" for t in turns if not t.is_ad)


async def _enrich_window(win: Window, turns: list[Turn], ctx: dict) -> dict:
    body = _render_turns(turns)
    if not body:
        return {"summary": "", "qa_pairs": [], "insights": []}
    system = prompts.ENRICH_SYSTEM.format(host=settings().host_name, guests=", ".join(ctx["guests"]) or "unknown")
    return await llm.chat_json(system, f"CHAPTER: {win.title or '(untitled)'}\n\n{body}", max_tokens=4000)


# ---------------------------------------------------------------- main

async def process_video(video: dict) -> None:
    s = settings()
    vid = video["id"]
    lines, diarized = await _load_lines(video)
    if not lines:
        raise RuntimeError("no transcript lines")
    duration = video.get("duration_s") or int(lines[-1].end) + 1
    links = await db.select("video_guests", "guest_id", video_id=vid)
    guests = await db.select_in("guests", "name", "id", [l["guest_id"] for l in links])
    profile = video.get("profile") or {}
    ctx = {
        "title": video["title"],
        "guests": [g["name"] for g in guests],
        "keyterms": [g["name"] for g in guests] + (profile.get("keyterms") or []),
    }
    windows = build_windows(video.get("chapters") or [], duration)

    # normalise (cached so re-runs of enrich/embedding don't pay for translation again)
    cache_key = turns_cache_key(vid, video["transcript_source"])
    cached = await db.get_json(cache_key)
    if cached and len(cached) == len(windows):
        per_window = [[Turn(**t) for t in w] for w in cached]
    else:
        per_window = await llm.gather_limited(
            [tolerate_filter(_normalise_window(w, lines, diarized, ctx), [],
                            f"normalise {vid} {fmt_ts(w.start_s)}-{fmt_ts(w.end_s)}") for w in windows], s.concurrency)
        await db.put_json(cache_key, [[t.__dict__ for t in w] for w in per_window])
    name_sole_guest(per_window, ctx["guests"])

    enriched = await llm.gather_limited(
        [tolerate_filter(_enrich_window(w, t, ctx), {"summary": "", "qa_pairs": [], "insights": []},
                         f"enrich {vid} {fmt_ts(w.start_s)}-{fmt_ts(w.end_s)}") for w, t in zip(windows, per_window)],
        s.concurrency)

    # chunks per window
    ep = f"FO{video['episode_no']}" if video.get("episode_no") else "Figuring Out"
    header_base = f"Episode {ep}: {video['title']}\nGuests: {', '.join(ctx['guests']) or 'unknown'}\n"
    window_chunks = []
    for w, turns in zip(windows, per_window):
        cs = chunk_turns(turns)
        window_chunks.append([(c, f"{header_base}Chapter: {w.title or 'untitled'}\n\n{c.text}") for c in cs])

    # episode summary from segment summaries
    seg_summaries = "\n".join(f"[{fmt_ts(w.start_s)}] {w.title or ''}: {e.get('summary', '')}"
                              for w, e in zip(windows, enriched) if e.get("summary"))
    ep_sum = await llm.chat_json(prompts.EPISODE_SUMMARY_SYSTEM,
                                 f"EPISODE: {video['title']}\n\n{seg_summaries}", max_tokens=1500)

    # embeddings, one batched pass per kind
    seg_texts = [f"{video['title']} | {w.title or ''}\n{e.get('summary', '')}" for w, e in zip(windows, enriched)]
    chunk_texts = [ctxt for cs in window_chunks for _, ctxt in cs]
    qa_items = [(i, q) for i, e in enumerate(enriched) for q in (e.get("qa_pairs") or []) if q.get("question")]
    ins_items = [(i, x) for i, e in enumerate(enriched) for x in (e.get("insights") or [])
                 if x.get("text") and x.get("kind") in INSIGHT_KINDS]
    seg_emb = await llm.embed(seg_texts)
    chunk_emb = await llm.embed(chunk_texts)
    qa_emb = await llm.embed([q["question"] for _, q in qa_items])
    ins_emb = await llm.embed([x["text"] for _, x in ins_items])
    ep_emb = await llm.embed_one(f"{video['title']}\n{ep_sum.get('summary', '')}")

    # replace previous rows for this episode
    for table in ("insights", "qa_pairs", "chunks", "segments"):
        await db.delete(table, video_id=vid)

    seg_rows = await db.insert("segments", [{
        "video_id": vid, "idx": i, "start_s": w.start_s, "end_s": w.end_s, "title": w.title,
        "summary": e.get("summary"), "source": w.source, "embedding": llm.to_pgvector(emb),
    } for i, (w, e, emb) in enumerate(zip(windows, enriched, seg_emb))])
    seg_id_by_idx = {r["idx"]: r["id"] for r in seg_rows}

    chunk_rows, k = [], 0
    for wi, cs in enumerate(window_chunks):
        for c, ctxt in cs:
            chunk_rows.append({
                "video_id": vid, "segment_id": seg_id_by_idx[wi], "idx": k,
                "speaker_role": c.speaker_role, "speaker_name": c.speaker_name,
                "speaker_conf": 1.0 if diarized else None,
                "start_s": c.start_s, "end_s": max(c.end_s, c.start_s), "text_en": c.text,
                "contextual_text": ctxt, "is_ad": c.is_ad, "embedding": llm.to_pgvector(chunk_emb[k]),
            })
            k += 1
    inserted = await db.insert("chunks", chunk_rows)
    inserted.sort(key=lambda r: r["idx"])
    chunks_by_window: dict[int, list[dict]] = {}
    for r in inserted:
        wi = next(i for i, sid in seg_id_by_idx.items() if sid == r["segment_id"])
        chunks_by_window.setdefault(wi, []).append(r)

    def chunk_at(wi: int, t: float) -> dict | None:
        cs = chunks_by_window.get(wi) or []
        hit = None
        for c in cs:
            if c["start_s"] <= t:
                hit = c
        return hit or (cs[0] if cs else None)

    qa_rows = []
    for (wi, q), emb in zip(qa_items, qa_emb):
        w = windows[wi]
        t = parse_ts(q.get("start"), w.start_s, w.end_s) or w.start_s
        first = chunk_at(wi, t)
        ids = []
        if first:
            cs = chunks_by_window[wi]
            pos = cs.index(first)
            ids = [c["id"] for c in cs[pos:pos + 2]]  # question chunk + the one after (answers run long)
        qa_rows.append({"video_id": vid, "segment_id": seg_id_by_idx[wi], "question": q["question"],
                        "answer_summary": q.get("answer_summary"), "answer_chunk_ids": ids,
                        "start_s": int(t), "embedding": llm.to_pgvector(emb)})
    if qa_rows:
        await db.insert("qa_pairs", qa_rows)

    ins_rows = []
    for (wi, x), emb in zip(ins_items, ins_emb):
        w = windows[wi]
        t = parse_ts(x.get("start"), w.start_s, w.end_s) or w.start_s
        c = chunk_at(wi, t)
        ins_rows.append({"video_id": vid, "chunk_id": c["id"] if c else None, "kind": x["kind"],
                         "text": x["text"], "speaker_name": x.get("speaker"), "start_s": int(t),
                         "embedding": llm.to_pgvector(emb)})
    if ins_rows:
        await db.insert("insights", ins_rows)

    await db.update("videos", {
        "summary": ep_sum.get("summary"), "takeaways": ep_sum.get("takeaways"),
        "summary_embedding": llm.to_pgvector(ep_emb),
        "processed_at": datetime.now(timezone.utc).isoformat(), "last_error": None,
    }, id=vid)
    log.info("processed %s: %d segments, %d chunks, %d Q&A, %d insights",
             vid, len(seg_rows), len(chunk_rows), len(qa_rows), len(ins_rows))


async def run(limit: int | None = None, video_id: str | None = None, force: bool = False) -> None:
    rows = await db.select("videos", "*")
    if video_id:
        todo = [r for r in rows if r["id"] == video_id]
    else:
        todo = [r for r in rows if r["selected"] and r.get("transcript_source")
                and (force or not r.get("processed_at"))]
        # Tier A first: they are the ones users will hit most
        todo.sort(key=lambda r: (r.get("tier") != "A", -(r.get("popularity_pct") or 0)))
    if limit:
        todo = todo[:limit]
    log.info("processing %d episodes", len(todo))
    for v in todo:
        try:
            await process_video(v)
        except Exception as e:
            log.exception("process failed %s", v["id"])
            await db.update("videos", {"last_error": f"process: {e}"[:500], "attempts": v["attempts"] + 1},
                            id=v["id"])
