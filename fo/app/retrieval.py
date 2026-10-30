"""Query planning + multi-index retrieval.

intent        indexes used
specific   -> qa_pairs (host questions) + hybrid chunks, filtered to the guest when named
compare    -> hybrid chunks per guest, in parallel
theme      -> insights (advice/books/...) resolved back to their transcript chunks + hybrid chunks
recommend  -> segment summaries ranked by similarity, popularity and depth
Sources passed to the answer model are always transcript text (or segment summaries for recommend),
never the LLM-extracted insight sentence on its own, so every citation points at what was said."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict, dataclass, field

from .. import db, llm, prompts
from ..config import settings
from ..textutil import fmt_ts

log = logging.getLogger(__name__)

REC_W_SIM, REC_W_POP, REC_W_DEPTH = 0.75, 0.15, 0.10


@dataclass
class Plan:
    standalone_question: str
    search_query: str
    intent: str
    guests: list[str] = field(default_factory=list)
    insight_kinds: list[str] = field(default_factory=list)
    answer_language: str = "hinglish"


@dataclass
class Source:
    kind: str                     # chunk | segment
    ref_id: int
    video_id: str
    start_s: int
    end_s: int
    text: str
    speaker: str | None = None
    speaker_role: str | None = None
    score: float = 0.0
    highlight: str | None = None  # insight that led us here (theme questions)
    n: int = 0                    # [S#] number, assigned after rerank
    video_title: str = ""
    episode_no: int | None = None
    duration_s: int | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["timestamp"] = fmt_ts(self.start_s)
        d["url"] = f"https://www.youtube.com/watch?v={self.video_id}&t={self.start_s}s"
        d["snippet"] = self.text[:280]
        d.pop("text")
        return d


_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
VALID_INTENTS = {"specific", "compare", "theme", "recommend", "chitchat", "out_of_scope"}


async def plan_query(history: list[dict], message: str) -> Plan:
    convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-6:])
    out = await llm.chat_json(
        prompts.ROUTER_SYSTEM.format(host=settings().host_name),
        f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\nLATEST USER MESSAGE:\n{message}",
        max_tokens=600,
    )
    intent = out.get("intent") if out.get("intent") in VALID_INTENTS else "specific"
    lang = out.get("answer_language") if out.get("answer_language") in prompts.LANGUAGE_INSTRUCTIONS else "hinglish"
    if lang == "hindi" and not _DEVANAGARI.search(message):
        lang = "hinglish"  # Roman-script Hindi is Hinglish; the router sometimes answers in Devanagari anyway
    return Plan(
        standalone_question=out.get("standalone_question") or message,
        search_query=out.get("search_query") or message,
        intent=intent,
        guests=[g for g in out.get("guests") or [] if isinstance(g, str) and g.strip()],
        insight_kinds=[k for k in out.get("insight_kinds") or [] if isinstance(k, str)],
        answer_language=lang,
    )


async def resolve_guests(names: list[str]) -> list[dict]:
    found = await asyncio.gather(*(db.rpc("find_guests", {"q": n, "max_results": 1}) for n in names))
    out, seen = [], set()
    for rows in found:
        if rows and rows[0]["id"] not in seen:
            seen.add(rows[0]["id"])
            out.append(rows[0])
    return out


def _chunk_source(r: dict, score: float | None = None) -> Source:
    return Source("chunk", r["id"], r["video_id"], r["start_s"], r["end_s"], r["text_en"],
                  r.get("speaker_name"), r.get("speaker_role"), score if score is not None else r.get("score", 0))


async def _chunks_by_ids(ids: list[int]) -> list[dict]:
    return await db.select_in("chunks", "id,video_id,start_s,end_s,text_en,speaker_name,speaker_role,is_ad", "id", ids)


async def _hybrid(emb: str, query: str, n: int, guest_ids=None) -> list[Source]:
    rows = await db.rpc("hybrid_search_chunks", {
        "query_embedding": emb, "query_text": query, "match_count": n, "filter_guest_ids": guest_ids})
    return [_chunk_source(r) for r in rows]


async def retrieve(plan: Plan) -> tuple[list[Source], list[dict]]:
    emb_vec, guests = await asyncio.gather(llm.embed_one(plan.search_query), resolve_guests(plan.guests))
    emb = llm.to_pgvector(emb_vec)
    gids = [g["id"] for g in guests] or None
    sources: list[Source] = []

    if plan.intent == "compare" and guests and len(guests) >= 2:
        per = await asyncio.gather(*(_hybrid(emb, plan.search_query, 6, [g["id"]]) for g in guests))
        sources = [s for group in per for s in group]

    elif plan.intent == "recommend":
        rows = await db.rpc("search_segments", {"query_embedding": emb, "match_count": 14,
                                                "filter_guest_ids": gids})
        for r in rows:
            score = (REC_W_SIM * r["similarity"] + REC_W_POP * (r.get("popularity_pct") or 0)
                     + REC_W_DEPTH * (r.get("depth_score") or 0))
            sources.append(Source("segment", r["id"], r["video_id"], r["start_s"], r["end_s"],
                                  f"Chapter \"{r.get('title') or 'untitled'}\": {r.get('summary') or ''}",
                                  score=score))
        sources.sort(key=lambda s: s.score, reverse=True)

    elif plan.intent == "theme":
        kinds = [k for k in plan.insight_kinds if k in {"advice", "story", "stat", "book", "framework", "prediction"}]
        ins, hyb = await asyncio.gather(
            db.rpc("search_insights", {"query_embedding": emb, "match_count": 20,
                                       "filter_kinds": kinds or None, "filter_guest_ids": gids}),
            _hybrid(emb, plan.search_query, 8, gids),
        )
        chunk_rows = {r["id"]: r for r in await _chunks_by_ids([i["chunk_id"] for i in ins if i.get("chunk_id")])}
        for i in ins:
            r = chunk_rows.get(i.get("chunk_id"))
            if r and not r["is_ad"]:
                s = _chunk_source(r, i["similarity"])
                s.highlight = i["text"]
                sources.append(s)
        sources += hyb

    else:  # specific (and compare with < 2 resolved guests)
        qa, hyb = await asyncio.gather(
            db.rpc("search_qa_pairs", {"query_embedding": emb, "match_count": 6, "filter_guest_ids": gids}),
            _hybrid(emb, plan.search_query, 12, gids),
        )
        ans_ids = [cid for q in qa for cid in (q.get("answer_chunk_ids") or [])]
        qa_chunks = {r["id"]: r for r in await _chunks_by_ids(ans_ids)}
        for q in qa:
            for cid in q.get("answer_chunk_ids") or []:
                r = qa_chunks.get(cid)
                if r and not r["is_ad"]:
                    s = _chunk_source(r, q["similarity"])
                    s.highlight = f"Q: {q['question']}"
                    sources.append(s)
        sources += hyb

    sources = _dedupe(sources)
    await _attach_video_meta(sources)
    return sources, guests


def _dedupe(sources: list[Source]) -> list[Source]:
    best: dict[tuple[str, int], Source] = {}
    for s in sources:
        key = (s.kind, s.ref_id)
        if key not in best:
            best[key] = s
        elif s.highlight and not best[key].highlight:
            best[key].highlight = s.highlight
    return list(best.values())


async def _attach_video_meta(sources: list[Source]) -> None:
    vids = list({s.video_id for s in sources})
    meta = {v["id"]: v for v in await db.select_in("videos", "id,title,episode_no,duration_s", "id", vids)}
    for s in sources:
        m = meta.get(s.video_id) or {}
        s.video_title, s.episode_no, s.duration_s = m.get("title", ""), m.get("episode_no"), m.get("duration_s")


async def rerank(question: str, sources: list[Source], keep: int) -> list[Source]:
    if not sources:
        return []
    if not settings().rerank_enabled or len(sources) <= 3:
        return sources[:keep]
    candidates = sources[:24]
    passages = "\n\n".join(
        f"[{i}] ({s.video_title[:80]} @ {fmt_ts(s.start_s)})\n{s.text[:1200]}" for i, s in enumerate(candidates))
    try:
        out = await llm.chat_json(prompts.RERANK_SYSTEM, f"QUESTION: {question}\n\nPASSAGES:\n{passages}",
                                  max_tokens=300)
        order = [int(i) for i in out.get("relevant_ids") or [] if str(i).isdigit() and int(i) < len(candidates)]
    except Exception as e:
        log.warning("rerank failed, using retrieval order: %s", e)
        return candidates[:keep]
    if not order:
        # nothing judged relevant: give the answer model a few so it can say what *is* covered
        return candidates[:4]
    seen, ranked = set(), []
    for i in order:
        if i not in seen:
            seen.add(i)
            ranked.append(candidates[i])
    return ranked[:keep]
