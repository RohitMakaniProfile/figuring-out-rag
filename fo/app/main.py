"""FastAPI server: streaming chat over Server-Sent Events + discovery endpoints.

Event order for one answer:  status* -> sources -> token* -> citations -> followups -> done
Run:  uvicorn fo.app.main:app --reload"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from .. import db, llm, prompts
from ..config import settings
from ..textutil import cited_ids, fmt_ts
from .retrieval import Source, plan_query, rerank, retrieve

log = logging.getLogger("fo.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """First visitor should not wait ~6 s for topics: compute the home-page data once at start.
    A failure is fine, the endpoints compute it on demand."""
    async def warm():
        for fn in (stats, guests, clusters):
            try:
                await fn()
            except Exception as e:
                log.warning("home cache warm-up skipped (%s): %s", fn.__name__, type(e).__name__)
    task = asyncio.create_task(warm())
    yield
    task.cancel()


app = FastAPI(title="Figuring Out companion", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "").split(",") if o] or ["http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
STATIC = Path(__file__).parent / "static"


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=40)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sources_block(sources: list[Source]) -> str:
    out = []
    for s in sources:
        ep = f"FO{s.episode_no}" if s.episode_no else "episode"
        who = s.speaker or ("unknown speaker" if s.speaker_role in (None, "unknown") else s.speaker_role)
        if s.kind == "segment":
            kind = "chapter summary"
        elif s.speaker_role == "mixed":
            # several people talk in this passage; a "main speaker" would be read as the author of all of it
            kind = "transcript with several speakers: attribute each line to the name printed before it"
        else:
            kind = f"transcript, main speaker: {who}"
        out.append(f"[S{s.n}] {ep} \"{s.video_title}\" at {fmt_ts(s.start_s)} ({kind})\n{s.text}")
    return "\n\n".join(out)


async def _followups(question: str, answer: str, lang_instr: str) -> list[str]:
    try:
        out = await llm.chat_json(prompts.FOLLOWUP_SYSTEM.format(language_instruction=lang_instr),
                                  f"QUESTION: {question}\n\nANSWER: {answer[:2500]}", max_tokens=300)
        return [f for f in out.get("followups") or [] if isinstance(f, str)][:3]
    except Exception:
        return []


async def _answer_stream(messages: list[Msg]):
    s = settings()
    t0 = time.perf_counter()
    history = [m.model_dump() for m in messages[:-1]]
    question = messages[-1].content
    try:
        yield sse("status", {"text": "Understanding the question"})
        plan = await plan_query(history, question)
        lang_instr = prompts.LANGUAGE_INSTRUCTIONS[plan.answer_language]
        log.info("plan: %s", plan)

        if plan.intent in ("chitchat", "out_of_scope"):
            sys = prompts.CHITCHAT_SYSTEM.format(host=s.host_name, language_instruction=lang_instr)
            msgs = [{"role": "system", "content": sys}, *history[-6:], {"role": "user", "content": question}]
            async for tok in llm.chat_stream(msgs, deployment=s.mini_deployment, max_tokens=400):
                yield sse("token", {"text": tok})
            yield sse("done", {"ms": int((time.perf_counter() - t0) * 1000)})
            return

        who = f" with {', '.join(plan.guests)}" if plan.guests else ""
        yield sse("status", {"text": f"Searching episodes{who}"})
        candidates, guests = await retrieve(plan)
        sources = await rerank(plan.standalone_question, candidates, s.max_context_sources)
        for i, src in enumerate(sources, 1):
            src.n = i
        yield sse("sources", {"sources": [src.public() for src in sources], "intent": plan.intent,
                              "guests": [g["name"] for g in guests]})

        if not sources:
            sys = prompts.CHITCHAT_SYSTEM.format(host=s.host_name, language_instruction=lang_instr)
            note = ("The archive search found nothing relevant to this question. Say so honestly and "
                    "suggest asking about a guest or topic the podcast covers.")
            msgs = [{"role": "system", "content": sys + "\n\n" + note}, {"role": "user", "content": question}]
            async for tok in llm.chat_stream(msgs, deployment=s.mini_deployment, max_tokens=300):
                yield sse("token", {"text": tok})
            yield sse("done", {"ms": int((time.perf_counter() - t0) * 1000)})
            return

        yield sse("status", {"text": f"Reading {len(sources)} moments from "
                                     f"{len({x.video_id for x in sources})} episodes"})
        sys = prompts.ANSWER_SYSTEM.format(host=s.host_name, language_instruction=lang_instr)
        user = (f"SOURCES:\n{_sources_block(sources)}\n\n"
                f"QUESTION (answer this): {plan.standalone_question}\n"
                f"(User's original wording: {question})")
        msgs = [{"role": "system", "content": sys}, *history[-6:], {"role": "user", "content": user}]
        answer, first = [], True
        async for tok in llm.chat_stream(msgs, max_tokens=700):
            if first:
                log.info("first token after %d ms", (time.perf_counter() - t0) * 1000)
                first = False
            answer.append(tok)
            yield sse("token", {"text": tok})

        full = "".join(answer)
        valid = {src.n for src in sources}
        yield sse("citations", {"used": [n for n in cited_ids(full) if n in valid]})
        yield sse("followups", {"items": await _followups(plan.standalone_question, full, lang_instr)})
        yield sse("done", {"ms": int((time.perf_counter() - t0) * 1000)})
    except asyncio.CancelledError:  # client went away
        raise
    except Exception as e:
        log.exception("chat failed")
        yield sse("error", {"text": f"Could not finish the answer: {type(e).__name__}. Try again."})


@app.post("/api/chat")
async def chat(body: ChatIn):
    if body.messages[-1].role != "user":
        body.messages.append(Msg(role="user", content="continue"))
    return StreamingResponse(_answer_stream(body.messages), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL_S = 300   # the home page reads these on every visit; the corpus changes only when someone ingests


async def _cached(key: str, factory):
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1]
    value = await factory()          # an error is raised, never cached
    _CACHE[key] = (time.monotonic(), value)
    return value


async def _live_videos() -> dict[str, dict]:
    vids = await db.select("videos", "id,title,episode_no,summary,popularity_pct,depth_score,selected,processed_at")
    return {v["id"]: v for v in vids if v["selected"] and v["processed_at"]}


async def _guest_names_by_video(video_ids: set[str]) -> dict[str, list[str]]:
    links = [l for l in await db.select("video_guests", "video_id,guest_id") if l["video_id"] in video_ids]
    people = {g["id"]: g["name"] for g in await db.select_in("guests", "id,name", "id", sorted({l["guest_id"] for l in links}))}
    out: dict[str, list[str]] = {}
    for l in links:
        if l["guest_id"] in people:
            out.setdefault(l["video_id"], []).append(people[l["guest_id"]])
    return out


MIN_TOPIC_WEIGHT = 0.5   # the tagger gives 1-3 topics per episode; a 0.1-0.3 tag is a passing mention, not "this episode is about it"


def strong_links(links: list[dict]) -> list[dict]:
    """Each episode's main topic plus any other topic it is substantially about."""
    best: dict[str, float] = {}
    for l in links:
        best[l["video_id"]] = max(best.get(l["video_id"], 0), l["weight"] or 0)
    return [l for l in links if (l["weight"] or 0) >= MIN_TOPIC_WEIGHT or (l["weight"] or 0) == best[l["video_id"]]]


_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.I)   # "what was your experience…": only makes sense inside its episode


def pick_questions(questions: list[str], n: int = 3, skip: set[str] | None = None) -> list[str]:
    """Spread n self-contained, readable questions over an episode instead of taking its first (intro) ones.
    `skip` holds questions already shown elsewhere on the page."""
    skip = skip or set()
    ok = [q.strip() for q in questions if q.strip() and q.strip() not in skip]
    good = [q for q in ok if 25 <= len(q) <= 120 and not _SECOND_PERSON.search(q)]
    pool = good or ok
    step = max(1, len(pool) // n)
    return pool[::step][:n]


async def _questions_by_video(video_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in await db.select_in("qa_pairs", "id,video_id,question", "video_id", video_ids):
        out.setdefault(r["video_id"], []).append(r["question"])
    return out


@app.get("/api/stats")
async def stats():
    async def build():
        live = set(await _live_videos())
        names = await _guest_names_by_video(live)
        # guests of the searchable episodes only: the guests table also holds every unprocessed episode's guest
        return {"episodes": len(live), "guests": len({n for ns in names.values() for n in ns})}
    return await _cached("stats", build)


@app.get("/api/guests")
async def guests():
    """Guests of the searchable episodes, each with the episodes they appear in (so the UI can show whose thoughts these are)."""
    async def build():
        vids = await _live_videos()
        links = [l for l in await db.select("video_guests", "video_id,guest_id") if l["video_id"] in vids]
        people = {g["id"]: g for g in await db.select_in("guests", "id,name,role", "id", sorted({l["guest_id"] for l in links}))}
        out: dict[int, dict] = {}
        for l in links:
            g = people.get(l["guest_id"])
            if not g:
                continue
            v = vids[l["video_id"]]
            entry = out.setdefault(g["id"], {"name": g["name"], "role": g.get("role"), "episodes": []})
            entry["episodes"].append({"id": v["id"], "title": v["title"], "episode_no": v["episode_no"]})
        return sorted(out.values(), key=lambda x: x["name"].lower())
    return await _cached("guests", build)


@app.get("/api/clusters")
async def clusters():
    """Topics for the home page. Every episode, guest and sample question comes from what is searchable right now,
    so a suggested question can always be answered (the taxonomy itself was built on more episodes than are ingested)."""
    async def build():
        cl = await db.select("clusters", "id,slug,name,persona,description")
        vids = await _live_videos()
        links = strong_links(await db.select_in("video_clusters", "video_id,cluster_id,weight", "video_id", sorted(vids)))
        questions = await _questions_by_video(sorted(vids))
        names = await _guest_names_by_video(set(vids))
        out, shown = [], set()
        for c in cl:
            eps = [vids[l["video_id"]] for l in links if l["cluster_id"] == c["id"] and l["video_id"] in vids]
            eps.sort(key=lambda v: (v.get("depth_score") or 0) + (v.get("popularity_pct") or 0), reverse=True)
            if not eps:
                continue
            picks: list[str] = []
            for v in eps:                        # one question per episode first, so a topic shows several guests
                picks += pick_questions(questions.get(v["id"], []), 1, shown | set(picks))
            picks = picks[:3]
            shown |= set(picks)                  # an episode can sit in two topics; do not show the same question twice
            guests_in = list(dict.fromkeys(n for v in eps for n in names.get(v["id"], [])))
            out.append({**c, "sample_questions": picks, "episode_count": len(eps),
                        "guests": guests_in[:6],
                        "top_episodes": [{"id": v["id"], "title": v["title"], "episode_no": v["episode_no"]} for v in eps[:5]]})
        return out or await _episode_shelves(vids, questions, names)
    return await _cached("clusters", build)


async def _episode_shelves(vids: dict[str, dict], questions: dict[str, list[str]], names: dict[str, list[str]]) -> list[dict]:
    """No clusters (taxonomy not built / too few episodes): one shelf per searchable episode, same shape."""
    return [{"id": v["id"], "slug": v["id"], "name": v["title"], "persona": None, "description": v.get("summary"),
             "sample_questions": pick_questions(questions.get(v["id"], []), 3), "episode_count": 1,
             "guests": names.get(v["id"], []),
             "top_episodes": [{"id": v["id"], "title": v["title"], "episode_no": v["episode_no"]}]} for v in vids.values()]


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
