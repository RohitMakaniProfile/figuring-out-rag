"""Retrieval/answer sanity eval. Extend eval/questions.jsonl to ~80 questions covering every intent.

  python -m eval.run_eval

Checks per question (no LLM judge, so it is cheap and deterministic):
  intent       the router picked the expected intent
  guests       cited sources include every expected guest (by speaker name or episode title)
  cited        the answer cites at least one source (not for out_of_scope), and every [S#] exists
  grounded     for out_of_scope: no sources were retrieved
`unanswerable: true` + `absent: [keywords]` = the corpus cannot answer; passes only if no cited passage mentions them
(the start of the answer is printed so a human can confirm it actually refuses).
Add `expect_text` (a keyword that must appear in a cited chunk) to a question for timestamp-correctness checks."""
import asyncio
import json
import pathlib
import re

from fo import db, llm, prompts
from fo.app.retrieval import plan_query, rerank, retrieve
from fo.config import settings


async def _guest_names(video_ids: set[str]) -> dict[str, str]:
    links = await db.select_in("video_guests", "video_id,guest_id", "video_id", sorted(video_ids))
    names = {g["id"]: g["name"] for g in await db.select_in("guests", "id,name", "id", sorted({l["guest_id"] for l in links}))}
    out: dict[str, str] = {}
    for l in links:
        out[l["video_id"]] = f"{out.get(l['video_id'], '')} {names.get(l['guest_id'], '')}".strip()
    return out


def name_in(name: str, haystack: str) -> bool:
    """Every real word of the guest's name appears (not just the last word: 'Singh' is three guests)."""
    return all(t in haystack for t in (w for w in name.lower().replace(".", "").split() if len(w) > 2))


async def run_one(case: dict) -> dict:
    s = settings()
    plan = await plan_query([], case["q"])
    want = case["intent"] if isinstance(case["intent"], list) else [case["intent"]]
    res = {"q": case["q"], "intent_ok": plan.intent in want, "plan_intent": plan.intent}
    if plan.intent in ("chitchat", "out_of_scope"):
        res["grounded_ok"] = "out_of_scope" in want
        res["pass"] = res["intent_ok"]
        return res
    cands, _ = await retrieve(plan)
    sources = await rerank(plan.standalone_question, cands, s.max_context_sources)
    for i, src in enumerate(sources, 1):
        src.n = i
    block = "\n\n".join(f"[S{x.n}] {x.video_title} @ {x.start_s}s ({x.speaker})\n{x.text}" for x in sources)
    sys = prompts.ANSWER_SYSTEM.format(host=s.host_name, language_instruction=prompts.LANGUAGE_INSTRUCTIONS[plan.answer_language])
    answer = "".join([t async for t in llm.chat_stream(
        [{"role": "system", "content": sys}, {"role": "user", "content": f"SOURCES:\n{block}\n\nQUESTION: {plan.standalone_question}"}])])
    guest_names = await _guest_names({x.video_id for x in sources})
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", answer)}
    valid = {x.n for x in sources}
    cited_sources = [x for x in sources if x.n in cited]
    # guest = the episode's guest (video_guests), not the chunk's speaker label: captions leave many chunks unnamed
    haystack = " ".join(f"{x.speaker or ''} {x.video_title} {guest_names.get(x.video_id, '')}" for x in cited_sources).lower()
    if case.get("unanswerable"):  # not in the corpus: no cited passage may mention the subject (answer text is printed)
        res["refused_ok"] = not any(k.lower() in x.text.lower() for x in cited_sources for k in case.get("absent", []))
        res["answer_start"] = answer[:140].replace("\n", " ")
        res["pass"] = res["intent_ok"] and res["refused_ok"]
        return res
    res["cited_ok"] = bool(cited) and cited <= valid
    if not res["cited_ok"]:
        res["answer_start"] = answer[:200].replace("\n", " ")
    res["guests_ok"] = all(name_in(g, haystack) for g in case.get("expect_guests", []))
    if case.get("expect_text"):
        res["text_ok"] = any(case["expect_text"].lower() in x.text.lower() for x in cited_sources)
    res["pass"] = res["intent_ok"] and res["cited_ok"] and res["guests_ok"] and res.get("text_ok", True)
    return res


async def main():
    cases = [json.loads(l) for l in pathlib.Path(__file__).with_name("questions.jsonl").read_text().splitlines() if l.strip()]
    results = await llm.gather_limited([run_one(c) for c in cases], 3)
    for r in results:
        print("PASS" if r["pass"] else "FAIL", r)
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
