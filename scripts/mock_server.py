"""Preview the UI with canned data (no Azure / Supabase needed).

  python scripts/mock_server.py   ->  http://localhost:8001"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
          "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ.setdefault(k, "https://example.com")

import uvicorn  # noqa: E402

from fo.app import main  # noqa: E402
from fo.app.retrieval import Plan, Source  # noqa: E402

CLUSTERS = [
    {"id": 1, "slug": "geopolitics-india", "name": "India and the world", "persona": "UPSC aspirants and news junkies",
     "description": "", "sample_questions": ["Khan Sir ne China ke baare mein kya bola?",
                                             "Jaishankar ke hisaab se US visa issue kya hai?"]},
    {"id": 2, "slug": "brain-health", "name": "Brain, focus and sleep", "persona": "Students who want to focus better",
     "description": "", "sample_questions": ["Huberman ka morning routine kya hai?", "Focus 10x kaise karein?"]},
]
LINKS = [{"video_id": "E3oG313_kps", "cluster_id": 1, "weight": 1}, {"video_id": "Y566_T-YlNQ", "cluster_id": 2, "weight": 1}]
VIDEOS = [{"id": "E3oG313_kps", "title": "Khan Sir Podcast | FO343", "episode_no": 343, "popularity_pct": 1, "depth_score": .6,
           "processed_at": "x", "selected": True},
          {"id": "Y566_T-YlNQ", "title": "Andrew Huberman | FO556", "episode_no": 556, "popularity_pct": .9,
           "depth_score": .9, "processed_at": "x", "selected": True}]


async def fake_select(table, columns="*", **eq):
    return {"clusters": CLUSTERS, "video_clusters": LINKS, "videos": VIDEOS, "guests": [{"id": 1}, {"id": 2}]}.get(table, [])


async def fake_plan(history, message):
    return Plan(message, message, "specific", ["Andrew Huberman"], [], "hinglish")


async def fake_retrieve(plan):
    t = "Andrew Huberman: Get sunlight in your eyes within the first hour after waking."
    srcs = [
        Source("chunk", 1, "Y566_T-YlNQ", 376, 460, t, "Andrew Huberman", "guest", .9,
               highlight="Q: What is Huberman's morning cortisol hack?",
               video_title="Andrew Huberman: Become Mentally Dangerous With These Daily Habits | FO556 Raj Shamani",
               episode_no=556, duration_s=10016),
        Source("chunk", 2, "Y566_T-YlNQ", 1441, 1520, "Andrew Huberman: Physiological sigh, two inhales one long exhale.",
               "Andrew Huberman", "guest", .8, video_title="Andrew Huberman: Become Mentally Dangerous | FO556",
               episode_no=556, duration_s=10016),
        Source("chunk", 3, "4Vz6L8B73i4", 900, 980, "Dr Sahar Yousef: Protect the first 90 minutes for deep work.",
               "Sahar Yousef", "guest", .7,
               video_title="Neuroscientist's Guide To 10X Your Focus & Memory | Dr Sahar Yousef | FO559 Raj Shamani",
               episode_no=559, duration_s=10506),
    ]
    return srcs, [{"id": 1, "name": "Andrew Huberman"}]


async def fake_rerank(q, sources, keep):
    return sources


async def fake_stream(messages, deployment=None, max_tokens=0):
    text = ("Huberman ka main point simple hai: subah uthne ke pehle ghante mein **sunlight aankhon mein** jaani chahiye, "
            "isse cortisol ka natural peak sahi time pe aata hai aur din bhar energy rehti hai [S1].\n\n"
            "Overthinking ke liye wo physiological sigh suggest karte hain: do baar inhale, ek lambi exhale [S2].\n\n"
            "Dr Sahar Yousef bhi isi direction mein kehti hain ki din ke pehle 90 minute deep work ke liye bachao [S3].")
    for word in text.split(" "):
        await asyncio.sleep(0.02)
        yield word + " "


async def fake_json(*a, **k):
    return {"followups": ["Sleep ke baare mein Huberman ne kya bola?", "Physiological sigh kab use karein?",
                          "Sahar Yousef ka deep work routine?"]}


main.db.select = fake_select
main.plan_query, main.retrieve, main.rerank = fake_plan, fake_retrieve, fake_rerank
main.llm.chat_stream, main.llm.chat_json = fake_stream, fake_json

if __name__ == "__main__":
    uvicorn.run(main.app, host="127.0.0.1", port=8001, log_level="warning")
