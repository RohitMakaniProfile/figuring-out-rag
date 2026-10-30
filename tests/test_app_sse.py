"""End-to-end test of the /api/chat event stream with the LLM and DB mocked out."""
import json
import os

import pytest

for k, v in {
    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com", "AZURE_OPENAI_API_KEY": "x",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "main", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "emb",
    "SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "x",
}.items():
    os.environ.setdefault(k, v)

from fastapi.testclient import TestClient  # noqa: E402

from fo.app import main, retrieval  # noqa: E402
from fo.app.retrieval import Plan, Source  # noqa: E402


def parse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        ev = next(l[7:] for l in block.splitlines() if l.startswith("event: "))
        data = json.loads(next(l[6:] for l in block.splitlines() if l.startswith("data: ")))
        events.append((ev, data))
    return events


@pytest.fixture
def client(monkeypatch):
    async def fake_plan(history, message):
        intent = "chitchat" if message.lower().startswith("hi") else "specific"
        return Plan(message, "China infrastructure", intent, ["Khan Sir"], [], "hinglish")

    async def fake_retrieve(plan):
        s = Source("chunk", 11, "E3oG313_kps", 2460, 2520, "Khan Sir: China builds roads fast.",
                   "Khan Sir", "guest", 0.9, video_title="Khan Sir Podcast | FO343", episode_no=343, duration_s=7130)
        return [s], [{"id": 1, "name": "Khan Sir"}]

    async def fake_rerank(q, sources, keep):
        return sources

    async def fake_stream(messages, deployment=None, max_tokens=0):
        for tok in ["Khan Sir kehte hain China ", "roads bahut fast banata hai [S1]. ", "Fake [S9]."]:
            yield tok

    async def fake_json(*a, **k):
        return {"followups": ["India ka kya?", "Kab bola?", "Aur kya?"]}

    monkeypatch.setattr(main, "plan_query", fake_plan)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(main, "rerank", fake_rerank)
    monkeypatch.setattr(main.llm, "chat_stream", fake_stream)
    monkeypatch.setattr(main.llm, "chat_json", fake_json)
    return TestClient(main.app)


def test_answer_stream_order_and_citations(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Khan sir ne China pe kya bola?"}]})
    assert r.status_code == 200
    events = parse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "status" and "sources" in kinds and kinds[-1] == "done"
    assert kinds.index("sources") < kinds.index("token") < kinds.index("citations")
    src = dict(events)["sources"]["sources"][0]
    assert src["url"].endswith("&t=2460s") and src["timestamp"] == "41:00" and src["n"] == 1
    assert dict(events)["citations"]["used"] == [1]          # [S9] does not exist -> dropped
    assert len(dict(events)["followups"]["items"]) == 3


def test_chitchat_skips_retrieval(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    kinds = [e for e, _ in parse(r.text)]
    assert "sources" not in kinds and kinds[-1] == "done"


def test_validation(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
