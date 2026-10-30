"""Regression tests for bugs found during the pilot review. No network, everything mocked."""
import os
import types

import pytest

for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
          "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ.setdefault(k, "https://example.com")

from fastapi.testclient import TestClient  # noqa: E402

from fo import db  # noqa: E402
from fo.app import main  # noqa: E402
from fo.ingest.process import _turns_from_llm, parse_ts  # noqa: E402
from fo.textutil import Window  # noqa: E402

WIN = Window(600, 900, "x", "chapters")


# --- timestamps copied back with the brackets the prompt showed them in ---------------------------

@pytest.mark.parametrize("raw", ["[06:16]", "(06:16)", " [ 06:16 ] ", "06:16"])
def test_parse_ts_accepts_bracketed_markers(raw):
    assert parse_ts(raw, 300, 900) == 376


def test_bracketed_timestamps_do_not_collapse_to_window_start():
    out = {"turns": [{"start": "[10:00]", "speaker": "A", "role": "guest", "confidence": 0.9, "text": "one"},
                     {"start": "[12:30]", "speaker": "A", "role": "guest", "confidence": 0.9, "text": "two"}]}
    turns = _turns_from_llm(out, WIN, "Host")
    assert [t.start_s for t in turns] == [600, 750]   # before the fix every turn was pinned to 600


# --- non-numeric confidence killed the whole episode ----------------------------------------------

@pytest.mark.parametrize("conf", ["high", "n/a", None, [], "0,9"])
def test_unusable_confidence_means_unattributed_not_a_crash(conf):
    out = {"turns": [{"start": "10:00", "speaker": "A", "role": "guest", "confidence": conf, "text": "hi"}]}
    (t,) = _turns_from_llm(out, WIN, "Host")
    assert (t.speaker_role, t.speaker_name) == ("unknown", None)


def test_non_dict_turn_entries_are_skipped():
    out = {"turns": ["junk", None, {"start": "10:00", "speaker": "A", "role": "guest", "confidence": 1, "text": "ok"}]}
    assert [t.text for t in _turns_from_llm(out, WIN, "Host")] == ["ok"]


# --- Supabase silently returns at most 1000 rows --------------------------------------------------

class FakeQuery:
    def __init__(self, rows, log):
        self.rows, self.log, self.lo, self.hi = rows, log, 0, None

    def select(self, cols):
        return self

    def eq(self, k, v):
        self.rows = [r for r in self.rows if r.get(k) == v]
        return self

    def order(self, col):
        self.log.append(("order", col))
        return self

    def range(self, lo, hi):
        self.lo, self.hi = lo, hi
        return self

    def execute(self):
        # emulate PostgREST: never more than 1000 rows per response
        return types.SimpleNamespace(data=self.rows[self.lo:self.hi + 1][:1000])


def _fake_sb(rows, log):
    return types.SimpleNamespace(table=lambda name: FakeQuery(rows, log))


def test_select_returns_every_row_past_the_1000_cap(monkeypatch):
    import asyncio
    rows = [{"video_id": f"v{i % 600}", "cluster_id": i, "weight": 1} for i in range(2500)]
    log = []
    monkeypatch.setattr(db, "sb", lambda: _fake_sb(rows, log))
    got = asyncio.run(db.select("video_clusters", "video_id,cluster_id,weight"))
    assert len(got) == 2500 and got == rows
    assert [c for _, c in log[:3]] == ["video_id", "cluster_id", "weight"]   # composite key: order by all


def test_select_exact_multiple_of_page_terminates_and_orders_by_id(monkeypatch):
    import asyncio
    rows = [{"id": i} for i in range(2000)]
    log = []
    monkeypatch.setattr(db, "sb", lambda: _fake_sb(rows, log))
    assert len(asyncio.run(db.select("videos", "*"))) == 2000
    assert {c for _, c in log} == {"id"}


def test_select_applies_eq_filter(monkeypatch):
    import asyncio
    rows = [{"id": i, "video_id": "a" if i % 2 else "b"} for i in range(10)]
    monkeypatch.setattr(db, "sb", lambda: _fake_sb(rows, []))
    assert len(asyncio.run(db.select("chunks", "id,video_id", video_id="a"))) == 5


# --- API input limits -----------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_blank_message_is_rejected_before_any_llm_call(client, content):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": content}]})
    assert r.status_code == 422


def test_ui_only_sends_a_bounded_history():
    html = (main.STATIC / "index.html").read_text()
    assert "history.slice(-11)" in html       # a long chat used to hit the API's 40-message cap and die with 422
    assert "messages: history }" not in html


# --- mixed-speaker chunks were presented to the answer model as having a "main speaker" -------------

def test_sources_block_does_not_name_a_main_speaker_for_mixed_chunks():
    from fo.app.retrieval import Source
    mixed = Source("chunk", 1, "v", 130, 235, "Speaker: q?\nMacron: a.", "Emmanuel Macron", "mixed", 1.0, n=1,
                   video_title="T", episode_no=473)
    solo = Source("chunk", 2, "v", 300, 340, "Macron: a.", "Emmanuel Macron", "guest", 1.0, n=2, video_title="T")
    block = main._sources_block([mixed, solo])
    first, second = block.split("\n\n")
    assert "main speaker" not in first and "several speakers" in first
    assert "main speaker: Emmanuel Macron" in second           # single-speaker chunks keep the label


# --- router answered a Roman-script question in Devanagari ------------------------------------------

@pytest.mark.parametrize("message,router_lang,expected", [
    ("Aaj Delhi ka weather kaisa hai?", "hindi", "hinglish"),
    ("आज दिल्ली का मौसम कैसा है?", "hindi", "hindi"),
    ("Khan Sir ne kya bola?", "hinglish", "hinglish"),
    ("What did he say?", "english", "english"),
])
def test_hindi_answer_language_requires_devanagari_input(monkeypatch, message, router_lang, expected):
    import asyncio

    from fo.app import retrieval

    async def fake_json(*a, **k):
        return {"intent": "specific", "answer_language": router_lang}

    monkeypatch.setattr(retrieval.llm, "chat_json", fake_json)
    assert asyncio.run(retrieval.plan_query([], message)).answer_language == expected


# --- profile only ever saw episodes with captions, so a blocked YouTube IP meant no clusters ---------

def test_profile_todo_selection():
    from fo.ingest.profile import pick_todo
    rows = [{"id": "a", "captions_at": "t", "profiled_at": None}, {"id": "b", "captions_at": None, "profiled_at": None},
            {"id": "c", "captions_at": "t", "profiled_at": "t"}, {"id": "d", "captions_at": None, "profiled_at": "t"}]
    ids = lambda **kw: [r["id"] for r in pick_todo(rows, kw.get("video"), kw.get("force", False), kw.get("unc", False))]
    assert ids() == ["a"]                                   # default behaviour unchanged
    assert ids(unc=True) == ["a", "b"]                      # new flag: also uncaptioned, still skips done ones
    assert ids(unc=True, force=True) == ["a", "b", "c", "d"]
    assert ids(video="d") == ["d"]


# --- one content-filtered window failed the whole episode (Azure 400 content_filter) ---------------

def _bad_request(body):
    import httpx
    from openai import BadRequestError
    req = httpx.Request("POST", "http://x")
    return BadRequestError("Error code: 400 - filtered by content management policy",
                           response=httpx.Response(400, request=req), body=body)


def test_content_filter_window_is_skipped_not_fatal():
    import asyncio

    from fo.ingest.process import tolerate_filter

    async def filtered():
        raise _bad_request({"code": "content_filter"})

    async def ok():
        return ["fine"]

    assert asyncio.run(tolerate_filter(filtered(), [], "w1")) == []
    assert asyncio.run(tolerate_filter(ok(), [], "w2")) == ["fine"]


def test_other_errors_still_propagate():
    import asyncio

    from fo.ingest.process import tolerate_filter

    async def other_400():
        raise _bad_request({"code": "invalid_prompt"}).__class__("bad schema", response=_bad_request({}).response, body={})

    async def boom():
        raise RuntimeError("network")

    for coro in (other_400(), boom()):
        with pytest.raises(Exception):
            asyncio.run(tolerate_filter(coro, [], "w"))


def test_one_filtered_window_keeps_the_rest_of_the_gather():
    import asyncio

    from fo import llm
    from fo.ingest.process import tolerate_filter

    async def window(i):
        if i == 1:
            raise _bad_request({"code": "content_filter"})
        return f"w{i}"

    out = asyncio.run(llm.gather_limited([tolerate_filter(window(i), "SKIPPED", f"w{i}") for i in range(3)], 2))
    assert out == ["w0", "SKIPPED", "w2"]


# --- speaker names: a lone guest should not stay anonymous -----------------------------------------

def _t(role, name=None):
    from fo.textutil import Turn
    return Turn(0, 1, role, name, "x")


def test_single_guest_names_every_guest_turn():
    from fo.ingest.process import name_sole_guest
    pw = [[_t("guest"), _t("host", "Raj Shamani"), _t("unknown"), _t("guest", "Someone Else")]]
    name_sole_guest(pw, ["Khan Sir"])
    assert [(t.speaker_role, t.speaker_name) for t in pw[0]] == [
        ("guest", "Khan Sir"), ("host", "Raj Shamani"), ("unknown", None), ("guest", "Khan Sir")]


@pytest.mark.parametrize("guests", [[], ["A", "B"]])
def test_multi_or_no_guest_stays_unnamed(guests):
    from fo.ingest.process import name_sole_guest
    pw = [[_t("guest")]]
    name_sole_guest(pw, guests)
    assert pw[0][0].speaker_name is None


# --- taxonomy on a small corpus ---------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(13, [2, 3, 4]), (6, [2]), (100, list(range(2, 13)))])
def test_k_range_adapts_to_corpus_size(n, expected):
    from fo.ingest import taxonomy
    assert list(taxonomy.k_range(n)) == expected


def test_k_range_empty_when_too_few_episodes():
    from fo.ingest import taxonomy
    assert list(taxonomy.k_range(5)) == []


def test_build_taxonomy_small_corpus_is_a_clear_error_not_a_kmeans_crash(monkeypatch):
    import asyncio

    from fo.ingest import taxonomy

    async def fake_select(table, cols="*", **kw):
        return [{"id": f"v{i}", "title": "t", "profile": {"summary": "s"}} for i in range(4)]

    monkeypatch.setattr(taxonomy.db, "select", fake_select)
    with pytest.raises(RuntimeError, match="need at least"):
        asyncio.run(taxonomy.build_taxonomy())


# --- /api/clusters with no clusters: episode shelves -------------------------------------------------

def test_ui_renders_episode_shelf_fallback():
    html = (main.STATIC / "index.html").read_text()
    assert "shortText(c.description" in html and "c.persona ?" in html


# --- captions have no speaker labels: host/guest needs the strong model and an explicit interview rule -------

@pytest.mark.parametrize("diarized,expected", [(False, "CHAT"), (True, None)])
def test_caption_normalise_uses_chat_deployment(monkeypatch, diarized, expected):
    import asyncio

    from fo.ingest import process
    from fo.textutil import Line

    seen = {}

    async def fake_json(system, user, **kw):
        seen["dep"], seen["system"] = kw.get("deployment"), system
        return {"turns": []}

    monkeypatch.setattr(process.llm, "chat_json", fake_json)
    monkeypatch.setattr(process, "lines_in_range", lambda *a: [object()])
    monkeypatch.setattr(process, "render_lines_for_llm", lambda l: "x")
    monkeypatch.setattr(process, "settings", lambda: types.SimpleNamespace(host_name="Raj", chat_deployment="CHAT"))
    asyncio.run(process._normalise_window(WIN, [], diarized, {"title": "t", "guests": ["G"], "keyterms": []}))
    assert seen["dep"] == expected
    assert ("SPLIT the stretch" in seen["system"]) is (not diarized)   # interview rule only for caption transcripts


def test_turns_cache_key_is_versioned_so_old_mini_turns_are_not_reused():
    from fo.ingest.process import NORMALISE_VERSION, turns_cache_key
    assert turns_cache_key("abc", "youtube_captions") == f"turns/abc.youtube_captions.{NORMALISE_VERSION}.json"
    assert turns_cache_key("abc", "youtube_captions") != "turns/abc.youtube_captions.json"


# --- a stalled Azure call left the chat UI on "Reading..." for minutes (180 s timeout x 5 retries) -----------

def test_chat_stream_uses_a_short_timeout_and_few_retries(monkeypatch):
    import asyncio

    from fo import llm

    seen = {}

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class Completions:
        async def create(self, **kw):
            return Stream()

    class Client:
        chat = types.SimpleNamespace(completions=Completions())

        def with_options(self, **kw):
            seen.update(kw)
            return self

    monkeypatch.setattr(llm, "client", lambda: Client())
    monkeypatch.setattr(llm, "settings", lambda: types.SimpleNamespace(chat_deployment="d", llm_temperature=None))

    async def run():
        return [t async for t in llm.chat_stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(run()) == []
    assert seen == {"timeout": llm.CHAT_TIMEOUT_S, "max_retries": llm.CHAT_RETRIES}
    assert llm.CHAT_TIMEOUT_S <= 60 and llm.CHAT_RETRIES <= 2


def test_ui_has_a_stall_watchdog():
    assert "lastEvent" in (main.STATIC / "index.html").read_text()


# --- UI shows whose thoughts these are: /api/guests ---------------------------------------------------------

def test_ui_shows_guest_names():
    html = (main.STATIC / "index.html").read_text()
    assert "/api/guests" in html and "ep-guest" in html


# --- answers must be short and plain (they used to be long, report-like paragraphs) ------------------------------

def test_answer_prompt_demands_short_plain_answers():
    from fo import prompts
    p = prompts.ANSWER_SYSTEM
    assert "HARD LIMIT" in p and "3 short sentences" in p and "no recap" in p.lower()
    assert "Direct answer:" in p                       # named as something NOT to write
    assert "[S2]" in p                                 # citations are still required
    assert all("simple" in v or "plain" in v for v in prompts.LANGUAGE_INSTRUCTIONS.values())


# --- Supabase HTTP/2 stream resets (httpx.RemoteProtocolError) broke chat requests -------------------------------

def test_db_reads_retry_transport_errors_but_not_other_errors(monkeypatch):
    import asyncio

    import httpx

    real_sleep = asyncio.sleep

    async def no_wait(_):
        await real_sleep(0)

    monkeypatch.setattr(db.asyncio, "sleep", no_wait)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("StreamReset")
        return "ok"

    assert asyncio.run(db.run_read(flaky)) == "ok" and calls["n"] == 3

    calls["n"] = 0

    def always():
        calls["n"] += 1
        raise httpx.RemoteProtocolError("StreamReset")

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(db.run_read(always))
    assert calls["n"] == db.READ_ATTEMPTS

    def bad():
        calls["n"] += 1
        raise ValueError("not transport")

    calls["n"] = 0
    with pytest.raises(ValueError):
        asyncio.run(db.run_read(bad))
    assert calls["n"] == 1


def test_db_writes_are_never_retried():
    import inspect
    src = inspect.getsource(db)
    for fn in ("insert", "upsert", "update", "delete"):
        body = inspect.getsource(getattr(db, fn))
        assert "run_read" not in body, fn


# --- home page data: topics, guests and stats come from what is searchable right now -----------------------------

@pytest.fixture(autouse=True)
def fresh_cache():
    main._CACHE.clear()
    yield
    main._CACHE.clear()


def _stub(monkeypatch, clusters, links=None):
    vids = [{"id": "a", "title": "Ep A", "episode_no": 1, "summary": "About A", "selected": True, "processed_at": "t",
             "popularity_pct": 0, "depth_score": 0},
            {"id": "b", "title": "Ep B", "episode_no": 2, "summary": "About B", "selected": False, "processed_at": "t"},
            {"id": "c", "title": "Ep C", "episode_no": 3, "summary": "About C", "selected": True, "processed_at": None},
            {"id": "d", "title": "Ep D", "episode_no": 4, "summary": "About D", "selected": True, "processed_at": "t",
             "popularity_pct": 0, "depth_score": 0}]
    qa = [{"id": i, "video_id": "a", "question": f"What does Guest A say about topic number {i}?"} for i in range(9)] + \
         [{"id": 100, "video_id": "d", "question": "What is Guest D's main advice for founders?"}, {"id": 101, "video_id": "d", "question": "Why?"}]
    vg = [{"video_id": "a", "guest_id": 1}, {"video_id": "d", "guest_id": 2}, {"video_id": "c", "guest_id": 3},
          {"video_id": "x", "guest_id": 1}]
    default_links = [{"video_id": "a", "cluster_id": 1, "weight": 1}, {"video_id": "b", "cluster_id": 1, "weight": 1},
                     {"video_id": "d", "cluster_id": 1, "weight": 1}]
    calls = {"n": 0}

    async def fake_select(table, cols="*", **kw):
        calls["n"] += 1
        return {"clusters": clusters, "videos": vids, "video_guests": vg}[table]

    async def fake_in(table, cols, key, values):
        if table == "video_clusters":
            return [l for l in (links if links is not None else default_links) if l["video_id"] in values]
        if table == "qa_pairs":
            return [q for q in qa if q["video_id"] in values]
        if table == "guests":
            assert 3 not in values                      # the unprocessed episode's guest is never fetched
            return [{"id": 1, "name": "Zed", "role": "r1"}, {"id": 2, "name": "amy", "role": None}]
        raise AssertionError(table)

    monkeypatch.setattr(main.db, "select", fake_select)
    monkeypatch.setattr(main.db, "select_in", fake_in)
    return calls


CLUSTER = {"id": 1, "slug": "x", "name": "X", "persona": "p", "description": "d", "sample_questions": ["stale Hinglish question?"]}


def test_topics_use_real_questions_and_guests_of_searchable_episodes(client, monkeypatch):
    _stub(monkeypatch, [CLUSTER])
    (c,) = client.get("/api/clusters").json()
    assert c["episode_count"] == 2                       # b is not selected: not counted
    assert "stale Hinglish question?" not in c["sample_questions"]
    assert c["sample_questions"] == ["What does Guest A say about topic number 0?", "What is Guest D's main advice for founders?"]
    assert sorted(c["guests"]) == ["Zed", "amy"] and {e["id"] for e in c["top_episodes"]} == {"a", "d"}


def test_topic_without_searchable_episodes_is_hidden(client, monkeypatch):
    _stub(monkeypatch, [CLUSTER], links=[{"video_id": "b", "cluster_id": 1, "weight": 1}])
    body = client.get("/api/clusters").json()
    assert [s["id"] for s in body] == ["a", "d"]         # no topic has an episode: falls back to per-episode shelves
    assert body[0]["name"] == "Ep A" and body[0]["guests"] == ["Zed"] and len(body[0]["sample_questions"]) == 3


def test_stats_and_guests_count_only_searchable_episodes(client, monkeypatch):
    _stub(monkeypatch, [CLUSTER])
    assert client.get("/api/stats").json() == {"episodes": 2, "guests": 2}
    g = client.get("/api/guests").json()
    assert [x["name"] for x in g] == ["amy", "Zed"] and g[1]["episodes"] == [{"id": "a", "title": "Ep A", "episode_no": 1}]


def test_home_endpoints_are_cached_and_errors_are_not(client, monkeypatch):
    calls = _stub(monkeypatch, [CLUSTER])
    client.get("/api/clusters"); n = calls["n"]
    client.get("/api/clusters")
    assert calls["n"] == n                               # second visit does no database work

    main._CACHE.clear()

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(main.db, "select", boom)
    with pytest.raises(RuntimeError):
        TestClient(main.app, raise_server_exceptions=True).get("/api/clusters")
    assert "clusters" not in main._CACHE


def test_pick_questions_prefers_readable_ones_spread_over_the_episode():
    qs = ["Why?"] + [f"What does the guest say about thing number {i}?" for i in range(9)]
    assert main.pick_questions(qs, 3) == [qs[1], qs[4], qs[7]]
    assert main.pick_questions(["Why?", "How?"], 3) == ["Why?", "How?"]    # nothing readable: still returns something


def test_select_in_pages_past_the_1000_row_cap(monkeypatch):
    import asyncio

    class InQuery(FakeQuery):
        def in_(self, key, values):
            self.rows = [r for r in self.rows if r.get(key) in values]
            return self

    rows = [{"id": i, "video_id": "v1" if i % 2 else "v2"} for i in range(2600)]
    monkeypatch.setattr(db, "sb", lambda: types.SimpleNamespace(table=lambda n: InQuery(rows, [])))
    got = asyncio.run(db.select_in("qa_pairs", "id,video_id", "video_id", ["v1", "v2"]))
    assert len(got) == 2600 and len({r["id"] for r in got}) == 2600


def test_ui_has_loading_and_retry_states_for_topics():
    html = (main.STATIC / "index.html").read_text()
    assert "Retry" in html and "skeleton" in html


def test_pick_questions_skips_second_person_and_already_shown_ones():
    qs = ["What was your experience with the new film and why now?", "What does the guest say about the film industry today?",
          "How does the guest plan to fund the next movie project?"]
    assert main.pick_questions(qs, 3) == qs[1:]                                  # "your ..." only makes sense inside its episode
    assert main.pick_questions(qs, 3, skip={qs[1]}) == [qs[2]]                   # already used by another topic
    assert main.pick_questions(["Why you?"], 1) == ["Why you?"]                  # nothing better: still returns something


def test_topic_guests_are_unique(client, monkeypatch):
    _stub(monkeypatch, [CLUSTER])
    import fo.app.main as m
    # a guest appearing in two episodes of one topic is listed once
    orig = m._guest_names_by_video

    async def dup(ids):
        return {"a": ["Zed"], "d": ["Zed", "amy"]}

    monkeypatch.setattr(m, "_guest_names_by_video", dup)
    (c,) = client.get("/api/clusters").json()
    assert c["guests"] == ["Zed", "amy"]


# --- topics should describe what the chatbot can answer from, not the whole 570-episode archive ---------------------

def test_taxonomy_can_be_limited_to_searchable_episodes(monkeypatch):
    import asyncio

    from fo.ingest import taxonomy

    rows = [{"id": "a", "title": "A", "profile": {"x": 1}, "selected": True, "processed_at": "t"},
            {"id": "b", "title": "B", "profile": {"x": 1}, "selected": True, "processed_at": None},
            {"id": "c", "title": "C", "profile": {"x": 1}, "selected": False, "processed_at": "t"},
            {"id": "d", "title": "D", "profile": None, "selected": True, "processed_at": "t"}]

    async def fake_select(*a, **k):
        return rows

    monkeypatch.setattr(taxonomy.db, "select", fake_select)
    assert [v["id"] for v in asyncio.run(taxonomy.profiled_videos(True))] == ["a"]
    assert [v["id"] for v in asyncio.run(taxonomy.profiled_videos(False))] == ["a", "b", "c"]


def test_topic_membership_uses_main_topic_and_strong_secondary_tags_only():
    links = [{"video_id": "a", "cluster_id": 1, "weight": 0.7}, {"video_id": "a", "cluster_id": 2, "weight": 0.3},
             {"video_id": "b", "cluster_id": 2, "weight": 0.2},                       # only tag: it is the main topic even if weak
             {"video_id": "c", "cluster_id": 1, "weight": 1.0}, {"video_id": "c", "cluster_id": 3, "weight": 0.6},
             {"video_id": "c", "cluster_id": 2, "weight": 0.1}, {"video_id": "d", "cluster_id": 1, "weight": None}]
    got = {(l["video_id"], l["cluster_id"]) for l in main.strong_links(links)}
    assert got == {("a", 1), ("b", 2), ("c", 1), ("c", 3), ("d", 1)}
