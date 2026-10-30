"""Caption fetching: transcript-API blocked -> yt-dlp fallback. No network, everything mocked."""
import os
import sys
import types

import pytest

for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
          "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ.setdefault(k, "https://example.com")

from youtube_transcript_api import IpBlocked, NoTranscriptFound  # noqa: E402

from fo.ingest import captions  # noqa: E402
from fo.textutil import captions_to_lines  # noqa: E402

VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello there

00:00:14.000 --> 00:00:16.000
>> Well, they knew
"""


@pytest.fixture(autouse=True)
def reset_blocked(monkeypatch):
    monkeypatch.setattr(captions, "_api_blocked", False)


def test_choose_track_prefers_manual():
    assert captions.choose_track({"en": [{}]}, {"en-orig": [{}]}) == ("en", False, "en")


def test_choose_track_hindi_uses_orig_not_translation():
    # a Hindi episode: yt-dlp lists an English *auto-translation* ("en") next to the spoken "hi-orig"
    auto = {"en": [{}], "hi": [{}], "hi-orig": [{}], "fr": [{}]}
    assert captions.choose_track({}, auto) == ("hi-orig", True, "hi")


def test_choose_track_rejects_translations_only():
    assert captions.choose_track({}, {"en": [{}], "hi": [{}]}) is None
    assert captions.choose_track({}, {}) is None


class _FakeApi:
    def __init__(self, exc_on="list"):
        self.exc_on, self.list_calls = exc_on, 0

    def list(self, video_id):
        self.list_calls += 1
        if self.exc_on == "list":
            raise IpBlocked(video_id)
        return _FakeList(self.exc_on)


class _FakeTranscript:
    language_code, is_generated = "en", True

    def fetch(self):
        raise IpBlocked("v")


class _FakeList:
    def __init__(self, mode):
        self.mode = mode

    def __iter__(self):
        return iter([_FakeTranscript()])

    def find_manually_created_transcript(self, langs):
        raise NoTranscriptFound("v", langs, None)

    def find_generated_transcript(self, langs):
        return _FakeTranscript()


@pytest.mark.parametrize("where", ["list", "fetch"])
def test_ip_block_falls_back_to_ytdlp_and_remembers(monkeypatch, where):
    api = _FakeApi(exc_on=where)
    monkeypatch.setattr(captions, "_api", lambda: api)
    monkeypatch.setattr(captions, "_fetch_ytdlp", lambda vid: {"language": "en", "lines": [vid], "via": "yt-dlp"})
    assert captions._fetch("AAA")["via"] == "yt-dlp"      # IpBlocked in list() AND in fetch() are both caught
    assert captions._api_blocked is True
    calls = api.list_calls
    assert captions._fetch("BBB")["lines"] == ["BBB"]
    assert api.list_calls == calls                       # blocked API is not retried (each retry costs ~65 s)


def _fake_yt_dlp(monkeypatch, info, fail_times=0):
    state = {"extract": 0}

    class FakeYDL:
        def __init__(self, opts):
            state["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            state["extract"] += 1
            if state["extract"] <= fail_times:
                raise RuntimeError("HTTP Error 429")
            return info

        def urlopen(self, url):
            return types.SimpleNamespace(read=lambda: VTT.encode())

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setattr("time.sleep", lambda s: None)
    return state


def test_ytdlp_returns_parsed_lines(monkeypatch):
    info = {"subtitles": {}, "automatic_captions": {
        "en-orig": [{"ext": "json3", "url": "u0"}, {"ext": "vtt", "url": "u1"}], "en": [{"ext": "vtt", "url": "u2"}]}}
    _fake_yt_dlp(monkeypatch, info)
    d = captions._fetch_ytdlp("AAA")
    assert d["language"] == "en" and d["is_generated"] is True and d["available"] == ["en"]
    assert [r["text"] for r in d["lines"]] == ["Hello there", ">> Well, they knew"]
    assert captions_to_lines(d["lines"])[1].turn_start is True


def test_ytdlp_no_spoken_track_returns_empty_not_error(monkeypatch):
    _fake_yt_dlp(monkeypatch, {"subtitles": {}, "automatic_captions": {"en": [{"ext": "vtt", "url": "u"}]}})
    d = captions._fetch_ytdlp("AAA")
    assert d["lines"] == [] and d["language"] is None


def test_ytdlp_retries_then_gives_up(monkeypatch):
    info = {"subtitles": {"en": [{"ext": "vtt", "url": "u"}]}, "automatic_captions": {}}
    state = _fake_yt_dlp(monkeypatch, info, fail_times=2)
    assert captions._fetch_ytdlp("AAA")["lines"]          # 2 transient failures, 3rd attempt works
    assert state["extract"] == 3
    state = _fake_yt_dlp(monkeypatch, info, fail_times=99)
    with pytest.raises(captions.Blocked):
        captions._fetch_ytdlp("AAA")
    assert state["extract"] == captions.YTDLP_RETRIES
