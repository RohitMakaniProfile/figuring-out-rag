from fo.textutil import (
    Line, Turn, build_windows, captions_to_lines, chunk_turns, cited_ids, clean_description,
    fmt_ts, parse_chapters, parse_episode_no, parse_iso8601_duration, render_lines_for_llm,
    scribe_words_to_lines,
)

# Format copied from the real FO556 (Andrew Huberman) description
HUBERMAN_DESC = """Get your hand-picked playbook here: https://www.figuringout.co/pdf/fo-556

--------------

Timestamps -

00:00 - Intro
02:53 - Who Is Andrew Huberman?
06:16 - The Morning Cortisol Hack
1:11:27 - Why Fame Can Destroy Performers
2:46:17 - Outro

---------------

In today's episode, we sit down with Andrew Huberman, professor & neuroscientist at Stanford.
Instagram: https://www.instagram.com/hubermanlab/
"""


def test_iso_duration():
    assert parse_iso8601_duration("PT2H46M56S") == 2 * 3600 + 46 * 60 + 56
    assert parse_iso8601_duration("PT58S") == 58
    assert parse_iso8601_duration("P1DT1S") == 86401
    assert parse_iso8601_duration("garbage") is None


def test_episode_no():
    assert parse_episode_no("Andrew Huberman: ... | FO556 Raj Shamani") == 556
    assert parse_episode_no("Zakir Khan | FO 148 | Raj Shamani") == 148
    assert parse_episode_no("Shorts: random clip") is None


def test_chapters_real_format():
    ch = parse_chapters(HUBERMAN_DESC, duration_s=10016)
    assert [c["start_s"] for c in ch] == [0, 173, 376, 4287, 9977]
    assert ch[2]["title"] == "The Morning Cortisol Hack"


def test_chapters_rejects_non_chapters():
    assert parse_chapters("05:00 something\n06:00 other") == []        # does not start at 0
    assert parse_chapters("00:00 a\n01:00 b") == []                    # fewer than 3


def test_clean_description_drops_links():
    out = clean_description(HUBERMAN_DESC)
    assert "http" not in out and "Andrew Huberman, professor" in out


def test_windows_split_long_chapter():
    ch = [{"start_s": 0, "title": "Intro"}, {"start_s": 60, "title": "Long"}, {"start_s": 3000, "title": "End"}]
    wins = build_windows(ch, duration_s=3100, max_window_s=600)
    long_parts = [w for w in wins if w.title == "Long"]
    assert len(long_parts) == 5 and long_parts[0].start_s == 60 and long_parts[-1].end_s == 3000
    assert all(w.source == "chapters" for w in wins)
    assert build_windows([], 1000, default_window_s=480)[-1].end_s == 1000


def test_captions_and_render():
    lines = captions_to_lines([
        {"text": "[Music]", "start": 0, "duration": 2},
        {"text": "नमस्ते दोस्तों", "start": 2.5, "duration": 3},
        {"text": "आज हमारे साथ", "start": 40, "duration": 3},
    ])
    assert len(lines) == 2
    rendered = render_lines_for_llm(lines)
    assert "[00:02]" in rendered and "[00:40]" in rendered


def test_scribe_words_group_by_speaker():
    words = [
        {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing", "speaker_id": "speaker_0"},
        {"text": "sir", "start": 0.5, "end": 0.8, "type": "word", "speaker_id": "speaker_0"},
        {"text": "Thanks", "start": 1.0, "end": 1.3, "type": "word", "speaker_id": "speaker_1"},
    ]
    lines = scribe_words_to_lines(words)
    assert [(l.speaker, l.text) for l in lines] == [("speaker_0", "Hello sir"), ("speaker_1", "Thanks")]
    assert "<speaker_1>" in render_lines_for_llm(lines)


def test_chunking_respects_ads_and_sizes():
    turns = [
        Turn(0, 30, "host", "Raj Shamani", "Question one? " * 20),
        Turn(30, 200, "guest", "Andrew Huberman", "Answer sentence here. " * 120),
        Turn(200, 230, "host", "Raj Shamani", "This episode is sponsored by X. " * 10, is_ad=True),
        Turn(230, 400, "guest", "Andrew Huberman", "More insight. " * 150),
    ]
    chunks = chunk_turns(turns, target_words=280, max_words=420)
    assert all(len(c.text.split()) <= 420 + 40 for c in chunks)          # labels add a few words
    assert any(c.is_ad for c in chunks) and not all(c.is_ad for c in chunks)
    for c in chunks:
        assert len({t.is_ad for t in c.turns}) == 1                      # never mixed
    assert chunks[0].start_s == 0 and chunks[-1].end_s == 400
    assert "Raj Shamani:" in chunks[0].text and "Andrew Huberman:" in chunks[0].text   # Q + A packed together


def test_misc():
    assert fmt_ts(376) == "06:16" and fmt_ts(4287) == "1:11:27"
    assert cited_ids("a [S2] b [S1] c [S2]") == [2, 1]


# Same structure as a real YouTube auto-caption VTT (overlapping cues, ">>" = speaker change)
VTT = """WEBVTT

00:00:01.075 --> 00:00:03.376
So when you were there,

00:00:02.416 --> 00:00:04.880
when you killed him, the

00:00:14.562 --> 00:00:16.560
>> Well, they knew

00:00:15.593 --> 00:00:17.824
when we said it.

00:00:15.593 --> 00:00:17.824
when we said it.
"""


def test_parse_vtt_overlapping_cues_and_turn_markers():
    from fo.textutil import captions_to_lines, parse_vtt, render_lines_for_llm
    raw = parse_vtt(VTT)
    assert [r["text"] for r in raw] == ["So when you were there,", "when you killed him, the",
                                        ">> Well, they knew", "when we said it."]   # exact repeat dropped
    assert raw[0]["start"] == 1.075 and round(raw[0]["duration"], 3) == 2.301
    lines = captions_to_lines(raw)
    assert [l.turn_start for l in lines] == [False, False, True, False]
    assert lines[2].text == "Well, they knew"
    assert "[00:14] \u00bb Well, they knew" in render_lines_for_llm(lines)
