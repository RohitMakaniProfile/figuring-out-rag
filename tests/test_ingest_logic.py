import os

for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
          "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ.setdefault(k, "https://example.com")

from fo import prompts  # noqa: E402
from fo.ingest.process import _turns_from_llm, parse_ts  # noqa: E402
from fo.ingest.profile import sample_transcript  # noqa: E402
from fo.ingest.scribe import clean_keyterms  # noqa: E402
from fo.ingest.taxonomy import percentile_ranks  # noqa: E402
from fo.textutil import Window  # noqa: E402


def test_parse_ts_clamps_and_rejects():
    assert parse_ts("06:16", 300, 900) == 376
    assert parse_ts("1:00:00", 300, 900) is None          # outside window
    assert parse_ts(302, 305, 900) == 305                 # small drift clamped
    assert parse_ts("abc", 0, 10) is None


def test_turns_from_llm_speakers_and_order():
    out = {"turns": [
        {"start": "10:00", "speaker": "Raj", "role": "host", "confidence": 0.9, "text": "Why China?", "is_ad": False},
        {"start": "10:20", "speaker": "Khan Sir", "role": "guest", "confidence": 0.95, "text": "Because roads.", "is_ad": False},
        {"start": "09:00", "speaker": "Khan Sir", "role": "guest", "confidence": 0.3, "text": "Out of order.", "is_ad": False},
        {"start": "11:00", "speaker": "Raj", "role": "host", "confidence": 0.9, "text": "", "is_ad": False},
    ]}
    turns = _turns_from_llm(out, Window(600, 900, "China", "chapters"), host="Raj Shamani")
    assert [t.speaker_name for t in turns] == ["Raj Shamani", "Khan Sir", None]
    assert turns[2].speaker_role == "unknown"            # low confidence is not attributed
    assert turns[2].start_s == 620                        # out-of-order start pinned to previous
    assert turns[0].end_s == 620 and turns[-1].end_s == 900


def test_keyterms_constraints():
    terms = clean_keyterms(["Khan Sir", "Khan Sir", "x" * 60, "one two three four five six", "a<b>", ""])
    assert terms == ["Khan Sir", "ab"]


def test_percentiles_with_ties():
    assert percentile_ranks([10, 20, 20, 30]) == [0.0, 0.5, 0.5, 1.0]
    assert percentile_ranks([5]) == [1.0]


def test_sample_transcript_spreads():
    lines = [{"start": i * 10, "text": f"w{i} " * 50} for i in range(100)]
    s = sample_transcript(lines, n_excerpts=4, words_each=100)
    assert s.count("[") == 4 and "w0" in s and "w75" in s


def test_prompts_format_cleanly():
    prompts.PROFILE_SYSTEM.format(host="Raj Shamani")
    prompts.NORMALIZE_SYSTEM.format(host="R", guests="G", source_note="n", keyterms="k",
                                    speaker_rule=prompts.NORMALIZE_RULE_CAPTIONS, win_start="00:00", win_end="10:00")
    prompts.ENRICH_SYSTEM.format(host="R", guests="G")
    prompts.ROUTER_SYSTEM.format(host="R")
    prompts.ANSWER_SYSTEM.format(host="R", language_instruction="x")
    prompts.CHITCHAT_SYSTEM.format(host="R", language_instruction="x")
    assert '"followups"' in prompts.FOLLOWUP_SYSTEM.format(language_instruction="x")
