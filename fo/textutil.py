"""Pure helpers (no I/O) so they can be unit tested."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------- time ----------

_ISO_DUR = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def parse_iso8601_duration(s: str | None) -> int | None:
    if not s:
        return None
    m = _ISO_DUR.match(s)
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def hms_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def fmt_ts(sec: float | int) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------- metadata ----------

_EPISODE = re.compile(r"\bFO\s?-?\s?(\d{1,4})\b")


def parse_episode_no(title: str) -> int | None:
    m = _EPISODE.search(title or "")
    return int(m.group(1)) if m else None


_CHAPTER = re.compile(
    r"^\s*[\(\[]?((?:\d{1,2}:)?\d{1,2}:\d{2})[\)\]]?\s*(?:[-–—:|•]\s*)?(\S.*?)\s*$"
)


def parse_chapters(description: str | None, duration_s: int | None = None) -> list[dict]:
    """YouTube-style chapters: first at 0:00, at least 3, strictly increasing, inside the video."""
    if not description:
        return []
    found: list[dict] = []
    for line in description.splitlines():
        m = _CHAPTER.match(line)
        if not m:
            continue
        start = hms_to_seconds(m.group(1))
        title = m.group(2).strip(" -–—")
        if not title:
            continue
        if found and start <= found[-1]["start_s"]:
            # a second timestamp block (or garbage) – keep the first coherent run only
            if start == 0:
                break
            continue
        found.append({"start_s": start, "title": title})
    if len(found) < 3 or found[0]["start_s"] != 0:
        return []
    if duration_s:
        found = [c for c in found if c["start_s"] < duration_s]
    return found


_URL = re.compile(r"https?://\S+")


def clean_description(desc: str | None, max_chars: int = 2500) -> str:
    """Drop links and link-only lines; keep the prose that describes the guest and topics."""
    if not desc:
        return ""
    lines = []
    for line in desc.splitlines():
        stripped = _URL.sub("", line).strip(" -•⁠\u2060")
        if stripped and not set(stripped) <= set("-=_•. "):
            lines.append(stripped)
    return "\n".join(lines)[:max_chars]


# ---------- transcript lines ----------

_NOISE = re.compile(r"\[(?:music|applause|laughter|संगीत|हंसी)\]", re.I)


@dataclass
class Line:
    start: float
    end: float
    text: str
    speaker: str | None = None  # diarized speaker id when available
    turn_start: bool = False    # captions: YouTube ASR put ">>" here = the speaker changed


def captions_to_lines(raw: list[dict]) -> list[Line]:
    """youtube-transcript-api raw data -> Line list."""
    out: list[Line] = []
    for r in raw:
        text = _NOISE.sub("", r.get("text", "")).replace("\n", " ").strip()
        turn = False
        while text.startswith(">>") or text.startswith("&gt;&gt;"):
            text, turn = text.lstrip(">").replace("&gt;", "").strip(), True
        if not text:
            continue
        start = float(r["start"])
        out.append(Line(start=start, end=start + float(r.get("duration", 0) or 0), text=text, turn_start=turn))
    return out


_VTT_TIME = re.compile(r"^(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})")
_VTT_TAG = re.compile(r"<[^>]+>")


def parse_vtt(vtt: str) -> list[dict]:
    """WebVTT -> [{start, duration, text}] in youtube-transcript-api's raw shape.
    YouTube ASR cues overlap and rolling captions repeat the previous line; repeats are dropped."""
    out: list[dict] = []
    prev_line = ""
    cue_lines: list[str] = []
    start = end = 0.0

    def flush():
        nonlocal prev_line, cue_lines
        kept = []
        for ln in cue_lines:
            ln = _VTT_TAG.sub("", ln).strip()
            if ln and ln != prev_line:
                kept.append(ln)
            if ln:
                prev_line = ln
        if kept:
            out.append({"start": start, "duration": max(end - start, 0.0), "text": " ".join(kept)})
        cue_lines = []

    for raw in vtt.splitlines():
        m = _VTT_TIME.match(raw.strip())
        if m:
            flush()
            h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
            start = int(h1 or 0) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            end = int(h2 or 0) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
        elif not raw.strip():
            flush()
            cue_lines = []
        elif "-->" not in raw and not raw.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            cue_lines.append(raw)
    flush()
    return out


def scribe_words_to_lines(words: list[dict], max_words: int = 60, gap_s: float = 1.5) -> list[Line]:
    """Group Scribe word tokens into short speaker-consistent lines."""
    out: list[Line] = []
    cur: Line | None = None
    count = 0
    for w in words:
        if w.get("type") != "word":
            continue
        spk = w.get("speaker_id")
        text = (w.get("text") or "").strip()
        if not text:
            continue
        start, end = float(w.get("start") or 0), float(w.get("end") or 0)
        new_line = (
            cur is None
            or spk != cur.speaker
            or count >= max_words
            or start - cur.end > gap_s
        )
        if new_line:
            if cur:
                out.append(cur)
            cur, count = Line(start=start, end=end, text=text, speaker=spk), 1
        else:
            cur.text += " " + text
            cur.end = end
            count += 1
    if cur:
        out.append(cur)
    return out


def lines_in_range(lines: list[Line], start: float, end: float) -> list[Line]:
    return [l for l in lines if start <= l.start < end]


def render_lines_for_llm(lines: list[Line], marker_every_s: float = 30.0) -> str:
    """Compact text with [mm:ss] markers (and speaker ids when diarized) for the normaliser."""
    parts: list[str] = []
    last_marker = -1e9
    last_speaker: str | None = None
    for l in lines:
        prefix = ""
        if l.speaker is not None and l.speaker != last_speaker:
            prefix = f"\n[{fmt_ts(l.start)}] <{l.speaker}> "
            last_marker, last_speaker = l.start, l.speaker
        elif l.turn_start:
            prefix = f"\n[{fmt_ts(l.start)}] \u00bb "
            last_marker = l.start
        elif l.start - last_marker >= marker_every_s:
            prefix = f"\n[{fmt_ts(l.start)}] "
            last_marker = l.start
        parts.append(prefix + l.text)
    return " ".join(parts).strip()


# ---------- segmentation ----------

@dataclass
class Window:
    start_s: int
    end_s: int
    title: str | None
    source: str  # 'chapters' | 'window'
    part: int = 0


def build_windows(chapters: list[dict], duration_s: int, max_window_s: int = 600,
                  default_window_s: int = 480) -> list[Window]:
    """Chapters become segments; very long chapters are split so each LLM call stays small."""
    wins: list[Window] = []
    if chapters:
        bounds = [c["start_s"] for c in chapters] + [duration_s]
        for i, c in enumerate(chapters):
            s, e = bounds[i], bounds[i + 1]
            if e <= s:
                continue
            n = max(1, round((e - s) / max_window_s)) if (e - s) > max_window_s * 1.5 else 1
            step = (e - s) / n
            for k in range(n):
                wins.append(Window(int(s + k * step), int(s + (k + 1) * step) if k < n - 1 else e,
                                   c["title"], "chapters", k))
    else:
        s = 0
        while s < duration_s:
            e = min(duration_s, s + default_window_s)
            wins.append(Window(s, e, None, "window"))
            s = e
    return wins


# ---------- chunking ----------

@dataclass
class Turn:
    start_s: float
    end_s: float
    speaker_role: str          # host | guest | unknown
    speaker_name: str | None
    text: str
    is_ad: bool = False


@dataclass
class Chunk:
    start_s: int
    end_s: int
    speaker_role: str
    speaker_name: str | None
    text: str                  # speaker-labelled English text
    is_ad: bool = False
    turns: list[Turn] = field(default_factory=list)


_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_long_turn(t: Turn, max_words: int) -> list[Turn]:
    words = t.text.split()
    if len(words) <= max_words:
        return [t]
    sentences = _SENT.split(t.text)
    groups: list[str] = []
    cur: list[str] = []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            groups.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        groups.append(" ".join(cur))
    # a single sentence longer than max_words: hard split
    final: list[str] = []
    for g in groups:
        gw = g.split()
        for i in range(0, len(gw), max_words):
            final.append(" ".join(gw[i : i + max_words]))
    total = sum(len(g.split()) for g in final) or 1
    dur = max(t.end_s - t.start_s, 0)
    out, cursor = [], t.start_s
    for g in final:
        span = dur * len(g.split()) / total
        out.append(Turn(cursor, cursor + span, t.speaker_role, t.speaker_name, g, t.is_ad))
        cursor += span
    return out


def _label(t: Turn) -> str:
    who = t.speaker_name or {"host": "Host", "guest": "Guest"}.get(t.speaker_role, "Speaker")
    return f"{who}: {t.text}"


def _make_chunk(turns: list[Turn]) -> Chunk:
    roles = {t.speaker_role for t in turns}
    words_by_name: dict[str | None, int] = {}
    for t in turns:
        words_by_name[t.speaker_name] = words_by_name.get(t.speaker_name, 0) + len(t.text.split())
    dominant = max(words_by_name, key=words_by_name.get)
    return Chunk(
        start_s=int(turns[0].start_s),
        end_s=int(round(turns[-1].end_s)),
        speaker_role=roles.pop() if len(roles) == 1 else "mixed",
        speaker_name=dominant,
        text="\n".join(_label(t) for t in turns),
        is_ad=all(t.is_ad for t in turns),
        turns=turns,
    )


def chunk_turns(turns: list[Turn], target_words: int = 280, max_words: int = 420) -> list[Chunk]:
    """Pack whole turns into ~target-size chunks; never mix ad and non-ad turns in one chunk."""
    pieces: list[Turn] = []
    for t in turns:
        pieces.extend(_split_long_turn(t, max_words))
    chunks: list[Chunk] = []
    cur: list[Turn] = []
    cur_words = 0
    for p in pieces:
        w = len(p.text.split())
        boundary = cur and (cur[-1].is_ad != p.is_ad)
        if cur and (boundary or cur_words + w > max_words or cur_words >= target_words):
            chunks.append(_make_chunk(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += w
    if cur:
        # merge a tiny tail into the previous chunk when it fits
        if chunks and cur_words < target_words // 4 and not chunks[-1].is_ad and not cur[-1].is_ad \
                and len(chunks[-1].text.split()) + cur_words <= max_words:
            chunks[-1] = _make_chunk(chunks[-1].turns + cur)
        else:
            chunks.append(_make_chunk(cur))
    return chunks


# ---------- answers ----------

_CITE = re.compile(r"\[S(\d+)\]")


def cited_ids(text: str) -> list[int]:
    seen: list[int] = []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
