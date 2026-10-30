# Figuring Out companion

A multi-index RAG chatbot over the **Figuring Out** podcast (Raj Shamani). Ask in Hinglish, Hindi or English.
Answers use only podcast content. Every claim carries a timestamp chip that opens the YouTube video at that second.

The bot is a search-and-cite assistant, not an impersonation: it says "Khan Sir says…" and never speaks as Raj or a guest.

## What is already done

The database is **already live** on Supabase project `RajProject` (`orhcvinszqrbfmwqlsgi`, ap-south-1):
tables, HNSW indexes, retrieval functions and a private `transcripts` bucket. The SQL is in `supabase/migrations/`
if you ever need to rebuild it on another project. Do not apply it twice to RajProject.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-ingest.txt   # web app only: requirements.txt
cp .env.example .env        # fill in keys yourself; the service-role key must never reach the browser
python -m pytest -q         # 84 tests, no network needed
```

Needs an Azure **text-embedding-3-large** (or -small) deployment. `ada-002` cannot produce 1024 dims.
A cheap mini chat deployment in `AZURE_OPENAI_MINI_DEPLOYMENT` cuts ingestion cost a lot.

Run ingestion **from your own machine**: YouTube blocks datacenter IPs for captions.

## Quickest path (two commands)

```bash
python scripts/doctor.py     # checks Azure chat + embeddings, Supabase DB + Storage, ElevenLabs key, YouTube captions
bash scripts/pilot.sh        # 3 real episodes (English, Hindi, short) end to end, then marks them searchable
uvicorn fo.app.main:app --reload
```

`doctor` prints exactly what to fix. `pilot.sh` stops at the first failure; if Scribe fails for a video it
continues on that episode's caption transcript. Use your own episodes with `bash scripts/pilot.sh ID1 ID2`.
The step-by-step commands below are the same thing, one stage at a time.

## Run order (start with a pilot)

```bash
python -m fo.ingest.cli discover                 # playlist -> episodes, chapters, view counts
python -m fo.ingest.cli captions --limit 15      # free YouTube captions
python -m fo.ingest.cli profile  --limit 15      # guests, summary, depth, keyterms
```

**Pilot check before spending on Scribe.** Pick 3 Hindi episodes and compare caption quality with Scribe:

```bash
python -m fo.ingest.cli scribe --video <VIDEO_ID>
```

Open `scribe/<id>.json` and the captions JSON from the Supabase Storage `transcripts` bucket. If captions are good
enough after the LLM normalise step, you can run most episodes on captions only and save the Scribe cost.

```bash
# then the whole library
python -m fo.ingest.cli captions && python -m fo.ingest.cli profile
python -m fo.ingest.cli taxonomy                 # data-driven clusters + tags (k adapts: 2..min(12, episodes//3), needs >= 6 profiled episodes)
python -m fo.ingest.cli taxonomy --searchable    # same, but only for episodes the chatbot can answer from (replaces existing clusters)
python -m fo.ingest.cli score
python -m fo.ingest.cli select                   # Tier A ≈ top popular + top deep per cluster + recent
python -m fo.ingest.cli scribe                   # Tier A only, diarized
python -m fo.ingest.cli process                  # normalise, chunk, enrich, embed, store
python -m fo.ingest.cli publish --video ID       # make a processed episode searchable (select does this for Tier A)
python -m fo.ingest.cli status                   # progress, errors, DB size vs 500 MB
# later: python -m fo.ingest.cli select --all && python -m fo.ingest.cli process   (Tier B on captions)
```

Every stage is resumable. Errors are stored per episode (`videos.last_error`) and shown by `status`.

## Serve

```bash
uvicorn fo.app.main:app --reload     # http://localhost:8000
python -m eval.run_eval              # golden-question sanity checks; grow questions.jsonl to ~80
python scripts/mock_server.py        # UI preview with canned data, no keys needed
```

## How a question is answered

1. **Router** (mini model): standalone question, English search query, intent, guest names, answer language.
2. **Retrieval by intent**: `specific` uses host-question index + hybrid chunks; `compare` searches each guest separately;
   `theme` uses extracted insights resolved back to transcript chunks; `recommend` ranks chapter summaries by
   similarity, popularity and depth.
3. **Rerank** (mini model) drops passages that only share keywords.
4. **Answer** streams over SSE: `status → sources → token* → citations → followups → done`. Citations that do not
   match a retrieved source are dropped. If nothing relevant exists, the bot says so.

## Cost (Scribe v2 is $0.22/h; keyterms add 20%)

Tier A ≈ 150 episodes × ~1.6 h ≈ 240 h ≈ $65 with keyterms. All ~570 episodes ≈ $150–$250. Azure usage is separate;
the normalise pass dominates, so use the mini deployment. Check current prices on the ElevenLabs pricing page before a big run.

## Known limits

- Captions-based episodes have **inferred** speakers. Low-confidence turns are stored as `unknown` and the bot says
  "the episode mentions…" instead of naming someone.
- Supabase free projects pause after about a week of inactivity, and the DB cap is 500 MB. Run `status` after the pilot
  and extrapolate. Raw transcripts live in Storage, not the DB.
- `source_url` makes ElevenLabs fetch the audio from YouTube. If it fails for some videos, the episode keeps its caption
  transcript and `last_error` says why. Check that using this pipeline on third-party content fits YouTube's and the
  creator's terms before any public launch; for a personal project link back to the video, as the UI does.
- This code was tested with mocked LLM/DB calls and against the live schema with dummy rows. **The ingestion stages have
  not yet run against real YouTube, Azure or ElevenLabs.** Expect a round of small fixes during the pilot.
