"""All settings come from the environment (.env). Nothing secret or deployment-specific lives in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name} (see .env.example)")
    return val


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    # Azure OpenAI
    azure_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT", required=True))
    azure_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY", required=True))
    azure_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    chat_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_CHAT_DEPLOYMENT", required=True))
    # Cheap model for ingestion passes, routing, rerank. Falls back to the main deployment.
    mini_deployment: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_MINI_DEPLOYMENT") or _env("AZURE_OPENAI_CHAT_DEPLOYMENT", required=True)
    )
    embedding_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", required=True))
    # Must match halfvec(1024) in the DB schema. Change both together.
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "1024")))
    # Reasoning deployments (o-series / gpt-5) reject custom temperature: leave empty to omit.
    llm_temperature: float | None = field(default_factory=lambda: _float_or_none("LLM_TEMPERATURE"))

    # Supabase
    supabase_url: str = field(default_factory=lambda: _env("SUPABASE_URL", required=True))
    supabase_key: str = field(default_factory=lambda: _env("SUPABASE_SERVICE_ROLE_KEY", required=True))
    storage_bucket: str = field(default_factory=lambda: _env("SUPABASE_STORAGE_BUCKET", "transcripts"))

    # Sources
    youtube_api_key: str | None = field(default_factory=lambda: _env("YOUTUBE_API_KEY"))
    playlist_id: str = field(default_factory=lambda: _env("FO_PLAYLIST_ID", "PLE0Jo6NF_JYO5-phess8GKafKMtPv3tfZ"))
    host_name: str = field(default_factory=lambda: _env("FO_HOST_NAME", "Raj Shamani"))
    min_episode_seconds: int = field(default_factory=lambda: int(_env("FO_MIN_EPISODE_SECONDS", "1200")))
    elevenlabs_api_key: str | None = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))
    scribe_model: str = field(default_factory=lambda: _env("SCRIBE_MODEL", "scribe_v2"))
    # Optional HTTP(S) proxy for YouTube caption fetching (YouTube blocks most cloud IPs)
    youtube_proxy_url: str | None = field(default_factory=lambda: _env("YOUTUBE_PROXY_URL"))

    # Corpus selection (Phase 1 = Tier A)
    select_top_popular_per_cluster: int = field(default_factory=lambda: int(_env("SELECT_TOP_POPULAR", "8")))
    select_top_deep_per_cluster: int = field(default_factory=lambda: int(_env("SELECT_TOP_DEEP", "8")))
    select_recent: int = field(default_factory=lambda: int(_env("SELECT_RECENT", "25")))

    # Retrieval / answering
    rerank_enabled: bool = field(default_factory=lambda: _env("RERANK_ENABLED", "true").lower() == "true")
    max_context_sources: int = field(default_factory=lambda: int(_env("MAX_CONTEXT_SOURCES", "10")))
    concurrency: int = field(default_factory=lambda: int(_env("INGEST_CONCURRENCY", "4")))


@lru_cache
def settings() -> Settings:
    return Settings()
