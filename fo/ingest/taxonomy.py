"""Stage 4 — taxonomy, scores and corpus selection.

taxonomy: embed episode profiles -> KMeans (k chosen by silhouette) -> LLM names the clusters ->
          LLM tags every episode with 1-3 clusters.
score:    popularity = percentile of log(views) inside its publish-year cohort (old episodes don't win
          just by age); engagement = likes/views percentile; depth = insight density + length.
select:   Tier A = per cluster top-N popular + top-N deep, plus the most recent episodes."""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .. import db, llm, prompts
from ..config import settings

log = logging.getLogger(__name__)

K_MIN = int(os.getenv("TAXONOMY_K_MIN", "2"))
K_MAX = int(os.getenv("TAXONOMY_K_MAX", "12"))
SAMPLES_PER_CLUSTER = 6
# depth = how much transferable insight an episode holds; density dominates, length is a tiebreaker
DEPTH_W_DENSITY, DEPTH_W_LENGTH = 0.75, 0.25


def _profile_text(v: dict) -> str:
    p = v.get("profile") or {}
    return f"{v['title']}\n{p.get('summary', '')}\nTopics: {', '.join(p.get('topics') or [])}"


def percentile_ranks(values: list[float]) -> list[float]:
    """0..1 percentile rank, ties share the average rank."""
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def k_range(n: int) -> range:
    """Candidate cluster counts for n episodes: at least ~3 episodes per cluster, never below K_MIN."""
    return range(K_MIN, min(K_MAX, n // 3) + 1)


async def profiled_videos(searchable_only: bool = False) -> list[dict]:
    """Profiled episodes; with searchable_only just the ones the chatbot can answer from (selected and processed),
    which is what the home page topics should describe."""
    rows = await db.select("videos", "id,title,profile,selected,processed_at")
    return [v for v in rows if v.get("profile") and (not searchable_only or (v["selected"] and v["processed_at"]))]


async def build_taxonomy(searchable_only: bool = False) -> None:
    videos = await profiled_videos(searchable_only)
    ks = k_range(len(videos))
    if not ks:
        raise RuntimeError(f"only {len(videos)} profiled episodes; need at least {K_MIN * 3} to build a taxonomy")
    X = np.array(await llm.embed([_profile_text(v) for v in videos]))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    best = None
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        score = silhouette_score(X, km.labels_, metric="cosine")
        log.info("k=%d silhouette=%.3f", k, score)
        if best is None or score > best[0]:
            best = (score, k, km)
    _, k, km = best
    log.info("chose k=%d", k)

    blocks = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        dist = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        reps = [videos[i] for i in idx[np.argsort(dist)[:SAMPLES_PER_CLUSTER]]]
        sample = "\n".join(f"- {v['title']} :: {(v['profile'] or {}).get('summary', '')[:300]}" for v in reps)
        blocks.append(f"### cluster_index {c} ({len(idx)} episodes)\n{sample}")
    labelled = await llm.chat_json(prompts.TAXONOMY_SYSTEM, "\n\n".join(blocks),
                                   deployment=settings().chat_deployment, max_tokens=4000)

    await db.run(lambda: db.sb().table("clusters").delete().neq("id", -1).execute())  # cascades tags
    rows, seen = [], set()
    for c in labelled.get("clusters", []):
        slug = c["slug"]
        if slug in seen:
            slug = f"{slug}-{c['cluster_index']}"
        seen.add(slug)
        rows.append({"slug": slug, "name": c["name"], "persona": c.get("persona"),
                     "description": c.get("description"), "sample_questions": c.get("sample_questions") or []})
    clusters = await db.insert("clusters", rows)
    log.info("clusters: %s", [c["name"] for c in clusters])
    await tag_all(clusters, videos)


async def tag_all(clusters: list[dict] | None = None, videos: list[dict] | None = None, searchable_only: bool = False) -> None:
    clusters = clusters or await db.select("clusters", "id,slug,name,description")
    videos = videos or await profiled_videos(searchable_only)
    by_slug = {c["slug"]: c["id"] for c in clusters}
    taxonomy = "\n".join(f"- {c['slug']}: {c['name']} — {c.get('description') or ''}" for c in clusters)

    async def tag(v):
        try:
            out = await llm.chat_json(prompts.TAG_SYSTEM,
                                      f"TAXONOMY:\n{taxonomy}\n\nEPISODE:\n{_profile_text(v)}", max_tokens=300)
        except Exception as e:
            log.warning("tag failed %s: %s", v["id"], e)
            return []
        rows = []
        for t in (out.get("clusters") or [])[:3]:
            cid = by_slug.get(t.get("slug"))
            if cid:
                rows.append({"video_id": v["id"], "cluster_id": cid,
                             "weight": float(min(max(t.get("weight", 1.0), 0.0), 1.0))})
        return rows

    results = await llm.gather_limited([tag(v) for v in videos], settings().concurrency * 2)
    rows = [r for rs in results for r in rs]
    await db.upsert("video_clusters", rows, on_conflict="video_id,cluster_id")
    log.info("tagged %d episodes with %d cluster links", len(videos), len(rows))


async def score_all() -> None:
    videos = await db.select("videos", "id,published_at,duration_s,view_count,like_count,profile")
    cohorts: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        cohorts[(v.get("published_at") or "unknown")[:4]].append(v)

    updates: dict[str, dict] = {v["id"]: {} for v in videos}
    for vs in cohorts.values():
        pop = percentile_ranks([math.log1p(v.get("view_count") or 0) for v in vs])
        eng = percentile_ranks([(v.get("like_count") or 0) / max(v.get("view_count") or 1, 1) for v in vs])
        for v, p, e in zip(vs, pop, eng):
            updates[v["id"]].update(popularity_pct=round(p, 4), engagement_pct=round(e, 4))

    length_pct = percentile_ranks([float(v.get("duration_s") or 0) for v in videos])
    for v, lp in zip(videos, length_pct):
        density = (v.get("profile") or {}).get("insight_density")
        if isinstance(density, (int, float)):
            d = (min(max(density, 1), 10) - 1) / 9
            updates[v["id"]]["depth_score"] = round(DEPTH_W_DENSITY * d + DEPTH_W_LENGTH * lp, 4)

    for vid, vals in updates.items():
        if vals:
            await db.update("videos", vals, id=vid)
    log.info("scored %d episodes", len(updates))


async def select_corpus(all_episodes: bool = False) -> None:
    s = settings()
    videos = await db.select("videos", "id,title,published_at,popularity_pct,depth_score,profiled_at")
    links = await db.select("video_clusters", "video_id,cluster_id")
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for l in links:
        by_cluster[l["cluster_id"]].append(l["video_id"])
    vmap = {v["id"]: v for v in videos}

    tier_a: set[str] = set()
    for vids in by_cluster.values():
        vs = [vmap[i] for i in vids if i in vmap]
        tier_a |= {v["id"] for v in sorted(vs, key=lambda v: v.get("popularity_pct") or 0, reverse=True)
                   [: s.select_top_popular_per_cluster]}
        tier_a |= {v["id"] for v in sorted(vs, key=lambda v: v.get("depth_score") or 0, reverse=True)
                   [: s.select_top_deep_per_cluster]}
    recent = sorted(videos, key=lambda v: v.get("published_at") or "", reverse=True)[: s.select_recent]
    tier_a |= {v["id"] for v in recent}

    for v in videos:
        a = v["id"] in tier_a
        await db.update("videos", {"tier": "A" if a else "B", "selected": a or all_episodes}, id=v["id"])
    log.info("Tier A: %d episodes; selected: %d", len(tier_a), len(videos) if all_episodes else len(tier_a))
