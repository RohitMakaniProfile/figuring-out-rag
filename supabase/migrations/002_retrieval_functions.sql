-- Retrieval functions (already applied to project RajProject).
-- Search functions are plpgsql because Supabase forbids function-level SET hnsw.iterative_scan;
-- set_config(..., true) at call time works and keeps filtered HNSW searches from returning nothing.

create or replace function public.allowed_videos(filter_video_ids text[] default null,
  filter_guest_ids bigint[] default null, filter_cluster_ids int[] default null)
returns setof text language sql stable set search_path = public, extensions as $$
  select v.id from videos v
  where v.selected
    and (filter_video_ids is null or v.id = any(filter_video_ids))
    and (filter_guest_ids is null or exists (select 1 from video_guests vg where vg.video_id = v.id and vg.guest_id = any(filter_guest_ids)))
    and (filter_cluster_ids is null or exists (select 1 from video_clusters vc where vc.video_id = v.id and vc.cluster_id = any(filter_cluster_ids)))
$$;

create or replace function public.hybrid_search_chunks(
  query_embedding extensions.halfvec(1024), query_text text, match_count int default 12,
  filter_video_ids text[] default null, filter_guest_ids bigint[] default null, filter_cluster_ids int[] default null,
  full_text_weight float default 1.0, semantic_weight float default 1.0, rrf_k int default 50)
returns table (id bigint, video_id text, segment_id bigint, speaker_role text, speaker_name text,
  speaker_conf real, start_s int, end_s int, text_en text, score float)
language plpgsql stable set search_path = public, extensions as $$
declare tsq tsquery;
begin
  perform set_config('hnsw.iterative_scan', 'relaxed_order', true);
  if coalesce(trim(query_text), '') <> '' and plainto_tsquery('english', query_text)::text <> '' then
    tsq := to_tsquery('english', replace(plainto_tsquery('english', query_text)::text, '&', '|'));
  end if;
  return query
  with allowed as (select a from allowed_videos(filter_video_ids, filter_guest_ids, filter_cluster_ids) a),
  full_text as (
    select c.id, row_number() over (order by ts_rank_cd(c.fts, tsq) desc) as rank_ix
    from chunks c where tsq is not null and c.fts @@ tsq and not c.is_ad and c.video_id in (select a from allowed)
    order by rank_ix limit match_count * 2),
  semantic as (
    select s.id, row_number() over (order by s.dist) as rank_ix from (
      select c.id, c.embedding <=> query_embedding as dist from chunks c
      where c.embedding is not null and not c.is_ad and c.video_id in (select a from allowed)
      order by c.embedding <=> query_embedding limit match_count * 2) s)
  select c.id, c.video_id, c.segment_id, c.speaker_role, c.speaker_name, c.speaker_conf, c.start_s, c.end_s, c.text_en,
    (coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight + coalesce(1.0 / (rrf_k + se.rank_ix), 0.0) * semantic_weight)::float as score
  from full_text ft full outer join semantic se on ft.id = se.id join chunks c on c.id = coalesce(ft.id, se.id)
  order by score desc limit match_count;
end $$;

create or replace function public.search_qa_pairs(query_embedding extensions.halfvec(1024), match_count int default 8,
  filter_video_ids text[] default null, filter_guest_ids bigint[] default null, filter_cluster_ids int[] default null)
returns table (id bigint, video_id text, question text, answer_summary text, answer_chunk_ids bigint[], start_s int, similarity float)
language plpgsql stable set search_path = public, extensions as $$
begin
  perform set_config('hnsw.iterative_scan', 'relaxed_order', true);
  return query select p.id, p.video_id, p.question, p.answer_summary, p.answer_chunk_ids, p.start_s,
    (1 - (p.embedding <=> query_embedding))::float
  from qa_pairs p where p.embedding is not null
    and p.video_id in (select a from allowed_videos(filter_video_ids, filter_guest_ids, filter_cluster_ids) a)
  order by p.embedding <=> query_embedding limit match_count;
end $$;

create or replace function public.search_insights(query_embedding extensions.halfvec(1024), match_count int default 20,
  filter_kinds text[] default null, filter_video_ids text[] default null, filter_guest_ids bigint[] default null,
  filter_cluster_ids int[] default null)
returns table (id bigint, video_id text, chunk_id bigint, kind text, text text, speaker_name text, start_s int, similarity float)
language plpgsql stable set search_path = public, extensions as $$
begin
  perform set_config('hnsw.iterative_scan', 'relaxed_order', true);
  return query select i.id, i.video_id, i.chunk_id, i.kind, i.text, i.speaker_name, i.start_s,
    (1 - (i.embedding <=> query_embedding))::float
  from insights i where i.embedding is not null and (filter_kinds is null or i.kind = any(filter_kinds))
    and i.video_id in (select a from allowed_videos(filter_video_ids, filter_guest_ids, filter_cluster_ids) a)
  order by i.embedding <=> query_embedding limit match_count;
end $$;

create or replace function public.search_segments(query_embedding extensions.halfvec(1024), match_count int default 10,
  filter_cluster_ids int[] default null, filter_guest_ids bigint[] default null)
returns table (id bigint, video_id text, title text, summary text, start_s int, end_s int, video_title text,
  popularity_pct real, depth_score real, similarity float)
language plpgsql stable set search_path = public, extensions as $$
begin
  perform set_config('hnsw.iterative_scan', 'relaxed_order', true);
  return query select s.id, s.video_id, s.title, s.summary, s.start_s, s.end_s, v.title, v.popularity_pct, v.depth_score,
    (1 - (s.embedding <=> query_embedding))::float
  from segments s join videos v on v.id = s.video_id
  where s.embedding is not null and s.video_id in (select a from allowed_videos(null, filter_guest_ids, filter_cluster_ids) a)
  order by s.embedding <=> query_embedding limit match_count;
end $$;

create or replace function public.find_guests(q text, max_results int default 3, min_score real default 0.35)
returns table (id bigint, name text, score real) language sql stable set search_path = public, extensions as $$
  select g.id, g.name, m.score from guests g
  cross join lateral (select greatest(similarity(lower(g.name), lower(q)), word_similarity(lower(q), lower(g.name)),
    coalesce((select max(greatest(similarity(lower(a), lower(q)), word_similarity(lower(q), lower(a)))) from unnest(g.aliases) a), 0)) as score) m
  where m.score >= min_score order by m.score desc limit max_results
$$;

create or replace function public.db_size_report() returns table (relation text, total_mb numeric)
language sql stable set search_path = public, extensions as $$
  select 'DATABASE TOTAL', round(pg_database_size(current_database()) / 1048576.0, 1)
  union all select c.relname::text, round(pg_total_relation_size(c.oid) / 1048576.0, 1)
  from pg_class c join pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'r'
$$;

-- backend only (service role)
revoke execute on function public.allowed_videos(text[], bigint[], int[]) from public, anon, authenticated;
revoke execute on function public.hybrid_search_chunks(extensions.halfvec, text, int, text[], bigint[], int[], float, float, int) from public, anon, authenticated;
revoke execute on function public.search_qa_pairs(extensions.halfvec, int, text[], bigint[], int[]) from public, anon, authenticated;
revoke execute on function public.search_insights(extensions.halfvec, int, text[], text[], bigint[], int[]) from public, anon, authenticated;
revoke execute on function public.search_segments(extensions.halfvec, int, int[], bigint[]) from public, anon, authenticated;
revoke execute on function public.find_guests(text, int, real) from public, anon, authenticated;
revoke execute on function public.db_size_report() from public, anon, authenticated;
grant execute on function public.allowed_videos(text[], bigint[], int[]) to service_role;
grant execute on function public.hybrid_search_chunks(extensions.halfvec, text, int, text[], bigint[], int[], float, float, int) to service_role;
grant execute on function public.search_qa_pairs(extensions.halfvec, int, text[], bigint[], int[]) to service_role;
grant execute on function public.search_insights(extensions.halfvec, int, text[], text[], bigint[], int[]) to service_role;
grant execute on function public.search_segments(extensions.halfvec, int, int[], bigint[]) to service_role;
grant execute on function public.find_guests(text, int, real) to service_role;
grant execute on function public.db_size_report() to service_role;
