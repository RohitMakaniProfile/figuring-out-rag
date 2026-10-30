-- Figuring Out RAG: core schema (already applied to project RajProject)
create extension if not exists vector with schema extensions;
create extension if not exists pg_trgm with schema extensions;

create table public.videos (
  id text primary key, episode_no int, title text not null, description text,
  published_at timestamptz, duration_s int, view_count bigint, like_count bigint, comment_count bigint,
  chapters jsonb not null default '[]'::jsonb, caption_langs text[], primary_language text, profile jsonb,
  popularity_pct real, engagement_pct real, depth_score real,
  tier text check (tier in ('A','B')), selected boolean not null default false,
  transcript_source text check (transcript_source in ('youtube_captions','scribe_v2')),
  speaker_quality text check (speaker_quality in ('diarized','inferred')),
  summary text, takeaways jsonb, summary_embedding extensions.halfvec(1024),
  captions_at timestamptz, profiled_at timestamptz, transcribed_at timestamptz, processed_at timestamptz,
  last_error text, attempts int not null default 0,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create table public.guests (id bigserial primary key, name text not null unique, aliases text[] not null default '{}',
  role text, created_at timestamptz not null default now());
create table public.video_guests (video_id text not null references public.videos(id) on delete cascade,
  guest_id bigint not null references public.guests(id) on delete cascade, primary key (video_id, guest_id));
create table public.clusters (id serial primary key, slug text not null unique, name text not null, persona text,
  description text, sample_questions text[] not null default '{}', created_at timestamptz not null default now());
create table public.video_clusters (video_id text not null references public.videos(id) on delete cascade,
  cluster_id int not null references public.clusters(id) on delete cascade, weight real not null default 1,
  primary key (video_id, cluster_id));
create table public.segments (id bigserial primary key, video_id text not null references public.videos(id) on delete cascade,
  idx int not null, start_s int not null, end_s int not null, title text, summary text,
  source text not null check (source in ('chapters','window')), embedding extensions.halfvec(1024), unique (video_id, idx));
create table public.chunks (id bigserial primary key, video_id text not null references public.videos(id) on delete cascade,
  segment_id bigint references public.segments(id) on delete cascade, idx int not null,
  speaker_role text not null check (speaker_role in ('host','guest','mixed','unknown')), speaker_name text, speaker_conf real,
  start_s int not null, end_s int not null, text_en text not null, contextual_text text not null,
  is_ad boolean not null default false, embedding extensions.halfvec(1024),
  fts tsvector generated always as (to_tsvector('english', contextual_text)) stored, unique (video_id, idx));
create table public.qa_pairs (id bigserial primary key, video_id text not null references public.videos(id) on delete cascade,
  segment_id bigint references public.segments(id) on delete cascade, question text not null, answer_summary text,
  answer_chunk_ids bigint[] not null default '{}', start_s int not null, embedding extensions.halfvec(1024));
create table public.insights (id bigserial primary key, video_id text not null references public.videos(id) on delete cascade,
  chunk_id bigint references public.chunks(id) on delete set null,
  kind text not null check (kind in ('advice','story','stat','book','framework','prediction')),
  text text not null, speaker_name text, start_s int not null, embedding extensions.halfvec(1024));

create index videos_selected_idx on public.videos (selected) where selected;
create index videos_published_idx on public.videos (published_at desc);
create index guests_name_trgm on public.guests using gin (name extensions.gin_trgm_ops);
create index video_guests_guest on public.video_guests (guest_id);
create index video_clusters_cluster on public.video_clusters (cluster_id);
create index segments_video on public.segments (video_id);
create index chunks_video on public.chunks (video_id);
create index chunks_fts on public.chunks using gin (fts);
create index qa_video on public.qa_pairs (video_id);
create index insights_video on public.insights (video_id);
create index insights_kind on public.insights (kind);
create index chunks_emb_hnsw on public.chunks using hnsw (embedding extensions.halfvec_cosine_ops);
create index qa_emb_hnsw on public.qa_pairs using hnsw (embedding extensions.halfvec_cosine_ops);
create index insights_emb_hnsw on public.insights using hnsw (embedding extensions.halfvec_cosine_ops);
create index segments_emb_hnsw on public.segments using hnsw (embedding extensions.halfvec_cosine_ops);
create index videos_emb_hnsw on public.videos using hnsw (summary_embedding extensions.halfvec_cosine_ops);

create or replace function public.touch_updated_at() returns trigger language plpgsql set search_path = '' as $$
begin new.updated_at := now(); return new; end $$;
create trigger videos_touch before update on public.videos for each row execute function public.touch_updated_at();

-- backend uses the service-role key; no anon/authenticated access
alter table public.videos enable row level security;
alter table public.guests enable row level security;
alter table public.video_guests enable row level security;
alter table public.clusters enable row level security;
alter table public.video_clusters enable row level security;
alter table public.segments enable row level security;
alter table public.chunks enable row level security;
alter table public.qa_pairs enable row level security;
alter table public.insights enable row level security;

insert into storage.buckets (id, name, public) values ('transcripts', 'transcripts', false) on conflict (id) do nothing;
