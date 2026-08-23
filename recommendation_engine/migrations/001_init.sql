-- 001_init.sql — recommendations schema: catalog, signals, scoring inputs, caches.
--
-- Idempotent throughout (IF NOT EXISTS / exception-guarded CREATE TYPE) so a
-- partially-applied migration can be re-run safely. Applied by
-- `python -m recommendation_engine.migrations.migrate`.

CREATE SCHEMA IF NOT EXISTS recommendations;

-- pgvector >= 0.8.0 is required for hnsw.iterative_scan, which is what keeps
-- recall usable when a metadata filter is applied alongside the KNN. The
-- version is asserted at runtime by /health, not here — CREATE EXTENSION
-- cannot express a minimum version.
CREATE EXTENSION IF NOT EXISTS vector;

-- Fuzzy title/author resolution for the ad-hoc "I liked these" path.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DO $$ BEGIN
  CREATE TYPE recommendations.item_source AS ENUM ('platform', 'cmu');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ══════════════════════════════════════════════════════════════════════════
-- Catalog — one row per book, embedded ONCE at ingestion.
-- Runtime embeds queries only; it never re-embeds catalog rows.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.items (
    id               bigserial PRIMARY KEY,
    source           recommendations.item_source NOT NULL,
    source_id        text NOT NULL,   -- Firestore storyId | CMU wikipedia article id

    -- Normalized preprocessing schema. We never embed a raw plot summary:
    -- summaries are 429 words on average and full of incident, which swamps
    -- the premise/theme/tone signal the recommender actually ranks on.
    title            text NOT NULL,
    author           text,
    genres           text[] NOT NULL DEFAULT '{}',  -- crosswalked platform categories
    raw_genres       text[] NOT NULL DEFAULT '{}',  -- original labels, kept for audit
    core_premise     text,
    themes           text[] NOT NULL DEFAULT '{}',  -- key themes / tropes
    tone             text[] NOT NULL DEFAULT '{}',

    -- Filter metadata
    word_count       integer,
    chapter_count    integer,
    language         text,
    target_audience  text,
    published_year   integer,   -- 34% of CMU dates are year-only; no fake precision

    -- Eligibility gates the partial HNSW index below. Set false for unpublished
    -- or deleted platform stories, and for records whose normalization came back
    -- low-confidence (a thin summary yields a confidently hallucinated premise,
    -- which is a poisoned vector that looks perfectly fine).
    is_eligible      boolean NOT NULL DEFAULT true,
    confidence       real,

    -- Embedding provenance. embed_input is stored verbatim so a result can be
    -- explained and reproduced; its sha lets a re-run skip unchanged rows.
    embed_input      text NOT NULL,
    embed_input_sha  text NOT NULL,
    embedding        vector(768),
    embed_model      text,
    -- RETRIEVAL_DOCUMENT for every catalog row. Recorded because mixing task
    -- types across a corpus degrades recall silently — the backfill refuses to.
    embed_task_type  text,
    embedded_at      timestamptz,

    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),

    UNIQUE (source, source_id)
);

-- Partial HNSW index: ineligible rows never enter the graph at all, which
-- removes the single highest-selectivity filter from the recall problem instead
-- of asking iterative_scan to solve it.
--
-- Cheap here (the table is empty at migration time). The bulk backfill drops
-- and recreates this with a raised maintenance_work_mem — building the graph
-- once after load is far faster than maintaining it across 15k inserts.
CREATE INDEX IF NOT EXISTS items_embedding_hnsw
    ON recommendations.items USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE is_eligible;

CREATE INDEX IF NOT EXISTS items_genres_gin
    ON recommendations.items USING gin (genres);
CREATE INDEX IF NOT EXISTS items_themes_gin
    ON recommendations.items USING gin (themes);
CREATE INDEX IF NOT EXISTS items_author_trgm
    ON recommendations.items USING gin (author gin_trgm_ops);
CREATE INDEX IF NOT EXISTS items_title_trgm
    ON recommendations.items USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS items_source_elig
    ON recommendations.items (source, is_eligible);
CREATE INDEX IF NOT EXISTS items_word_count
    ON recommendations.items (word_count) WHERE is_eligible;


-- ══════════════════════════════════════════════════════════════════════════
-- Scoring term 2 — popularity / engagement prior. Materialized on a schedule
-- because it is global (not per-request) and volume-damped arithmetic has no
-- business running inside the retrieval path.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.item_stats (
    item_id        bigint PRIMARY KEY
                   REFERENCES recommendations.items(id) ON DELETE CASCADE,
    likes          integer NOT NULL DEFAULT 0,
    ratings_count  integer NOT NULL DEFAULT 0,
    avg_rating     real,     -- 1..5
    completions    integer NOT NULL DEFAULT 0,
    -- Anonymous global counter, incrementable by UNAUTHENTICATED clients per
    -- firestore.rules. Carried for reference but weighted near-zero; see the
    -- popularity weights in `config` below.
    views          integer NOT NULL DEFAULT 0,
    -- likes + ratings_count + completions. Drives the cold-start ramp α.
    n_interactions integer NOT NULL DEFAULT 0,
    pop_score      real NOT NULL DEFAULT 0,  -- materialized, [0,1]
    refreshed_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS item_stats_pop
    ON recommendations.item_stats (pop_score DESC);


-- ══════════════════════════════════════════════════════════════════════════
-- Reader signals, exported server-side from Firestore.
--
-- `users/{uid}/readingProgress` is strictly private (firestore.rules warns that
-- `allow read` grants `list`, which would let anyone enumerate a reader's whole
-- history), so this table can only ever be populated via the Admin SDK.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.interactions (
    user_id     text   NOT NULL,
    item_id     bigint NOT NULL
                REFERENCES recommendations.items(id) ON DELETE CASCADE,
    kind        text   NOT NULL,  -- like | rating | progress | completion
    weight      real   NOT NULL,  -- engagement strength used by the CF seed set
    value       real,             -- rating 1..5, or scrollPercent 0..1
    occurred_at timestamptz NOT NULL,
    PRIMARY KEY (user_id, item_id, kind)
);
CREATE INDEX IF NOT EXISTS interactions_item
    ON recommendations.interactions (item_id);


-- ══════════════════════════════════════════════════════════════════════════
-- Scoring term 3 — item-item collaborative filtering.
--
-- STUB BY DESIGN. The table, the rebuild query and the scoring term are all
-- real, but `w_cf_ceiling` is 0 in `config` below, so CF contributes nothing
-- yet. The platform has no impression/click log — only binary likes, immutable
-- ratings, and readingProgress *current state* — so co-occurrence would be far
-- too sparse to beat the popularity prior. Keeping the shape means lighting it
-- up later is a config flip plus a backfill, not a rescoring rewrite.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.item_cooccurrence (
    item_a bigint NOT NULL
           REFERENCES recommendations.items(id) ON DELETE CASCADE,
    item_b bigint NOT NULL
           REFERENCES recommendations.items(id) ON DELETE CASCADE,
    cooc   integer NOT NULL,  -- readers who engaged with both
    sim    real    NOT NULL,  -- shrunk cosine: cooc / (sqrt(n_a * n_b) + lambda)
    PRIMARY KEY (item_a, item_b),
    -- One direction stored, both queried. Halves the table and makes the
    -- rebuild's symmetry a constraint rather than a convention.
    CHECK (item_a < item_b)
);
CREATE INDEX IF NOT EXISTS cooc_b_sim
    ON recommendations.item_cooccurrence (item_b, sim DESC);


-- ══════════════════════════════════════════════════════════════════════════
-- Precomputed taste vector — the reason behavioral recs need no runtime
-- embedding call at all.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.user_taste (
    user_id         text PRIMARY KEY,
    taste_embedding vector(768) NOT NULL,
    -- Excluded from results (never recommend what they already read) and also
    -- the seed set S_u for the CF term.
    seed_item_ids   bigint[] NOT NULL,
    -- Items to actively suppress, e.g. anything rated <= 2.
    suppressed_item_ids bigint[] NOT NULL DEFAULT '{}',
    n_signals       integer NOT NULL,
    computed_at     timestamptz NOT NULL DEFAULT now()
);


-- ══════════════════════════════════════════════════════════════════════════
-- Explanation cache, keyed deterministically on
--   sha256(model | prompt_ver | source:source_id | embed_input_sha | query_fp)
-- so the same reader asking the same thing is free, and the key self-invalidates
-- when the item's normalized text or the prompt version changes.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.explanation_cache (
    cache_key   text PRIMARY KEY,
    item_id     bigint NOT NULL
                REFERENCES recommendations.items(id) ON DELETE CASCADE,
    explanation text NOT NULL,
    model       text NOT NULL,
    prompt_ver  integer NOT NULL,
    hit_count   integer NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);


-- ══════════════════════════════════════════════════════════════════════════
-- Ingest bookkeeping — makes the 15.5k-record LLM normalization pass resumable.
-- Keyed on the sha of the TRUNCATED summary, so a crashed run re-reads the
-- cache and skips, and a prompt_ver bump forces regeneration.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.normalization_cache (
    summary_sha  text PRIMARY KEY,
    core_premise text,
    themes       text[] NOT NULL DEFAULT '{}',
    tone         text[] NOT NULL DEFAULT '{}',
    confidence   real,
    model        text NOT NULL,
    prompt_ver   integer NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendations.ingest_runs (
    id          bigserial PRIMARY KEY,
    kind        text NOT NULL,  -- cmu_backfill | platform_sync | stats_refresh | cooc_rebuild
    status      text NOT NULL,  -- running | completed | failed
    cursor      text,           -- resume point
    counts      jsonb NOT NULL DEFAULT '{}',
    error       text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS ingest_runs_kind_started
    ON recommendations.ingest_runs (kind, started_at DESC);


-- ══════════════════════════════════════════════════════════════════════════
-- Scoring knobs. In the database rather than env vars so ranking can be retuned
-- without a redeploy — and so a bad tune is one UPDATE away from being reverted.
-- ══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS recommendations.config (
    key         text PRIMARY KEY,
    value       real NOT NULL,
    description text,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

INSERT INTO recommendations.config (key, value, description) VALUES
    ('w_pop_ceiling',      0.25, 'Max popularity weight at full behavioral ramp'),
    ('w_cf_ceiling',       0.00, 'Max CF weight. 0 = stubbed; set 0.20 once an event log exists'),
    ('ramp_n_min',         5,    'Below this interaction count, scoring is 100% semantic'),
    ('ramp_n50',           20,   'Interaction count at which behavioral gets half its ceiling'),
    ('bayes_prior_c',      20,   'Bayesian rating prior weight; matches ramp_n50 by design'),
    ('cf_shrinkage_lambda',10,   'Stops a single 1-of-1 co-occurrence from scoring 1.0'),
    ('pop_w_bayes',        0.60, 'Popularity sub-weight: volume-damped rating'),
    ('pop_w_engagement',   0.40, 'Popularity sub-weight: log-compressed likes + completions'),
    ('pop_completion_mult',3.00, 'A completion is worth this many likes'),
    ('rrf_k',              60,   'Reciprocal Rank Fusion constant'),
    ('mmr_lambda',         0.70, 'MMR relevance/diversity tradeoff'),
    ('cmu_target_catalog', 5000, 'Platform items at which CMU hits its weight floor'),
    ('cmu_weight_floor',   0.25, 'Minimum multiplier applied to CMU-sourced items')
ON CONFLICT (key) DO NOTHING;
