# Deployment — services, security, scale, production readiness

> **Status: designed, not built.** Deliberately deferred so the design could settle
> first. Everything here follows patterns already proven in this workspace —
> `taleTribe-agents` and `creditProxy` both deploy this way — so it's an extension of
> house style, not a new stack.

---

## Part 1: Target architecture

```
Browser
  │
  ├── ranking + sync explanations ──→ Firebase Functions ──OIDC──→ Cloud Run
  │                                   (Stage A: service stays PRIVATE)
  └── streaming explanations ─────────Firebase ID token───────────→ Cloud Run
                                      (Stage B: requires public ingress)
                                                │
                    ┌───────────────────────────┴──────────────────────┐
                    ▼                                                  ▼
            story-data's Neon project                          Gemini API
            (pgvector, `recommendations` schema)               embeddings +
            RW primary  ← ingest                               flash-lite
            RO compute  ← serving
```

**The database is story-data's, not this service's.** The `recommendations`
schema lives in it and story-data migrates it; recs connects with its own role
(`recs_service`, migration `000020`) which has usage on that schema and nothing
else. Reader signals arrive from `story-data sync-recs`, not from an export
pipeline — see "Data privacy" below.

| Service | Role | Why this one |
|---|---|---|
| **Cloud Run** | the service itself | Already how `taleTribe-agents` and all four creditProxy services deploy. Scale-to-zero, request-scoped billing, native OIDC. |
| **Neon Postgres** + pgvector | catalog, vectors, signals | creditProxy already uses Neon in production (`NEON_DATABASE_URL` → `creditproxy-postgres-dsn`), so it's an existing operational dependency, not a new vendor. Supports read replicas and pgvector ≥ 0.8. |
| **Artifact Registry** | container images | `us-central1-docker.pkg.dev/story-6f89f/...`, same as agents. |
| **Secret Manager** | DSNs, API key | Same create-or-update-if-changed pattern as agents/creditProxy. |
| **Cloud Scheduler** + Cloud Run Jobs | nightly sync, stats refresh | Ingest must not run in a request path. |
| **Firebase Functions** | the browser's front door | Existing auth bridge (`agentService.ts` pattern). |
| **Cloud Logging / Monitoring** | observability | structlog already emits JSON, which becomes queryable for free. |
| **GitHub Actions** + Terraform | CI/CD | Existing WIF keyless setup; state in `gs://story-6f89f-tfstate`. |

**Deliberately not used:** a dedicated vector database (Postgres is already here and the
workload is half relational), Cloud SQL (Neon is the established Postgres), Redis
(nothing needs a shared cache — the explanation cache is in Postgres and the query-vector
cache is per-instance by design).

---

## Part 2: The container

Follows the agents `Dockerfile` almost exactly, with three differences that matter.

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
# Export ONLY the deps this service needs. `--with recommendations` pulls asyncpg
# and pgvector; the story agent's image must never grow them, and this image must
# never grow torch/diffusers from the image-gen group.
RUN pip install --no-cache-dir poetry poetry-plugin-export \
 && poetry export -f requirements.txt --only main --with recommendations \
      --without-hashes -o /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip uninstall -y poetry poetry-plugin-export

# Explicit COPY per package. A missing COPY ships a silently degraded image —
# the same warning the agents Dockerfile carries.
COPY agents/storyAgent/brain/embedding_provider.py agents/storyAgent/brain/
COPY agents/__init__.py agents/storyAgent/__init__.py agents/storyAgent/brain/__init__.py ./
COPY rate_limit.py ./
COPY recommendation_engine/ ./recommendation_engine/

# Never run as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import os,httpx; httpx.get(f'http://localhost:{os.getenv(\"PORT\",\"8100\")}/health',timeout=5)" || exit 1

CMD ["python", "-m", "recommendation_engine.server"]
```

**Three deliberate differences from the agents image:**

1. **`--with recommendations`.** The optional Poetry group is the mechanism keeping the
   two images' dependency sets disjoint.
2. **A non-root user.** The agents image runs as root; new services shouldn't. Cheap to
   add now, annoying later.
3. **A narrower COPY set.** This service needs exactly one module from `agents/` —
   `embedding_provider.py`, for the 768-dimension contract. Copying the whole story agent
   would drag in Firestore, MCP and the LLM provider for nothing.

**Image size discipline.** There is no corpus to exclude any more — the catalog is read
from story-data at ingest time rather than shipped in the image. Keep `.dockerignore`
narrow anyway: `tests/`, `recommendation_engine/docs/`, `.env`, and the caches.

---

## Part 3: Infrastructure as code

A **new Terraform root module** at `recommendation_engine/terraform/`, not additions to
the agents module. Separate state, separate blast radius, independent `apply`.

```hcl
terraform {
  backend "gcs" {
    bucket = "story-6f89f-tfstate"
    prefix = "novelsync-recs"        # agents uses "novelsync-agents"
  }
  required_providers { google = { source = "hashicorp/google", version = "~> 5.0" } }
}

resource "google_cloud_run_v2_service" "recs" {
  name     = "novelsync-recs"
  location = "us-central1"
  ingress  = var.enable_public_access ? "INGRESS_TRAFFIC_ALL" : "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.recs.email

    scaling {
      min_instance_count = var.min_instances   # 1 — see cold start below
      max_instance_count = var.max_instances   # 10 — bounds DB connections
    }

    max_instance_request_concurrency = 40
    timeout                          = "300s"   # SSE streams are long-lived

    containers {
      image = var.image                        # pinned to ${{ github.sha }}
      resources {
        limits    = { cpu = "1", memory = "1Gi" }
        cpu_idle  = false                       # keep the pool + page cache warm
        startup_cpu_boost = true
      }

      env { name = "ENVIRONMENT"      value = "production" }
      env { name = "RECS_SERVICE_URL" value = local.service_url }   # OIDC audience
      env { name = "RECS_DB_POOL_MAX" value = "5" }

      env {
        name = "RECS_DATABASE_URL_RO"
        value_source { secret_key_ref { secret = "recs-postgres-dsn-ro", version = "latest" } }
      }
      # ... recs-postgres-dsn-rw, recs-gemini-api-key

      startup_probe  { http_get { path = "/health" } timeout_seconds = 5  failure_threshold = 10 }
      liveness_probe { http_get { path = "/health" } timeout_seconds = 3  period_seconds = 30 }
    }
  }

  lifecycle {
    # Copied from the agents module: public ingress must never be reachable
    # without production auth actually being switched on.
    precondition {
      condition     = !var.enable_public_access || var.environment == "production"
      error_message = "enable_public_access requires ENVIRONMENT=production"
    }
  }
}
```

**Why a `lifecycle.precondition`.** Public ingress plus a non-production `ENVIRONMENT`
means `oidc_audience` returns `None`, so `verify_internal_token` becomes a **no-op** — an
open, unauthenticated service. The precondition makes that combination impossible to
apply rather than merely inadvisable. The agents module uses the same guard for the same
reason.

---

## Part 4: CI/CD

Two workflows, mirroring the agents repo, both **path-filtered** on
`recommendation_engine/**` so unrelated commits don't redeploy.

### `pr-check-recs.yml`

`black --check` · `isort --check-only` · `ruff check` · `pytest tests/test_rec_*.py` ·
`terraform fmt -check` · `terraform validate`.

Runs with `USE_MOCK=true` and **no** `RECS_TEST_DATABASE_URL`, so integration tests
self-skip. To exercise them in CI, add a `pgvector/pgvector:pg16` service container and
set the variable — worth doing, since integration tests found the one class of bug unit
tests couldn't.

### `deploy-recs.yml` (push to `main`)

```
test ──→ build ──→ push ──→ secrets ──→ MIGRATE ──→ terraform apply ──→ smoke
```

1. **Auth**: Workload Identity Federation (`secrets.WIF_PROVIDER` +
   `WIF_SERVICE_ACCOUNT`), `id-token: write`. **No long-lived service-account JSON key
   anywhere** — the existing repos already work this way.
2. **Artifact Registry**: idempotent create, plus a cleanup policy keeping 3 images.
3. **Secrets**: create-or-update **only if the value changed**, so unchanged secrets
   don't accrue versions.
4. **Migrations** — see below.
5. **Terraform**: `plan` then `apply`, with `-var="image=...:${{ github.sha }}"`. Images
   are pinned to a commit SHA, never `latest`.
6. **Smoke test**: mint an ID token, `curl /health` with 5 retries, and **assert the
   body**, not just a 200 — `pgvector_ok`, `hnsw_index_present`,
   `embedding_dimension_ok`. Those are exactly the conditions whose absence is silent.

### This service does not migrate

The `recommendations` schema is created by story-data's goose migrations
(`000019`, `000020`), applied by story-data's own deploy under an advisory lock.
recs has no migration runner; `/health` reports `schema_present: false` and goes
red if it is started against a database story-data has not migrated.

**The ordering constraint that follows:** a recs revision that needs a new column
cannot ship before the story-data migration that adds it. In practice that means
schema changes for recs are ordinary story-data PRs, deployed first.

**Migrations must be forward-only and additive.** Cloud Run can roll a revision back
instantly; it cannot roll a schema back. So: add columns, don't drop them; add tables,
don't rename them. A revision rollback has to remain safe against the newer schema.

### The two feed jobs

Neither runs in the serving process, and neither is scheduled yet.

| Job | Where it runs | What it needs |
|---|---|---|
| `story-data sync-recs` | story-data's image, as a Cloud Run **Job** | the RW DSN. Full re-derivation; scans the social tables. |
| `python -m recommendation_engine.ingest.platform` | this image, as a Cloud Run **Job** | the RW DSN + `GOOGLE_AI_STUDIO_API_KEY`. Incremental on a stored cursor. |
| `python -m recommendation_engine.sync.seed --refresh-only` | this image | recomputes `item_stats`, `pop_score`, `user_taste` from `interactions`. Run **after** the other two. |

Cloud Scheduler → Cloud Run Jobs is the smaller wiring than an authenticated
endpoint, and it keeps a full table scan off the request path. Order matters:
ingest (catalog) → sync-recs (signals) → refresh (aggregates).

**Never run `--rebuild-index` against the serving database.** It drops the HNSW
graph before recreating it, so every query in between falls back to a sequential
scan. It is a maintenance command for a Neon branch, not a scheduled step.

---

## Part 5: Security

### Identity: three distinct layers

| Layer | Mechanism | Where verified |
|---|---|---|
| Service → service | Google **OIDC ID token**, audience = `RECS_SERVICE_URL`, plus an email allowlist | `verify_internal_token` |
| Reader identity (Stage A) | Firebase ID token forwarded as `X-Firebase-Token` | Functions verify; service trusts the OIDC caller |
| Reader identity (Stage B) | Firebase ID token verified **in-process** | the service itself |

Note there is **no shared secret** anywhere — the whole chain is short-lived tokens. That
matches `agentService.ts`, where despite the name "internal service token" the mechanism
is a minted OIDC identity token.

### Ingress: Stage A keeps the service completely private

This is the most consequential security decision, and it's worth stating plainly:

**Stage A** (sync explanations through Functions) means the service can run with
`ingress = INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER` and IAM `run.invoker` granted **only**
to the Functions service account. Not reachable from the internet at all. Zero public
attack surface.

**Stage B** (streamed explanations) requires the browser to connect directly, because
Functions gen2 buffers responses and cannot proxy SSE. That means public ingress, and
therefore:

- in-process Firebase ID token verification (reuse the MCP server's
  `verify_firebase_token`, which needs no firebase-admin);
- a CORS allowlist of the three production origins, not `*`;
- a **per-IP** throttle on the stream endpoint (the MCP server's
  `OAuthThrottleMiddleware` is the existing pattern) — the per-*user* bucket doesn't help
  before a token is validated;
- the `cache_key` capability property: a stream requires a key only an authenticated,
  rate-limited call could have minted.

**Recommendation: ship Stage A first and treat Stage B as an explicit, separately
reviewed decision.** Streaming is a UX nicety; a private service is a security property.
Gate it behind `VITE_ENABLE_REC_STREAMING` so it's one env var from being reverted.

### Data privacy

**Reading history is the sensitive data here.** `reading_progress` in story-data is
private per-user history — a table this service is not permitted to read at all.
Consequences:

- The derivation is **story-data's**, in `internal/store/recommendations.go`, and
  migration `000020` grants `recs_service` usage on the `recommendations` schema
  only (plus `public` solely to resolve the `vector` type — no table grant). That
  grant is what makes co-locating the two schemas safe; without it, sharing a
  database silently hands recs a second, unaudited path to every reader's history.
- Once derived, `recommendations.interactions` is a reading-history database. It deserves
  the same care as the source tables: restricted role, no ad-hoc analyst access, and
  a documented retention position.
- **Consider pseudonymising `user_id`** — HMAC with a pepper in Secret Manager rather than
  the raw Firebase uid. It limits damage from a Postgres compromise and makes deletion
  requests a matter of dropping rows by hashed key. But **it's a one-way door**: rotating
  the pepper invalidates every row, and joining back to a uid for debugging becomes
  impossible. It also has to be applied inside story-data's derivation, which is the
  only thing that sees the raw uid. Decide before the first real sync, not after.
- **`/internal/users/forget`** for GDPR/CCPA erasure: delete from `interactions` and
  `user_taste` by user key. `purge_synthetic(prefix=...)` is already the shape of this.

### Least privilege

A dedicated service account with exactly: `roles/secretmanager.secretAccessor` on its
three secrets, `roles/datastore.user` (only if it reads Firestore directly), and nothing
else. **No Postgres superuser** — recs connects as `recs_service`, which has DML on
the `recommendations` schema and nothing else; it does not need `CREATE`, because it
does not migrate. story-data's own role owns the schema and runs the migrations.

### Supply chain and secrets hygiene

Images pinned to a commit SHA. `poetry.lock` committed and installed from. WIF instead of
key files. Secrets only ever via `value_source.secret_key_ref`, never a plain `env` value
— a plain value is visible in the Cloud Run console and in Terraform state.

**Terraform state contains secret *references*, but plan output can leak values.** Keep
the state bucket private and don't post `terraform plan` output into PR comments.

---

## Part 6: Scalability

### Cloud Run

`min_instance_count = 1` and `cpu_idle = false`. Both are needed together: the first
avoids cold starts (~2–5s including pool setup), the second stops CPU being throttled
between requests, which would let the connection pool and page cache go cold anyway.

`max_instance_request_concurrency = 40`. This is an async service whose work is mostly
awaiting Postgres and Gemini, so an instance handles many concurrent requests happily.
The ceiling is really the connection pool, below.

### Connection math — the constraint that bites first

```
peak connections = max_instances × RECS_DB_POOL_MAX
                 = 10 × 5 = 50
                 + ingest/sync jobs (2 × 5 = 10)
                 = 60
```

Postgres connections are expensive, and Neon's direct-endpoint limit scales with compute
size. Two mitigations, and we want both:

1. **Use Neon's pooled endpoint** (`-pooler` host) for serving. It's PgBouncer in
   transaction mode, which multiplexes many clients onto few backends.
2. **`statement_cache_size=0`** on the asyncpg pool — **already implemented**. Transaction
   pooling plus asyncpg's prepared-statement cache produces intermittent
   `prepared statement "__asyncpg_stmt_x__" already exists` errors that appear only under
   concurrency, i.e. never in local testing.

And the reason `SET LOCAL` is used for every HNSW setting rather than a session `SET`:
under transaction pooling a session setting is discarded or leaks to an unrelated caller,
and the query would silently run at pgvector's default `ef_search = 40`. Already
implemented, and the reason it exists is this deployment topology.

### Sizing Postgres from measured numbers

Measured on the live database: **4,296 bytes per row** of HNSW index (768 floats = 3,072
bytes, plus `m = 16` graph links), and 31 MB total relation size at 1,987 rows —
measured on the old CMU-seeded catalog, and still the right per-row figure to
extrapolate from, since a row's size does not depend on where the story came from.

| Catalog | HNSW index | Total relation | Neon compute |
|---|---|---|---|
| 2k (today) | 8 MB | 31 MB | 0.25 CU |
| 16.5k (full corpus) | ~71 MB | ~260 MB | 0.5–1 CU |
| 200k | ~860 MB | ~3.2 GB | **≥ 2 CU** (8 GB) |
| 1M | ~4.3 GB | ~16 GB | **≥ 4 CU**, or truncate to 256 dims |

The number to watch is whether the **HNSW index stays resident in memory**. Once the
graph spills to disk, latency degrades sharply and non-linearly — a graph traversal turns
into random I/O. 1 Neon CU = 1 vCPU + 4 GB, so size compute against the index, not the
table.

**The scaling lever is Matryoshka truncation**: add a `vector(256)` **column** with its own
index — a third of the vector bytes. Add a column; never mutate the 768 one, which would
invalidate everything at once.

### Read/write split

`RECS_DATABASE_URL_RO` points at a dedicated Neon read-only compute; `_RW` at the primary.
`Database` already opens two pools and shares one when the DSNs match, so this is
configuration, not code. Ingest and sync hit the primary; serving never does. Vector scans
therefore never share compute with the billing workload.

**Autosuspend must be disabled on the serving compute.** Neon suspends idle computes and
resume takes 500 ms–3 s, which would dominate every request after a quiet period.

### The planner crossover

Currently the planner chooses a sequential scan over the HNSW index, because at 2,000 rows
brute force genuinely is cheaper. Estimated costs are within ~6% (780 vs 828), so the
crossover is close.

Practical implications for deployment:

- Latency measured today reflects brute force and **will not resemble** post-crossover
  behaviour. Don't set SLOs from current numbers.
- `ef_search` and `iterative_scan` are inert until the flip. They're correct and waiting.
- Worth a **periodic `EXPLAIN` check** in monitoring: if the index is present but unused,
  say so, rather than assuming it's working. See the alerting section.

### Scaling the LLM paths

Ranking scales with Postgres. HyDE and explanations scale with Gemini quota, which is
**per-project and shared with the story agent** — a recommendation traffic spike could
degrade chapter generation. Mitigations: the explanation cache (hit rate is the metric to
watch), the 6/min per-user bucket, and if it becomes real, a separate API key or project
for this service.

---

## Part 7: Production readiness

### Health and probes

`/health` already returns **503** unless pgvector ≥ 0.8.0, the HNSW index exists, and the
embedder reports 768 dimensions. Wired to the Cloud Run **startup probe**, that means a
misconfigured revision never receives traffic.

Use a generous `failure_threshold` on startup (the pool must open) and a light
`liveness_probe` — liveness should detect a wedged process, not re-litigate dependencies.
A liveness probe that fails because Postgres is briefly unreachable will restart-loop the
service and make an outage worse.

### Observability

structlog already emits JSON, so Cloud Logging indexes the fields for free. Turn these
into log-based metrics:

| Signal | Why it matters |
|---|---|
| `/health` non-200 | the composite dependency check |
| `degraded: true` rate | retrieval timeouts under selective filters |
| p95 latency **per mode** | behavioral / ad-hoc / HyDE have different profiles |
| `LLM_RATE_LIMITED` rate | either abuse or a bucket set too tight |
| explanation cache hit rate | the primary cost control; a drop means a spend increase |
| `knn_timeout` | `statement_timeout` being hit |
| Gemini 429s / retry counts | quota pressure, shared with the story agent |
| `ingest_runs` where `status='failed'` | a silently stale catalog |

### Alert on the silent failures specifically

This project's dominant failure mode is *no error, quietly worse results*. Generic
uptime monitoring will not catch any of these, so each gets its own check:

```sql
-- 1. Mixed embedders in one vector space (should always return exactly one row)
SELECT embed_model, count(*) FROM recommendations.items
 WHERE embedding IS NOT NULL GROUP BY embed_model;

-- 2. Catalog going stale
SELECT max(finished_at) FROM recommendations.ingest_runs WHERE status='completed';

-- 3. Rows eligible but unembedded (invisible to the partial index)
SELECT count(*) FROM recommendations.items WHERE is_eligible AND embedding IS NULL;

-- 4. Is the HNSW index actually being used? (see the planner crossover above)
EXPLAIN SELECT id FROM recommendations.items WHERE is_eligible
 ORDER BY embedding <=> $1 LIMIT 10;
```

Plus a **scoring-config sanity check**: `w_pop_ceiling + w_cf_ceiling ≤ 0.45`. The
`config` table is editable at runtime by design, which means it's editable *wrongly* at
runtime. `ScoringConfig` clamps on read, but an alert makes a bad edit visible rather than
merely survivable.

### SLOs

Set them per mode, because one number would be dishonest — ad-hoc must embed the query,
and a Gemini round trip alone is 80–250 ms:

| Mode | Target p95 | Dominated by |
|---|---|---|
| Behavioral | **< 150 ms** | Postgres only — no embedding call |
| Ad-hoc in-corpus | **< 500 ms** | one embedding round trip |
| HyDE | **< 2 s** | LLM generation + embedding |
| Explanation TTFB (stream) | **< 1 s** | LLM first token |

Re-baseline after the planner crossover.

### Rollback

Cloud Run keeps revisions, so rollback is a traffic split — near-instant. Consider
`--no-traffic` deploys plus a manual promote for risky changes.

**But schema and data changes don't roll back.** Forward-only migrations, additive
changes, and a re-run of the ingest is always safe (it's idempotent and skips rows whose
`embed_input_sha` is unchanged). A `PROMPT_VERSION` bump is the one genuinely expensive,
hard-to-undo operation — it invalidates the normalization cache and costs a full
re-derive of every story.

### Cost controls

| Control | Mechanism |
|---|---|
| Explanation spend | deterministic cache + 6/min bucket + bounded `max_output_tokens` |
| Runaway scaling | `max_instance_count` |
| Postgres | Neon compute cap and autoscaling bounds |
| Ingest | `--limit`, the stored cursor, and the `embed_input_sha` skip |

**Explanations bypass creditProxy and are therefore unmetered.** There is no ledger entry,
so the cache and the rate bucket are the *only* things between a client-side loop and a
bill. Worth a **daily per-user explanation cap** in addition to the per-minute bucket, and
a budget alert on the Gemini project.

### Disaster recovery

Everything in this database is **derivable**. That's an unusually comfortable position:

| Data | Recovery |
|---|---|
| `items` (catalog) | re-run `ingest.platform --full` — one embedding pass over the published catalog |
| `interactions` | re-run `story-data sync-recs`; `story_likes`/`story_ratings`/`reading_progress` are the system of record |
| `item_stats`, `user_taste`, `item_cooccurrence` | `--refresh-only`, minutes, free |
| `explanation_cache` | regenerates on demand |
| `normalization_cache` | the one worth backing up — it is the accumulated LLM spend |

So the RPO can be relaxed. Enable Neon PITR anyway (it's cheap), but the real backup is
that story-data's own tables hold the source of truth and every derivation is
reproducible — and since both now live in one Neon project, a PITR restore recovers
the source and the derivations together rather than leaving them at different points
in time.

---

## Part 8: Ordered checklist

**Infrastructure**
- [ ] Neon: database + `recommendations` schema, dedicated RO compute, **autosuspend off**, pgvector ≥ 0.8
- [ ] Measure Cloud Run → Neon RTT; if > 15 ms, reconsider the region
- [ ] Secrets: `recs-postgres-dsn-rw`, `recs-postgres-dsn-ro`, `recs-gemini-api-key`
- [ ] Service account with only the three secret bindings
- [ ] Artifact Registry repo + cleanup policy

**Code**
- [ ] `Dockerfile` with `--with recommendations`, non-root user, narrow COPY set
- [ ] `.dockerignore` excluding `tests/`, `recommendation_engine/docs/`, `.env`
- [ ] Terraform root at prefix `novelsync-recs`, with the `enable_public_access` precondition
- [ ] `deploy-recs.yml` and `pr-check-recs.yml`, path-filtered
- [ ] No migration step here — story-data owns the schema and must deploy first
- [ ] Confirm `statement_cache_size=0` against the pooled endpoint under load

**Data**
- [ ] Confirm the Neon project ships pgvector >= 0.8 (`hnsw.iterative_scan`)
- [ ] Create the `recs_service` role, then re-run story-data's migrations so `000020` grants it
- [ ] Point `RECS_DATABASE_URL` at story-data's project; `RECS_DATABASE_URL_RO` at a read compute
- [ ] Ingest the catalog (`ingest.platform --full`)
- [ ] Decide `user_id` pseudonymisation **before** the first sync — one-way door, and it
      belongs in story-data's derivation, the only thing that sees the raw uid
- [ ] Schedule the three jobs in order: ingest → `sync-recs` → `--refresh-only`
- [ ] `--purge` the synthetic readers

**Operations**
- [ ] Log-based metrics + the four silent-failure queries
- [ ] Per-mode SLOs and dashboards
- [ ] Gemini budget alert; daily per-user explanation cap
- [ ] Verify the smoke test asserts the `/health` **body**, not just the status
- [ ] Ship Stage A. Treat Stage B as a separate security review.

---

## Part 9: Known gaps

Honest list of what isn't solved.

**No load test.** Every latency number is single-request and pre-crossover. Concurrency
behaviour, pool saturation and the pooled-endpoint interaction are all untested under
load.

**Gemini quota is shared with the story agent.** A recommendation spike could degrade
chapter generation. No isolation today beyond rate buckets.

**Explanations are unmetered.** Accepted, but it means cost control is entirely
preventative.

**`verify_internal_token` is duplicated** between this service and the agents service.
Logged as debt; the fix is extracting a shared `service_auth.py`.

**Single region.** No multi-region story, and Neon adds a second regional dependency.
Fine for now; worth knowing.

**Quality is unmeasured** — see [evaluation.md](evaluation.md). Deploying an unmeasured
ranker is defensible for an MVP, but it means a regression could ship unnoticed. The
recall sweep and a genre-holdout tripwire would be the minimum bar before this is
load-bearing for readers.
