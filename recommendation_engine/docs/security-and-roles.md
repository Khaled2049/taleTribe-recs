# Security, roles, and the privacy boundary

The single most important fact about this service:

> **taleTribe-recs shares a database with story-data, and must never be able to read
> `reading_progress`.**

Everything below follows from that. The reasoning is scattered across migration
comments, `deployment.md` Part 5, and `migration-to-story-data.md`; this document is
the consolidated version.

---

## Why a database role is the thing that matters

When `recommendations` was its own database on `:5434`, the boundary was free — there
was no connection from recs to product data because there was no shared database.
Folding the schema into story-data removed that accident of topology.

What remains is a choice. Two schemas in one database means one connection string
away from every reader's history, unless something stops it. **A grant is what makes
it a boundary rather than an intention.**

### The data, ranked by sensitivity

| Table | Owner | Sensitivity | May recs read it? |
|---|---|---|---|
| `stories` (published), `story_tags`, `chapters`, `chapter_summaries` | story-data | **Public.** `ListPublicStories` serves them to any caller. | Yes — the ingest pulls them |
| `story_likes`, `story_ratings` | story-data | Semi-public; attributable to a user | **No.** Derived by story-data |
| `reading_progress` | story-data | **Private per-user history.** What someone reads and how far. | **Never** |
| `recommendations.interactions` | recs schema, written by story-data | Derived reading history — still sensitive | Yes |
| `recommendations.*` (everything else) | recs | Derived | Yes |

### Catalog is pulled; signals are pushed

```
recs  ──PULLS──→  stories, chapters, …        (public data, so no boundary crossed)
story-data  ──PUSHES──→  recommendations.interactions   (private data, derived first)
```

The asymmetry is the design. Published stories are public, so recs polling them
crosses nothing. Reader signals are not, so they arrive pre-derived by the only
service permitted to compute them — `internal/store/recommendations.go`. Everything
downstream of `interactions` (`item_stats`, `user_taste`, `item_cooccurrence`) is
computed by recs and never leaves the `recommendations` schema.

---

## The `recs_service` role

Defined in `story-data/migrations/000020_recommendations_role_grant.sql`:

```sql
GRANT USAGE ON SCHEMA public TO recs_service;
GRANT USAGE ON SCHEMA recommendations TO recs_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA recommendations TO recs_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA recommendations TO recs_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA recommendations GRANT ... TO recs_service;
```

Four things about this are worth understanding rather than skimming:

**`USAGE ON SCHEMA public` is not a leak.** It is required so the role can resolve the
`vector` type and the `hnsw`/`gin` operator classes, which the extensions install in
`public`. It grants access to no table, and no table grant is issued.

**`ALTER DEFAULT PRIVILEGES` covers future tables.** Without it, a table added by a
later migration would be invisible to recs until someone remembered to grant it —
which would present as a mysterious permission error long after the migration landed.

**No `CREATE`.** recs does not migrate, so it does not need it. story-data's own role
owns the schema and runs goose.

**The whole thing is guarded on the role existing**, so it is a no-op in local
development and in tests, where one superuser owns everything. That is convenient and
it is also why the gap below has never been noticed.

### ⚠ Known gap: the ingest cannot run as this role

`ingest/platform.py` reads `stories`, `story_tags`, `chapters` and
`chapter_summaries` — all in `public` — and `_retire_unpublished` runs
`UPDATE recommendations.items … FROM stories`. The grant above issues **no `SELECT`
on any `public` table**.

The first time `recs_service` actually exists, the ingest fails with
`permission denied for table stories`. It has never been observed because the role
does not exist anywhere yet; local development runs as `postgres`.

The migration plan contains both halves of the contradiction: decision 0.2 says
"`GRANT USAGE ON SCHEMA recommendations` and nothing on `public`", while decision 0.3
says "recs polling `stories` is not a privacy boundary crossing." Both are
defensible; only one is implemented.

**Resolution is a deploy blocker, and there are two honest options:**

1. Add `GRANT SELECT ON stories, story_tags, chapters, chapter_summaries TO
   recs_service` to a new story-data migration. Simple, and consistent with decision
   0.3 — these are public data. It widens the role slightly for the serving path too.
2. Give the *ingest job* a second, separate role with those reads, and leave the
   serving role confined. More precise, more moving parts. Preferable if the ingest
   ever runs somewhere the serving container does not.

Option 1 is the smaller change and matches the stated principle. Either way it must
be decided before the role is created, because the checklist item "create the
`recs_service` role, then re-run migrations" is what makes the failure appear.

---

## The request-path trust chain

The browser never reaches this service. Two hops, each with a distinct job:

```
Browser
  │  Firebase ID token (the reader's own identity)
  ▼
Firebase Function  recommendStories / explainRecommendations
  │                functions/src/endpoints/recommendations.ts
  │  ← THE TRUST BOUNDARY
  │  Google OIDC token, audience = RECS_SERVICE_URL
  ▼
taleTribe-recs  verify_internal_token()
  │  recs_service role
  ▼
Postgres — recommendations schema only
```

### Hop 1 — the Function is the trust boundary

Two properties make it one, and both must survive any future edit:

**`user_id` comes from the verified token, never the body.** The Function sets
`user_id: userId` where `userId` was resolved by `requireAuth` from the caller's
Firebase ID token. The Zod schema is `.strict()`, so a `user_id` in the request body
is rejected outright rather than silently overriding.

Without this, `/recommend/behavioral` — which returns recommendations derived from a
reader's private history — would accept any uid a browser cared to type. **`user_id`
arriving in a request body is trusted by recs precisely because the Function is the
boundary.** Anyone touching the route models must preserve that.

**The browser never learns the Cloud Run URL or holds an OIDC token.** The Function
mints the identity token server-side with `GoogleAuth.getIdTokenClient(audience)`.
`getRecommendationServiceUrl()` additionally refuses a localhost URL outside the
emulator, so a misconfigured deploy fails loudly rather than calling nothing.

Input ceilings are enforced here too, and they are the real caps on LLM spend:
`topK` ≤ 50, `books` ≤ 10, `prompt` ≤ 2,000 chars, `itemIds` ≤ 25.

### Hop 2 — `verify_internal_token`

In production it validates the OIDC token's signature, expiry, and **audience**
(`RECS_SERVICE_URL`), then requires the token's `email` claim to be on a configured
allowlist. Audience checking is what stops a token minted for some other Cloud Run
service being replayed here.

**It is a no-op when `RECS_SERVICE_URL` is unset**, which is what lets local
development run with no credentials. That is a deliberate convenience with a sharp
edge: a production deployment that forgets to set `RECS_SERVICE_URL` is *completely
unauthenticated*. `RecSettings` production validators reject the local DSN and require
a Gemini key; **`recs_service_url` should be treated as equally mandatory** and Cloud
Run invoker IAM should be restricted to the Functions service account regardless, so
the network layer fails closed even if the application layer is misconfigured.

---

## Threat table

| Threat | Status | What stops it |
|---|---|---|
| Reader requests another reader's behavioral recommendations | **Blocked** | Function sets `user_id` from the verified Firebase token; Zod `.strict()` rejects it in the body |
| recs reads `reading_progress` | **Blocked by design, gap in practice** | `000020` grants no `public` access — but see the ingest gap above |
| Browser calls Cloud Run directly | **Blocked** (once deployed) | OIDC audience check + Cloud Run invoker IAM |
| OIDC token replayed from another service | **Blocked** | Audience pinned to `RECS_SERVICE_URL`; caller email allowlist |
| Unauthenticated production deploy | **Not enforced** | `verify_internal_token` silently no-ops when `RECS_SERVICE_URL` is empty |
| Ungoverned Gemini spend | **Weakly mitigated** | Explanation cache + 6 req/min bucket. No daily cap, no budget alert. |
| Prompt injection via story text | **Not mitigated** | See below |
| Reading history leaks from `interactions` | **Not mitigated beyond the role** | No pseudonymisation; raw Firebase uids |

### Prompt injection is not mitigated

Two paths feed **user-authored text** into a Gemini prompt:

- **Normalization** (`ingest/normalize_llm.py`) reads story descriptions and chapter
  summaries. An author can write a story description containing instructions.
- **Explanations** (`explain.py`) include the item's LLM-derived premise, themes and
  tone in the prompt, plus the reader's own free-text query.

The blast radius is genuinely small, which is why this is recorded rather than
urgent: normalization output is constrained to a controlled vocabulary
(`vocabularies.py`) and filtered on read, and an explanation is one sentence rendered
as text in a card. There is no tool use, no code execution, and no privileged data in
either prompt's context.

The realistic attacks are **catalog poisoning** (an author crafts a description that
makes the model emit themes matching every query, inflating their reach) and
**explanation defacement** (an author gets attacker-controlled prose shown to readers
as TaleTribe's own recommendation copy). Both argue for treating `core_premise` and
explanation text as untrusted user content in the UI — escape it, never render it as
markup — and for the confidence gate already in place.

### `interactions` is a reading-history database

Once story-data derives it, `recommendations.interactions` holds, per Firebase uid,
which stories a person read and how far. It deserves the same care as the source
tables: a restricted role, no ad-hoc analyst access, and a documented retention
position.

**Pseudonymising `user_id`** — HMAC with a pepper in Secret Manager rather than the
raw uid — limits damage from a database compromise and makes erasure a matter of
dropping rows by hashed key. It is a **one-way door**: rotating the pepper invalidates
every row, and joining back to a uid for debugging becomes impossible. It also has to
be applied inside story-data's derivation, the only thing that sees the raw uid.

**Decide before the first real sync, not after.** After that, changing it means
discarding the accumulated signals.

For GDPR/CCPA erasure the shape already exists: `purge_synthetic(prefix=...)` is a
delete-by-user-key over `interactions` and `user_taste`. An
`/internal/users/forget` endpoint is the same query with a different predicate.

---

## Secrets and least privilege

Three secrets, and nothing else:

| Secret | Used by |
|---|---|
| `recs-postgres-dsn-rw` | ingest, sync |
| `recs-postgres-dsn-ro` | request serving |
| `recs-gemini-api-key` | HyDE, explanations, normalization, embeddings |

The service account should hold `roles/secretmanager.secretAccessor` on exactly those
three and nothing more. **No Postgres superuser** — recs connects as `recs_service`,
which has DML on one schema and no `CREATE`.

The read/write split is a privilege boundary as well as a performance one: the
serving path uses the RO DSN, so a bug in a request handler cannot write to the
catalog.

### Local development is deliberately unguarded

`RECS_DATABASE_URL` defaults to story-data's local stack, `verify_internal_token`
no-ops without `RECS_SERVICE_URL`, and everything runs as `postgres`. That is what
makes zero-config local dev possible — and it is why `RecSettings` has explicit
production validators that reject the local DSN when `ENVIRONMENT=production`. Read
`config.py`'s validators before adding a new default.

---

## What to check before going to production

- [ ] **Resolve the ingest grant gap** (above) — otherwise the first real ingest fails
- [ ] `RECS_SERVICE_URL` set, so `verify_internal_token` is not a no-op
- [ ] Cloud Run invoker IAM restricted to the Functions service account
- [ ] Caller email allowlist populated
- [ ] `recs_service` created; migrations re-run so `000020` has a grantee
- [ ] Decide `user_id` pseudonymisation — one-way door
- [ ] Daily per-user explanation cap and a Gemini budget alert
- [ ] Confirm `core_premise` and explanation text are escaped in the UI

Related: [jobs.md](jobs.md) for what each job is permitted to touch,
[deployment.md](deployment.md) Part 5 for the infrastructure view,
[migration-to-story-data.md](migration-to-story-data.md) for why the schemas share a
database at all.
