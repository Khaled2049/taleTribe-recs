# Deployment

Production deployment is defined in recommendation_engine/terraform and
.github/workflows/deploy.yml. Story-data owns and applies the database schema.

## Architecture

~~~text
Browser -> Firebase Functions -> Google OIDC -> novelsync-recs
                                              -> Neon
                                              -> Gemini

Cloud Scheduler (paused for initial rollout)
  -> novelsync-recs-pipeline
       1. novelsync-recs-ingest
       2. novelsync-story-data-sync-recs
       3. novelsync-recs-refresh
~~~

Cloud Run accepts HTTPS ingress because Firebase Functions calls its URL, but
it is private by IAM: no allUsers binding exists. Only the Functions runtime
and deployment identities can invoke it, and the application also validates
the OIDC audience and caller email.

## Managed resources

| Kind | Name |
|---|---|
| Cloud Run service | novelsync-recs |
| Catalog job | novelsync-recs-ingest |
| Aggregate job | novelsync-recs-refresh |
| Signal job (story-data) | novelsync-story-data-sync-recs |
| Workflow | novelsync-recs-pipeline |
| Scheduler | novelsync-recs-nightly |
| Terraform state | gs://story-6f89f-tfstate/novelsync-recs |

The service scales to zero, as novelsync-agents does: no minimum instance, and
CPU is billed only while a request is being served. It runs at most ten
instances, concurrency 40, one CPU, 1 GiB memory, and at most five connections
per database pool. The first request after an idle spell pays a cold start of a
few seconds; set min_instances = 1 (keeping cpu_idle = true) to trade a small
idle charge for a warm instance.

## Prerequisites

Secret Manager must contain enabled versions of:

- recs-postgres-dsn-rw: pooled recs_service primary DSN.
- recs-postgres-dsn-ro: pooled recs_service read DSN.
- google-ai-studio-api-key: production Gemini key.

The GitHub repository needs WIF_PROVIDER and WIF_SERVICE_ACCOUNT, and the WIF
service-account policy must trust Khaled2049/taleTribe-recs.

Terraform supplies all non-secret runtime values. Never put a database DSN in
GitHub, Terraform variables, a browser bundle, or a committed env file.

## CI/CD

PR validation runs Black, isort, Ruff, the full pytest suite against an
ephemeral pgvector database migrated by story-data, Terraform validation, and
a Docker build.

The main workflow repeats the tests, verifies all prerequisites, pushes an
image tagged with the commit SHA, applies Terraform, mints an OIDC token, and
checks the complete health response.

The scheduler is created paused. RECS_SCHEDULER_PAUSED is read from GitHub
repository variables; an absent value means true.

## Rollout order

1. Deploy story-data so migrations 19 through 23 and its signal job exist.
2. Deploy taleTribe-recs.
3. Run the initial load below.
4. Deploy taleTribe-frontend so Functions receive the service URL.
5. Exercise both recommendation Functions.
6. Run the pipeline once by hand and confirm all three steps succeed. The
   initial load runs the jobs with your own credentials, so this is the first
   run under the workflow's service account:
   `gcloud workflows run novelsync-recs-pipeline --project=story-6f89f --location=us-central1`
7. Set RECS_SCHEDULER_PAUSED=false and rerun the recs deployment.

Do not deploy recs before story-data. Its startup probe intentionally fails
until the recommendations schema and HNSW index exist.

## Initial load

~~~bash
gcloud run jobs execute novelsync-recs-ingest \
  --project=story-6f89f --region=us-central1 \
  --args=-m,recommendation_engine.ingest.platform,--full --wait

gcloud run jobs execute novelsync-story-data-sync-recs \
  --project=story-6f89f --region=us-central1 --wait

gcloud run jobs execute novelsync-recs-refresh \
  --project=story-6f89f --region=us-central1 --wait
~~~

Scheduled runs use incremental ingest.

## Verification

The authenticated health response must show:

~~~text
status = ok
database.connected = true
database.schema_present = true
database.pgvector_ok = true
database.hnsw_index_present = true
embedding_dimension_ok = true
embed_model_ok = true
database.item_count > 0
database.eligible_count > 0
~~~

The database boundary needs no manual check. A missing recommendations-schema
grant fails this health check (schema_present reads information_schema, which
lists only tables the role can access), a missing catalog grant fails the
ingest job with permission denied, and story-data's
TestRecommendationRolePrivileges asserts in CI that recs_service cannot read
reading_progress, story_likes or story_ratings.

## Enabling and stopping the schedule

~~~bash
gh variable set RECS_SCHEDULER_PAUSED \
  --repo Khaled2049/taleTribe-recs --body false
~~~

Rerun the recs deployment to apply the value. Set it back to true and apply
again to prevent future nightly executions.

## Rollback

Reapply an earlier immutable image tag with Terraform. Only the three most recent
images are kept: each deploy applies an Artifact Registry cleanup policy
(KEEP_IMAGE_COUNT in the deploy workflow) that deletes the rest. Migrations are additive
so older service revisions remain compatible. All recommendation data can be
re-derived in the same order as the initial load.
