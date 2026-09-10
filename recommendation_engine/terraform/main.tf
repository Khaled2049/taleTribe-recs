data "google_project" "current" {
  project_id = var.project_id
}

locals {
  # New Cloud Run services receive this stable project-number URL. Pinning the
  # same value in the container and the Functions caller keeps the OIDC audience
  # exact without introducing a self-reference to the service's computed URI.
  service_url = "https://${var.service_name}-${data.google_project.current.number}.${var.region}.run.app"
  allowed_service_accounts = join(",", [
    var.firebase_functions_service_account,
    var.deployment_service_account,
  ])

  # Every container builds RecSettings, whose production validator requires
  # these, so the service and both jobs share one definition.
  common_env = {
    ENVIRONMENT              = "production"
    RECS_SERVICE_URL         = local.service_url
    ALLOWED_SERVICE_ACCOUNTS = local.allowed_service_accounts
  }

  # Env var -> Secret Manager secret. The service adds its read-replica DSN;
  # the batch jobs read through the primary in code (they must see their own
  # writes), so these two are all they need.
  secret_env = {
    RECS_DATABASE_URL        = data.google_secret_manager_secret.database_rw.secret_id
    GOOGLE_AI_STUDIO_API_KEY = data.google_secret_manager_secret.gemini.secret_id
  }

  # The ingest and refresh jobs differ only in what they run and how long they
  # may take.
  batch_jobs = {
    ingest = {
      args    = ["-m", "recommendation_engine.ingest.platform"]
      timeout = "3600s"
    }
    refresh = {
      args    = ["-m", "recommendation_engine.sync.seed", "--refresh-only"]
      timeout = "900s"
    }
  }
}

resource "google_project_service" "required" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "workflowexecutions.googleapis.com",
    "workflows.googleapis.com",
    "cloudscheduler.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

data "google_secret_manager_secret" "database_rw" {
  project   = var.project_id
  secret_id = var.database_rw_secret_id
}

data "google_secret_manager_secret" "database_ro" {
  project   = var.project_id
  secret_id = var.database_ro_secret_id
}

data "google_secret_manager_secret" "gemini" {
  project   = var.project_id
  secret_id = var.gemini_secret_id
}

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "novelsync-recs-run"
  display_name = "NovelSync recommendations runtime"
}

resource "google_service_account" "workflow" {
  project      = var.project_id
  account_id   = "novelsync-recs-workflow"
  display_name = "NovelSync recommendations workflow"
}

resource "google_service_account" "scheduler" {
  project      = var.project_id
  account_id   = "novelsync-recs-scheduler"
  display_name = "NovelSync recommendations scheduler"
}

resource "google_secret_manager_secret_iam_member" "runtime_secrets" {
  for_each = {
    database_rw = data.google_secret_manager_secret.database_rw.id
    database_ro = data.google_secret_manager_secret.database_ro.id
    gemini      = data.google_secret_manager_secret.gemini.id
  }

  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "recs" {
  project  = var.project_id
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "300s"
    max_instance_request_concurrency = 40

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        # Billed only while serving, as novelsync-agents is. An idle instance gets
        # almost no CPU, so the one piece of work that outlives a response —
        # ExplanationCache's fire-and-forget hit_count bump — can run late, or be
        # lost if the instance is shut down first.
        cpu_idle          = true
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = merge(local.common_env, {
          RECS_DB_POOL_MIN                       = "1"
          RECS_DB_POOL_MAX                       = "5"
          MAX_REQUESTS_PER_MINUTE_PER_USER       = "30"
          MAX_LLM_REQUESTS_PER_MINUTE_PER_USER   = "6"
          RECS_MAX_SEARCHES_PER_DAY_PER_USER     = "10"
          RECS_MAX_EXPLANATIONS_PER_DAY_PER_USER = "30"
          RECS_MAX_SEARCHES_PER_DAY_PLATFORM     = "1000"
          RECS_MAX_EXPLANATIONS_PER_DAY_PLATFORM = "1000"
        })
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = merge(local.secret_env, {
          RECS_DATABASE_URL_RO = data.google_secret_manager_secret.database_ro.secret_id
        })
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 12

        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_iam_member.runtime_secrets,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = toset([
    var.firebase_functions_service_account,
    var.deployment_service_account,
  ])

  project  = var.project_id
  name     = google_cloud_run_v2_service.recs.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${each.value}"
}

resource "google_cloud_run_v2_job" "batch" {
  for_each = local.batch_jobs

  project  = var.project_id
  name     = "novelsync-recs-${each.key}"
  location = var.region

  template {
    task_count  = 1
    parallelism = 1

    template {
      service_account = google_service_account.runtime.email
      max_retries     = 1
      timeout         = each.value.timeout

      containers {
        image   = var.image
        command = ["python"]
        args    = each.value.args

        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.key
            value = env.value
          }
        }
        dynamic "env" {
          for_each = local.secret_env
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_iam_member.runtime_secrets,
  ]
}

resource "google_cloud_run_v2_job_iam_member" "workflow_jobs" {
  for_each = toset(concat(
    [for job in google_cloud_run_v2_job.batch : job.name],
    [var.story_data_sync_job_name],
  ))

  project  = var.project_id
  name     = each.value
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# roles/run.invoker above lets the workflow start a job (run.jobs.run) but not
# read the operation that call returns, and the jobs.run connector polls that
# operation until the job finishes. Without this, step 1 starts the ingest and
# the workflow then fails, so reader signals and aggregates silently stop
# refreshing. Project-level because operations live under the location, not
# under the job, so a job-level grant would not cover them. Read-only.
resource "google_project_iam_member" "workflow_run_viewer" {
  project = var.project_id
  role    = "roles/run.viewer"
  member  = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_workflows_workflow" "pipeline" {
  project         = var.project_id
  name            = "novelsync-recs-pipeline"
  region          = var.region
  description     = "Ingest catalog, derive reader signals, then refresh recommendation aggregates."
  service_account = google_service_account.workflow.id

  source_contents = templatefile("${path.module}/pipeline.yaml.tftpl", {
    project_id               = var.project_id
    region                   = var.region
    ingest_job_name          = google_cloud_run_v2_job.batch["ingest"].name
    story_data_sync_job_name = var.story_data_sync_job_name
    refresh_job_name         = google_cloud_run_v2_job.batch["refresh"].name
  })

  depends_on = [
    google_project_service.required,
    google_cloud_run_v2_job_iam_member.workflow_jobs,
    google_project_iam_member.workflow_run_viewer,
  ]
}

resource "google_project_iam_member" "scheduler_workflow_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "nightly" {
  project          = var.project_id
  region           = var.region
  name             = "novelsync-recs-nightly"
  description      = "Run the ordered recommendation data pipeline."
  schedule         = var.scheduler_schedule
  time_zone        = var.scheduler_time_zone
  paused           = var.scheduler_paused
  attempt_deadline = "320s"

  retry_config {
    retry_count          = 2
    min_backoff_duration = "30s"
    max_backoff_duration = "300s"
    max_doublings        = 2
  }

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/${google_workflows_workflow.pipeline.name}/executions"
    body        = base64encode("{}")

    headers = {
      "Content-Type" = "application/json"
    }

    oauth_token {
      service_account_email = google_service_account.scheduler.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [
    google_project_service.required,
    google_project_iam_member.scheduler_workflow_invoker,
  ]
}
