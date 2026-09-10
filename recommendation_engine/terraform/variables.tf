variable "project_id" {
  description = "GCP project containing the recommendation service."
  type        = string
  default     = "story-6f89f"
}

variable "region" {
  description = "GCP region for Cloud Run, Workflows, and Cloud Scheduler."
  type        = string
  default     = "us-central1"
}

variable "image" {
  description = "Immutable recommendation container image, normally tagged with the Git commit SHA."
  type        = string
}

variable "service_name" {
  description = "Cloud Run recommendation service name."
  type        = string
  default     = "novelsync-recs"
}

variable "firebase_functions_service_account" {
  description = "Gen 2 Firebase Functions runtime identity allowed to invoke recs."
  type        = string
  default     = "793308156964-compute@developer.gserviceaccount.com"
}

variable "deployment_service_account" {
  description = "GitHub deployment identity allowed to perform the authenticated smoke test."
  type        = string
  default     = "github-actions@story-6f89f.iam.gserviceaccount.com"
}

variable "story_data_sync_job_name" {
  description = "Cloud Run job owned by story-data that derives recommendation signals."
  type        = string
  default     = "novelsync-story-data-sync-recs"
}

variable "database_rw_secret_id" {
  type    = string
  default = "recs-postgres-dsn-rw"
}

variable "database_ro_secret_id" {
  type    = string
  default = "recs-postgres-dsn-ro"
}

variable "gemini_secret_id" {
  description = "Existing shared Google AI Studio key in Secret Manager."
  type        = string
  default     = "google-ai-studio-api-key"
}

variable "min_instances" {
  description = "Warm instances kept running. 0 scales to zero, as novelsync-agents does; the first request after an idle spell pays a cold start."
  type        = number
  default     = 0

  validation {
    condition     = var.min_instances >= 0
    error_message = "min_instances must be non-negative."
  }
}

variable "max_instances" {
  type    = number
  default = 10

  validation {
    condition     = var.max_instances >= 1 && var.max_instances >= var.min_instances
    error_message = "max_instances must be at least 1 and no smaller than min_instances."
  }
}

variable "scheduler_paused" {
  description = "Keep true through initial load and validation; set false in a reviewed apply to start nightly runs."
  type        = bool
  default     = true
}

variable "scheduler_schedule" {
  description = "Nightly recommendation pipeline schedule."
  type        = string
  default     = "0 3 * * *"
}

variable "scheduler_time_zone" {
  type    = string
  default = "America/Denver"
}
