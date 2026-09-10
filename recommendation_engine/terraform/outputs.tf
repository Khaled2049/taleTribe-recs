output "service_url" {
  description = "Stable recommendation service URL and OIDC audience."
  value       = local.service_url
}

output "service_name" {
  value = google_cloud_run_v2_service.recs.name
}

output "ingest_job_name" {
  value = google_cloud_run_v2_job.batch["ingest"].name
}

output "refresh_job_name" {
  value = google_cloud_run_v2_job.batch["refresh"].name
}

output "workflow_name" {
  value = google_workflows_workflow.pipeline.name
}

output "scheduler_name" {
  value = google_cloud_scheduler_job.nightly.name
}

output "scheduler_paused" {
  value = google_cloud_scheduler_job.nightly.paused
}
