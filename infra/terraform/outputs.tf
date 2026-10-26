output "judge_url" {
  description = "HTTPS link for the judges (only when public_endpoint = true)."
  value       = var.public_endpoint ? "https://${aws_cloudfront_distribution.app[0].domain_name}/" : "private: no public link (public_endpoint = false); use port_forward_command"
}

output "port_forward_command" {
  description = "Private access: forwards the dashboard to http://127.0.0.1:<app_port>/ on your machine through SSM."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters portNumber=${var.app_port},localPortNumber=${var.app_port}"
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "instance_id" {
  value = aws_instance.app.id
}

output "ssm_session_command" {
  description = "Shell on the instance without SSH."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.app.id}"
}

output "evidence_bucket" {
  value = aws_s3_bucket.evidence.bucket
}

output "tracks_table" {
  value = aws_dynamodb_table.tracks.name
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}

output "dashboard_url" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "baseline_x86_instance_ids" {
  value = aws_instance.baseline_x86[*].id
}

output "baseline_arm_stock_instance_ids" {
  value = aws_instance.baseline_arm_stock[*].id
}

output "origin_secret" {
  description = "Header value CloudFront adds (X-Floorsight-Origin); the app may require it."
  value       = random_password.origin_secret.result
  sensitive   = true
}
