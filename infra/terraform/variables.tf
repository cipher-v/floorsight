variable "region" {
  description = "AWS region. Must offer Graviton4 (c8g), Bedrock access to the chosen Claude model, and the COOL AMI."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for every resource name (neutral, no product or character names)."
  type        = string
  default     = "anisha-robotics"
}

variable "project_tag" {
  description = "Value of the Project tag on every resource; the AWS Budget filters on it (user:Project)."
  type        = string
  default     = "anisha-robotics-floorsight"
}

variable "public_endpoint" {
  description = "THE PUBLIC-LINK SWITCH. false (default): no inbound port, no Elastic IP, no CloudFront; the dashboard binds to 127.0.0.1 and is reached only through an SSM port-forward. true: CloudFront HTTPS link + Elastic IP + app port open to CloudFront's origin-facing IPs only. Flip it only after the owner approves publishing."
  type        = bool
  default     = false
}

variable "preservation_hold" {
  description = "Owner hold: nothing is deleted automatically. true disables the S3 evidence expiry, the DynamoDB TTL and log expiry, keeps buckets non-empty-destroy-safe and keeps the root volume if the instance is terminated."
  type        = bool
  default     = true
}

variable "cool_ami_id" {
  description = "AMI id of the COOL (Cloud-Optimized OpenCV Library) Marketplace listing for this region. Copy it from the Marketplace subscription's launch page. No default on purpose."
  type        = string

  validation {
    condition     = can(regex("^ami-[0-9a-f]{8,17}$", var.cool_ami_id))
    error_message = "cool_ami_id must look like ami-0123456789abcdef0."
  }
}

variable "instance_type" {
  description = "Graviton instance for the app (COOL is listed for Graviton4)."
  type        = string
  default     = "c8g.xlarge"

  validation {
    condition     = can(regex("^[a-z]+[0-9]+g[a-z]*\\.[0-9a-z]+$", var.instance_type))
    error_message = "instance_type must be a Graviton (arm64) type such as c8g.xlarge."
  }
}

variable "root_volume_gb" {
  description = "Root volume; must be at least the AMI snapshot size (60 GiB for the COOL v3 image)."
  type        = number
  default     = 80
}

variable "baseline_x86_instance_type" {
  description = "x86 instance for the benchmark baseline (stock OpenCV 5 wheel)."
  type        = string
  default     = "c7i.xlarge"
}

variable "baseline_x86_count" {
  description = "Set to 1 only while running the 3-way benchmark, then back to 0."
  type        = number
  default     = 0
}

variable "baseline_arm_stock_count" {
  description = "Optional plain Ubuntu 24.04 arm64 instance (same instance_type) with the stock OpenCV wheel. 0 = run the stock-wheel leg in a separate venv on the COOL instance instead."
  type        = number
  default     = 0
}

variable "bedrock_model_id" {
  description = "Bedrock model id the agent calls through the Messages-API (Mantle) endpoint."
  type        = string
  default     = "anthropic.claude-opus-5-5"
}

variable "bedrock_fallback_model_id" {
  description = "Model the agent's client-side refusal fallback retries with (FALLBACK_MODEL_ID in floorsight/llm/bedrock.py). Empty = none."
  type        = string
  default     = "anthropic.claude-opus-5"
}

variable "bedrock_extra_actions" {
  description = "Extra IAM actions for Bedrock, if AWS adds any beyond bedrock-mantle:CreateInference."
  type        = list(string)
  default     = []
}

variable "agent_llm" {
  description = "FLOORSIGHT_LLM on the instance: bedrock (Claude on Bedrock) or local (offline policy, no Bedrock calls). Default local: the demo replays in a loop and every agent step is one Bedrock call - see the Bedrock cost row in docs/aws/DEPLOY.md before switching."
  type        = string
  default     = "local"

  validation {
    condition     = contains(["bedrock", "local"], var.agent_llm)
    error_message = "agent_llm must be bedrock or local."
  }
}

variable "judge_token" {
  description = "FLOORSIGHT_TOKEN: if set, the dashboard needs ?token=<value>. Share the full link with the judges only."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alert_email" {
  description = "Optional: subscribed to the SNS alert topic (AWS sends a confirmation mail that must be clicked) and to the budget, if Terraform creates it. Empty = no email."
  type        = string
  default     = ""

  validation {
    condition     = var.alert_email == "" || can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email))
    error_message = "alert_email must be empty or an email address."
  }
}

variable "create_budget" {
  description = "Create the monthly AWS Budget here. false when a budget was created by hand first (DEPLOY.md step 1), to avoid two budgets."
  type        = bool
  default     = false
}

variable "budget_limit_usd" {
  description = "Monthly cost budget; notifications at 50 %, 80 % actual and 100 % forecast."
  type        = number
  default     = 50
}

variable "app_repo_url" {
  description = "Optional git URL (https, no credentials) to clone the code from. Empty = take the code bundle from the private artifacts bucket (code_bundle_key)."
  type        = string
  default     = ""
}

variable "app_git_ref" {
  description = "Branch, tag or commit to deploy when app_repo_url is set (use the same commit for every benchmark leg)."
  type        = string
  default     = "main"
}

variable "code_bundle_key" {
  description = "S3 key (in the artifacts bucket) of the code bundle .tgz the instance unpacks, e.g. code/floorsight-20261023-r1.tgz. Used when app_repo_url is empty."
  type        = string
  default     = ""
}

variable "app_start_command" {
  description = "Command systemd runs (from /opt/floorsight/app), without --host/--port (added from public_endpoint and app_port). Empty = install the service but do not start it."
  type        = string
  default     = ""
}

variable "app_port" {
  type    = number
  default = 8000
}

variable "evidence_retention_days" {
  description = "S3 evidence expiry; ignored while preservation_hold is true."
  type        = number
  default     = 1
}

variable "track_ttl_seconds" {
  type    = number
  default = 86400
}

variable "log_retention_days" {
  description = "CloudWatch log retention; ignored (never expire) while preservation_hold is true."
  type        = number
  default     = 7
}
