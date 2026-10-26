# Floorsight on AWS.
# Graviton4 instance from the COOL Marketplace AMI; private by default (public_endpoint = false:
# no inbound port, reach the dashboard with an SSM port-forward). public_endpoint = true adds the
# CloudFront HTTPS link + Elastic IP. Private S3 buckets (evidence; versioned artifacts: code
# bundle, clips, run receipts), DynamoDB track history, SNS alerts, least-privilege instance role
# (incl. Claude on Bedrock for the agent), CloudWatch logs + dashboard, optional AWS Budget.
# preservation_hold = true (default) turns every automatic deletion off. See docs/aws/DEPLOY.md.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  # "us.anthropic.x" (cross-region inference profile) -> "anthropic.x" (the foundation model behind it)
  bedrock_ids      = [for m in [var.bedrock_model_id, var.bedrock_fallback_model_id] : m if m != ""]
  bedrock_models   = distinct([for m in local.bedrock_ids : replace(m, "/^(us|eu|apac|global|jp|au)\\./", "")])
  bedrock_profiles = [for m in local.bedrock_ids : m if can(regex("^(us|eu|apac|global|jp|au)\\.", m))]
  cw_namespace     = "Floorsight"
  site_id          = "demo"
}

# --------------------------------------------------------------------------- #
# Evidence bucket: private, SSE-S3, no versioning, objects expire after 1 day
# --------------------------------------------------------------------------- #

resource "aws_s3_bucket" "evidence" {
  bucket_prefix = "${var.name}-evidence-"
  force_destroy = !var.preservation_hold # under the hold, destroy refuses to drop stored evidence
}

resource "aws_s3_bucket_public_access_block" "evidence" {
  bucket                  = aws_s3_bucket.evidence.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "evidence" {
  count  = var.preservation_hold ? 0 : 1 # owner hold: no automatic expiry
  bucket = aws_s3_bucket.evidence.id
  rule {
    id     = "expire-evidence"
    status = "Enabled"
    filter {}
    expiration {
      days = var.evidence_retention_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "evidence_bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.evidence.arn, "${aws_s3_bucket.evidence.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "evidence" {
  bucket     = aws_s3_bucket.evidence.id
  policy     = data.aws_iam_policy_document.evidence_bucket.json
  depends_on = [aws_s3_bucket_public_access_block.evidence]
}

# --------------------------------------------------------------------------- #
# Artifacts bucket: code bundle, source clips, run outputs and receipts.
# Private, versioned (an overwrite keeps the old version), never expires.
# --------------------------------------------------------------------------- #

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${var.name}-artifacts-"
  force_destroy = false
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

data "aws_iam_policy_document" "artifacts_bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket     = aws_s3_bucket.artifacts.id
  policy     = data.aws_iam_policy_document.artifacts_bucket.json
  depends_on = [aws_s3_bucket_public_access_block.artifacts]
}

# --------------------------------------------------------------------------- #
# Track history (WorldState snapshots), on-demand, TTL auto-delete unless held
# --------------------------------------------------------------------------- #

resource "aws_dynamodb_table" "tracks" {
  name         = "${var.name}-tracks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "t_ms"

  attribute {
    name = "session_id"
    type = "S"
  }
  attribute {
    name = "t_ms"
    type = "N"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = !var.preservation_hold
  }
}

# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

resource "aws_sns_topic" "alerts" {
  name = "${var.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #

resource "aws_cloudwatch_log_group" "app" {
  name              = "/${var.name}/app"
  retention_in_days = var.preservation_hold ? 0 : var.log_retention_days # 0 = never expire
}

# --------------------------------------------------------------------------- #
# IAM: instance role, least privilege
# --------------------------------------------------------------------------- #

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.name}-app"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "app" {
  statement {
    sid       = "EvidenceObjects"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.evidence.arn}/*"]
  }

  statement {
    sid       = "ArtifactsList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }

  statement {
    sid       = "ArtifactsObjects"
    actions   = ["s3:GetObject", "s3:PutObject"] # no delete: the bucket is versioned and held
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }

  statement {
    sid       = "TrackTable"
    actions   = ["dynamodb:PutItem", "dynamodb:Query"]
    resources = [aws_dynamodb_table.tracks.arn]
  }

  statement {
    sid       = "Alerts"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]
  }

  # The agent calls the Messages-API endpoint (bedrock-mantle.<region>.api.aws), which authorizes
  # bedrock-mantle:CreateInference (AWS Bedrock user guide, "Prerequisites"; Anthropic "Claude in
  # Amazon Bedrock"). Its resource ARN format is not documented there, hence "*" for this one action.
  statement {
    sid       = "BedrockMantleInference"
    actions   = ["bedrock-mantle:CreateInference"]
    resources = ["*"]
  }

  statement {
    sid     = "BedrockAgentModel"
    actions = concat(["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], var.bedrock_extra_actions)
    resources = concat(
      # the agent model and its refusal-fallback model, in any region (inference profiles route between regions)
      [for m in local.bedrock_models : "arn:${local.partition}:bedrock:*::foundation-model/${m}"],
      # inference profiles, if a us./eu./global. profile id is configured
      [for p in local.bedrock_profiles : "arn:${local.partition}:bedrock:${var.region}:${local.account_id}:inference-profile/${p}"],
    )
  }

  # Narration for the demo video (Amazon Polly text-to-speech); no resource-level permissions exist for it.
  statement {
    sid       = "VideoNarration"
    actions   = ["polly:SynthesizeSpeech", "polly:DescribeVoices"]
    resources = ["*"]
  }

  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData has no resource-level permissions; scoped by namespace instead
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = [local.cw_namespace]
    }
  }

  statement {
    sid       = "AppLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.app.arn}:*"]
  }
}

resource "aws_iam_role_policy" "app" {
  name   = "${var.name}-app"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app.json
}

# Session Manager instead of SSH (no port 22, no key pair).
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.name}-app"
  role = aws_iam_role.app.name
}

# Baseline benchmark instances only need SSM (no app data access).
resource "aws_iam_role" "bench" {
  name               = "${var.name}-bench"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "bench_ssm" {
  role       = aws_iam_role.bench.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "bench" {
  name = "${var.name}-bench"
  role = aws_iam_role.bench.name
}

# --------------------------------------------------------------------------- #
# Network: default VPC, subnets in AZs that offer the instance types
# --------------------------------------------------------------------------- #

data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_instance_type_offerings" "app" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "app" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.app.locations
  }
}

data "aws_ec2_instance_type_offerings" "x86" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.baseline_x86_instance_type]
  }
}

data "aws_subnets" "x86" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.x86.locations
  }
}

data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "app" {
  name_prefix = "${var.name}-app-"
  description = "Floorsight app: no inbound unless public_endpoint (then app port from CloudFront only); no SSH (SSM)"
  vpc_id      = data.aws_vpc.default.id
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "app_from_cloudfront" {
  count             = var.public_endpoint ? 1 : 0 # private mode: no inbound rule at all
  security_group_id = aws_security_group.app.id
  description       = "CloudFront origin-facing"
  prefix_list_id    = data.aws_ec2_managed_prefix_list.cloudfront.id
  ip_protocol       = "tcp"
  from_port         = var.app_port
  to_port           = var.app_port
}

resource "aws_vpc_security_group_egress_rule" "app_all" {
  security_group_id = aws_security_group.app.id
  description       = "AWS APIs (S3, DynamoDB, SNS, Bedrock, SSM, CloudWatch), apt, git, pip"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_security_group" "bench" {
  name_prefix = "${var.name}-bench-"
  description = "Benchmark baselines: no inbound at all (SSM only)"
  vpc_id      = data.aws_vpc.default.id
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_egress_rule" "bench_all" {
  security_group_id = aws_security_group.bench.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #

resource "random_password" "origin_secret" {
  length  = 32
  special = false
}

locals {
  app_env = {
    AWS_REGION                  = var.region
    AWS_DEFAULT_REGION          = var.region
    FLOORSIGHT_S3_BUCKET        = aws_s3_bucket.evidence.bucket
    FLOORSIGHT_S3_PREFIX        = "evidence/"
    FLOORSIGHT_SNS_TOPIC_ARN    = aws_sns_topic.alerts.arn
    FLOORSIGHT_DDB_TABLE        = aws_dynamodb_table.tracks.name
    FLOORSIGHT_DDB_TTL_S        = tostring(var.track_ttl_seconds)
    FLOORSIGHT_CW_NAMESPACE     = local.cw_namespace
    FLOORSIGHT_SITE_ID          = local.site_id
    FLOORSIGHT_ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.bucket
    FLOORSIGHT_BEDROCK_MODEL_ID = var.bedrock_model_id
    FLOORSIGHT_LLM              = var.agent_llm # "bedrock" switches floorsight.llm to Claude on Bedrock
    FLOORSIGHT_TOKEN            = var.judge_token
    FLOORSIGHT_ORIGIN_SECRET    = var.public_endpoint ? random_password.origin_secret.result : ""
    # same hold for the app's local history/evidence purge as for S3 expiry, DynamoDB TTL and log expiry
    FLOORSIGHT_PRESERVATION_HOLD = var.preservation_hold ? "on" : "off"
  }
  app_host    = var.public_endpoint ? "0.0.0.0" : "127.0.0.1" # private: only an SSM port-forward reaches it
  app_command = var.app_start_command == "" ? "" : "${var.app_start_command} --host ${local.app_host} --port ${var.app_port}"
  bundle_s3   = var.code_bundle_key == "" ? "" : "s3://${aws_s3_bucket.artifacts.bucket}/${var.code_bundle_key}"
}

resource "aws_instance" "app" {
  ami                         = var.cool_ami_id
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.app.ids)[0]
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  associate_public_ip_address = true  # outbound only (S3, Bedrock, apt, pip); no inbound rule unless public_endpoint
  user_data_replace_on_change = false # a changed bootstrap must never replace (and wipe) the instance

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    role              = "app"
    opencv_source     = "cool"
    repo_url          = var.app_repo_url
    git_ref           = var.app_git_ref
    bundle_s3         = local.bundle_s3
    region            = var.region
    app_start_command = local.app_command
    log_group         = aws_cloudwatch_log_group.app.name
    env               = local.app_env
  })

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = !var.preservation_hold
    tags = {
      Name = "${var.name}-app-01-root"
    }
  }

  tags = {
    Name = "${var.name}-app-01"
  }

  lifecycle {
    ignore_changes = [user_data, ami] # bootstrap/AMI updates are applied in place over SSM, never by replacement
    precondition {
      condition     = var.app_repo_url != "" || var.code_bundle_key != ""
      error_message = "Set app_repo_url or code_bundle_key so the instance has code to run."
    }
    precondition {
      condition     = length(data.aws_subnets.app.ids) > 0
      error_message = "No default subnet in an AZ that offers ${var.instance_type} in ${var.region}."
    }
    precondition {
      # the origin secret only proves a request came through CloudFront; without a token anyone with the link
      # could read evidence and approve, reject or add rules
      condition     = !var.public_endpoint || length(var.judge_token) >= 20
      error_message = "public_endpoint = true needs judge_token set to a random string of at least 20 characters (e.g. openssl rand -hex 16)."
    }
  }
}

# Stable address: CloudFront keeps pointing at the same origin across stop/start.
resource "aws_eip" "app" {
  count    = var.public_endpoint ? 1 : 0
  domain   = "vpc"
  instance = aws_instance.app.id
  tags = {
    Name = "${var.name}-app"
  }
}

data "aws_ami" "ubuntu_x86" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_ami" "ubuntu_arm" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

locals {
  bench_user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    role              = "baseline"
    opencv_source     = "pip"
    repo_url          = var.app_repo_url
    git_ref           = var.app_git_ref
    bundle_s3         = local.bundle_s3
    region            = var.region
    app_start_command = ""
    log_group         = ""
    env               = { AWS_REGION = var.region }
  })
}

resource "aws_instance" "baseline_x86" {
  count                       = var.baseline_x86_count
  ami                         = data.aws_ami.ubuntu_x86.id
  instance_type               = var.baseline_x86_instance_type
  subnet_id                   = sort(data.aws_subnets.x86.ids)[0]
  vpc_security_group_ids      = [aws_security_group.bench.id]
  iam_instance_profile        = aws_iam_instance_profile.bench.name
  associate_public_ip_address = true # outbound only (apt/git/pip); no inbound rules
  user_data                   = local.bench_user_data

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  tags = {
    Name = "${var.name}-baseline-x86"
  }
}

resource "aws_instance" "baseline_arm_stock" {
  count                       = var.baseline_arm_stock_count
  ami                         = data.aws_ami.ubuntu_arm.id
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.app.ids)[0]
  vpc_security_group_ids      = [aws_security_group.bench.id]
  iam_instance_profile        = aws_iam_instance_profile.bench.name
  associate_public_ip_address = true
  user_data                   = local.bench_user_data

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  tags = {
    Name = "${var.name}-baseline-arm-stock"
  }
}

# --------------------------------------------------------------------------- #
# CloudFront: the HTTPS judge link (default *.cloudfront.net certificate)
# --------------------------------------------------------------------------- #

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}

resource "aws_cloudfront_distribution" "app" {
  count           = var.public_endpoint ? 1 : 0
  enabled         = true
  comment         = "Floorsight dashboard (judge link)"
  price_class     = "PriceClass_100"
  is_ipv6_enabled = true
  http_version    = "http2"

  origin {
    origin_id   = "floorsight-app"
    domain_name = aws_eip.app[0].public_dns

    custom_origin_config {
      http_port                = var.app_port
      https_port               = 443
      origin_protocol_policy   = "http-only" # CloudFront -> EC2 leg is plain HTTP on 8000 (see DEPLOY.md)
      origin_ssl_protocols     = ["TLSv1.2"]
      origin_read_timeout      = 60
      origin_keepalive_timeout = 60
    }

    # The app can reject requests that did not come through this distribution.
    custom_header {
      name  = "X-Floorsight-Origin"
      value = random_password.origin_secret.result
    }
  }

  default_cache_behavior {
    target_origin_id         = "floorsight-app"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = var.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title   = "Camera fps"
          region  = var.region
          view    = "timeSeries"
          stat    = "Average"
          period  = 60
          metrics = [[{ expression = "SEARCH('{${local.cw_namespace},Site,Camera} MetricName=\"CameraFps\"', 'Average', 60)", id = "fps" }]]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Pipeline latency (ms)"
          region = var.region
          view   = "timeSeries"
          period = 60
          metrics = [
            [local.cw_namespace, "PipelineLatency", "Site", local.site_id, { stat = "p50", label = "p50" }],
            [local.cw_namespace, "PipelineLatency", "Site", local.site_id, { stat = "p95", label = "p95" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title   = "Events by type"
          region  = var.region
          view    = "timeSeries"
          stat    = "Sum"
          period  = 300
          metrics = [[{ expression = "SEARCH('{${local.cw_namespace},Site,EventType} MetricName=\"Events\"', 'Sum', 300)", id = "ev" }]]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title   = "Instance CPU %"
          region  = var.region
          view    = "timeSeries"
          stat    = "Average"
          period  = 60
          metrics = [["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.app.id]]
        }
      },
      {
        type = "log", x = 0, y = 12, width = 24, height = 6
        properties = {
          title  = "App log"
          region = var.region
          view   = "table"
          query  = "SOURCE '${aws_cloudwatch_log_group.app.name}' | fields @timestamp, @message | sort @timestamp desc | limit 100"
        }
      },
    ]
  })
}

# --------------------------------------------------------------------------- #
# Cost guard-rail (notifies; does not stop anything)
# --------------------------------------------------------------------------- #

resource "aws_budgets_budget" "monthly" {
  count        = var.create_budget && var.alert_email != "" ? 1 : 0
  name         = "${var.name}-floorsight-monthly"
  budget_type  = "COST"
  limit_amount = format("%.2f", var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$${var.project_tag}"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
