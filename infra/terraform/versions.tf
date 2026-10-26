terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.70.0, < 7.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }

  # Local state on purpose: one person deploys a short-lived demo.
  # State contains the origin secret - keep terraform.tfstate out of git (see .gitignore).
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project_tag
      ManagedBy = "terraform"
      Purpose   = "opencv-ai-competition-2026"
    }
  }
}
