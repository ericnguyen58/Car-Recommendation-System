# Car Recommender — AWS infrastructure (Terraform)
#
# Provisions: an S3 bucket for model/data artifacts, an ECR repository for
# the API's container image, two Secrets Manager secret containers, an App
# Runner service running the API, and an IAM role GitHub Actions assumes via
# OIDC (no long-lived AWS keys in the repo).
#
# Apply this yourself with your own AWS account — it isn't run from the
# coding session. After `terraform apply`:
#   1. Populate the two secrets (values are deliberately NOT set here, so
#      real secrets never touch .tf files or state history):
#        aws secretsmanager put-secret-value --secret-id car-recommender/anthropic-api-key --secret-string "sk-ant-..."
#        aws secretsmanager put-secret-value --secret-id car-recommender/api-password      --secret-string "<a password>"
#   2. Upload model artifacts once (until the pipeline's export/train stages
#      are wired to push to S3 directly):
#        aws s3 cp models/rf_recommender.joblib s3://$(terraform output -raw artifacts_bucket)/models/rf_recommender.joblib
#        aws s3 cp models/label_encoders.joblib s3://$(terraform output -raw artifacts_bucket)/models/label_encoders.joblib
#        aws s3 cp models/feature_list.json     s3://$(terraform output -raw artifacts_bucket)/models/feature_list.json
#        aws s3 cp data/final/model_ready.csv   s3://$(terraform output -raw artifacts_bucket)/data/model_ready.csv
#   3. Set the `AWS_ROLE_ARN` GitHub repo variable to `terraform output -raw github_actions_role_arn`
#      — this is what arms .github/workflows/deploy.yml (it no-ops until set).

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name used to prefix/tag every resource."
  type        = string
  default     = "car-recommender"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the deploy role, as \"owner/name\"."
  type        = string
  default     = "ericnguyen58/Car-Recommendation-System"
}

variable "container_port" {
  description = "Port the API container listens on (matches the Dockerfile's uvicorn --port)."
  type        = number
  default     = 8000
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# S3 — model/data artifact storage (ml/model_store.py's "s3" backend)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# ECR — API container image
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---------------------------------------------------------------------------
# Secrets Manager — secret containers only; values are set out-of-band (see
# header comment) so a real API key/password is never written to .tf state.
# ---------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name = "${var.project_name}/anthropic-api-key"
}

resource "aws_secretsmanager_secret" "api_password" {
  name = "${var.project_name}/api-password"
}

# ---------------------------------------------------------------------------
# App Runner — runs the FastAPI backend from the ECR image
# ---------------------------------------------------------------------------
data "aws_caller_identity" "current" {}

resource "aws_iam_role" "apprunner_ecr_access" {
  name = "${var.project_name}-apprunner-ecr-access"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_iam_role" "apprunner_instance" {
  name = "${var.project_name}-apprunner-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "tasks.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# What the running container is allowed to do: read artifacts from S3, read
# the two secrets — nothing else. Matches ml/model_store.py + core/secrets.py.
resource "aws_iam_role_policy" "apprunner_instance" {
  name = "${var.project_name}-apprunner-instance-policy"
  role = aws_iam_role.apprunner_instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          aws_secretsmanager_secret.anthropic_api_key.arn,
          aws_secretsmanager_secret.api_password.arn,
        ]
      },
    ]
  })
}

resource "aws_apprunner_service" "api" {
  service_name = "${var.project_name}-api"

  source_configuration {
    auto_deployments_enabled = true
    image_repository {
      repository_type = "ECR"
      image_identifier = "${aws_ecr_repository.api.repository_url}:latest"
      image_configuration {
        port = tostring(var.container_port)
        runtime_environment_variables = {
          DEPLOY_ENV                       = "aws"
          MODEL_STORE_BACKEND               = "s3"
          AWS_S3_BUCKET                     = aws_s3_bucket.artifacts.bucket
          AWS_REGION                        = var.aws_region
          ANTHROPIC_API_KEY_SECRET_NAME     = aws_secretsmanager_secret.anthropic_api_key.name
          API_PASSWORD_SECRET_NAME          = aws_secretsmanager_secret.api_password.name
        }
      }
    }
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }
}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC — lets .github/workflows/deploy.yml assume an AWS role
# with no long-lived credentials stored in GitHub.
# ---------------------------------------------------------------------------
data "tls_certificate" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github_actions.certificates[0].sha1_fingerprint]
}

resource "aws_iam_role" "github_actions_deploy" {
  name = "${var.project_name}-github-actions-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github_actions.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name = "${var.project_name}-github-actions-deploy-policy"
  role = aws_iam_role.github_actions_deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:BatchGetImage",
        ]
        Resource = [aws_ecr_repository.api.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["apprunner:StartDeployment", "apprunner:DescribeService"]
        Resource = [aws_apprunner_service.api.arn]
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "apprunner_service_url" {
  value = aws_apprunner_service.api.service_url
}

output "apprunner_service_arn" {
  description = "Set this as the AWS_APPRUNNER_SERVICE_ARN repo variable so deploy.yml can trigger redeploys explicitly"
  value       = aws_apprunner_service.api.arn
}

output "github_actions_role_arn" {
  description = "Set this as the AWS_ROLE_ARN repo variable to arm .github/workflows/deploy.yml"
  value       = aws_iam_role.github_actions_deploy.arn
}
