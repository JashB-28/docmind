#!/usr/bin/env bash
# Runs ON the EC2 box (invoked by SSM from the deploy workflow).
# Logs into ECR, pulls the freshly built image, and restarts the stack.
set -euo pipefail

cd /opt/docmind

AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Authenticate Docker to ECR using the instance role (no stored credentials).
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# Point the prod compose file at the just-pushed image and roll it out.
export ECR_IMAGE="${REGISTRY}/docmind:latest"
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker image prune -f
echo "Deployed ${ECR_IMAGE}"
