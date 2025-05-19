#!/bin/bash

# Exit on error
set -e

# Configuration
IMAGE_NAME="glia/ps-router"
IMAGE_TAG="latest"

# Determine the script directory and project root (Docker build context)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="${PROJECT_ROOT}"
DOCKERFILE_FILE="${SCRIPT_DIR}/Dockerfile"
AWS_REGION="us-east-2"  # Change this to your desired AWS region

# Check if AWS account ID is set
if [ -z "${AWS_ACCOUNT_ID}" ]; then
    echo "Error: AWS_ACCOUNT_ID environment variable is not set. Please set it in your shell environment."
    exit 1
fi

# ECR repository name
ECR_REPO_NAME="${IMAGE_NAME}"
ECR_REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# Login to ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com


echo "Building Docker image for router..."

export DOCKER_BUILDKIT=1

# Build the Docker image
# docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile ${DOCKERFILE_PATH}
docker buildx build --platform linux/amd64 \
    -t "${ECR_REPO_URI}:${IMAGE_TAG}" \
    --push \
    -f "${DOCKERFILE_FILE}" \
    "${DOCKERFILE_PATH}"

# echo "Successfully built ${IMAGE_NAME}:${IMAGE_TAG}"

# # Tag the image for ECR
# echo "Tagging image for ECR..."
# echo "docker tag ${IMAGE_NAME} ${ECR_REPO_URI}"
# docker tag ${IMAGE_NAME} ${ECR_REPO_URI}

# # Push the image to ECR
# echo "Pushing image to ECR..."
# docker push ${ECR_REPO_URI}:${IMAGE_TAG}

echo "Successfully published ${ECR_REPO_URI}:${IMAGE_TAG}"
