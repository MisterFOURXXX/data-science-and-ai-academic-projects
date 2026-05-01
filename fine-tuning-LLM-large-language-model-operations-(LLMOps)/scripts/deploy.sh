#!/bin/bash

set -e

echo "Starting deployment process..."

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "Building Docker image..."
docker build -t llmops-rag-model:latest .

echo "Testing Docker image locally..."
docker run --rm llmops-rag-model:latest python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "Logging to ECR..."
aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com

echo "Creating ECR repository if not exists..."
aws ecr describe-repositories --repository-names llmops-rag-model 2>/dev/null || \
    aws ecr create-repository --repository-name llmops-rag-model

ECR_URI=$(aws ecr describe-repositories --repository-names llmops-rag-model --query 'repositories[0].repositoryUri' --output text)

echo "Tagging and pushing image to ECR..."
docker tag llmops-rag-model:latest $ECR_URI:latest
docker push $ECR_URI:latest

echo "Deploying to AWS Lambda..."
python deployment/aws_deploy.py

echo "Deployment completed successfully!"