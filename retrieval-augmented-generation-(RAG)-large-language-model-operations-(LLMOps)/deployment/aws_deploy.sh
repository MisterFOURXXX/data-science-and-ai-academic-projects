#!/bin/bash

set -e

AWS_REGION="us-east-1"
ECR_REPOSITORY="llmops-rag"
ECS_CLUSTER="llmops-cluster"
ECS_SERVICE="llmops-rag-service"
TASK_FAMILY="llmops-rag-task"

echo "Building Docker image..."
docker build -t $ECR_REPOSITORY:latest -f deployment/Dockerfile.aws .

echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

echo "Creating ECR repository if not exists..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION || \
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION

echo "Tagging and pushing image..."
docker tag $ECR_REPOSITORY:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

echo "Creating ECS cluster if not exists..."
aws ecs describe-clusters --clusters $ECS_CLUSTER --region $AWS_REGION | grep -q $ECS_CLUSTER || \
    aws ecs create-cluster --cluster-name $ECS_CLUSTER --region $AWS_REGION

echo "Registering task definition..."
cat > task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "taskRoleArn": "ecsTaskExecutionRole",
    "executionRoleArn": "ecsTaskExecutionRole",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "4096",
    "memory": "16384",
    "runtimePlatform": {
        "operatingSystemFamily": "LINUX",
        "cpuArchitecture": "X86_64"
    },
    "containerDefinitions": [
        {
            "name": "llmops-rag",
            "image": "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest",
            "memory": 16384,
            "cpu": 4096,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                {"name": "MLFLOW_TRACKING_URI", "value": "https://dagshub.com/your-username/your-repo.mlflow"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/llmops-rag",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF

TASK_DEFINITION_ARN=$(aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION | jq -r '.taskDefinition.taskDefinitionArn')

echo "Creating or updating service..."
aws ecs describe-services --services $ECS_SERVICE --cluster $ECS_CLUSTER --region $AWS_REGION | grep -q $ECS_SERVICE && \
    aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --task-definition $TASK_FAMILY --desired-count 1 --region $AWS_REGION || \
    aws ecs create-service --cluster $ECS_CLUSTER --service-name $ECS_SERVICE --task-definition $TASK_FAMILY --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" --region $AWS_REGION

echo "Deployment completed successfully!"