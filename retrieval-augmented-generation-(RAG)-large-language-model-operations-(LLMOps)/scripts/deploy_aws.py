#!/usr/bin/env python
import argparse
import subprocess
import json
import boto3
import time
from pathlib import Path

def get_aws_client(service, region='us-east-1'):
    return boto3.client(service, region_name=region)

def create_ecr_repository(repository_name, region='us-east-1'):
    ecr = get_aws_client('ecr', region)
    try:
        response = ecr.create_repository(repositoryName=repository_name)
        repository_uri = response['repository']['repositoryUri']
        print(f"Created ECR repository: {repository_uri}")
        return repository_uri
    except ecr.exceptions.RepositoryAlreadyExistsException:
        response = ecr.describe_repositories(repositoryNames=[repository_name])
        repository_uri = response['repositories'][0]['repositoryUri']
        print(f"ECR repository already exists: {repository_uri}")
        return repository_uri

def build_and_push_docker_image(repository_uri, dockerfile_path='deployment/Dockerfile.aws'):
    subprocess.run([
        'docker', 'build', '-t', f'{repository_uri}:latest', '-f', dockerfile_path, '.'
    ], check=True)
    
    subprocess.run([
        'docker', 'push', f'{repository_uri}:latest'
    ], check=True)
    
    print(f"Pushed image to {repository_uri}:latest")

def create_ecs_cluster(cluster_name, region='us-east-1'):
    ecs = get_aws_client('ecs', region)
    try:
        response = ecs.create_cluster(clusterName=cluster_name)
        print(f"Created ECS cluster: {cluster_name}")
        return response['cluster']
    except ecs.exceptions.ClusterNotFoundException:
        print(f"ECS cluster already exists: {cluster_name}")
        return None

def register_task_definition(repository_uri, task_family, region='us-east-1'):
    ecs = get_aws_client('ecs', region)
    
    task_definition = {
        'family': task_family,
        'taskRoleArn': 'ecsTaskExecutionRole',
        'executionRoleArn': 'ecsTaskExecutionRole',
        'networkMode': 'awsvpc',
        'requiresCompatibilities': ['FARGATE'],
        'cpu': '4096',
        'memory': '16384',
        'runtimePlatform': {
            'operatingSystemFamily': 'LINUX',
            'cpuArchitecture': 'X86_64'
        },
        'containerDefinitions': [
            {
                'name': 'llmops-rag',
                'image': f'{repository_uri}:latest',
                'memory': 16384,
                'cpu': 4096,
                'portMappings': [
                    {
                        'containerPort': 8000,
                        'protocol': 'tcp'
                    }
                ],
                'environment': [
                    {'name': 'CUDA_VISIBLE_DEVICES', 'value': '0'},
                    {'name': 'MLFLOW_TRACKING_URI', 'value': 'https://dagshub.com/your-username/your-repo.mlflow'}
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/{task_family}',
                        'awslogs-region': region,
                        'awslogs-stream-prefix': 'ecs'
                    }
                }
            }
        ]
    }
    
    response = ecs.register_task_definition(**task_definition)
    task_definition_arn = response['taskDefinition']['taskDefinitionArn']
    print(f"Registered task definition: {task_definition_arn}")
    return task_definition_arn

def create_or_update_service(cluster_name, service_name, task_definition_arn, subnet_ids, security_group_ids, region='us-east-1'):
    ecs = get_aws_client('ecs', region)
    
    service_config = {
        'cluster': cluster_name,
        'serviceName': service_name,
        'taskDefinition': task_definition_arn,
        'desiredCount': 1,
        'launchType': 'FARGATE',
        'networkConfiguration': {
            'awsvpcConfiguration': {
                'subnets': subnet_ids,
                'securityGroups': security_group_ids,
                'assignPublicIp': 'ENABLED'
            }
        }
    }
    
    try:
        response = ecs.create_service(**service_config)
        print(f"Created service: {service_name}")
        return response['service']
    except ecs.exceptions.ServiceNotFoundException:
        response = ecs.update_service(
            cluster=cluster_name,
            service=service_name,
            taskDefinition=task_definition_arn,
            desiredCount=1
        )
        print(f"Updated service: {service_name}")
        return response['service']

def deploy_to_aws(repository_name='llmops-rag', cluster_name='llmops-cluster', service_name='llmops-rag-service'):
    print("Starting AWS deployment...")
    
    repository_uri = create_ecr_repository(repository_name)
    
    build_and_push_docker_image(repository_uri)
    
    create_ecs_cluster(cluster_name)
    
    task_definition_arn = register_task_definition(repository_uri, f'{repository_name}-task')
    
    subnet_ids = input("Enter subnet IDs (comma-separated): ").split(',')
    security_group_ids = input("Enter security group IDs (comma-separated): ").split(',')
    
    create_or_update_service(cluster_name, service_name, task_definition_arn, subnet_ids, security_group_ids)
    
    print("Waiting for service to stabilize...")
    time.sleep(30)
    
    print("Deployment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to AWS ECS")
    parser.add_argument("--repository", default="llmops-rag", help="ECR repository name")
    parser.add_argument("--cluster", default="llmops-cluster", help="ECS cluster name")
    parser.add_argument("--service", default="llmops-rag-service", help="ECS service name")
    
    args = parser.parse_args()
    
    deploy_to_aws(args.repository, args.cluster, args.service)