import os
import yaml
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    questions_path: str = "data/download/Questions.csv"
    answers_path: str = "data/download/Answers.csv"
    score_threshold: int = 5
    top_n_questions: int = 50
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass
class VectorstoreConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    chroma_persist_dir: str = "./chroma_db"
    retrieval_k: int = 10

@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    quantization_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5.0e-5
    dpo_beta: float = 0.1
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./qwen-dpo-final"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    batch_size: int = 8
    num_workers: int = 4

@dataclass
class EvaluationConfig:
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    generation_models: List[str] = field(default_factory=lambda: ["bert-base-uncased"])
    semantic_model: str = "all-MiniLM-L6-v2"
    test_size: int = 50

@dataclass
class MonitoringConfig:
    log_metrics: bool = True
    log_parameters: bool = True
    log_artifacts: bool = True
    mlflow_tracking_uri: str = "https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow"
    mlflow_experiment_name: str = "llmops-rag-pipeline-experiment"

@dataclass
class DeploymentConfig:
    aws_region: str = "us-east-1"
    ecr_repository: str = "llmops-rag"
    ecs_cluster: str = "llmops-cluster"
    instance_type: str = "g4dn.xlarge"
    min_capacity: int = 1
    max_capacity: int = 3
    target_cpu_utilization: int = 70
    container_port: int = 8000

class Config:
    def __init__(self, config_path: str = "params.yaml"):
        self.config_path = config_path
        self._load_config()
        self._setup_mlflow_auth()
    
    def _setup_mlflow_auth(self):
        tracking_uri = self.monitoring.mlflow_tracking_uri
        if "dagshub" in tracking_uri:
            username = os.getenv("DAGSHUB_USERNAME")
            token = os.getenv("DAGSHUB_TOKEN")
            if username and token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                params = yaml.safe_load(f)
        else:
            params = {}
        
        training_params = params.get('training', {})
        
        self.data = DataConfig(**params.get('data', {}))
        self.vectorstore = VectorstoreConfig(**params.get('vectorstore', {}))
        self.model = ModelConfig(**params.get('model', {}))
        self.training = TrainingConfig(**training_params)
        self.inference = InferenceConfig(**params.get('inference', {}))
        self.evaluation = EvaluationConfig(**params.get('evaluation', {}))
        self.monitoring = MonitoringConfig(**params.get('monitoring', {}))
        self.deployment = DeploymentConfig(**params.get('deployment', {}))

config = Config()