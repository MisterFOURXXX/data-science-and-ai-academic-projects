## Data Science and AI Academic Projects

This repository is a comprehensive collection of academic projects that apply theoretical knowledge to real-world problems. It contains well-documented notebooks that provide detailed explanations of theories, concepts, and code implementation details for each project.

## Core Principles

Each project in this repository is designed as a comprehensive learning resource. The notebooks are structured to move from foundational knowledge to practical application, ensuring deep understanding of the subject matter. The focus is on understanding the "why" behind the "how."

## Covered Area Knowledge

This repository includes a diverse collection of projects demonstrating unique applications of data-driven methodologies. The projects are categorized by primary focus, though many integrate concepts from multiple domains to solve business and real-world problems.

**Data Science and Machine Learning**

Projects in this area focus on data science end-to-end pipeline including exploratory data analysis, feature engineering, feature engineering techniques, machine learning algorithms, and model evaluation for solving classification and regression problem.

**Deep Learning and Neural Networks**

Projects in this area focus on designing and implementing various neural network architectures to solve complex classification and regression tasks. Additionally, these projects involve building sequence-based models such as RNNs, LSTMs, and GRUs for advanced time-series forecasting. Each implementation leverages industry-standard frameworks including PyTorch, TensorFlow, and Keras to optimize model performance and predictive accuracy.

**Natural Language Processing**

Projects in this area focus on the automated processing and analysis of human language to bridge the gap between unstructured text and actionable insights. These projects implement core NLP tasks including Text Classification and Sentiment Analysis to categorize content, alongside Information Extraction to identify key entities and relationships within documents. Advanced workflows explore Topic Modeling to discover underlying themes across large datasets, as well as Text Summarization and Text Generation to synthesize and create coherent, contextually relevant language.

**MLOps (Machine Learning Operations)**

Projects in this area address the operational challenges of deploying and maintaining machine learning models in production environments. Key topics include:
- **Model Versioning and Artifact Management** - Tracking model versions, training configurations, and evaluation metrics using tools like DVC and MLflow
- **CI/CD for Machine Learning** - Automating retraining, validation, and deployment pipelines
- **Model Monitoring** - Tracking data drift, concept drift, and performance degradation in production
- **Scalable Inference** - Implementing batch processing and API-based serving with latency and throughput optimization
- **Infrastructure as Code** - Containerization with Docker and orchestration with Kubernetes

**LLMs and Generative AI**

Projects explore state-of-the-art LLM architectures, Retrieval-Augmented Generation (RAG), and advanced prompt engineering, covering the entire LLM lifecycle:
- **Pre-training LLMs and LLMs Architecture** - The foundation of this repository. Projects explore the architecture of Transformer-based models, focusing on the transition from foundational Pre-trained LLMs to specialized downstream applications through a deep understanding of attention mechanisms and positional encodings.
- **LLM Fine-Tuning** - A deep dive into modern adaptation techniques. This includes Supervised Fine-Tuning for instruction following, and Parameter-Efficient Fine-Tuning to adapt large-scale models with minimal computational overhead and memory footprints.
- **Fine-Tuning and Deployment Optimization Techniques** - Projects in this section focus on making LLMs production-ready. This includes Quantization to reduce model size, Pruning to remove redundant weights, Knowledge Distillation, and other techniques to optimize LLMs. These techniques ensure high-performance inference on consumer-grade hardware.
- **Reinforcement Learning for Alignment** - Advanced training using the Transformer Reinforcement Learning library (trl). This includes implementing almost of transformer reinforcement learning types and experimental features from this library, alongside reward modeling to align model outputs with human preferences, safety guidelines, and specific performance constraints.
- **LLM Model Evaluation** - Intrinsic evaluation using perplexity and loss curves alongside extrinsic evaluation using BLEU, ROUGE, BERTScore, and METEOR, as well as model-based evaluation where one LLM evaluates another

**LLMOps (Large Language Model Operations)**

Projects in this area extend MLOps principles to address the unique challenges of deploying and managing LLMs in production:
- **Resource-Efficient Fine-Tuning** - Implementing QLoRA with 4-bit quantization and gradient checkpointing to fine-tune models on consumer-grade GPUs
- **RAG Pipeline Operations** - Building production-ready retrieval-augmented generation systems with vector databases (ChromaDB, FAISS) and embedding models
- **Prompt Engineering and Versioning** - Managing prompt templates as code, tracking prompt performance across versions, and implementing structured output formats
- **LLM Evaluation Automation** - Continuous assessment of model outputs across quality, safety, and relevance dimensions using programmatic metrics
- **Cost Optimization** - Techniques for reducing inference latency and token usage through batching, caching, and model quantization
- **Model Registry for LLMs** - Managing model optimization for inference separately from base models for efficient versioning, A/B testing, and hot-swapping in production

**RAG (Retrieval-Augmented Generation)**

Projects focus on building knowledge-augmented generation systems that combine semantic search with LLM generation:
- **Vector Database Creation** - Document chunking strategies, embedding model selection, and persistence with ChromaDB
- **Retrieval Evaluation** - Ranking metrics including Hit Rate, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and nDCG
- **End-to-End RAG Evaluation** - Measuring retrieval quality and generation quality on held-out test sets
- **Production RAG Pipelines** - Batch processing, parallel retrieval, and latency monitoring

## Key Features

- **Theoretical Explanations:** Every notebook starts with clear explanations of core theories, concepts, mathematical foundations, algorithmic principles, and architectural patterns.
- **Practical Application:** All projects use authentic, publicly available datasets, exploring real-world data challenges including cleaning, preprocessing, and feature engineering.
- **End-to-End Pipeline:** Notebooks guide through the complete project lifecycle from problem formulation and data acquisition to modeling, evaluation, and deployment readiness.
- **Reproducible Workflows:** Projects include pinned dependencies, deterministic random seeds, and documented hardware requirements.
- **Comprehensive Evaluation:** Multi-metric assessment includes task-specific metrics, resource utilization (CPU, GPU, memory, latency), and model performance (accuracy, F1, BLEU, ROUGE, BERTScore).
- **MLOps and LLMOps Integration:** Projects demonstrate production-ready practices including experiment tracking with MLflow, data version control with DVC, containerization with Docker, cloud deployment (AWS ECS/ECR), hyperparameter optimization, pipeline automation, GPU resource monitoring, and model registry management.

## Getting Started

To explore these projects, navigate to the notebook of interest. Notebooks can be viewed directly on GitHub or the repository can be cloned to run them locally. Each notebook is designed as a standalone guide to its respective topic with an end-to-end project pipeline.

**LICENSE: MIT License  - see LICENSE file for details.**
