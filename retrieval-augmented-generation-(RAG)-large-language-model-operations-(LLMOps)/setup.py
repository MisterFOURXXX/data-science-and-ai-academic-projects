from setuptools import setup, find_packages

setup(
    name="llmops-rag-pipeline",
    version="1.0.0",
    author="MisterFour",
    description="LLMOps RAG Pipeline with DVC, MLflow, and DagsHub",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "langchain",
        "chromadb",
        "sentence-transformers",
        "scikit-learn",
        "polars",
        "beautifulsoup4",
        "evaluate",
        "fastapi",
        "uvicorn",
        "mlflow",
        "dagshub",
        "dvc"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy"
        ],
        "aws": [
            "boto3"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)