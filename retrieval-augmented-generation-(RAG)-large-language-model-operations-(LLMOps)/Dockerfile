FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh

RUN /opt/conda/bin/conda create -n llmops python=3.10 -y \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/envs/llmops/bin:${PATH}"

COPY requirements.txt requirements_dev.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_dev.txt \
    && pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir dvc dvc-s3 mlflow dagshub

WORKDIR /app

COPY . .

RUN dvc remote add -d storage s3://your-bucket/dvc-storage

EXPOSE 8000 5000

CMD ["python", "-m", "uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]