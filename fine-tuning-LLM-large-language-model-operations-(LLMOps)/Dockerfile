FROM pytorch/pytorch:2.9.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/qwen-dpo-final
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "deployment/inference_api.py"]