# Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install wheels to a local folder to copy them later (reduces image size)
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime
FROM python:3.11-slim

# Optimization: Only install the bare minimum libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the installed site-packages from the builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure the installed packages are in the path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Optimization: Reduce Python bytecode/log overhead
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Optimization: Use 'httptools' and 'uvloop' for maximum FastAPI speed
CMD ["uvicorn", "src.deployment.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--workers", "2"]