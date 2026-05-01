from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from contextlib import asynccontextmanager
import sys
sys.path.append("/app")

from src.rag.pipeline import OptimizedRAGPipeline
from src.config import config

class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 512

class QueryResponse(BaseModel):
    answer: str
    query_time_ms: float

class BatchQueryRequest(BaseModel):
    queries: List[str]
    max_tokens: Optional[int] = 512

class BatchQueryResponse(BaseModel):
    answers: List[str]
    total_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    device: str

rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    print("Loading RAG pipeline...")
    rag_pipeline = OptimizedRAGPipeline(
        config,
        config.vectorstore.chroma_persist_dir,
        config.training.output_dir
    )
    print("RAG pipeline loaded successfully")
    yield
    print("Shutting down...")

app = FastAPI(title="LLMOps RAG API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=rag_pipeline is not None,
        vectorstore_loaded=rag_pipeline is not None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    import time
    start_time = time.time()
    answer = rag_pipeline.get_answer(request.query)
    query_time = (time.time() - start_time) * 1000
    return QueryResponse(answer=answer, query_time_ms=query_time)

@app.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    import time
    start_time = time.time()
    answers = rag_pipeline.batch_query(request.queries)
    total_time = (time.time() - start_time) * 1000
    return BatchQueryResponse(answers=answers, total_time_ms=total_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)