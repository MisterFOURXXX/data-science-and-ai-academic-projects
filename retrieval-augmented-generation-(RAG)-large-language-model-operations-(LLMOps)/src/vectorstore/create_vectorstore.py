import pickle
import json
import hashlib
import re
import sys
import os
import shutil
from pathlib import Path
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config

def create_documents_from_dataframe(df, dataset_type):
    documents = []
    for row in df.iter_rows(named=True):
        enriched_content = f"Question: {row['question_body']}\nAnswer: {row['answer_body']}"
        
        metadata = {
            "source": "stackoverflow",
            "dataset_type": dataset_type,
            "question_id": hashlib.md5(row['question_body'].encode()).hexdigest(),
            "score": row['answer_score'],
            "length": len(row['answer_body']),
            "has_code": 1 if re.search(r'```|def |class |function|import |#include', row['answer_body']) else 0,
            "index": row.get('index', 0)
        }
        
        documents.append(Document(page_content=enriched_content, metadata=metadata))
    return documents

def create_vectorstore():
    chroma_path = Path(config.vectorstore.chroma_persist_dir)
    
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    
    chroma_path.mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    
    with open("data/splits/train.pkl", "rb") as f:
        train_pl = pickle.load(f)
    
    train_documents = create_documents_from_dataframe(train_pl, "train")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.vectorstore.chunk_size,
        chunk_overlap=config.vectorstore.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    
    split_docs = text_splitter.split_documents(train_documents)
    
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.vectorstore.embedding_model,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': config.vectorstore.embedding_batch_size},
    )
    
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        client=chroma_client,
        collection_name="stackoverflow_coding_train",
        persist_directory=str(chroma_path),
    )
    
    metrics = {
        "num_documents": len(train_documents),
        "num_chunks": len(split_docs),
        "avg_chunk_length": sum(len(doc.page_content) for doc in split_docs) / len(split_docs),
        "embedding_dimension": 384,
        "chunk_size": config.vectorstore.chunk_size,
        "chunk_overlap": config.vectorstore.chunk_overlap
    }
    
    with open("metrics/vectorstore_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Created vectorstore with {len(split_docs)} chunks")
    return vectorstore

if __name__ == "__main__":
    create_vectorstore()