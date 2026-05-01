#!/usr/bin/env python
import requests
import time
import json
import argparse
from typing import List, Dict, Any

def test_health(base_url: str = "http://localhost:8000"):
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_single_query(base_url: str = "http://localhost:8000", query: str = "How to sort a list in Python?"):
    start_time = time.time()
    response = requests.post(
        f"{base_url}/query",
        json={"query": query}
    )
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"\nSingle Query Test:")
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed_time:.2f} ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer'][:200]}...")
        print(f"Query time from API: {data['query_time_ms']:.2f} ms")
    
    return response.json() if response.status_code == 200 else None

def test_batch_query(base_url: str = "http://localhost:8000", queries: List[str] = None):
    if queries is None:
        queries = [
            "How to reverse a string in Python?",
            "How to read a file in Python?",
            "How to create a list comprehension?"
        ]
    
    start_time = time.time()
    response = requests.post(
        f"{base_url}/batch_query",
        json={"queries": queries}
    )
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"\nBatch Query Test:")
    print(f"Status: {response.status_code}")
    print(f"Total time: {elapsed_time:.2f} ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Number of answers: {len(data['answers'])}")
        print(f"API reported time: {data['total_time_ms']:.2f} ms")
        
        for i, answer in enumerate(data['answers']):
            print(f"\nQuery {i+1}: {queries[i]}")
            print(f"Answer: {answer[:150]}...")
    
    return response.json() if response.status_code == 200 else None

def run_load_test(base_url: str = "http://localhost:8000", num_requests: int = 10, query: str = "Test query"):
    print(f"\nLoad Test: {num_requests} requests")
    
    times = []
    successes = 0
    
    for i in range(num_requests):
        start_time = time.time()
        response = requests.post(
            f"{base_url}/query",
            json={"query": f"{query} {i}"}
        )
        elapsed_time = (time.time() - start_time) * 1000
        times.append(elapsed_time)
        
        if response.status_code == 200:
            successes += 1
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{num_requests} requests")
    
    print(f"\nLoad Test Results:")
    print(f"Success rate: {successes/num_requests*100:.1f}%")
    print(f"Average time: {sum(times)/len(times):.2f} ms")
    print(f"Min time: {min(times):.2f} ms")
    print(f"Max time: {max(times):.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["health", "single", "batch", "load"], default="health", help="Test type")
    parser.add_argument("--query", default="How to sort a list in Python?", help="Query for single test")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests for load test")
    
    args = parser.parse_args()
    
    if args.test == "health":
        test_health(args.url)
    elif args.test == "single":
        test_single_query(args.url, args.query)
    elif args.test == "batch":
        test_batch_query(args.url)
    elif args.test == "load":
        run_load_test(args.url, args.num_requests, args.query)