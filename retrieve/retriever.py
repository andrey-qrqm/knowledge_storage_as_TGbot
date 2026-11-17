from FlagEmbedding import BGEM3FlagModel, FlagReranker
from itertools import batched, chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os, uuid
from dotenv import load_dotenv
import logging
import asyncio

from transformers import AutoModelForSequenceClassification

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

load_dotenv()

base_model = BGEM3FlagModel(
    'BAAI/bge-m3',  
    use_fp16=True # Setting use_fp16 to True speeds up computation with a slight performance degradation
    ) 

reranker = FlagReranker(
    'BAAI/bge-reranker-v2-m3', 
    use_fp16=True# Setting use_fp16 to True speeds up computation with a slight performance degradation
) 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_KEY"),
    cloud_inference=True
)

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    # Embed the query
    query_embedding = base_model.encode(
        query,
        max_length=512,
    )["dense_vecs"]

    # Perform a search in Qdrant
    search_result = client.query_points(
        collection_name="test_tg",
        query=query_embedding,
        with_vectors=False,
        with_payload=True,
        limit=top_k,
    )

    result = []
    for point in search_result.points:
        result.append({
            "score_vector": point.score,
            "payload": point.payload
        })
    
    for answer in result:
        logging.info(f"Retrieved context with score {answer['score_vector']}: {answer['payload']}")
        print(answer['payload']['index'], answer['payload']['chunk'])  
    
    return result

def double_chunk(results: list[dict], chunk_size: int = 256, overlap: int = 50) -> list[dict]:
    """Further split retrieved chunks into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    new_results = []
    for r in results:
        p = r["payload"]
        for n, chunk in enumerate(splitter.split_text(p["chunk"])):
            new_results.append({
                "score_vector": r["score_vector"],
                "payload": {
                    "index": p["index"],
                    "chunk_id": f"{p['index']}_{n}",
                    "chunk": chunk,
                    "timestamp": p.get("timestamp"),
                },
                "score_rerank": None  # Placeholder for reranker score
            })
    return new_results

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    for r in results:
        p = r["payload"]
        rerank_input = [(query, p["chunk"])]
        score = reranker.compute_score(rerank_input, normalize=True)[0]
        r["score_rerank"] = score
        logging.info(f"Reranked chunk {p['index']} with score {float(score):.3f}")

    return results


def top_k_reranked(results: list[dict], k: int = 3) -> list[dict]:
    """Return the top k results based on reranker scores."""
    sorted_results = sorted(results, key=lambda x: x["score_rerank"], reverse=True)
    return sorted_results[:k]

def build_context(results: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    total = 0
    for rank, r in enumerate(results, start=1):
        p = r["payload"]
        header = f"[{rank}] (chunk {p['index']}, score {r['score_vector']:.3f}, rerank {r['score_rerank']:.3f})"
        if p.get("author"):
            header += f", author: {p['author']}"
        if p.get("timestamp"):
            header += f", {p['timestamp']}"
        header += ")"

        block = f"{header}\n{p['chunk'].strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)

def to_json(results: list[dict]) -> str:
    """Convert results to a JSON string for easier inspection."""
    return json.dumps(results, indent=2)

#results = retrieve_context("What does KG stand for?", top_k=3)
#chunked_results = double_chunk(results, chunk_size=256, overlap=50)
#reranked_scores = rerank_results("What does KG stand for?", chunked_results)
#top_reranked = top_k_reranked(reranked_scores, k=3)
#context = build_context(top_reranked, max_chars=6000)

#print("Retrieved contexts:\n", context)
