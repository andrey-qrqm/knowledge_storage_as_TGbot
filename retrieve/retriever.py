from FlagEmbedding import BGEM3FlagModel
from itertools import batched, chain
import json
import os, uuid
from dotenv import load_dotenv
import logging
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

load_dotenv()

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

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
    """
    Retrieve the most relevant context from Qdrant based on the query.

    Args:
        query (str): The input query for which to retrieve context.
        top_k (int): The number of top relevant contexts to retrieve.

    Returns:
        list[dict]: A list of dictionaries containing the retrieved context.
    """
    # Embed the query
    query_embedding = model.encode(
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
            "score": point.score,
            "payload": point.payload
        })
    
    for answer in result:
        logging.info(f"Retrieved context with score {answer['score']}: {answer['payload']}")
        print(answer['payload']['index'], answer['payload']['chunk'])  
    
    return result

def build_context(results: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    total = 0
    for rank, r in enumerate(results, start=1):
        p = r["payload"]
        header = f"[{rank}] (chunk {p['index']}, score {r['score']:.3f}"
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

#results = retrieve_context("What does KG stand for?", top_k=3)
#context = build_context(results)
#print("Context for LLM:\n", context)

