from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

PATH_INFO = "../tg/data/info.json"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_KEY"),
    cloud_inference=True
)

#Load data from the JSON file
def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} entries from {file_path}.")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}")
            return []

def chunking(data, chunk_size, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    for entry in data:
        text = entry.get("info")
        if not text:
            logging.warning(f"Entry {entry.get('index')} has no 'info', skipping.")
            continue
        for n, chunk in enumerate(splitter.split_text(text)):
            yield {
                "index": entry["index"],            # parent note id
                "chunk_id": f"{entry['index']}_{n}",  # unique per chunk
                "chunk": chunk,
                "timestamp": entry.get("timestamp"),
            }

def batch_print(data, batch_size=2):
    for batch in batched(chunking(data, chunk_size=1000), batch_size):
        logging.info(f"Processing batch of size {len(batch)}")
        for item in batch:
            logging.info(f"Index: {item['index']}, Chunk ID: {item['chunk_id']}, Timestamp: {item.get('timestamp')}")
            

def batch_embed(data, chunk_size=1000, batch_size=16):
    for batch in batched(chunking(data, chunk_size=chunk_size), batch_size):
        
        to_embed = [item for item in batch if item.get("is_embedded") is not True]
        skipped = len(batch) - len(to_embed)

        if skipped:
            logging.info(f"Skipping {skipped} already embedded chunks")

        if not to_embed:
            continue  # nothing to do for this batch
        
        texts = [item["chunk"] for item in to_embed]

        vecs = model.encode(
            texts,
            batch_size=len(texts),
            max_length=512,
        )["dense_vecs"]

        out = []
        for item, vec in zip(to_embed, vecs):
            out.append({**item, "embedding": vec.tolist()})

        logging.info(f"Embedded batch of {len(out)} chunks")
        yield out


data = load_data(PATH_INFO)
batch_print(data, batch_size=2)

all_chunks = list(chain.from_iterable(batch_embed(data)))
vectors = [item["embedding"] for item in all_chunks]
print(len(all_chunks), len(all_chunks[0]["embedding"])) 

def upsert_to_qdrant(client, all_chunks):
    points = []
    for i, item in enumerate(all_chunks):
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, item["chunk_id"])),
            vector=item["embedding"],
            payload={
                "index": item["index"],
                "chunk": item["chunk"],
                "timestamp": item.get("timestamp"),
                "author": item.get("author"),
                "user_id": item.get("user_id"),
            }
        )
        points.append(point)
    logging.info(f"Upserting {len(points)} points to Qdrant")
    
    try:
        client.upsert(
            collection_name="test_tg",
            points=points,
        )
        logging.info(f"Successfully upserted {len(points)} points to Qdrant collection 'test_tg'")
    except Exception as e:
        logging.error(f"Failed to upsert points to Qdrant: {e}")

upsert_to_qdrant(client, all_chunks)
