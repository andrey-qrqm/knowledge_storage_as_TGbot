from models import LLMResponse, LLMRequest

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


import logging

import ollama
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from huggingface_hub import login, whoami
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    filename="src/logs/service.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s"
)

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
CHAT_MODEL = os.getenv("OLLAMA_CHATMODEL")

_ollama_client = ollama.Client(host=OLLAMA_HOST)

def is_ollama_available() -> bool:
    try:
        _ollama_client.list()
    except (ConnectionError, ollama.ResponseError):
        return False
    return True

def embed_query(query: str) -> list[float]:
   return embed_model.encode(query).tolist()

def get_llm_response(request: LLMRequest) -> LLMResponse:
    # QUERY EMBED
    # GET CONTEXT
    # GET RESPONSE FROM LLM

    if is_ollama_available():
        logging.info("OLLAMA is available. Proceeding with LLM response generation.")
        llm_resp = _ollama_client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.question}
            ]
        )
        resp = llm_resp["message"]["content"].strip()
        return LLMResponse(answer=resp, sources=["source1", "source2"])
    else:
        logging.error("OLLAMA is not available. Cannot generate LLM response.")
        raise ConnectionError("OLLAMA service is not available.")