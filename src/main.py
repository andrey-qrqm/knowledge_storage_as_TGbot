from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import httpx

from routes.rag_qa import router as rag_qa_router
from routes.proxy import proxy_router as proxy_router

import os
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        base_url=OLLAMA_HOST,
        timeout=httpx.Timeout(None, connect=5.0),  # generation can be slow
    )
    yield
    await app.state.http.aclose()

app = FastAPI(lifespan=lifespan)

app.include_router(rag_qa_router)
app.include_router(proxy_router)  # Add this line to include the router with the prefix

@app.get("/")
async def read_root():  
    return {"message": "Hello, World!"}
    