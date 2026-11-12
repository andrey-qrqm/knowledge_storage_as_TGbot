from fastapi import APIRouter, Depends, HTTPException
from typing import AsyncGenerator, Annotated

from models import LLMRequest, LLMResponse
from services.question import get_llm_response

router = APIRouter(prefix="/api")

async def get_session() -> AsyncGenerator:
    # Placeholder for session management logic
    yield

@router.post("/rag_qa")
async def rag_qa(
    data: LLMRequest,
    ):
    response = get_llm_response(data)
    return response
