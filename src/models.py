from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
	answer: str
	sources: list[str] = Field(default_factory=list)

class LLMRequest(BaseModel):
    question: str