from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def trim_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value):
        return SearchRequest.trim_query(value)


class Chunk(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    url: str
    content: str
    rrf_score: float


class SearchResponse(BaseModel):
    results: list[Chunk]


class Source(BaseModel):
    id: int
    chunk_id: int
    document_id: int
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
