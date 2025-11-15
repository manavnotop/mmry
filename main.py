from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from mmry.client import MemoryClient

app = FastAPI()
client = MemoryClient()


class MemoryInput(BaseModel):
    text: str
    metadata: dict | None = None
    user_id: Optional[str] = None


class QueryInput(BaseModel):
    query: str
    top_k: int = 3
    user_id: Optional[str] = None


class UpdateInput(BaseModel):
    memory_id: str
    new_text: str
    user_id: Optional[str] = None


@app.post("/memory/create")
async def create_memory(inp: MemoryInput) -> Dict[str, Any]:
    return client.create_memory(inp.text, inp.metadata, inp.user_id)


@app.post("/memory/query")
async def query_memory(inp: QueryInput):
    return {"results": client.query_memory(inp.query, inp.top_k, inp.user_id)}


@app.post("/memory/update")
async def update_memory(inp: UpdateInput):
    return client.update_memory(inp.memory_id, inp.new_text, inp.user_id)


@app.post("/memory/all")
async def list_all(user_id: Optional[str] = None):
    return client.list_all(user_id)


@app.get("/health")
async def health(user_id: Optional[str] = None):
    return client.get_health(user_id)
