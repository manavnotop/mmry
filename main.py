from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

from memory_manager import MemoryManager

app = FastAPI()
manager = MemoryManager()

class MemoryInput(BaseModel):
    text: str
    metadata: dict | None = None

class QueryInput(BaseModel):
    query: str
    top_k: int = 3

class UpdateInput(BaseModel):
    memory_id: str
    new_text: str

@app.post("/memory/create")
async def create_memory(inp: MemoryInput) -> Dict[str, Any]:
    return manager.create_memory(inp.text, inp.metadata)

@app.post("/memory/query")
async def query_memory(inp: QueryInput):
    return { "results": manager.query_memory(inp.query, inp.top_k)}

@app.post("/memory/update")
async def update_memory(inp: UpdateInput):
    return manager.update_memory(inp.memory_id, inp.new_text)

@app.post("/memory/all")
async def list_all():
    return manager.list_all()
