from typing import Any, Dict, Optional

from mmry.memory_manager import MemoryManager
from mmry.vector_store.qdrant import Qdrant


class MemoryClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        vector_db_config = config.get("vector_db", {})
        api_key = config.get("api_key")
        self.manager = MemoryManager(db=Qdrant(**vector_db_config), api_key=api_key)

    def create_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        return self.manager.create_memory(text, metadata)

    def query_memory(self, query: str, top_k: int = 3):
        return self.manager.query_memory(query, top_k)

    def update_memory(self, memory_id: str, new_text: str):
        return self.manager.update_memory(memory_id, new_text)

    def list_all(self):
        return self.manager.list_all()
