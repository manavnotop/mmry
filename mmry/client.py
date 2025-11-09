from typing import Any, Dict, Optional

from mmry.memory_manager import MemoryManager
from mmry.vector_store.qdrant import Qdrant


class MemoryClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        vector_db_config = (config or {}).get("vector_db", {})
        self.manager = MemoryManager(Qdrant(**vector_db_config))

    def create_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        return self.manager.create_memory(text, metadata)

    def query_memory(self, query: str, top_k: int = 3):
        return self.manager.query_memory(query, top_k)

    def update_memory(self, memory_id: str, new_text: str):
        return self.manager.update_memory(memory_id, new_text)

    def list_all(self):
        return self.manager.list_all()
