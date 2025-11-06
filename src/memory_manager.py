from typing import Any, Dict, List, Optional

from src.memory_store import MemoryStore


class MemoryManager:
    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.store = MemoryStore()
        self.threshold = similarity_threshold

    def create_memory(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """create a new memory or update the existing one"""

        similar: List[Dict[str, Any]] = self.store.search(text, top_k=1)
        if similar and similar[0]["score"] > self.threshold:
            """ in this case just update the memory """

            mem_id = similar[0]["id"]
            self.store.update_memory(mem_id, text)
            return {
                "status": "updated",
                "id": mem_id,
                "old": similar[0]["payload"],
                "new_text": text,
            }
        else:
            mem_id: str = self.store.add_memory(text, metadata)
            return {"status": "created", "id": mem_id, "text": text}

    def query_memory(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.store.search(query, top_k)

    def update_memory(self, memory_id: str, new_text: str) -> None:
        return self.store.update_memory(memory_id, new_text)

    def list_all(self) -> List[Dict[str, Any]]:
        return self.store.get_all()
