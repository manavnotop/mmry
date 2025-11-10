from typing import Any, Dict, List, Optional

from mmry.base.vectordb_base import VectorDBBase
from mmry.llms.openrouter_summariser import OpenRouterSummarizer
from mmry.utils.decay import apply_memory_decay
from mmry.utils.scoring import rerank_results
from mmry.vector_store.qdrant import Qdrant


class MemoryManager:
    def __init__(
        self,
        db: VectorDBBase | None = None,
        similarity_threshold: float = 0.8,
        summarizer: Optional[OpenRouterSummarizer] = None,
    ):
        self.store = db or Qdrant()
        self.threshold = similarity_threshold
        self.summarizer = summarizer

    def create_memory(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        summarized = text
        if self.summarizer:
            summarized = self.summarizer.summarize(text)

        metadata = metadata or {}
        metadata["raw_text"] = text
        metadata["summary"] = summarized

        similar = self.store.search(text, top_k=1)
        if similar and similar[0]["score"] > self.threshold:
            mem_id = similar[0]["id"]
            self.store.update_memory(mem_id, text)
            return {
                "status": "updated",
                "id": mem_id,
                "old": similar[0]["payload"],
                "new_text": text,
            }
        mem_id = self.store.add_memory(text, metadata)
        return {"status": "created", "id": mem_id, "text": text}

    def query_memory(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.store.search(query, top_k)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        return reranked

    def update_memory(self, memory_id: str, new_text: str) -> None:
        return self.store.update_memory(memory_id, new_text)

    def list_all(self) -> List[Dict[str, Any]]:
        return self.store.get_all()
