from typing import Any, Dict, List, Optional

from mmry.base.vectordb_base import VectorDBBase
from mmry.llms.openrouter_summariser import OpenRouterSummarizer
from mmry.utils.decay import apply_memory_decay
from mmry.utils.health import MemoryHealth
from mmry.utils.logger import MemoryLogger
from mmry.utils.scoring import rerank_results
from mmry.vector_store.qdrant import Qdrant


class MemoryManager:
    def __init__(
        self,
        db: VectorDBBase | None = None,
        similarity_threshold: float = 0.8,
        summarizer: Optional[OpenRouterSummarizer] = None,
        log_path: str = "memory_events.jsonl",
    ):
        self.store = db or Qdrant()
        self.threshold = similarity_threshold
        self.summarizer = summarizer
        self.logger = MemoryLogger(log_path)

    def create_memory(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Summarize → Check Similarity → Merge or Add"""

        self.logger.log("create_request", {"text": text})
        summarized = self.summarizer.summarize(text) if self.summarizer else text
        metadata = metadata or {}
        metadata["raw_text"] = text
        metadata["summary"] = summarized

        similar = self.store.search(summarized, top_k=1)

        if similar and similar[0]["score"] > self.threshold:
            old = similar[0]["payload"]["text"]
            mem_id = similar[0]["id"]

            if self.merger:
                merged_text = self.merger.merge_memories(old, summarized)
            else:
                merged_text = summarized

            self.store.update_memory(mem_id, merged_text)
            return {
                "status": "merged",
                "id": mem_id,
                "old": old,
                "new": summarized,
                "merged": merged_text,
            }

        mem_id = self.store.add_memory(summarized, metadata)
        result = {"status": "created", "id": mem_id, "summary": summarized}
        self.logger.log("create_result", result)
        return result

    def query_memory(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self.logger.log("query_request", {"query": query})
        results = self.store.search(query, top_k)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        memories = [r["payload"]["text"] for r in reranked]

        context_summary = None
        if self.context_builder:
            context_summary = self.context_builder.build_context(memories)

        result = {
            "query": query,
            "context_summary": context_summary,
            "memories": reranked,
        }
        self.logger.log(
            "query_result", {"query": query, "top_k": len(result["memories"])}
        )
        return result

    def update_memory(self, memory_id: str, new_text: str) -> None:
        return self.store.update_memory(memory_id, new_text)

    def list_all(self) -> List[Dict[str, Any]]:
        return self.store.get_all()

    def get_health(self) -> Dict[str, Any]:
        memories = self.store.get_all()
        health = MemoryHealth(memories)
        stats = health.summary()
        self.logger.log("health_snapshot", stats)
        return stats
