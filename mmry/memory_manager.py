import os
import re
from typing import Any, Dict, List, Optional

from mmry.base.vectordb_base import VectorDBBase
from mmry.llms.openrouter_context_builder import OpenRouterContextBuilder
from mmry.llms.openrouter_merger import OpenRouterMerger
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
        api_key: Optional[str] = None,
        summarizer: Optional[OpenRouterSummarizer] = None,
        merger: Optional[OpenRouterMerger] = None,
        context_builder: Optional[OpenRouterContextBuilder] = None,
        log_path: str = "memory_events.jsonl",
    ):
        self.store = db or Qdrant()
        self.threshold = similarity_threshold
        self.logger = MemoryLogger(log_path)

        # Get API key from parameter, environment, or None
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # Initialize LLM components if api_key is available
        # Allow explicit components to override auto-initialization
        if api_key:
            self.summarizer = summarizer or OpenRouterSummarizer(api_key=api_key)
            self.merger = merger or OpenRouterMerger(api_key=api_key)
            self.context_builder = context_builder or OpenRouterContextBuilder(
                api_key=api_key
            )
        else:
            # If no API key, use provided components or None
            self.summarizer = summarizer
            self.merger = merger
            self.context_builder = context_builder

    def create_memory(
        self,
        text: str | List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a memory from text or conversation.

        Args:
            text: Either a string or a list of conversation dicts with 'role' and 'content' keys.
            metadata: Optional metadata to attach to the memory.

        Returns:
            Dict with status, id, and summary information.
        """
        self.logger.log("create_request", {"text": text})

        # Handle summarization for both text and conversations
        if self.summarizer:
            summarized = self.summarizer.summarize(text)
            # Clean the summary to remove markdown and formatting for better searchability
            summarized = self._clean_summary(summarized)
        else:
            # If no summarizer, convert conversation to string if needed
            if isinstance(text, list):
                # Format conversation as a simple string representation
                conversation_str = "\n".join(
                    [
                        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                        for msg in text
                    ]
                )
                summarized = conversation_str
            else:
                summarized = text

        metadata = metadata or {}
        metadata["raw_text"] = text if isinstance(text, str) else str(text)
        metadata["raw_conversation"] = text if isinstance(text, list) else None
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

    def _clean_summary(self, summary: str) -> str:
        """
        Clean summary text by removing markdown formatting and converting
        numbered lists to simple sentences for better vector search similarity.
        """
        # Remove markdown bold/italic
        summary = re.sub(r"\*\*([^*]+)\*\*", r"\1", summary)
        summary = re.sub(r"\*([^*]+)\*", r"\1", summary)
        summary = re.sub(r"__([^_]+)__", r"\1", summary)
        summary = re.sub(r"_([^_]+)_", r"\1", summary)

        # Convert numbered/bulleted lists to sentences
        # Match patterns like "1. text" or "- text" or "• text"
        lines = summary.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove list markers (1., 2., -, •, etc.)
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^[-•]\s*", "", line)
            cleaned_lines.append(line)

        # Join with periods and spaces for better semantic search
        cleaned = ". ".join(cleaned_lines)
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned
