import os
import re
from typing import Any, Dict, List, Optional

from mmry.base.llm_base import ContextBuilderBase, MergerBase, SummarizerBase
from mmry.base.vectordb_base import VectorDBBase
from mmry.config import MemoryConfig
from mmry.factory import LLMFactory, VectorDBFactory
from mmry.utils.decay import apply_memory_decay
from mmry.utils.health import MemoryHealth
from mmry.utils.logger import MemoryLogger
from mmry.utils.scoring import rerank_results


class MemoryManager:
    """Main class to manage memories using vector storage and LLMs."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        db: VectorDBBase | None = None,
        summarizer: Optional[SummarizerBase] = None,
        merger: Optional[MergerBase] = None,
        context_builder: Optional[ContextBuilderBase] = None,
        log_path: str = "memory_events.jsonl",
    ):
        """
        Initialize the MemoryManager with configuration and components.

        Args:
            config: MemoryConfig object containing all configuration
            db: Optional pre-configured vector database instance
            summarizer: Optional pre-configured summarizer instance
            merger: Optional pre-configured merger instance
            context_builder: Optional pre-configured context builder instance
            log_path: Path for logging memory events
        """
        config = config or MemoryConfig()

        # Set up vector database
        if db:
            self.store = db
        elif config.vector_db_config:
            self.store = VectorDBFactory.create("qdrant", config.vector_db_config)
        else:
            # Create default Qdrant instance with default configuration
            from mmry.vector_store.qdrant import Qdrant

            self.store = Qdrant()  # Use default configuration

        self.threshold = config.similarity_threshold
        self.logger = MemoryLogger(config.log_path)

        # Get API key from parameter, environment, or None
        api_key = (
            config.llm_config.api_key
            if config.llm_config
            else os.getenv("OPENROUTER_API_KEY")
        )

        # Initialize LLM components if api_key is available
        # Allow explicit components to override auto-initialization
        if api_key and config.llm_config:
            llm_config = config.llm_config
            self.summarizer = summarizer or LLMFactory.create(llm_config, "summarizer")
            self.merger = merger or LLMFactory.create(llm_config, "merger")
            self.context_builder = context_builder or LLMFactory.create(
                llm_config, "context_builder"
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
            try:
                summarized = self.summarizer.summarize(text)
                # Clean the summary to remove markdown and formatting for better searchability
                summarized = self._clean_summary(summarized)
            except Exception as e:
                self.logger.log(
                    "summarizer_error",
                    {"error": str(e), "text_type": type(text).__name__},
                )
                # Fallback to basic text processing if summarizer fails
                if isinstance(text, list):
                    conversation_str = "\n".join(
                        [
                            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                            for msg in text
                        ]
                    )
                    summarized = conversation_str
                else:
                    summarized = text
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
                try:
                    merged_text = self.merger.merge_memories(old, summarized)
                except Exception as e:
                    self.logger.log("merger_error", {"error": str(e)})
                    # Fallback to using the new summary if merger fails
                    merged_text = summarized
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
        """Query memories based on a text query."""
        self.logger.log("query_request", {"query": query})
        results = self.store.search(query, top_k)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        memories = [r["payload"]["text"] for r in reranked]

        context_summary = None
        if self.context_builder:
            try:
                context_summary = self.context_builder.build_context(memories)
            except Exception as e:
                self.logger.log("context_builder_error", {"error": str(e)})
                # Fallback to joining memories if context builder fails
                context_summary = ". ".join(memories[:3])  # Use top 3 memories

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
        """Update an existing memory with new text."""
        return self.store.update_memory(memory_id, new_text)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all memories in the store."""
        return self.store.get_all()

    def get_health(self) -> Dict[str, Any]:
        """Get health metrics for the memory system."""
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
