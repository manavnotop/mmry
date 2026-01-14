import os
from typing import Any, Dict, List, Optional

from mmry.base.llm_base import ContextBuilderBase, MergerBase, SummarizerBase
from mmry.base.vectordb_base import VectorDBBase
from mmry.config import MemoryConfig
from mmry.factory import LLMFactory, VectorDBFactory
from mmry.utils.decay import apply_memory_decay
from mmry.utils.health import MemoryHealth
from mmry.utils.logger import MemoryLogger
from mmry.utils.scoring import rerank_results
from mmry.utils.text import clean_summary


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
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a memory from text or conversation.

        Args:
            text: Either a string or a list of conversation dicts with 'role' and 'content' keys.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with status, id, and summary information.
        """
        self.logger.log("create_request", {"text": text, "user_id": user_id})

        # Handle summarization for both text and conversations
        if self.summarizer:
            try:
                summarized = self.summarizer.summarize(text)
                # Clean the summary to remove markdown and formatting for better searchability
                summarized = clean_summary(summarized)
            except Exception as e:
                self.logger.log(
                    "summarizer_error",
                    {
                        "error": str(e),
                        "text_type": type(text).__name__,
                        "user_id": user_id,
                    },
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

        similar = self.store.search(summarized, top_k=1, user_id=user_id)

        if similar and similar[0]["score"] > self.threshold:
            old = similar[0]["payload"]["text"]
            mem_id = similar[0]["id"]

            if self.merger:
                try:
                    merged_text = self.merger.merge_memories(old, summarized)
                except Exception as e:
                    self.logger.log(
                        "merger_error", {"error": str(e), "user_id": user_id}
                    )
                    # Fallback to using the new summary if merger fails
                    merged_text = summarized
            else:
                merged_text = summarized

            self.store.update_memory(mem_id, merged_text, user_id=user_id)
            return {
                "status": "merged",
                "id": mem_id,
                "old": old,
                "new": summarized,
                "merged": merged_text,
            }

        mem_id = self.store.add_memory(summarized, metadata, user_id=user_id)
        result = {"status": "created", "id": mem_id, "summary": summarized}
        self.logger.log("create_result", result)
        return result

    def query_memory(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query memories based on a text query."""
        self.logger.log("query_request", {"query": query, "user_id": user_id})
        results = self.store.search(query, top_k, user_id=user_id)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        memories = [r["payload"]["text"] for r in reranked]

        context_summary = None
        if self.context_builder:
            try:
                context_summary = self.context_builder.build_context(memories)
            except Exception as e:
                self.logger.log(
                    "context_builder_error", {"error": str(e), "user_id": user_id}
                )
                # Fallback to joining memories if context builder fails
                context_summary = ". ".join(memories[:3])  # Use top 3 memories

        result = {
            "query": query,
            "context_summary": context_summary,
            "memories": reranked,
        }
        self.logger.log(
            "query_result",
            {"query": query, "top_k": len(result["memories"]), "user_id": user_id},
        )
        return result

    def update_memory(
        self, memory_id: str, new_text: str, user_id: Optional[str] = None
    ) -> None:
        """Update an existing memory with new text."""
        return self.store.update_memory(memory_id, new_text, user_id=user_id)

    def list_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all memories in the store."""
        return self.store.get_all(user_id=user_id)

    def delete_memory(
        self, memory_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete.
            user_id: Optional user ID to ensure correct user's memory is deleted.

        Returns:
            Dict with 'status' and 'deleted' keys.
        """
        self.logger.log("delete_request", {"memory_id": memory_id, "user_id": user_id})
        deleted = self.store.delete(memory_id, user_id=user_id)
        result = {"status": "deleted" if deleted else "not_found", "deleted": deleted}
        self.logger.log("delete_result", result)
        return result

    def get_health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health metrics for the memory system."""
        memories = self.store.get_all(user_id=user_id)
        health = MemoryHealth(memories)
        stats = health.summary()
        self.logger.log("health_snapshot", {"stats": stats, "user_id": user_id})
        return stats

    def create_memory_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple memories efficiently in a batch.

        Args:
            texts: List of text strings to create memories from.
            metadatas: Optional list of metadata dicts, one per text.
            user_ids: Optional list of user IDs, one per text.

        Returns:
            List of dicts with 'id', 'status', and 'summary' keys.
        """
        self.logger.log(
            "create_batch_request", {"count": len(texts), "user_ids": user_ids}
        )

        # Process texts - summarize if available
        processed_texts = texts
        if self.summarizer:
            processed_texts = []
            metadatas = metadatas or [{}] * len(texts)

            for i, text in enumerate(texts):
                try:
                    summarized = self.summarizer.summarize(text)
                    summarized = clean_summary(summarized)
                except Exception as e:
                    self.logger.log(
                        "summarizer_error_batch",
                        {
                            "error": str(e),
                            "index": i,
                            "user_id": user_ids[i] if user_ids else None,
                        },
                    )
                    summarized = text if isinstance(text, str) else str(text)

                # Update metadata with summary
                metadatas[i] = metadatas[i] or {}
                metadatas[i]["summary"] = summarized
                processed_texts.append(summarized)

        # Batch add to store
        memory_ids = self.store.add_batch(processed_texts, metadatas, user_ids)

        # Build results
        results = [
            {"status": "created", "id": mem_id, "summary": processed_texts[i]}
            for i, mem_id in enumerate(memory_ids)
        ]
        self.logger.log("create_batch_result", {"count": len(results)})
        return results
