import os
from typing import Any, Dict, List, Optional

from mmry.base.llm_base import ContextBuilderBase, MergerBase, SummarizerBase
from mmry.base.vectordb_base import VectorDBBase
from mmry.config import MemoryConfig
from mmry.factory import LLMFactory, VectorDBFactory
from mmry.utils.decay import apply_memory_decay
from mmry.utils.health import MemoryHealth
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
    ):
        """
        Initialize the MemoryManager with configuration and components.

        Args:
            config: MemoryConfig object containing all configuration
            db: Optional pre-configured vector database instance
            summarizer: Optional pre-configured summarizer instance
            merger: Optional pre-configured merger instance
            context_builder: Optional pre-configured context builder instance
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
            text: Either a string or a list of conversation dicts with 'role'
                and 'content' keys.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with status, id, and summary information.
        """
        # Handle summarization for both text and conversations
        if self.summarizer:
            try:
                summarized = self.summarizer.summarize(text)
                # Clean the summary to remove markdown formatting
                summarized = clean_summary(summarized)
            except Exception:
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
                except Exception:
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
        return {"status": "created", "id": mem_id, "summary": summarized}

    def query_memory(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query memories based on a text query."""
        results = self.store.search(query, top_k, user_id=user_id)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        memories = [r["payload"]["text"] for r in reranked]

        context_summary = None
        if self.context_builder:
            try:
                context_summary = self.context_builder.build_context(memories)
            except Exception:
                # Fallback to joining memories if context builder fails
                context_summary = ". ".join(memories[:3])  # Use top 3 memories

        return {
            "query": query,
            "context_summary": context_summary,
            "memories": reranked,
        }

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
        deleted = self.store.delete(memory_id, user_id=user_id)
        return {"status": "deleted" if deleted else "not_found", "deleted": deleted}

    def get_health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health metrics for the memory system."""
        memories = self.store.get_all(user_id=user_id)
        health = MemoryHealth(memories)
        return health.summary()

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
        # Process texts - summarize if available
        processed_texts = texts
        if self.summarizer:
            processed_texts = []
            metadatas = metadatas or [{}] * len(texts)

            for i, text in enumerate(texts):
                try:
                    summarized = self.summarizer.summarize(text)
                    summarized = clean_summary(summarized)
                except Exception:
                    summarized = text if isinstance(text, str) else str(text)

                # Update metadata with summary
                metadatas[i] = metadatas[i] or {}
                metadatas[i]["summary"] = summarized
                processed_texts.append(summarized)

        # Batch add to store
        memory_ids = self.store.add_batch(processed_texts, metadatas, user_ids)

        # Build results
        return [
            {"status": "created", "id": mem_id, "summary": processed_texts[i]}
            for i, mem_id in enumerate(memory_ids)
        ]

    async def create_memory_async(
        self,
        text: str | List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a memory from text or conversation (async version).

        Args:
            text: Either a string or a list of conversation dicts.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with status, id, and summary information.
        """
        # Handle summarization for both text and conversations
        if self.summarizer:
            try:
                # Check if summarizer has async method
                if hasattr(self.summarizer, "summarize_async"):
                    summarized = await self.summarizer.summarize_async(text)
                else:
                    summarized = self.summarizer.summarize(text)
                summarized = clean_summary(summarized)
            except Exception:
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
                    if hasattr(self.merger, "merge_memories_async"):
                        merged_text = await self.merger.merge_memories_async(
                            old, summarized
                        )
                    else:
                        merged_text = self.merger.merge_memories(old, summarized)
                except Exception:
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
        return {"status": "created", "id": mem_id, "summary": summarized}

    async def query_memory_async(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query memories based on a text query (async version).

        Args:
            query: Text to search for in memories.
            top_k: Number of results to return (default 3).
            user_id: Optional user identifier to filter memories.

        Returns:
            Dict with 'memories', 'context_summary', and 'query' keys.
        """
        results = self.store.search(query, top_k, user_id=user_id)
        decayed = [apply_memory_decay(r) for r in results]
        reranked = rerank_results(decayed)
        memories = [r["payload"]["text"] for r in reranked]

        context_summary = None
        if self.context_builder:
            try:
                if hasattr(self.context_builder, "build_context_async"):
                    context_summary = await self.context_builder.build_context_async(
                        memories
                    )
                else:
                    context_summary = self.context_builder.build_context(memories)
            except Exception:
                context_summary = ". ".join(memories[:3])

        return {
            "query": query,
            "context_summary": context_summary,
            "memories": reranked,
        }

    async def create_memory_batch_async(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple memories efficiently in a batch (async version).

        Args:
            texts: List of text strings to create memories from.
            metadatas: Optional list of metadata dicts, one per text.
            user_ids: Optional list of user IDs, one per text.

        Returns:
            List of dicts with 'id', 'status', and 'summary' keys.
        """
        processed_texts = texts
        if self.summarizer and hasattr(self.summarizer, "summarize_async"):
            processed_texts = []
            metadatas = metadatas or [{}] * len(texts)

            for i, text in enumerate(texts):
                try:
                    summarized = await self.summarizer.summarize_async(text)
                    summarized = clean_summary(summarized)
                except Exception:
                    summarized = text if isinstance(text, str) else str(text)

                metadatas[i] = metadatas[i] or {}
                metadatas[i]["summary"] = summarized
                processed_texts.append(summarized)

        memory_ids = self.store.add_batch(processed_texts, metadatas, user_ids)

        return [
            {"status": "created", "id": mem_id, "summary": processed_texts[i]}
            for i, mem_id in enumerate(memory_ids)
        ]
