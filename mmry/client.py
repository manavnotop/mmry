from typing import Any, Dict, List, Optional, Union

from mmry.config import MemoryConfig
from mmry.memory_manager import MemoryManager


class MemoryClient:
    """Client interface for the memory management system."""

    def __init__(self, config: Optional[Union[MemoryConfig, Dict[str, Any]]] = None):
        """
        Initialize the MemoryClient with configuration.

        Args:
            config: MemoryConfig object or dict with all configuration options
        """
        if config is None:
            config = MemoryConfig()
        elif isinstance(config, dict):
            config = self._dict_to_config(config)
        # else: config is already a MemoryConfig

        self.manager = MemoryManager(config=config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryClient":
        """
        Create a MemoryClient instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration options

        Returns:
            MemoryClient instance
        """
        config = cls._dict_to_config(config_dict)
        return cls(config)

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> MemoryConfig:
        """Convert a dictionary to MemoryConfig."""
        from mmry.config import LLMConfig, VectorDBConfig

        # Create LLM configuration if API key is provided
        llm_config = None
        if "api_key" in config_dict:
            llm_config = LLMConfig(
                api_key=config_dict["api_key"],
                model=config_dict.get("llm_model", "openai/gpt-oss-safeguard-20b"),
                base_url=config_dict.get(
                    "llm_base_url", "https://openrouter.ai/api/v1/chat/completions"
                ),
                timeout=config_dict.get("llm_timeout", 30),
            )

        # Create vector DB configuration
        vector_db_config_dict = config_dict.get("vector_db", {})
        vector_db_config = VectorDBConfig(
            url=vector_db_config_dict.get("url", "http://localhost:6333"),
            collection_name=vector_db_config_dict.get("collection_name", "mmry"),
            embed_model=vector_db_config_dict.get("embed_model", "all-MiniLM-L6-v2"),
            embed_model_type=vector_db_config_dict.get("embed_model_type", "local"),
            embed_api_key=vector_db_config_dict.get("embed_api_key"),
        )

        # Create memory configuration
        return MemoryConfig(
            llm_config=llm_config,
            vector_db_config=vector_db_config,
            similarity_threshold=config_dict.get("similarity_threshold", 0.8),
        )

    def create_memory(
        self,
        text: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a memory from text or conversation.

        Examples:
            >>> client = MemoryClient()
            >>> result = client.create_memory("I live in Mumbai")
            >>> print(result['id'])
            'abc123-...'

            >>> # With conversation
            >>> conv = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ]
            >>> client.create_memory(conv, user_id="user123")

            >>> # With metadata
            >>> client.create_memory(
            ...     "Important meeting tomorrow",
            ...     metadata={"category": "work", "priority": "high"},
            ...     user_id="user123"
            ... )

        Args:
            text: Either a string or a list of conversation dicts with 'role'
                and 'content' keys.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with 'status', 'id', and 'summary' keys.
        """
        return self.manager.create_memory(text, metadata, user_id)

    def query_memory(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query memories based on a text query.

        Examples:
            >>> client.query_memory("Where do I live?", top_k=3)
            {'memories': [...], 'context_summary': '...', 'query': 'Where do I live?'}

            >>> # User-specific query
            >>> client.query_memory("work preferences", user_id="user123")

        Args:
            query: Text to search for in memories.
            top_k: Number of results to return (default 3).
            user_id: Optional user identifier to filter memories.

        Returns:
            Dict with 'memories', 'context_summary', and 'query' keys.
        """
        return self.manager.query_memory(query, top_k, user_id)

    def update_memory(
        self, memory_id: str, new_text: str, user_id: Optional[str] = None
    ) -> None:
        """
        Update an existing memory with new text.

        Args:
            memory_id: ID of the memory to update
            new_text: New text to replace the existing memory
            user_id: Optional user identifier to ensure correct user's memory is updated
        """
        return self.manager.update_memory(memory_id, new_text, user_id)

    def list_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all memories in the store.

        Args:
            user_id: Optional user identifier to filter memories

        Returns:
            List of all memories
        """
        return self.manager.list_all(user_id)

    def get_health(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health metrics for the memory system.

        Args:
            user_id: Optional user identifier to get health stats for specific user

        Returns:
            Health metrics dictionary
        """
        return self.manager.get_health(user_id)

    def delete_memory(
        self, memory_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete
            user_id: Optional user identifier to ensure correct user's memory is deleted

        Returns:
            Dict with 'status' and 'deleted' keys
        """
        return self.manager.delete_memory(memory_id, user_id)

    def create_memory_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple memories efficiently in a batch.

        Examples:
            >>> client.create_memory_batch([
            ...     "I love Python",
            ...     "I work at Google",
            ...     "I live in NYC"
            ... ])
            [{'status': 'created', 'id': '...', 'summary': '...'}, ...]

            >>> # With user IDs
            >>> client.create_memory_batch(
            ...     ["Fact 1", "Fact 2"],
            ...     user_ids=["user1", "user1"]
            ... )

        Args:
            texts: List of text strings to create memories from.
            metadatas: Optional list of metadata dicts, one per text.
            user_ids: Optional list of user IDs, one per text.

        Returns:
            List of dicts with 'id', 'status', and 'summary' keys.
        """
        return self.manager.create_memory_batch(texts, metadatas, user_ids)

    async def create_memory_async(
        self,
        text: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a memory from text or conversation (async version).

        Args:
            text: Either a string or a list of conversation dicts with 'role'
                and 'content' keys.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with 'status', 'id', and 'summary' keys.
        """
        return await self.manager.create_memory_async(text, metadata, user_id)

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
        return await self.manager.query_memory_async(query, top_k, user_id)

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
        return await self.manager.create_memory_batch_async(texts, metadatas, user_ids)
