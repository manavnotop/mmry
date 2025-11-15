from typing import Any, Dict, List, Optional, Union

from mmry.config import LLMConfig, MemoryConfig, VectorDBConfig
from mmry.memory_manager import MemoryManager


class MemoryClient:
    """Client interface for the memory management system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MemoryClient with configuration.

        Args:
            config: Dictionary containing configuration options
        """
        config_dict = config or {}

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
        )

        # Create memory configuration
        memory_config = MemoryConfig(
            llm_config=llm_config,
            vector_db_config=vector_db_config,
            similarity_threshold=config_dict.get("similarity_threshold", 0.8),
            log_path=config_dict.get("log_path", "memory_events.jsonl"),
        )

        self.manager = MemoryManager(config=memory_config)

    def create_memory(
        self,
        text: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        """
        Create a memory from text or conversation.

        Args:
            text: Either a string or a list of conversation dicts with 'role' and 'content' keys.
            metadata: Optional metadata to attach to the memory.
            user_id: Optional user identifier to associate with this memory.

        Returns:
            Dict with status, id, and summary information.
        """
        return self.manager.create_memory(text, metadata, user_id)

    def query_memory(self, query: str, top_k: int = 3, user_id: Optional[str] = None):
        """
        Query memories based on a text query.

        Args:
            query: Text to search for in memories
            top_k: Number of results to return
            user_id: Optional user identifier to filter memories

        Returns:
            List of matching memories with context summary
        """
        return self.manager.query_memory(query, top_k, user_id)

    def update_memory(self, memory_id: str, new_text: str, user_id: Optional[str] = None):
        """
        Update an existing memory with new text.

        Args:
            memory_id: ID of the memory to update
            new_text: New text to replace the existing memory
            user_id: Optional user identifier to ensure correct user's memory is updated
        """
        return self.manager.update_memory(memory_id, new_text, user_id)

    def list_all(self, user_id: Optional[str] = None):
        """
        List all memories in the store.

        Args:
            user_id: Optional user identifier to filter memories

        Returns:
            List of all memories
        """
        return self.manager.list_all(user_id)

    def get_health(self, user_id: Optional[str] = None):
        """
        Get health metrics for the memory system.

        Args:
            user_id: Optional user identifier to get health stats for specific user

        Returns:
            Health metrics dictionary
        """
        return self.manager.get_health(user_id)
