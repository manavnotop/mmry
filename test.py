"""Mmry memory management tests."""

import os

from dotenv import load_dotenv

from mmry import MemoryClient, MemoryConfig, MemoryManager

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


class TestMemoryClient:
    """Test MemoryClient with pytest-style assertions."""

    def setup_method(self):
        """Create fresh client and clean state before each test."""
        self.client = MemoryClient(
            {
                "vector_db": {
                    "url": "http://localhost:6333",
                    "collection_name": "test_mmry",
                },
                "api_key": API_KEY,
                "similarity_threshold": 0.7,
            }
        )
        self.user1, self.user2 = "user1", "user2"

    def test_basic_memory_operations(self):
        """Test create, query, list, health."""
        result = self.client.create_memory("I live in Mumbai")
        assert result["status"] == "created"

        result = self.client.query_memory("Where do I live?", top_k=3)
        assert len(result["memories"]) >= 1

        memories = self.client.list_all()
        assert len(memories) >= 1

        health = self.client.get_health()
        assert health["memory_count"] >= 1

    def test_conversation_memory(self):
        """Test conversation summarization."""
        conv = [
            {"role": "user", "content": "Planning trip to Japan"},
            {"role": "assistant", "content": "Nice! Which cities?"},
            {"role": "user", "content": "Tokyo and Kyoto"},
        ]
        result = self.client.create_memory(conv)
        assert result["status"] == "created"

    def test_multi_user_isolation(self):
        """Test user memory isolation."""
        # Create memories for different users
        result1 = self.client.create_memory("I live in New York", user_id=self.user1)
        result2 = self.client.create_memory(
            "I prefer chocolate ice cream", user_id=self.user2
        )

        # Query user1's memory - should only return user1's memory
        user1_memories = self.client.query_memory("New York", user_id=self.user1)
        assert len(user1_memories["memories"]) >= 1
        # Verify the returned memory belongs to user1
        for mem in user1_memories["memories"]:
            assert mem["payload"].get("user_id") == self.user1

        # Query user2's memory - should only return user2's memory
        user2_memories = self.client.query_memory("chocolate", user_id=self.user2)
        assert len(user2_memories["memories"]) >= 1
        # Verify the returned memory belongs to user2
        for mem in user2_memories["memories"]:
            assert mem["payload"].get("user_id") == self.user2

        # Verify user1's memory is NOT in user2's query results
        user1_memory_ids = {m["id"] for m in user1_memories["memories"]}
        user2_memory_ids = {m["id"] for m in user2_memories["memories"]}
        assert user1_memory_ids.isdisjoint(user2_memory_ids), (
            "User memories should be isolated"
        )

    def test_metadata(self):
        """Test memory with metadata."""
        result = self.client.create_memory(
            "Test memory", metadata={"category": "test", "priority": 1}
        )
        assert result["status"] == "created"

    def test_direct_manager(self):
        """Test MemoryManager directly."""
        config = MemoryConfig(
            llm_config=None,
            vector_db_config=None,
            similarity_threshold=0.8,
        )
        manager = MemoryManager(config=config)
        result = manager.create_memory("Direct test")
        assert result["status"] == "created"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
