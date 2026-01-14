from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDBBase(ABC):
    """Base interface for interacting with all vector databases"""

    @abstractmethod
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        pass

    @abstractmethod
    def search(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_memory(
        self, memory_id: str, new_text: str, user_id: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def get_all(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def add_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add multiple memories in a single batch operation."""
        pass

    @abstractmethod
    def delete(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a memory by ID. Returns True if deleted, False if not found."""
        pass
