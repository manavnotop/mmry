from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDBBase(ABC):
    """Base interface for interacting with all vector databases"""

    @abstractmethod
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, new_text: str) -> None:
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        pass
