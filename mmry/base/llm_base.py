from abc import ABC, abstractmethod
from typing import Dict, List, Union


class LLMBase(ABC):
    """Abstract base for any LLM provider."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generic method to call the LLM with a prompt"""
        pass


class SummarizerBase(LLMBase):
    """Base class specifically for summarization tasks"""

    @abstractmethod
    def summarize(self, text: Union[str, List[Dict[str, str]]]) -> str:
        """Summarize text or conversation into a memory statement"""
        pass


class MergerBase(LLMBase):
    """Base class specifically for memory merging tasks"""

    @abstractmethod
    def merge_memories(self, old_memory: str, new_memory: str) -> str:
        """Merge two memory statements into a single, updated factual statement"""
        pass


class ContextBuilderBase(LLMBase):
    """Base class specifically for context building tasks"""

    @abstractmethod
    def build_context(self, memories: List[str]) -> str:
        """Summarize multiple memory statements into a coherent context"""
        pass
