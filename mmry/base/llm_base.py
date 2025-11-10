from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Abstract base for any LLM provider."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        pass
