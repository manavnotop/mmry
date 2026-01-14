"""Custom exceptions for mmry."""


class MmryError(Exception):
    """Base exception for mmry errors."""

    pass


class MemoryNotFoundError(MmryError):
    """Raised when a memory is not found."""

    pass


class MemoryDeleteError(MmryError):
    """Raised when memory deletion fails."""

    pass


class MemoryUpdateError(MmryError):
    """Raised when memory update fails."""

    pass


class LLMError(MmryError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when LLM API connection fails."""

    pass


class LLMTTimeoutError(LLMError):
    """Raised when LLM API times out."""

    pass


class VectorDBError(MmryError):
    """Base exception for vector database errors."""

    pass


class VectorDBConnectionError(VectorDBError):
    """Raised when vector database connection fails."""

    pass
