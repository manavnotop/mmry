from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    api_key: str
    model: str = "openai/gpt-oss-safeguard-20b"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 30


@dataclass
class VectorDBConfig:
    """Configuration for vector databases"""

    url: str = "http://localhost:6333"
    collection_name: str = "mmry"
    embed_model: str = "all-MiniLM-L6-v2"  # Default local model
    embed_model_type: Literal["local", "openrouter"] = (
        "local"  # Type of embedding model
    )
    embed_api_key: Optional[str] = None  # API key for remote embedding models


@dataclass
class MemoryConfig:
    """Overall memory system configuration"""

    llm_config: Optional[LLMConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None
    similarity_threshold: float = 0.8
