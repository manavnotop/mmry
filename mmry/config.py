from dataclasses import dataclass
from typing import Optional


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
    embed_model: str = "all-MiniLM-L6-v2"


@dataclass
class MemoryConfig:
    """Overall memory system configuration"""

    llm_config: Optional[LLMConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None
    similarity_threshold: float = 0.8
    log_path: str = "memory_events.jsonl"
