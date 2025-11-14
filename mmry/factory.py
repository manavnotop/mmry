from typing import Dict, Type

from mmry.base.llm_base import LLMBase
from mmry.base.vectordb_base import VectorDBBase
from mmry.config import LLMConfig, VectorDBConfig

# Registry dictionaries to store available implementations
VECTOR_DB_REGISTRY: Dict[str, Type[VectorDBBase]] = {}
LLM_REGISTRY: Dict[str, Type[LLMBase]] = {}


def register_vectordb(name: str):
    """Decorator to register a new vector database implementation."""

    def decorator(cls: Type[VectorDBBase]) -> Type[VectorDBBase]:
        VECTOR_DB_REGISTRY[name] = cls
        return cls

    return decorator


def register_llm(name: str):
    """Decorator to register a new LLM implementation."""

    def decorator(cls: Type[LLMBase]) -> Type[LLMBase]:
        LLM_REGISTRY[name] = cls
        return cls

    return decorator


# Register the Qdrant implementation at module level
def _register_implementations():
    """Register all available implementations."""
    # Import here to avoid circular imports
    from mmry.vector_store.qdrant import Qdrant

    # Register Qdrant vector database
    register_vectordb("qdrant")(Qdrant)

    # Register OpenRouter implementations
    from mmry.llms.openrouter_context_builder import OpenRouterContextBuilder
    from mmry.llms.openrouter_merger import OpenRouterMerger
    from mmry.llms.openrouter_summariser import OpenRouterSummarizer

    register_llm("openrouter_summarizer")(OpenRouterSummarizer)
    register_llm("openrouter_merger")(OpenRouterMerger)
    register_llm("openrouter_context_builder")(OpenRouterContextBuilder)


# Register implementations when module is imported
_register_implementations()


class VectorDBFactory:
    """Factory class for creating vector database instances."""

    @staticmethod
    def create(db_type: str, config: VectorDBConfig) -> VectorDBBase:
        """
        Create a vector database instance based on type and configuration.

        Args:
            db_type: Type of vector database (e.g., 'qdrant', 'pinecone')
            config: Configuration object for the vector database

        Returns:
            Instance of the requested vector database

        Raises:
            ValueError: If the requested database type is not registered
        """
        if db_type.lower() not in VECTOR_DB_REGISTRY:
            available_dbs = ", ".join(VECTOR_DB_REGISTRY.keys())
            raise ValueError(
                f"Unsupported vector database: {db_type}. Available: {available_dbs}"
            )

        db_class = VECTOR_DB_REGISTRY[db_type.lower()]
        return db_class(
            url=config.url,
            collection_name=config.collection_name,
            embed_model=config.embed_model,
        )


class LLMFactory:
    """Factory class for creating LLM instances."""

    @staticmethod
    def create(config: LLMConfig, task: str = "general") -> LLMBase:
        """
        Create an LLM instance based on configuration and task.

        Args:
            config: Configuration object for the LLM
            task: The specific task the LLM will be used for (affects implementation choice)

        Returns:
            Instance of the requested LLM

        Raises:
            ValueError: If the requested LLM type is not registered
        """
        # Choose implementation based on task
        task_to_provider = {
            "summarizer": "openrouter_summarizer",
            "merger": "openrouter_merger",
            "context_builder": "openrouter_context_builder",
        }

        llm_key = task_to_provider.get(task, "openrouter_summarizer")

        if llm_key not in LLM_REGISTRY:
            available_llms = ", ".join(LLM_REGISTRY.keys())
            raise ValueError(
                f"Unsupported LLM for task {task}: {llm_key}. Available: {available_llms}"
            )

        llm_class = LLM_REGISTRY[llm_key]
        return llm_class(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
        )
