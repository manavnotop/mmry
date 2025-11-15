from typing import Optional

from .embedding_base import EmbeddingModel
from .local_embedding import LocalEmbeddingModel
from .openrouter_embedding import OpenRouterEmbeddingModel


def create_embedding_model(
    model_type: str, model_name: str, api_key: Optional[str] = None
) -> EmbeddingModel:
    """
    Factory function to create an embedding model based on type.

    Args:
        model_type: Type of embedding model ('local' or 'openrouter')
        model_name: Name of the model to use
        api_key: API key for remote models (optional)

    Returns:
        An instance of an EmbeddingModel
    """
    if model_type == "local":
        return LocalEmbeddingModel(model_name)
    elif model_type == "openrouter":
        if not api_key:
            raise ValueError("API key is required for OpenRouter embedding models")
        return OpenRouterEmbeddingModel(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")
