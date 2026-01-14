from typing import List

import requests

from .embedding_base import EmbeddingModel


class OpenRouterEmbeddingModel(EmbeddingModel):
    """
    OpenRouter embedding model using their API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/text-embedding-3-small",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the OpenRouter embedding model.

        Args:
            api_key: OpenRouter API key
            model_name: Name of the embedding model to use
            base_url: Base URL for the OpenRouter API
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self._embedding_dim = None  # Will be determined from first API call

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenRouter API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepare the request payload
        payload = {"model": self.model_name, "input": texts}

        # Make the API call
        response = requests.post(
            f"{self.base_url}/embeddings", json=payload, headers=headers
        )

        response.raise_for_status()
        result = response.json()

        # Extract embeddings from response
        embeddings = [item["embedding"] for item in result["data"]]

        # Cache embedding dimension if not already cached
        if self._embedding_dim is None and embeddings:
            self._embedding_dim = len(embeddings[0])

        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.
        Makes an API call if dimension is not known yet.

        Returns:
            The dimension of the embeddings
        """
        if self.model_name in [
            "qwen/qwen3-7b-embedding:v1",
            "qwen/qwen3-8b-embedding:v1",
        ]:
            return 768  # Qwen3 7B/8B embedding model dimension
        elif self.model_name == "qwen/qwen3-14b-embedding:v1":
            return 1536  # Qwen3-14B embedding model dimension

        if self._embedding_dim is not None:
            return self._embedding_dim

        # Make a test call with a simple input to determine the embedding dimension
        try:
            test_embedding = self.embed(["test"])
            self._embedding_dim = len(test_embedding[0])
            return self._embedding_dim
        except Exception:
            # If we can't determine the dimension, return a reasonable default
            # This should typically not happen in practice as the API call would succeed
            return 1536  # Common dimension for OpenAI embeddings
