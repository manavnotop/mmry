from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.

        Returns:
            The dimension of the embeddings
        """
        pass
