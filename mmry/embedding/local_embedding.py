from typing import List

from sentence_transformers import SentenceTransformer

from .embedding_base import EmbeddingModel


class LocalEmbeddingModel(EmbeddingModel):
    """
    Local embedding model using Sentence Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding model.

        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using local model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        return self.model.encode(texts).tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.

        Returns:
            The dimension of the embeddings
        """
        return self.model.get_sentence_embedding_dimension()
