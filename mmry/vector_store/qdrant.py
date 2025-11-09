import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from mmry.base.vectordb_base import VectorDBBase


class Qdrant(VectorDBBase):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "mmry",
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.url = url
        self.collection = collection_name
        self.client = QdrantClient(url=url)
        self.embedder = SentenceTransformer()
        self.ensure_collection()

    def ensure_collection(self):
        dim = self.embedder.get_sentence_embedding_dimension()
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(
                    size=dim, distance=rest.Distance.COSINE
                ),
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts).tolist()

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        vector = self.embed([text])[0]
        memory_id = str(uuid.uuid4())
        payload = {"text": text}
        if metadata:
            payload.update(metadata)
        self.client.upsert(
            collection_name=self.collection,
            points=[rest.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return memory_id

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        vector = self.embed([query])[0]
        results = self.client.search(
            collection_name=self.collection, query_vector=vector, limit=top_k
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def update_memory(self, memory_id: str, new_text: str) -> None:
        vector = self.embed([new_text])[0]
        self.client.upsert(
            collection_name=self.collection,
            points=[
                rest.PointStruct(
                    id=memory_id, vector=vector, payload={"text": new_text}
                )
            ],
        )

    def get_all(self) -> List[Dict[str, Any]]:
        records = self.client.scroll(collection_name=self.collection, limit=100)[0]
        return [{"id": r.id, "payload": r.payload} for r in records]
