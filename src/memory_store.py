import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "mmry"


class MemoryStore:
    def __init__(self, url="http://localhost:6333", embed_model="all-MiniLM-L6-v2") -> None:
        self.client = QdrantClient(url=url)
        self.embedder = SentenceTransformer(embed_model)
        self.ensure_collection()

    def ensure_collection(self):
        dim = self.embedder.get_sentence_embedding_dimension()
        try:
            self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.client.recreate_collection(
                COLLECTION_NAME,
                vectors_config=rest.VectorParams(
                    size=dim, distance=rest.Distance.COSINE
                ),
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts).tolist()

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        vector: list[float] = self.embed([text])[0]
        memory_id = str(uuid.uuid4())
        payload: dict[str, str] = {"text": text}
        if metadata:
            payload.update(metadata)
        self.client.upsert(
            COLLECTION_NAME,
            points=[rest.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return memory_id

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        vector: list[float] = self.embed([query])[0]
        results = self.client.search(COLLECTION_NAME, query_vector=vector, limit=top_k)
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def update_memory(self, memory_id: str, new_text: str) -> None:
        vector: list[float] = self.embed([new_text])[0]
        self.client.upsert(
            COLLECTION_NAME,
            points=[
                rest.PointStruct(
                    id=memory_id, vector=vector, payload={"text": new_text}
                )
            ],
        )

    def get_all(self) -> List[Dict[str, Any]]:
        records = self.client.scroll(COLLECTION_NAME, limit=100)[0]
        return [{"id": r.id, "payload": r.payload} for r in records]
