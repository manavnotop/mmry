import datetime
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from mmry.base.vectordb_base import VectorDBBase
from mmry.embedding import create_embedding_model


class Qdrant(VectorDBBase):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "mmry",
        embed_model: str = "all-MiniLM-L6-v2",
        embed_model_type: str = "local",
        embed_api_key: Optional[str] = None,
    ):
        self.url = url
        self.collection = collection_name
        self.client = QdrantClient(url=url)
        self.embedder = create_embedding_model(
            model_type=embed_model_type, model_name=embed_model, api_key=embed_api_key
        )
        self.ensure_collection()

    def ensure_collection(self):
        dim = self.embedder.get_embedding_dimension()
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
        return self.embedder.embed(texts)

    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        vector = self.embed([text])[0]
        memory_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "created_at": datetime.datetime.now(datetime.UTC),
            "importance": 1.0,
        }
        if user_id:
            payload["user_id"] = user_id
        if metadata:
            payload.update(metadata)
        self.client.upsert(
            collection_name=self.collection,
            points=[rest.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return memory_id

    def search(
        self, query: str, top_k: int = 3, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        vector = self.embed([query])[0]

        # Prepare filtering conditions
        if user_id:
            # Create a filter to only return memories for the specified user
            search_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="user_id", match=rest.MatchValue(value=user_id)
                    )
                ]
            )
            results = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                query_filter=search_filter,
            )
        else:
            # For backward compatibility, search all memories if no user_id is provided
            results = self.client.search(
                collection_name=self.collection, query_vector=vector, limit=top_k
            )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def update_memory(
        self, memory_id: str, new_text: str, user_id: Optional[str] = None
    ) -> None:
        vector = self.embed([new_text])[0]
        # Retrieve existing payload to preserve created_at and importance
        try:
            existing = self.client.retrieve(
                collection_name=self.collection, ids=[memory_id]
            )[0]
            payload = existing.payload.copy() if existing.payload else {}
        except Exception:
            payload = {}

        payload["text"] = new_text
        # If user_id is provided, add it to the payload to ensure consistency
        if user_id:
            payload["user_id"] = user_id
        self.client.upsert(
            collection_name=self.collection,
            points=[rest.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )

    def get_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        # Prepare filtering conditions
        if user_id:
            search_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="user_id", match=rest.MatchValue(value=user_id)
                    )
                ]
            )
            records = self.client.scroll(
                collection_name=self.collection, limit=100, scroll_filter=search_filter
            )[0]
        else:
            # For backward compatibility, return all memories if no user_id is provided
            records = self.client.scroll(collection_name=self.collection, limit=100)[0]
        return [{"id": r.id, "payload": r.payload} for r in records]
