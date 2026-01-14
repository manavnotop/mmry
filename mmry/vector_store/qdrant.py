import datetime
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from mmry.base.vectordb_base import VectorDBBase


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
        # Import here to avoid circular import
        from mmry.factory import EmbeddingFactory

        self.embedder = EmbeddingFactory.create(
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

    def add_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple memories in a single batch operation.

        Args:
            texts: List of text strings to embed and store.
            metadatas: Optional list of metadata dicts, one per text.
            user_ids: Optional list of user IDs, one per text.

        Returns:
            List of memory IDs.
        """
        if not texts:
            return []

        # Batch embed all texts (much faster than individual calls)
        vectors = self.embed(texts)
        now = datetime.datetime.now(datetime.UTC)

        points = []
        for i, text in enumerate(texts):
            memory_id = str(uuid.uuid4())
            payload = {
                "text": text,
                "created_at": now,
                "importance": 1.0,
            }
            if user_ids and i < len(user_ids):
                payload["user_id"] = user_ids[i]
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            points.append(
                rest.PointStruct(id=memory_id, vector=vectors[i], payload=payload)
            )

        # Single batch upsert
        self.client.upsert(collection_name=self.collection, points=points)
        return [p.id for p in points]

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

        # Store version history before updating
        old_text = payload.get("text", "")
        old_updated_at = payload.get("updated_at") or payload.get("created_at")
        old_version = payload.get("version", 1)

        # Initialize history if not exists
        if "history" not in payload:
            payload["history"] = []

        # Add previous version to history
        version_entry = {
            "text": old_text,
            "updated_at": str(old_updated_at) if old_updated_at else None,
            "version": old_version,
        }
        payload["history"].append(version_entry)

        payload["text"] = new_text
        payload["version"] = old_version + 1
        payload["updated_at"] = datetime.datetime.now(datetime.UTC)
        # If user_id is provided, add it to the payload to ensure consistency
        if user_id:
            payload["user_id"] = user_id
        self.client.upsert(
            collection_name=self.collection,
            points=[rest.PointStruct(id=memory_id, vector=vector, payload=payload)],
        )

    def get_all(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all memories with optional pagination.

        Args:
            user_id: Optional user ID to filter memories.
            limit: Number of records to fetch per page (default 100).

        Returns:
            List of all memories with their payloads.
        """
        all_records = []
        offset = None

        while True:
            if user_id:
                search_filter = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="user_id", match=rest.MatchValue(value=user_id)
                        )
                    ]
                )
                records, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=limit,
                    scroll_filter=search_filter,
                    offset=offset,
                )
            else:
                records, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=limit,
                    offset=offset,
                )

            all_records.extend([{"id": r.id, "payload": r.payload} for r in records])

            # Break if no more pages
            if offset is None or len(records) < limit:
                break

        return all_records

    def get_memory_history(
        self, memory_id: str, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the version history of a memory.

        Args:
            memory_id: The ID of the memory.
            user_id: Optional user ID for ownership verification.

        Returns:
            List of historical versions with text, timestamp, and version info.
        """
        try:
            # Check ownership if user_id provided
            if user_id:
                existing = self.client.retrieve(
                    collection_name=self.collection, ids=[memory_id]
                )[0]
                if existing.payload.get("user_id") != user_id:
                    return []

            existing = self.client.retrieve(
                collection_name=self.collection, ids=[memory_id]
            )[0]
            return existing.payload.get("history", [])
        except Exception:
            return []

    def delete(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete.
            user_id: Optional user ID for ownership verification.

        Returns:
            True if deleted, False if not found.
        """
        try:
            # Check ownership if user_id provided
            if user_id:
                try:
                    existing = self.client.retrieve(
                        collection_name=self.collection, ids=[memory_id]
                    )[0]
                    if existing.payload.get("user_id") != user_id:
                        return False  # Not authorized to delete
                except Exception:
                    return False  # Memory not found

            # Delete the point
            self.client.delete(
                collection_name=self.collection,
                points_selector=rest.PointIdsList(points=[memory_id]),
            )
            return True
        except Exception:
            return False
