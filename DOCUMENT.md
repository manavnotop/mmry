# mmry Documentation

A memory management layer for AI agents and applications. Store, retrieve, and manage conversational memories using vector similarity search.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Memory Operations](#memory-operations)
7. [Error Handling](#error-handling)
8. [Advanced Usage](#advanced-usage)

---

## Overview

mmry (pronounced "memory") provides persistent memory for AI applications. It helps AI agents remember context across conversations without manual prompt engineering.

### Key Features

- **Multi-user support** - Isolated memories per user with `user_id` filtering
- **Semantic search** - Find relevant memories using vector embeddings
- **Automatic deduplication** - Similar memories are merged intelligently
- **Memory versioning** - Track changes over time with history
- **Batch operations** - Efficient bulk memory creation
- **Configurable similarity threshold** - Control when memories merge
- **Flexible LLM integration** - Use OpenRouter or local models
- **Async support** - Non-blocking operations with httpx

---

## Quick Start

### Basic Usage

```python
from mmry import MemoryClient

# Initialize client
client = MemoryClient({
    "vector_db": {"url": "http://localhost:6333"},
    "api_key": "your-openrouter-api-key",
})

# Create a memory
result = client.create_memory("I live in Mumbai and work at Google")
print(result)  # {'status': 'created', 'id': '...', 'summary': '...'}

# Query memories
results = client.query_memory("Where does the user live?")
print(results["context_summary"])  # "The user lives in Mumbai"
print(results["memories"])          # List of relevant memories

# List all memories
all_memories = client.list_all()
print(len(all_memories))

# Delete a memory
client.delete_memory(memory_id)

# Batch create
client.create_memory_batch([
    "User prefers dark mode",
    "User works as a software engineer",
    "User's favorite programming language is Python"
])
```

### Async Usage

```python
import asyncio
from mmry import MemoryClient

async def main():
    client = MemoryClient({"api_key": "your-key"})

    # Create memory asynchronously
    result = await client.create_memory_async(
        "I love working with Python"
    )

    # Query memories asynchronously
    results = await client.query_memory_async(
        "What programming language?"
    )

    # Batch create asynchronously
    await client.create_memory_batch_async([
        "Fact 1",
        "Fact 2",
        "Fact 3"
    ])

asyncio.run(main())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        mmry Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐                                            │
│   │  MemoryClient │  ← User-facing API                         │
│   └───────┬───────┘                                            │
│           │                                                    │
│   ┌───────▼───────┐                                            │
│   │ MemoryManager │  ← Core orchestration                      │
│   └───────┬───────┘                                            │
│           │                                                    │
│   ┌───────▼───────┐    ┌──────────────────┐                   │
│   │   VectorDB    │    │  LLM Components  │                   │
│   │   (Qdrant)    │◄───│ Summarizer       │                   │
│   │               │    │ Merger           │                   │
│   │  - Store      │    │ ContextBuilder   │                   │
│   │  - Search     │    └──────────────────┘                   │
│   │  - Retrieve   │                                            │
│   └───────────────┘                                            │
│                                                                 │
│   ┌───────────────┐    ┌──────────────────┐                   │
│   │  Embeddings   │───►│ Sentence-BERT    │  ← Local models   │
│   └───────────────┘    │ (default)        │                   │
│                        │ or OpenRouter    │                   │
│                        └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `MemoryClient` | `mmry/client.py` | Main user-facing API |
| `MemoryManager` | `mmry/memory_manager.py` | Orchestrates memory operations |
| `Qdrant` | `mmry/vector_store/qdrant.py` | Vector database storage |
| `OpenRouterLLMBase` | `mmry/llms/openrouter_base.py` | LLM API integration |
| `MemoryConfig` | `mmry/config.py` | Configuration dataclasses |
| `Factory` | `mmry/factory.py` | Factory patterns for components |

---

## Configuration

### VectorDBConfig

```python
from mmry import MemoryConfig, VectorDBConfig

config = MemoryConfig(
    vector_db_config=VectorDBConfig(
        url="http://localhost:6333",           # Qdrant URL
        collection_name="mmry",                # Collection name
        embed_model="all-MiniLM-L6-v2",        # Embedding model
        embed_model_type="local",              # "local" or "openrouter"
        embed_api_key=None,                    # API key for remote embeddings
    ),
    similarity_threshold=0.8,                  # Merge threshold (0.0-1.0)
)
```

### LLMConfig

```python
from mmry import MemoryConfig, LLMConfig

config = MemoryConfig(
    llm_config=LLMConfig(
        api_key="your-openrouter-api-key",
        model="openai/gpt-4o",                 # LLM model
        base_url="https://openrouter.ai/api/v1/chat/completions",
        timeout=30,                            # Request timeout
    ),
)
```

### Using Dictionary Configuration

```python
from mmry import MemoryClient

client = MemoryClient({
    "api_key": "your-key",
    "llm_model": "openai/gpt-4o",
    "vector_db": {
        "url": "http://localhost:6333",
        "collection_name": "mmry",
        "embed_model": "all-MiniLM-L6-v2",
        "embed_model_type": "local",
    },
    "similarity_threshold": 0.8,
})
```

---

## API Reference

### MemoryClient

#### `create_memory(text, metadata=None, user_id=None)`

Create a memory from text or conversation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` or `List[Dict]` | Text or conversation dicts with 'role' and 'content' |
| `metadata` | `Dict[str, Any]` | Optional metadata to attach |
| `user_id` | `str` | Optional user identifier |

**Returns:** `Dict[str, Any]` with 'status', 'id', and 'summary' keys

---

#### `query_memory(query, top_k=3, user_id=None)`

Query memories based on a text query.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Text to search for |
| `top_k` | `int` | Number of results (default 3) |
| `user_id` | `str` | Optional user filter |

**Returns:** `Dict[str, Any]` with 'memories', 'context_summary', and 'query' keys

---

#### `update_memory(memory_id, new_text, user_id=None)`

Update an existing memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | `str` | ID of memory to update |
| `new_text` | `str` | New text content |
| `user_id` | `str` | Optional user verification |

---

#### `delete_memory(memory_id, user_id=None)`

Delete a memory by ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_id` | `str` | ID of memory to delete |
| `user_id` | `str` | Optional user verification |

**Returns:** `Dict[str, Any]` with 'status' and 'deleted' keys

---

#### `list_all(user_id=None)`

List all memories.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | Optional user filter |

**Returns:** `List[Dict[str, Any]]` of all memories

---

#### `create_memory_batch(texts, metadatas=None, user_ids=None)`

Create multiple memories efficiently.

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | List of text strings |
| `metadatas` | `List[Dict]` | Optional metadata per text |
| `user_ids` | `List[str]` | Optional user IDs per text |

**Returns:** `List[Dict[str, Any]]` with 'id', 'status', 'summary' keys

---

#### Async Methods

All main methods have async counterparts:

- `create_memory_async()`
- `query_memory_async()`
- `create_memory_batch_async()`

---

## Memory Operations

### How Memory Creation Works

```
Input Text/Conversation
        │
        ▼
┌───────────────────┐
│ Summarizer (LLM)  │  ← Extract key facts, condense
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Embedding Model   │  ← Convert to vector
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Vector Search    │  ← Find similar memories
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
 Below      Above
threshold   threshold
    │           │
    ▼           ▼
 Store      ┌───────────────────┐
 as new     │ Merger (LLM)      │  ← Combine similar memories
 memory     └─────────┬─────────┘
                    │
                    ▼
            Update existing memory
```

**Steps:**
1. Text or conversation is summarized by LLM into key facts
2. Summary is embedded into a vector
3. Similar memories are searched (similarity > threshold)
4. If similar: merge with existing using LLM
5. If new: store in Qdrant with metadata

### How Memory Search Works

```
Query
 │
 ▼
┌───────────────────┐
│ Embedding Model   │  ← Convert query to vector
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Vector Search    │  ← Find top-k similar memories
│  (Qdrant)         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Rerank & Decay    │  ← Score by relevance and recency
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Context Builder   │  ← Combine memories into context
│      (LLM)        │  ← "User lives in Mumbai, works at Google"
└─────────┬─────────┘
          │
          ▼
   Context for LLM
```

---

## Error Handling

mmry provides a hierarchy of custom exceptions for structured error handling:

```python
from mmry.errors import (
    MmryError,
    MemoryNotFoundError,
    MemoryDeleteError,
    MemoryUpdateError,
    LLMError,
    LLMConnectionError,
    LLMTTimeoutError,
    VectorDBError,
    VectorDBConnectionError,
)

try:
    result = client.create_memory("Some text")
except LLMConnectionError:
    print("Failed to connect to LLM API")
except MemoryNotFoundError:
    print("Memory not found")
```

---

## Advanced Usage

### Multi-User Support

```python
# Create memories for different users
client.create_memory("I prefer dark mode", user_id="user1")
client.create_memory("I prefer light mode", user_id="user2")

# Query memories for specific user
results = client.query_memory("What theme?", user_id="user1")
# Returns only user1's memories
```

### Memory Versioning

```python
# Update a memory
client.update_memory(memory_id, "Updated information")

# Get memory history
history = qdrant.get_memory_history(memory_id)
# Returns list of previous versions
```

### Custom Embedding Models

```python
from mmry import MemoryConfig, VectorDBConfig

config = MemoryConfig(
    vector_db_config=VectorDBConfig(
        url="http://localhost:6333",
        embed_model="openai/text-embedding-3-small",
        embed_model_type="openrouter",
        embed_api_key="your-api-key",
    ),
)
```

### Custom LLM Configuration

```python
from mmry import MemoryConfig, LLMConfig

config = MemoryConfig(
    llm_config=LLMConfig(
        api_key="your-key",
        model="anthropic/claude-3-sonnet",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        timeout=60,
    ),
)
```

---

## Directory Structure

```
mmry/
├── client.py              # MemoryClient API
├── memory_manager.py      # Core logic
├── config.py              # Configuration classes
├── factory.py             # Factory patterns for LLM/VectorDB/Embeddings
├── errors.py              # Custom exceptions
├── llms/
│   ├── openrouter_base.py         # LLM base class
│   ├── openrouter_summariser.py   # Summarization
│   ├── openrouter_merger.py       # Memory merging
│   └── openrouter_context_builder.py  # Context building
├── vector_store/
│   └── qdrant.py          # Qdrant implementation
├── embedding/
│   ├── embedding_base.py  # Embedding interface
│   ├── local_embedding.py # Local sentence-transformers
│   └── openrouter_embedding.py  # OpenRouter embeddings
├── base/
│   ├── llm_base.py        # LLM base classes
│   └── vectordb_base.py   # VectorDB interface
└── utils/
    ├── text.py            # Text utilities
    ├── decay.py           # Memory decay scoring
    ├── scoring.py         # Reranking
    ├── health.py          # Health metrics
    └── datetime.py        # Datetime utilities
```

---

## Requirements

- Python 3.13+
- Qdrant (local or remote)
- OpenRouter API key (for LLM features)
