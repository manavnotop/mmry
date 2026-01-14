# mmry

A memory management layer for AI agents and applications. Store, retrieve, and manage conversational memories using vector similarity search.

## What is mmry?

mmry (pronounced "memory") provides a layer of persistent memory for AI applications. Similar to mem0 or supermemory, it helps AI agents remember context across conversations without manual prompt engineering.

## Key Features

- **Multi-user support** - Isolated memories per user with `user_id` filtering
- **Semantic search** - Find relevant memories using vector embeddings
- **Automatic deduplication** - Similar memories are merged intelligently
- **Memory versioning** - Track changes over time with history
- **Batch operations** - Efficient bulk memory creation
- **Configurable similarity threshold** - Control when memories merge
- **Flexible LLM integration** - Use OpenRouter or local models
- **Async support** - Non-blocking operations with httpx

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        mmry Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐                                             │
│   │  MemoryClient │  ← User-facing API                          │
│   └───────┬───────┘                                             │
│           │                                                     │
│   ┌───────▼───────┐                                             │
│   │ MemoryManager │  ← Core orchestration                       │
│   └───────┬───────┘                                             │
│           │                                                     │
│   ┌───────▼───────┐    ┌──────────────────┐                     │
│   │   VectorDB    │    │  LLM Components  │                     │
│   │   (Qdrant)    │◄───│ Summarizer       │                     │
│   │               │    │ Merger           │                     │
│   │  - Store      │    │ ContextBuilder   │                     │
│   │  - Search     │    └──────────────────┘                     │
│   │  - Retrieve   │                                             │
│   └───────────────┘                                             │
│                                                                 │
│   ┌───────────────┐    ┌──────────────────┐                     │
│   │  Embeddings   │───►│ Sentence-BERT    │  ← Local models     │
│   └───────────────┘    │ (default)        │                     │
│                        │ or OpenRouter    │                     │
│                        └──────────────────┘                     │
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

## How Memory Creation Works

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

## How Memory Search Works

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

## Quick Start

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

## Configuration

### VectorDB Config

```python
from mmry import MemoryConfig, VectorDBConfig

config = MemoryConfig(
    vector_db_config=VectorDBConfig(
        url="http://localhost:6333",           # Qdrant URL
        collection_name="mmry",                # Collection name
        embed_model="all-MiniLM-L6-v2",        # Embedding model
        embed_model_type="local",              # "local" or "openrouter"
    ),
    similarity_threshold=0.8,                  # Merge threshold (0.0-1.0)
)
```

### LLM Config

```python
from mmry import MemoryConfig, LLMConfig

config = MemoryConfig(
    llm_config=LLMConfig(
        api_key="your-openrouter-api-key",
        model="openai/gpt-4o",                # LLM model
        base_url="https://openrouter.ai/api/v1/chat/completions",
        timeout=30,                            # Request timeout
    ),
)
```

## API Reference

### MemoryClient

| Method | Description |
|--------|-------------|
| `create_memory(text, metadata, user_id)` | Create a memory from text or conversation |
| `query_memory(query, top_k, user_id)` | Search memories semantically |
| `update_memory(memory_id, new_text, user_id)` | Update an existing memory |
| `delete_memory(memory_id, user_id)` | Delete a memory by ID |
| `list_all(user_id)` | List all memories |
| `get_health(user_id)` | Get system health metrics |
| `create_memory_batch(texts, metadatas, user_ids)` | Create multiple memories |

## Directory Structure

```
mmry/
├── client.py              # MemoryClient API
├── memory_manager.py      # Core logic
├── config.py              # Configuration classes
├── factory.py             # Factory patterns for LLM/VectorDB
├── errors.py              # Custom exceptions
├── llms/
│   ├── openrouter_base.py         # LLM base class
│   ├── openrouter_summariser.py   # Summarization
│   ├── openrouter_merger.py       # Memory merging
│   └── openrouter_context_builder.py  # Context building
├── vector_store/
│   └── qdrant.py          # Qdrant implementation
└── utils/
    ├── text.py            # Text utilities
    ├── decay.py           # Memory decay scoring
    ├── scoring.py         # Reranking
    └── health.py          # Health metrics
```

## Running Tests

```bash
make test  # Starts Qdrant and runs pytest
```

## Requirements

- Python 3.13+
- Qdrant (local or remote)
- OpenRouter API key (for LLM features)
