# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mmry is a memory management system for AI agents that stores, retrieves, and manages conversational memories using vector similarity search (Qdrant). Key features:
- Multi-user support with `user_id` filtering
- Memory deduplication via similarity merging
- Memory decay scoring for relevance
- Dual embedding support (local sentence-transformers or OpenRouter API)

## Common Commands

```bash
# Code formatting and import sorting
make fix

# Run tests (starts Docker Qdrant container)
make test

# Start FastAPI server
uvicorn main:app
```

## Development

- **Package manager**: Use `uv run <command>` for all Python operations
- **Python**: 3.13+
- **Environment**: Set `OPENROUTER_API_KEY` in `.env` for LLM features

## Architecture

```
MemoryClient (mmry/client.py)
    └── MemoryManager (mmry/memory_manager.py)
        ├── create_memory() → Summarizer + Merger → Qdrant
        ├── query_memory() → VectorDB search + ContextBuilder + Reranking
        └── update_memory() / list_all()

Factory Pattern (mmry/factory.py):
- LLM providers: LLMFactory.create() - registry pattern with @register_llm
- Embeddings: EmbeddingFactory.create() - registry pattern with @register_embedding
- VectorDB: VectorDBFactory.create() - registry pattern with @register_vectordb
```

## Key Files

- [mmry/client.py](mmry/client.py) - User-facing API with MemoryConfig or dict
- [mmry/memory_manager.py](mmry/memory_manager.py) - Core memory orchestration
- [mmry/factory.py](mmry/factory.py) - Unified registry-based factory for LLM/VectorDB/Embeddings
- [mmry/llms/openrouter_base.py](mmry/llms/openrouter_base.py) - Shared base for LLM implementations
- [mmry/utils/text.py](mmry/utils/text.py) - Text utilities (clean_summary)
- [mmry/utils/datetime.py](mmry/utils/datetime.py) - Datetime utilities (parse_datetime)
