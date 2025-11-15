# Configuration for mmry evaluation pipeline
import os

# Default configuration values
DEFAULT_CONFIG = {
    # Data paths
    "data_path": "dataset/locomo10_rag.json",
    "output_path": "results/mmry_results.json",
    # API settings
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "model": os.getenv("MODEL", "openai/gpt-3.5-turbo"),
    # Memory settings
    "top_k": 3,
    "similarity_threshold": 0.7,
    "user_id": "mmry_eval",
    # Evaluation settings
    "max_workers": 10,
    # Vector database settings
    "qdrant_url": "http://localhost:6333",
    "collection_name": "mmry_evaluation",
    "embed_model": "all-MiniLM-L6-v2",
    "embed_model_type": "local",  # Can be "local" or "openrouter"
    "embed_api_key": os.getenv("OPENROUTER_API_KEY", None),
}
