import datetime
import math
from typing import Any, Dict, List


def hybrid_score(
    similarity: float,
    created_at: str | datetime.datetime,
    importance: float = 1.0,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
    decay_rate: float = 0.01,
) -> float:
    """
    Compute hybrid score combining semantic similarity, recency, and importance.
    Higher is better.
    """
    if isinstance(created_at, str):
        created_at = datetime.datetime.fromisoformat(created_at)

    now = datetime.datetime.now()
    delta_hours = (now - created_at).total_seconds / 3600
    recency_weight = math.exp(-decay_rate * delta_hours)

    final_score = alpha * similarity + beta * recency_weight + gamma * importance
    return round(final_score, 6)


def rerank_results(
    results: List[Dict[str, Any]],
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> list[Dict[str, Any]]:
    """
    Rerank a list of Qdrant search results by hybrid score.
    Each result must have: score (similarity), payload.created_at (ISO string)
    """
    for r in results:
        created_at = r["payload"].get("created_at")
        importance = float(r["payload"].get("importance", 1.0))
        sim = float(r["score"])
        r["final_score"] = hybrid_score(sim, created_at, importance, alpha, beta, gamma)
    return sorted(results, key=lambda x: x["final_score"], reverse=True)
