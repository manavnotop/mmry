import datetime
import math
from typing import Any, Dict


def compute_decay_factor(created_at: str, decay_rate: float = 0.01) -> float:
    """
    Returns a decay multiplier (0–1) based on how old a memory is.
    New memories ≈ 1.0, old ones decay toward 0.
    """
    created = datetime.datetime.fromisoformat(created_at)
    now = datetime.datetime.utcnow()
    delta_hours = (now - created).total_seconds() / 3600
    return math.exp(-decay_rate * delta_hours)


def apply_memory_decay(
    memory: Dict[str, Any], decay_rate: float = 0.01
) -> Dict[str, Any]:
    """Apply decay weighting to a memory entry."""
    if "created_at" not in memory["payload"]:
        return memory
    decay_factor = compute_decay_factor(memory["payload"]["created_at"], decay_rate)
    memory["decayed_score"] = memory["score"] * decay_factor
    return memory
