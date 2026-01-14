import datetime
import math
from typing import Any, Dict

from mmry.utils.datetime import parse_datetime


def compute_decay_factor(
    created_at: str | datetime.datetime, decay_rate: float = 0.01
) -> float:
    """
    Returns a decay multiplier (0–1) based on how old a memory is.
    New memories ≈ 1.0, old ones decay toward 0.
    """
    if not isinstance(created_at, (str, datetime.datetime)):
        return 1.0  # Default to no decay if invalid type

    created = parse_datetime(created_at)
    now = datetime.datetime.now(datetime.timezone.utc)
    delta_hours = (now - created).total_seconds() / 3600
    return math.exp(-decay_rate * delta_hours)


def apply_memory_decay(
    memory: Dict[str, Any], decay_rate: float = 0.01
) -> Dict[str, Any]:
    """Apply decay weighting to a memory entry."""
    created_at = memory["payload"].get("created_at")
    if created_at is None:
        return memory
    decay_factor = compute_decay_factor(created_at, decay_rate)
    memory["decayed_score"] = memory["score"] * decay_factor
    return memory
