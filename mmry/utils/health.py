import datetime
import statistics
from typing import Any, Dict, List


class MemoryHealth:
    """Compute health metrics for a collection of memory entries."""

    def __init__(self, memories: List[Dict[str, Any]]):
        self.memories = memories

    def count(self) -> int:
        return len(self.memories)

    def average_age_hours(self) -> float:
        now = datetime.datetime.now(datetime.timezone.utc)
        ages = []
        for m in self.memories:
            t = m["payload"].get("created_at")
            if not t:
                continue

            if isinstance(t, str):
                dt = datetime.datetime.fromisoformat(t)
            elif isinstance(t, datetime.datetime):
                dt = t
            else:
                continue

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)

            ages.append((now - dt).total_seconds() / 3600)
        return round(statistics.mean(ages), 3) if ages else 0.0

    def redundancy_score(self) -> float:
        """Rough estimate: std-dev of similarity scores (smaller ⇒ more redundancy)."""
        scores = [m.get("score", 0) for m in self.memories if "score" in m]
        if len(scores) < 2:
            return 0.0
        return round(statistics.stdev(scores), 4)

    def importance_distribution(self) -> Dict[str, float]:
        vals = [float(m["payload"].get("importance", 1.0)) for m in self.memories]
        if not vals:
            return {}
        return {
            "min": min(vals),
            "max": max(vals),
            "avg": round(statistics.mean(vals), 3),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "memory_count": self.count(),
            "avg_age_hours": self.average_age_hours(),
            "redundancy_score": self.redundancy_score(),
            "importance_distribution": self.importance_distribution(),
        }
