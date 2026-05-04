"""
Account tools — exposed to AccountAgent.

The wiring is what's evaluated, not the data source. We seed deterministic
mock data per user_id so the same query produces the same answer across runs.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

_PIPELINES = ["build-and-test", "deploy-staging", "deploy-prod", "lint-and-fmt", "e2e-suite"]
_BRANCHES = ["main", "develop", "feat/auth", "release/v1.2.3", "hotfix/db-pool"]
_STATUSES = ["passed", "passed", "failed", "passed", "cancelled"]


def _seed(user_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(user_id.encode(), digest_size=4).digest(), "big")


async def get_recent_builds(user_id: str, limit: int = 5) -> dict[str, Any]:
    """Return the most recent builds for a user, newest first.

    Args:
        user_id: Helix user identifier.
        limit: max number of builds to return (1-20).
    """
    seed = _seed(user_id)
    limit = max(1, min(20, limit))
    now = datetime.now(UTC)

    builds: list[dict[str, Any]] = []
    for i in range(limit):
        idx = (seed + i) % 5
        builds.append({
            "build_id": f"bld_{user_id[:6]}_{1000 + i}",
            "pipeline": _PIPELINES[idx],
            "branch": _BRANCHES[(seed + i) % len(_BRANCHES)],
            "status": _STATUSES[idx],
            "started_at": (now - timedelta(hours=2 * i + 1)).isoformat(),
            "duration_seconds": 60 + ((seed + i * 7) % 240),
        })
    return {"user_id": user_id, "count": len(builds), "builds": builds}


async def get_account_status(user_id: str, plan_tier: str = "free") -> dict[str, Any]:
    """Return current account status (plan, concurrency limits, storage usage).

    Args:
        user_id: Helix user identifier.
        plan_tier: caller-supplied plan tier (`free`, `pro`, `enterprise`).
            Used to size the response's limits when the user record is mocked.
    """
    limits = {
        "free":       (2,  10.0),
        "pro":        (8,  100.0),
        "enterprise": (32, 1000.0),
    }
    concurrent_limit, storage_limit = limits.get(plan_tier, limits["free"])
    seed = _seed(user_id)
    return {
        "user_id": user_id,
        "plan_tier": plan_tier,
        "concurrent_builds_used": seed % max(1, concurrent_limit),
        "concurrent_builds_limit": concurrent_limit,
        "storage_used_gb": round((seed % 1000) / 100.0, 2),
        "storage_limit_gb": storage_limit,
    }
