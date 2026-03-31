"""
client.py — ImmigrationEnv client for the Airport Immigration Processing Environment.

Provides a clean HTTP wrapper around the environment's REST API,
following the OpenEnv EnvClient pattern (sync + async).

Usage (sync):
    from client import ImmigrationEnv
    with ImmigrationEnv(base_url="http://localhost:7860") as env:
        result = env.reset(task_id="task1_document_check", seed=42)
        obs = result["observation"]
        step_result = env.step({
            "action_type": "clear",
            "passenger_id": obs["current_passenger"]["passenger_id"],
            "reason": "All documents valid."
        })
        print(step_result["reward"])

Usage (async):
    import asyncio
    from client import ImmigrationEnvAsync

    async def main():
        async with ImmigrationEnvAsync(base_url="http://localhost:7860") as env:
            result = await env.reset(task_id="task1_document_check")
            obs = result["observation"]
            step = await env.step({
                "action_type": "clear",
                "passenger_id": obs["current_passenger"]["passenger_id"],
                "reason": "All valid."
            })
            print(step["reward"])

    asyncio.run(main())
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# ─── Sync client ──────────────────────────────────────────────────────────────

class ImmigrationEnv:
    """Synchronous HTTP client for the Airport Immigration Processing Environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = None

    def __enter__(self):
        import requests
        self._session = requests.Session()
        return self

    def __exit__(self, *args):
        if self._session:
            self._session.close()
            self._session = None

    def _get(self, endpoint: str) -> Dict[str, Any]:
        import requests
        session = self._session or requests.Session()
        resp = session.get(f"{self.base_url}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
        import requests
        session = self._session or requests.Session()
        resp = session.post(f"{self.base_url}{endpoint}", json=payload or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check environment health."""
        return self._get("/health")

    def tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        return self._get("/tasks")

    def reset(
        self,
        task_id: str = "task1_document_check",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of task1_document_check, task2_flag_detection,
                     task3_queue_pressure, task4_adversarial
            seed: Random seed for reproducibility

        Returns:
            ResetResult dict with keys: observation, episode_id, task_id
        """
        return self._post("/reset", {"task_id": task_id, "seed": seed})

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take a step in the environment.

        Args:
            action: Dict with keys:
                action_type: "clear" | "hold" | "deny" | "escalate" | "request_document"
                passenger_id: ID of the current passenger
                reason: Brief justification string
                document_requested: (optional) doc name if action_type="request_document"

        Returns:
            StepResult dict with keys: observation, reward, done, info
        """
        return self._post("/step", {"action": action})

    def state(self) -> Dict[str, Any]:
        """Get the full current episode state."""
        return self._get("/state")

    def grade(self) -> Dict[str, Any]:
        """Grade the current episode. Returns score in [0.0, 1.0]."""
        return self._post("/grade")


# ─── Async client ─────────────────────────────────────────────────────────────

class ImmigrationEnvAsync:
    """Asynchronous HTTP client for the Airport Immigration Processing Environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = None

    async def __aenter__(self):
        try:
            import httpx
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        except ImportError:
            raise ImportError("httpx required for async client: pip install httpx")
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get(self, endpoint: str) -> Dict[str, Any]:
        resp = await self._client.get(endpoint)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
        resp = await self._client.post(endpoint, json=payload or {})
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> Dict[str, Any]:
        return await self._get("/health")

    async def tasks(self) -> Dict[str, Any]:
        return await self._get("/tasks")

    async def reset(
        self,
        task_id: str = "task1_document_check",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self._post("/reset", {"task_id": task_id, "seed": seed})

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return await self._post("/step", {"action": action})

    async def state(self) -> Dict[str, Any]:
        return await self._get("/state")

    async def grade(self) -> Dict[str, Any]:
        return await self._post("/grade")
