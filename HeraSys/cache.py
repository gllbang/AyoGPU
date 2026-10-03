from typing import Dict, Any
import asyncio
import ray

@ray.remote
class GlobalCacheActor:

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = value

    async def get(self, key: str) -> Any:
        async with self._lock:
            return self._cache.get(key)

    async def get_all(self) -> Dict[str, Any]:
        async with self._lock:
            return self._cache.copy()

    async def clear(self):
        async with self._lock:
            self._cache.clear()