"""Single Ray coordinator for clients sharing one global HeraSys scheduler."""

import ray

from .config import HeraConfig
from .runtime import HeraSys


@ray.remote(num_cpus=1, max_concurrency=1024)
class HeraSysActor:
    def __init__(self, pools, config=None, profiles=None):
        self.runtime = HeraSys(pools, config or HeraConfig(), profiles=profiles)

    async def execute(
        self, dag, inputs=None, *, query_id=None, namespace="default", timeout=300
    ):
        return await self.runtime.submit(
            dag, inputs, query_id=query_id, namespace=namespace, timeout=timeout
        )

    async def execute_factory(
        self, factory, *, query_id=None, namespace="default", timeout=300
    ):
        return await self.runtime.submit_factory(
            factory, query_id=query_id, namespace=namespace, timeout=timeout
        )

    def metrics(self):
        return {
            "active_queries": len(self.runtime.queries),
            "load": self.runtime.load,
            "physical_tasks": len(self.runtime.tasks),
            "cache_entries": len(self.runtime.cache),
            "queries": self.runtime.query_records,
            "events": self.runtime.events,
        }

    async def shutdown(self):
        await self.runtime.close()
