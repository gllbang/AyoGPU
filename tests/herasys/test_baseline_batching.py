import asyncio
from dataclasses import replace

from HeraSys.dags.node_commons import NodeOps
from HeraSys.engines.engine_types import EngineType
from HeraSys.backends import reference_pools
from HeraSys.workflows import primitive
from benchmarks.baselines import FIFOEngines


def test_fifo_batches_compatible_arrivals_across_other_classes_without_loss():
    async def scenario():
        pools = reference_pools()
        pool = pools[EngineType.AGGREGATOR]
        pools = {EngineType.AGGREGATOR: replace(pool, workers=pool.workers[:1])}
        events = []
        dispatcher = FIFOEngines(pools, events, batch_wait_s=0.02)
        a = primitive(
            "a",
            NodeOps.AGGREGATOR,
            EngineType.AGGREGATOR,
            ["input"],
            ["output"],
            mode="identity",
        )
        b = primitive(
            "b",
            NodeOps.AGGREGATOR,
            EngineType.AGGREGATOR,
            ["input"],
            ["output"],
            mode="indexes",
        )
        try:
            first = asyncio.create_task(dispatcher.execute("a0", a, {"input": [0]}))
            await asyncio.sleep(0.001)
            second = asyncio.create_task(dispatcher.execute("b0", b, {"input": [1]}))
            third = asyncio.create_task(dispatcher.execute("a1", a, {"input": [2]}))
            results = await asyncio.wait_for(asyncio.gather(first, second, third), 1)
            assert [r["output"]["output"] for r in results] == [[0], [1], [2]]
            assert [e["batch_size"] for e in events] == [2, 1]
        finally:
            await dispatcher.close()

    asyncio.run(scenario())
