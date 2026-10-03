"""Regressions against the explicitly patched isolated upstream checkout."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_batching_and_completed_status_survive_until_app_observation():
    checkout = Path(__file__).resolve().parents[2] / "baselines/Ayo"
    if not checkout.exists():
        pytest.skip("Run scripts/setup_ayo_full.sh to check the native Ayo patch")
    script = """
import asyncio
from types import SimpleNamespace
from Ayo.engines.embedder import collect_fifo
from Ayo.schedulers.graph_scheduler import GraphScheduler
from Ayo.queries.query_state import QueryStatus

async def scenario():
    engine = SimpleNamespace(request_queue=asyncio.Queue(), max_batch_size=4,
                             benchmark_batch_wait_s=.001)
    for i, size in enumerate([2, 1, 3, 1, 8, 1]):
        engine.request_queue.put_nowait(SimpleNamespace(id=i, size=size))
    batches = []
    while not engine.request_queue.empty() or getattr(engine, "benchmark_pending", None):
        batches.append([r.id for r in await collect_fifo(engine, lambda r: r.size)])
    assert batches == [[0, 1], [2, 3], [4], [5]], batches

    cls = GraphScheduler.__ray_metadata__.modified_class
    scheduler = cls.__new__(cls)
    async def cleanup():
        pass
    scheduler.query_runners = {"q": SimpleNamespace(
        query=SimpleNamespace(status=QueryStatus.COMPLETED), cleanup_runtime_context=cleanup)}
    await scheduler.cleanup_query("q")
    assert await scheduler.get_query_status("q") == QueryStatus.COMPLETED
    await scheduler.forget_query("q")
    assert not scheduler.query_runners
asyncio.run(scenario())
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout)},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
