import pytest
import requests

from HeraSys.backends import RayWorker
from HeraSys.gpu import GPUWorker


def test_google_errors_do_not_write_credentials_to_trace(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-must-not-leak")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-engine")

    def fail(*args, **kwargs):
        raise requests.RequestException("URL contains key=test-key-must-not-leak")

    monkeypatch.setattr(requests, "get", fail)
    worker = GPUWorker("tool")
    request = {
        "op": "web_search",
        "inputs": {"search_query": "test"},
        "config": {"provider": "google", "top_k": 5},
        "outputs": ["result"],
    }
    with pytest.raises(RuntimeError) as error:
        worker.execute_batch([request])
    assert "test-key-must-not-leak" not in str(error.value)
    assert "Google Search request failed" in str(error.value)


def test_ray_release_sends_only_kv_handles():
    import asyncio

    class Actor:
        calls = []

        class release:
            @staticmethod
            async def remote(outputs):
                Actor.calls.append(outputs)

    async def scenario():
        worker = RayWorker(Actor())
        await worker.release([{"vectors": [[1, 2, 3]], "answer": "text"}])
        assert not Actor.calls
        await worker.release([{"vectors": [[1, 2, 3]], "state": {"state_key": "kv-1"}}])
        assert Actor.calls == [[{"state": {"state_key": "kv-1"}}]]

    asyncio.run(scenario())


def test_deferred_kv_release_is_piggybacked_and_finally_flushed():
    import asyncio

    class Actor:
        batches = []
        releases = []

        class execute_batch:
            @staticmethod
            async def remote(requests, *, releases=None):
                Actor.batches.append((requests, releases))
                return ["result"]

        class release:
            @staticmethod
            async def remote(outputs):
                Actor.releases.append(outputs)

    async def scenario():
        worker = RayWorker(Actor(), defer_releases=True)
        first, last = (
            {"state": {"state_key": "first"}},
            {"state": {"state_key": "last"}},
        )
        await worker.release([first])
        assert not Actor.releases
        assert await worker.execute_batch(["next request"]) == ["result"]
        assert Actor.batches == [(["next request"], [first])]
        await worker.release([last])
        await worker.flush_releases()
        assert Actor.releases == [[last]]
        assert not worker.pending_releases

    asyncio.run(scenario())
