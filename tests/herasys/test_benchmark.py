import pytest

from benchmarks.herasys import (
    FIXTURE,
    mixed_workload,
    percentile,
    samples,
    warmup_batches,
    workload_samples,
)


def test_percentiles_and_reproducible_duplicate_schedule():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([4, 1, 2, 3], 0.99) == pytest.approx(3.97)
    assert percentile([], 0.99) is None
    requests = samples(FIXTURE, 100, 42, 0.1)
    assert requests == samples(FIXTURE, 100, 42, 0.1)
    assert len({request["question"] for request in requests}) == 90
    assert requests != samples(FIXTURE, 100, 43, 0.1)


def test_framework_outputs_match_reference(tmp_path):
    # The optional framework environment has Pydantic 2 and the real libraries.
    import pydantic

    if int(pydantic.__version__.split(".")[0]) < 2:
        pytest.skip("Run this integration check in .venv-herasys-baselines")
    pytest.importorskip("langgraph")
    pytest.importorskip("llama_index.core.workflow")
    import asyncio

    from HeraSys import HeraSys
    from HeraSys.backends import reference_pools
    from HeraSys.workflows import rag
    from benchmarks.baselines import FrameworkRuntime

    async def scenario():
        dag, inputs = rag(FIXTURE[0]["question"], FIXTURE[0]["documents"], max_tokens=8)
        results = []
        for framework in [None, "langgraph", "llamaindex"]:
            pools = reference_pools()
            runtime = (
                FrameworkRuntime(pools, framework) if framework else HeraSys(pools)
            )
            async with runtime:
                results.append(await runtime.submit(dag, inputs, query_id="q0"))
        assert results[0] == results[1] == results[2]

    asyncio.run(scenario())


def test_mixed_duplicates_stay_within_workflow_and_preserve_mix():
    from collections import Counter

    records = [
        {"question": f"rag {i}", "documents": [f"document {i}"]} for i in range(40)
    ]
    web = [{"question": f"web {i}"} for i in range(40)]
    items = workload_samples(records, web, 40, 42, 0.1, "mixed")
    assert items == workload_samples(records, web, 40, 42, 0.1, "mixed")
    unique = {}
    counts = Counter()
    for index, item in enumerate(items):
        kind = mixed_workload(index)
        counts[kind] += 1
        unique.setdefault(kind, set()).add(item["question"])
        assert item["question"].startswith("web" if kind == "web_search" else "rag")
        # Keep the original global-position corpus selection for nonduplicates;
        # do not introduce artificial overlap by restarting each stratum at row 0.
        assert mixed_workload(int(item["question"].split()[1])) == kind
    assert counts == {"rag_rerank": 20, "rag": 10, "web_search": 10}
    assert {kind: len(values) for kind, values in unique.items()} == {
        "rag_rerank": 18,
        "rag": 9,
        "web_search": 9,
    }


def test_mixed_warmup_covers_all_workflows_and_uses_web_corpus():
    from argparse import Namespace
    from collections import Counter

    args = Namespace(
        workloads=["mixed", "rag"],
        warmup=2,
        warmup_concurrency=8,
        profile_batch_sizes=[1, 4],
    )
    records = [{"question": f"rag {i}", "documents": [str(i)]} for i in range(8)]
    web = [{"question": f"web {i}"} for i in range(8)]
    batches = list(warmup_batches(args, records, web))
    assert Counter(kind for kind, _ in batches) == {
        "rag": 4,
        "rag_rerank": 4,
        "web_search": 4,
    }
    for kind, batch in batches:
        assert all(
            item["question"].startswith("web" if kind == "web_search" else "rag")
            for item in batch
        )
        assert len({item["question"] for item in batch}) == len(batch)


def test_generation_profiles_separate_rewrite_and_answer_and_load_legacy():
    from HeraSys.profiling import ProfileLibrary
    from HeraSys.workflows import web_search

    dag, _ = web_search("Question", max_tokens=128)
    rewrite = dag.get_node("rewrite_prefill")
    answer = dag.get_node("answer_prefill")
    profiles = ProfileLibrary()
    profiles.observe(rewrite, 0.1)
    profiles.observe(answer, 1.5)
    profiles.freeze()
    assert profiles.predict(rewrite) == pytest.approx(0.1)
    assert profiles.predict(answer) == pytest.approx(1.5)
    legacy = ProfileLibrary({profiles.legacy_key(answer): 2})
    assert legacy.predict(answer) == pytest.approx(answer.relative_ms / 1000 * 2)


def test_profile_collection_excludes_cold_pass_and_preserves_loaded_profiles():
    import asyncio
    from argparse import Namespace

    from HeraSys.backends import ReferenceWorker, reference_pools
    from HeraSys.profiling import ProfileLibrary
    from benchmarks.herasys import prepare_profiles

    class ColdFirstWorker(ReferenceWorker):
        async def execute_batch(self, requests):
            result = await super().execute_batch(requests)
            # Synthetic cold latency; this is a calibration regression check,
            # not a simulation used to claim GPU performance.
            for row in result:
                row["service_s"] = 1000 if len(self.calls) == 1 else 0.01
            return result

    async def scenario():
        pools = reference_pools()
        for pool in pools.values():
            pool.workers = [ColdFirstWorker()]
        args = Namespace(
            workloads=["rag"], warmup=1, warmup_concurrency=1,
            profile_batch_sizes=[1], profile_rounds=3,
            max_tokens=4, decode_quantum=4, pipeline_mode="dynamic",
            embedding_model="test-embed", reranker_model="test-rerank",
            llm_model="test-llm", timeout=3,
        )
        profiles = ProfileLibrary()
        coverage = await prepare_profiles(
            pools, args, FIXTURE, None, profiles, collect=True
        )
        assert [row["samples_collected"] for row in coverage] == [False, True, True, True]
        assert profiles.samples
        assert max(value for values in profiles.samples.values() for value in values) < 100
        before = {key: list(values) for key, values in profiles.samples.items()}
        coverage = await prepare_profiles(
            pools, args, FIXTURE, None, profiles, collect=False
        )
        assert [row["samples_collected"] for row in coverage] == [False]
        assert dict(profiles.samples) == before

    asyncio.run(scenario())


def test_fixture_prediction_uses_short_tool_result_without_changing_generation():
    from HeraSys.workflows import web_search

    fixture, _ = web_search("What is the capital of France?", max_tokens=128)
    google, _ = web_search(
        "What is the capital of France?", max_tokens=128, search_provider="google"
    )
    short = fixture.get_node("answer_prefill")
    long = google.get_node("answer_prefill")
    assert short.relative_ms < 5  # Fewer than 100 expected prompt tokens.
    assert long.relative_ms > 30  # Real snippets still use the long estimate.
    assert short.config == long.config  # Same prompt template and generation budget.


@pytest.mark.parametrize("mode", ["closed_loop", "waves"])
def test_arrival_modes_distinguish_refill_from_drained_waves(mode):
    import asyncio

    from benchmarks.herasys import drive_requests

    async def scenario():
        semaphore = asyncio.Semaphore(2)
        release_zero, one_done, two_started = (asyncio.Event() for _ in range(3))
        completed = []

        async def request(index, value):
            async with semaphore:
                assert value == index
                if index == 0:
                    await release_zero.wait()
                elif index == 1:
                    one_done.set()
                elif index == 2:
                    two_started.set()
                completed.append(index)

        task = asyncio.create_task(drive_requests(request, list(range(5)), 2, mode))
        await asyncio.wait_for(one_done.wait(), 1)
        if mode == "closed_loop":
            await asyncio.wait_for(two_started.wait(), 1)
        else:
            assert not two_started.is_set()
        release_zero.set()
        await asyncio.wait_for(task, 1)
        assert sorted(completed) == list(range(5))

    asyncio.run(scenario())
