import asyncio
from dataclasses import replace

import pytest

from HeraSys.dags.dag import DAG
from HeraSys.dags.node_commons import NodeOps
from HeraSys.engines.engine_types import EngineType
from HeraSys import HeraConfig, HeraSys, ResourcePool
from HeraSys.backends import ReferenceWorker, reference_pools
from HeraSys.policy import long_queries, priority, remaining, split_factor
from HeraSys.semantics import digest
from HeraSys.workflows import primitive, rag, web_search

DOCS = [
    "HeraSys shares workflow computations.",
    "Ayo decomposes workflows.",
    "Paris is the capital of France.",
    "Vectors represent document embeddings.",
]


def run(coro):
    return asyncio.run(coro)


def test_paper_equations():
    config = HeraConfig()
    assert priority(2, 3, True, config) == pytest.approx(1 / (2 + 1e-6) + 0.9 + 0.5)
    assert remaining({"a": 2, "b": 3}, {"a"}, 4, config) == pytest.approx(6, rel=1e-5)
    assert remaining({"a": 2, "b": 3}, set(), 0, config) == 5
    assert remaining({"a": 2, "b": 3}, {"a", "b"}, 4, config) == 0
    assert long_queries({"a": 1, "b": 1, "c": 1}, 1) == set()
    assert long_queries({"a": 1, "b": 1, "c": 10}, 1) == {"c"}
    assert split_factor(0, 10, config) == 4
    assert split_factor(1, 10, config) == 1
    assert digest((1, 2)) != digest({"__tuple__": [1, 2]})
    assert digest([1, 2]) != digest((1, 2))
    with pytest.raises(ValueError):
        HeraConfig(gamma=float("nan"))


def test_contribution_counts_enabled_branches_not_every_serial_descendant():
    async def scenario(scope):
        a = primitive(
            "a", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["x"], ["a_out"], ms=1
        )
        b = primitive(
            "b", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["a_out"], ["b_out"], ms=2
        )
        c = primitive(
            "c", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["a_out"], ["c_out"], ms=3
        )
        d = primitive(
            "d", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["b_out"], ["d_out"], ms=5
        )
        a >> b
        a >> c
        b >> d
        dag = DAG()
        dag.register_nodes(a, b, c, d)
        config = replace(HeraConfig(), fusion=False)
        if scope is not None:
            config = replace(config, contribution_scope=scope)
        async with HeraSys(reference_pools(), config) as runtime:
            future = runtime.submit(dag, {"x": "value"}, query_id="q")
            state = runtime.queries["q"]
            phi = runtime.tasks[state.mapping["a"]].owners[("q", "a")]
            assert await future == {"c_out": "value", "d_out": "value"}
        return phi

    # A enables two branches; D is a serial successor of B, not a third branch
    # directly enabled by A. Reverse depth already accounts for B's path to D.
    assert run(scenario(None)) == pytest.approx(0.002 + 0.003 + 0.1 + 2 * 0.1)
    assert run(scenario("direct")) == pytest.approx(0.002 + 0.003 + 0.1 + 2 * 0.1)
    assert run(scenario("transitive")) == pytest.approx(
        0.002 + 0.003 + 0.005 + 0.1 + 3 * 0.1
    )


def test_fusion_preserves_results_and_removes_dispatch():
    async def scenario(fusion):
        pools = reference_pools()
        async with HeraSys(pools, replace(HeraConfig(), fusion=fusion)) as runtime:
            dag, inputs = rag(
                "What is HeraSys?", DOCS, splits=2, max_tokens=8, quantum=4
            )
            futures = [runtime.submit(dag, inputs, query_id=f"q{i}") for i in range(5)]
            results = await asyncio.gather(*futures)
        executed = sum(
            len(call)
            for pool in pools.values()
            for worker in pool.workers
            for call in worker.calls
        )
        return results, executed, runtime.events

    baseline, baseline_count, _ = run(scenario(False))
    fused, fused_count, events = run(scenario(True))
    assert fused == baseline
    assert fused_count < baseline_count
    assert any(e["event"] == "fusion" for e in events)
    dispatch = [e for e in events if e["event"] == "dispatch"]
    assert any(max(e["contributions"]) > 1 for e in dispatch)


def test_signatures_ignore_names_but_respect_inputs_parameters_and_tenants():
    async def scenario():
        pools = reference_pools()
        async with HeraSys(pools) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, with_llm=False, splits=1)
            futures = [runtime.submit(dag, inputs, namespace="tenant-a")]
            renamed, _ = rag("HeraSys?", DOCS, with_llm=False, splits=1)
            for node in renamed.nodes:
                node.name = "renamed_" + node.name
            futures.append(runtime.submit(renamed, inputs, namespace="tenant-a"))
            futures.append(runtime.submit(dag, inputs, namespace="tenant-b"))
            changed, _ = rag("HeraSys?", DOCS, with_llm=False, splits=1)
            changed.get_node("retrieve").config["top_k"] = 1
            futures.append(runtime.submit(changed, inputs, namespace="tenant-a"))
            results = await asyncio.gather(*futures)
            assert results[0] == results[1] == results[2]
            assert len(results[3]["reranked_contexts"]) == 1
        return runtime

    runtime = run(scenario())
    records = runtime.query_records
    assert records[1]["reused_nodes"] > 0 or any(r["reused_nodes"] > 0 for r in records)
    assert any(
        e["event"] == "fusion" and e["node"].startswith("renamed_")
        for e in runtime.events
    )


def test_cache_bound_and_expiration():
    async def scenario():
        config = replace(HeraConfig(), cache_entries=2, cache_ttl_s=0.01)
        async with HeraSys(reference_pools(), config) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, with_llm=False, splits=1)
            await runtime.submit(dag, inputs)
            await asyncio.sleep(0.02)
            prior = sum(e["event"] == "cache_hit" for e in runtime.events)
            await runtime.submit(dag, inputs)
            assert sum(e["event"] == "cache_hit" for e in runtime.events) == prior
            assert len(runtime.cache) <= 2
        assert not runtime.tasks and not runtime.queries and not runtime.signatures

    run(scenario())


def test_cached_result_isolated_from_caller_mutation():
    async def scenario():
        async with HeraSys(reference_pools()) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, with_llm=False)
            first = await runtime.submit(dag, inputs)
            first["reranked_contexts"].clear()
            second = await runtime.submit(dag, inputs)
            assert second["reranked_contexts"]
            assert any(e["event"] == "cache_hit" for e in runtime.events)

    run(scenario())


def test_node_audit_survives_fusion_and_cached_suffix_pruning():
    async def scenario():
        config = replace(HeraConfig(), audit_nodes=True, cache_entries=1)
        async with HeraSys(reference_pools(), config) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, with_llm=False, splits=1)
            first, shared = await asyncio.gather(
                runtime.submit(dag, inputs, query_id="first"),
                runtime.submit(dag, inputs, query_id="shared"),
            )
            cached = await runtime.submit(dag, inputs, query_id="cached")
        assert first == shared == cached
        audits = {row["query_id"]: row["nodes"] for row in runtime.node_audits}
        assert audits["first"]["rerank"]["output_sha256"] == digest(first)
        assert audits["first"]["rerank"] == audits["shared"]["rerank"]
        assert audits["cached"]["rerank"]["output_sha256"] == digest(cached)
        assert audits["cached"]["rerank"]["input_sha256"] is None

    run(scenario())


def test_cached_sink_prunes_evicted_ancestors():
    async def scenario():
        pools = reference_pools()
        async with HeraSys(pools, replace(HeraConfig(), cache_entries=1)) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, with_llm=False)
            first = await runtime.submit(dag, inputs)
            count = sum(len(w.calls) for p in pools.values() for w in p.workers)
            assert await runtime.submit(dag, inputs) == first
            assert sum(len(w.calls) for p in pools.values() for w in p.workers) == count
            assert runtime.query_records[-1]["pruned_nodes"] > 0

    run(scenario())


def test_timeout_and_backend_failure_do_not_poison_following_queries():
    class FailingWorker(ReferenceWorker):
        async def execute_batch(self, requests):
            if requests[0]["inputs"].get("value") == "fail":
                raise RuntimeError("injected backend failure")
            return await super().execute_batch(requests)

    async def scenario():
        node = primitive(
            "echo",
            NodeOps.AGGREGATOR,
            EngineType.AGGREGATOR,
            ["value"],
            ["result"],
            mode="identity",
        )
        dag = DAG()
        dag.register_nodes(node)
        pool = ResourcePool([FailingWorker(0.02)])
        async with HeraSys({EngineType.AGGREGATOR: pool}) as runtime:
            results = await asyncio.gather(
                runtime.submit(dag, {"value": "fail"}),
                runtime.submit(dag, {"value": "fail"}),
                return_exceptions=True,
            )
            assert all(isinstance(v, RuntimeError) for v in results)
            with pytest.raises(TimeoutError):
                await runtime.submit(dag, {"value": "slow"}, timeout=0.001)
            assert await runtime.submit(dag, {"value": "ok"}) == {"result": "ok"}

    run(scenario())


def test_pipeline_and_decode_splits_preserve_output():
    async def scenario():
        results = []
        for splits, quantum in [(1, 8), (2, 4), (4, 2)]:
            async with HeraSys(reference_pools()) as runtime:
                dag, inputs = rag(
                    "HeraSys?", DOCS, splits=splits, max_tokens=8, quantum=quantum
                )
                results.append(await runtime.submit(dag, inputs))
        assert results[0] == results[1] == results[2]
        async with HeraSys(reference_pools()) as runtime:
            dag, inputs = web_search("What is HeraSys?", max_tokens=8, quantum=4)
            assert "answer" in await runtime.submit(dag, inputs)

    run(scenario())


def test_invalid_graph_admission_is_atomic():
    async def scenario():
        async with HeraSys(reference_pools()) as runtime:
            dag, _ = rag("HeraSys?", DOCS)
            with pytest.raises(ValueError, match="Missing input"):
                runtime.submit(dag, {})
            assert not runtime.queries and not runtime.tasks

    run(scenario())


def test_single_worker_reserves_capacity_for_heavy_query():
    async def scenario():
        pool = ResourcePool([ReferenceWorker(0.001)], max_batch_size=1)
        async with HeraSys(
            {EngineType.AGGREGATOR: pool}, replace(HeraConfig(), fusion=False)
        ) as runtime:
            futures = []
            for i, duration in enumerate([1, 1, 100]):
                dag = DAG()
                dag.register_nodes(
                    primitive(
                        "echo",
                        NodeOps.AGGREGATOR,
                        EngineType.AGGREGATOR,
                        ["value"],
                        ["result"],
                        ms=duration,
                        mode="identity",
                    )
                )
                futures.append(runtime.submit(dag, {"value": i}, query_id=f"q{i}"))
            await asyncio.gather(*futures)
        first = next(e for e in runtime.events if e["event"] == "dispatch")
        assert first["heavy"] == [True] and first["reserved"]

    run(scenario())


def test_bounded_llm_sessions_and_early_state_release():
    class HandleWorker(ReferenceWorker):
        def __init__(self):
            super().__init__()
            self.states = {}
            self.sequence = 0

        def _execute(self, r):
            if r["op"] == "llm_prefilling":
                text = r["inputs"][r["config"]["question_port"]]
            elif r["op"] == "llm_parallel_decoding":
                handle = next(iter(r["inputs"].values()))
                text = self.states[handle["state_key"]]
                if r["config"]["final"]:
                    return {r["outputs"][0]: text}
            else:
                return super()._execute(r)
            self.sequence += 1
            key = f"state{self.sequence}"
            self.states[key] = text
            return {r["outputs"][0]: {"state_key": key}}

        async def release(self, outputs):
            for output in outputs:
                for value in output.values():
                    if isinstance(value, dict) and "state_key" in value:
                        self.states.pop(value["state_key"], None)

    async def scenario():
        pools = reference_pools()
        worker = HandleWorker()
        pools[EngineType.LLM] = ResourcePool(
            [worker], max_batch_size=8, supports_state_fork=True, max_sessions=2
        )
        async with HeraSys(pools) as runtime:
            futures = []
            for i in range(12):
                dag, inputs = web_search(f"question {i}", max_tokens=8, quantum=4)
                futures.append(runtime.submit(dag, inputs, timeout=2))
            results = await asyncio.gather(*futures)
            assert all(
                result["answer"] == f"question {i}" for i, result in enumerate(results)
            )
        assert not worker.states

    run(scenario())


def test_late_duplicate_decodes_stay_with_independently_executed_prefills():
    async def scenario():
        decode_started = asyncio.Event()

        class SlowDecode(ReferenceWorker):
            async def execute_batch(self, requests):
                if requests[0]["op"] == "llm_parallel_decoding":
                    decode_started.set()
                    await asyncio.sleep(0.03)
                return await super().execute_batch(requests)

        pools = reference_pools()
        workers = [SlowDecode(), SlowDecode()]
        pools[EngineType.LLM] = ResourcePool(
            workers, max_batch_size=1, supports_state_fork=True
        )
        async with HeraSys(pools, replace(HeraConfig(), cache_entries=0)) as runtime:
            dag, inputs = rag("same question", DOCS, splits=1, max_tokens=8, quantum=4)
            first = runtime.submit(dag, inputs)
            await decode_started.wait()
            second = runtime.submit(dag, inputs)
            answers = await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert answers[0] == answers[1]
        for worker in workers:
            assert (
                sum(
                    request["op"] == "llm_prefilling"
                    for batch in worker.calls
                    for request in batch
                )
                == 1
            )

    run(scenario())


def test_completed_decode_keeps_capacity_until_its_source_kv_is_retired():
    class CapacityWorker(ReferenceWorker):
        def __init__(self):
            super().__init__()
            self.states = {}
            self.finishing = asyncio.Event()
            self.continue_decode = asyncio.Event()
            self.runtime = None
            self.sequence = 0

        async def execute_batch(self, requests):
            if requests[0]["op"] == "llm_prefilling":
                # One logical session fills our simulated physical KV pool.
                assert not self.states, "New prefill overtook outstanding KV release"
            elif not self.finishing.is_set():
                self.finishing.set()
                self.runtime._wake.set()
                await self.continue_decode.wait()
            return await super().execute_batch(requests)

        def _execute(self, request):
            if request["op"] == "llm_prefilling":
                self.sequence += 1
                key = str(self.sequence)
                self.states[key] = request["inputs"]["question"]
                return {request["outputs"][0]: {"state_key": key}}
            handle = next(iter(request["inputs"].values()))
            return {request["outputs"][0]: self.states[handle["state_key"]]}

        async def release(self, outputs):
            for output in outputs:
                for handle in output.values():
                    if isinstance(handle, dict) and "state_key" in handle:
                        self.states.pop(handle["state_key"], None)

    class CompletionDuringRetirement(HeraSys):
        async def _retire(self):
            await super()._retire()
            if worker.finishing.is_set() and not worker.continue_decode.is_set():
                # An actor result arrives after retirement inspected the source
                # handle, but before the same scheduler turn admits new work.
                worker.continue_decode.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)

    async def scenario():
        nonlocal worker
        worker = CapacityWorker()
        pools = reference_pools()
        pools[EngineType.LLM] = ResourcePool(
            [worker], max_batch_size=1, supports_state_fork=True, max_sessions=1
        )
        async with CompletionDuringRetirement(
            pools, HeraConfig.variant("ayo_static")
        ) as runtime:
            worker.runtime = runtime
            answers = await asyncio.gather(
                *(
                    runtime.submit(*rag(question, DOCS, max_tokens=4, quantum=4))
                    for question in ("first", "second")
                )
            )
            assert answers == [{"answer": "first"}, {"answer": "second"}]
        assert worker.continue_decode.is_set()
        assert not worker.states

    worker = None
    run(scenario())


def test_prefill_batch_ignores_format_but_preserves_generation_parameters():
    async def scenario():
        pools = reference_pools()
        worker = ReferenceWorker()
        pools[EngineType.LLM] = ResourcePool(
            [worker],
            max_batch_size=8,
            supports_state_fork=True,
            mixed_prefill_batches=True,
            batch_by_port_names=False,
        )
        async with HeraSys(pools, HeraConfig.variant("ayo_static")) as runtime:
            futures = []
            for index, (model, budget) in enumerate(
                (("model-a", 4), ("model-a", 4), ("model-b", 4), ("model-a", 8))
            ):
                question, context = f"question_{index}", f"context_{index}"
                prefill = primitive(
                    "prefill",
                    NodeOps.LLM_PREFILLING,
                    EngineType.LLM,
                    [question, context],
                    ["state"],
                    model=model,
                    question_port=question,
                    context_port=context,
                    max_tokens=budget,
                    prompt_template=f"template {index}: {{question}} {{context}}",
                    _hera_session="answer",
                )
                decode = primitive(
                    "decode",
                    NodeOps.LLM_PARTIAL_DECODING,
                    EngineType.LLM,
                    ["state"],
                    ["answer"],
                    model=model,
                    token_budget=budget,
                    final=True,
                    _hera_session="answer",
                )
                prefill >> decode
                dag = DAG(name=f"query_{index}")
                dag.register_nodes(prefill, decode)
                futures.append(
                    runtime.submit(dag, {question: f"q{index}", context: "text"})
                )
            assert await asyncio.gather(*futures) == [
                {"answer": f"[reference] q{i}"} for i in range(4)
            ]
        batches = [
            batch for batch in worker.calls if batch[0]["op"] == "llm_prefilling"
        ]
        assert sorted(map(len, batches)) == [1, 1, 2]
        assert all(
            len({request["config"]["model"] for request in batch}) == 1
            for batch in batches
        )
        assert all(
            len({request["config"]["max_tokens"] for request in batch}) == 1
            for batch in batches
        )

    run(scenario())


def test_heavy_reservation_lends_unused_batch_positions():
    async def scenario():
        worker = ReferenceWorker()
        pool = ResourcePool([worker], max_batch_size=3)
        async with HeraSys(
            {EngineType.AGGREGATOR: pool}, replace(HeraConfig(), fusion=False)
        ) as runtime:
            futures = []
            for name, ms in [("small_1", 1), ("small_2", 1), ("heavy", 100)]:
                dag = DAG()
                dag.register_nodes(
                    primitive(
                        name,
                        NodeOps.AGGREGATOR,
                        EngineType.AGGREGATOR,
                        ["value"],
                        ["result"],
                        ms=ms,
                        mode="identity",
                    )
                )
                futures.append(runtime.submit(dag, {"value": name}))
            await asyncio.gather(*futures)
        assert len(worker.calls) == 1
        assert [request["inputs"]["value"] for request in worker.calls[0]] == [
            "heavy",
            "small_1",
            "small_2",
        ]

    run(scenario())


def test_query_aging_advances_old_pipeline_before_new_ingestion():
    async def scenario(query_aging):
        started = asyncio.Event()

        class Worker(ReferenceWorker):
            async def execute_batch(self, requests):
                if requests[0]["inputs"].get("value") == "old":
                    started.set()
                    await asyncio.sleep(0.03)
                return await super().execute_batch(requests)

        worker = Worker()
        config = replace(
            HeraConfig(),
            fusion=False,
            resource_skew=False,
            adaptive_batching=False,
            max_ready_wait_s=None if query_aging else 0.005,
            max_query_wait_s=0.005 if query_aging else None,
        )
        async with HeraSys(
            {EngineType.AGGREGATOR: ResourcePool([worker], max_batch_size=1)}, config
        ) as runtime:
            old = DAG()
            first = primitive(
                "old_ingest",
                NodeOps.AGGREGATOR,
                EngineType.AGGREGATOR,
                ["value"],
                ["ready"],
                mode="identity",
            )
            second = primitive(
                "old_retrieve",
                NodeOps.AGGREGATOR,
                EngineType.AGGREGATOR,
                ["ready"],
                ["result"],
                mode="identity",
            )
            first >> second
            old.register_nodes(first, second)
            first_query = runtime.submit(old, {"value": "old"})
            await started.wait()
            newer = DAG()
            newer.register_nodes(
                primitive(
                    "new_ingest",
                    NodeOps.AGGREGATOR,
                    EngineType.AGGREGATOR,
                    ["value"],
                    ["result"],
                    mode="identity",
                )
            )
            second_query = runtime.submit(newer, {"value": "new"})
            await asyncio.gather(first_query, second_query)
        return [
            event["nodes"][0]
            for event in runtime.events
            if event["event"] == "dispatch"
        ]

    assert run(scenario(False)) == ["old_ingest", "new_ingest", "old_retrieve"]
    assert run(scenario(True)) == ["old_ingest", "old_retrieve", "new_ingest"]


def test_admission_factory_observes_burst_before_splitting():
    async def scenario():
        seen = []
        async with HeraSys(reference_pools(delay_s=0.001)) as runtime:

            def factory(load):
                seen.append(load)
                return rag(
                    "HeraSys?",
                    DOCS,
                    splits=split_factor(load, len(DOCS), runtime.config),
                    max_tokens=8,
                )

            results = await asyncio.gather(
                *(runtime.submit_factory(factory) for _ in range(20))
            )
            assert all(result == results[0] for result in results)
            assert seen[0] == pytest.approx(20 / runtime.config.load_scale)
            assert runtime.pending_admissions == 0
        assert not runtime.queries and not runtime.tasks

    run(scenario())


def test_factory_wait_counts_towards_query_deadline():
    async def scenario():
        pools = reference_pools()
        async with HeraSys(
            pools, replace(HeraConfig(), fusion_window_s=0.03)
        ) as runtime:
            with pytest.raises(TimeoutError):
                await runtime.submit_factory(
                    lambda load: rag("HeraSys?", DOCS, with_llm=False), timeout=0.005
                )
        assert all(
            not worker.calls for pool in pools.values() for worker in pool.workers
        )
        assert runtime.pending_admissions == 0

    run(scenario())


def test_eos_finishes_remaining_decode_chain_without_engine_roundtrips():
    class EarlyEOS(ReferenceWorker):
        def _execute(self, request):
            if request["op"] == "llm_parallel_decoding":
                return {
                    request["outputs"][0]: {
                        "state_key": "finished",
                        "_hera_finished": True,
                        "text": "finished answer",
                        "model": request["config"]["model"],
                    }
                }
            return super()._execute(request)

        async def release(self, outputs):
            pass

    async def scenario():
        pools = reference_pools()
        worker = EarlyEOS()
        pools[EngineType.LLM] = ResourcePool([worker], supports_state_fork=True)
        async with HeraSys(pools) as runtime:
            dag, inputs = rag("question", DOCS, max_tokens=128, quantum=16)
            assert await runtime.submit(dag, inputs) == {"answer": "finished answer"}
            assert (
                sum(
                    request["op"] == "llm_parallel_decoding"
                    for batch in worker.calls
                    for request in batch
                )
                == 1
            )
            assert runtime.query_records[-1]["pruned_nodes"] == 7
            calls = len(worker.calls)
            assert await runtime.submit(dag, inputs) == {"answer": "finished answer"}
            assert len(worker.calls) == calls

    run(scenario())


def test_shared_prefill_keeps_each_logical_session_on_its_worker():
    async def scenario():
        async with HeraSys(reference_pools()) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, max_tokens=8, quantum=4)
            other, _ = rag("HeraSys?", DOCS, max_tokens=8, quantum=8)
            for node in other.nodes:
                if node.config.get("_hera_session"):
                    node.config["_hera_session"] = "renamed_session"
                node.name = "other_" + node.name
            results = await asyncio.gather(
                runtime.submit(dag, inputs), runtime.submit(other, inputs)
            )
            assert results[0] == results[1]
        llm_dispatch = [
            e
            for e in runtime.events
            if e["event"] == "dispatch" and "llm" in e["engine"].lower()
        ]
        assert len({e["slot"] for e in llm_dispatch}) == 1
        assert any(
            e["event"] == "fusion" and "prefill" in e["node"] for e in runtime.events
        )

    run(scenario())


def test_optional_aging_serves_waiting_request_before_new_short_request():
    async def scenario(aging):
        worker = ReferenceWorker(0.04)
        config = replace(
            HeraConfig(),
            fusion=False,
            adaptive_batching=False,
            resource_skew=False,
            max_ready_wait_s=0.01 if aging else None,
        )
        async with HeraSys(
            {EngineType.AGGREGATOR: ResourcePool([worker], max_batch_size=1)}, config
        ) as runtime:

            def query(name, ms):
                dag = DAG()
                dag.register_nodes(
                    primitive(
                        name,
                        NodeOps.AGGREGATOR,
                        EngineType.AGGREGATOR,
                        ["value"],
                        ["result"],
                        ms=ms,
                        mode="identity",
                    )
                )
                return runtime.submit(dag, {"value": name})

            blocker = query("blocker", 1)
            await asyncio.sleep(0.005)
            old = query("old", 10000)
            await asyncio.sleep(0.02)
            short = query("short", 1)
            await asyncio.gather(blocker, old, short)
        return [call[0]["inputs"]["value"] for call in worker.calls]

    assert run(scenario(False)) == ["blocker", "short", "old"]
    assert run(scenario(True)) == ["blocker", "old", "short"]


def test_aging_applies_to_all_compatible_batch_positions():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockFirstBatch(ReferenceWorker):
            async def execute_batch(self, requests):
                if not entered.is_set():
                    entered.set()
                    await release.wait()
                return await super().execute_batch(requests)

        worker = BlockFirstBatch(0)
        config = replace(
            HeraConfig(),
            fusion=False,
            adaptive_batching=False,
            resource_skew=False,
            max_ready_wait_s=0.01,
        )
        async with HeraSys(
            {EngineType.AGGREGATOR: ResourcePool([worker], max_batch_size=2)}, config
        ) as runtime:

            def query(name, ms):
                dag = DAG()
                dag.register_nodes(
                    primitive(
                        name,
                        NodeOps.AGGREGATOR,
                        EngineType.AGGREGATOR,
                        ["value"],
                        ["result"],
                        ms=ms,
                        mode="identity",
                    )
                )
                return runtime.submit(dag, {"value": name})

            blocker = query("blocker", 1)
            await entered.wait()
            old = [query("old_1", 10000), query("old_2", 10000)]
            await asyncio.sleep(0.02)
            new = [query("new_1", 1), query("new_2", 1)]
            release.set()
            await asyncio.gather(blocker, *old, *new)
        calls = [
            [request["inputs"]["value"] for request in batch] for batch in worker.calls
        ]
        assert calls == [["blocker"], ["old_1", "old_2"], ["new_1", "new_2"]]

    run(scenario())


def test_value_fusion_across_different_pipeline_splits_and_tenant_isolation():
    async def scenario(fuse_values, namespace):
        pools = reference_pools(instances=1, delay_s=0.002)
        config = replace(HeraConfig(), value_fusion=fuse_values)
        async with HeraSys(pools, config) as runtime:
            one, inputs_one = rag("HeraSys?", DOCS, splits=1, max_tokens=8, quantum=4)
            two, inputs_two = rag("HeraSys?", DOCS, splits=2, max_tokens=8, quantum=4)
            output = await asyncio.gather(
                runtime.submit(one, inputs_one, namespace="first"),
                runtime.submit(two, inputs_two, namespace=namespace),
            )
        assert output[0] == output[1]
        assert (
            not runtime.tasks
            and not runtime.signatures
            and not runtime.value_signatures
        )
        llm_calls = [
            request
            for batch in pools[EngineType.LLM].workers[0].calls
            for request in batch
        ]
        return llm_calls, runtime.events

    unfused, _ = run(scenario(False, "first"))
    fused, events = run(scenario(True, "first"))
    isolated, isolated_events = run(scenario(True, "other"))
    count_prefills = lambda calls: sum(call["op"] == "llm_prefilling" for call in calls)
    assert count_prefills(unfused) == count_prefills(isolated) == 2
    assert count_prefills(fused) == 1
    assert any(event["event"] == "value_fusion" for event in events)
    assert not any(event["event"] == "value_fusion" for event in isolated_events)


def test_deterministic_decode_fusion_and_completed_answer_reuse():
    async def scenario():
        pools = reference_pools(delay_s=0.001)
        async with HeraSys(pools) as runtime:
            dag, inputs = rag("HeraSys?", DOCS, splits=1, max_tokens=8, quantum=4)
            results = await asyncio.gather(
                *(runtime.submit(dag, inputs) for _ in range(3))
            )
            assert results[0] == results[1] == results[2]
            calls = [
                r
                for pool in pools.values()
                for worker in pool.workers
                for batch in worker.calls
                for r in batch
            ]
            assert sum(r["op"] == "llm_parallel_decoding" for r in calls) == 2
            before = sum(
                len(worker.calls) for pool in pools.values() for worker in pool.workers
            )
            assert await runtime.submit(dag, inputs) == results[0]
            assert (
                sum(
                    len(worker.calls)
                    for pool in pools.values()
                    for worker in pool.workers
                )
                == before
            )
            stochastic, _ = rag("HeraSys?", DOCS, splits=1)
            stochastic.get_node("answer_prefill").config["temperature"] = 0.8
            assert not runtime._can_reuse(stochastic.get_node("answer_prefill"))

    run(scenario())


def test_value_fusion_does_not_confuse_engine_local_kv_handles():
    class HandleWorker(ReferenceWorker):
        def __init__(self):
            super().__init__(0.002)
            self.states = {}
            self.counter = 0

        def _execute(self, request):
            if request["op"] == "llm_prefilling":
                text = request["inputs"][request["config"]["question_port"]]
            else:
                handle = next(iter(request["inputs"].values()))
                text = self.states[handle["state_key"]]
                if request["config"].get("final"):
                    return {request["outputs"][0]: text}
            key = str(self.counter)
            self.counter += 1
            self.states[key] = text
            return {request["outputs"][0]: {"state_key": key}}

        async def release(self, outputs):
            for output in outputs:
                for value in output.values():
                    if isinstance(value, dict) and "state_key" in value:
                        self.states.pop(value["state_key"], None)

    async def scenario():
        pools = reference_pools()
        workers = [HandleWorker(), HandleWorker()]
        pools[EngineType.LLM] = ResourcePool(
            workers, max_batch_size=1, supports_state_fork=True, max_sessions=1
        )
        async with HeraSys(pools) as runtime:
            requests = [
                rag(question, DOCS, max_tokens=8, quantum=4)
                for question in ["first", "second"]
            ]
            answers = await asyncio.gather(
                *(runtime.submit(dag, inputs) for dag, inputs in requests)
            )
            assert answers == [{"answer": "first"}, {"answer": "second"}]
        assert all(not worker.states for worker in workers)

    run(scenario())
