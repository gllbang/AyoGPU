import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from HeraSys.dags.dag import DAG
from HeraSys.dags.node_commons import NodeOps
from HeraSys.engines.engine_types import EngineType
from HeraSys import HeraConfig, HeraSys, ResourcePool
from HeraSys.allocation import balancing_weight, choose_allocation
from HeraSys.backends import ReferenceWorker
from HeraSys.runtime import Task
from HeraSys.workflows import primitive


class Profile:
    """Explicit deterministic cost model for scheduler correctness, not GPU evidence."""

    def predict(self, node):
        return node.relative_ms / 1000

    def batch_time(self, nodes):
        return max(self.predict(node) for node in nodes)


def test_tail_feedback_reduces_throughput_weight_at_the_same_load():
    config = HeraConfig()
    assert balancing_weight(1, [5, 5, 6], config) == 1
    assert balancing_weight(1, [5, 6, 60], config) == pytest.approx(0.2)
    assert balancing_weight(0, [5, 6, 60], config) == 0
    assert balancing_weight(1, [5, 6, 60], replace(config, tail_feedback=False)) == 1


def test_joint_objective_changes_choice_when_tail_is_emphasized():
    old = primitive(
        "old", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["v"], ["out"], ms=10000
    )
    short = primitive(
        "short", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["v"], ["out"], ms=1000
    )
    tasks = [
        Task(1, old, "old", {}, set(), owners={("old", "ready"): 0}, priority=0.01),
        Task(2, short, "short", {}, set(), owners={("short", "ready"): 0}, priority=1),
    ]
    queries = {
        "old": SimpleNamespace(
            started=0, predictions={"ready": 10, "later": 90}, completed=set()
        ),
        "short": SimpleNamespace(
            started=100, predictions={"ready": 1}, completed=set()
        ),
    }

    def choose(balance):
        return (
            choose_allocation(
                [[tasks[1]], [tasks[0]]],
                slot=0,
                capacity=1,
                max_batch_size=1,
                profiles=Profile(),
                queries=queries,
                remaining={"old": 100, "short": 1},
                now=100,
                balance=balance,
                busy={},
                affinity=lambda _: None,
                reservation=0,
                config=replace(HeraConfig(), srtf_penalty=0, topology_penalty=0),
                eligible={1, 2},
                limit=lambda _: 1,
            )
            .tasks[0]
            .id
        )

    assert choose(1) == 2  # The short query minimizes mean completion time.
    assert choose(0) == 1  # Advancing the old query minimizes the projected tail.


def test_heavy_reservation_accounts_for_fraction_of_the_committed_batch():
    async def scenario():
        pool = ResourcePool(
            [ReferenceWorker(0.001), ReferenceWorker(0.001)], max_batch_size=8
        )
        config = replace(
            HeraConfig(), fusion=False, max_batch_wait_s=0, tail_feedback=False
        )
        async with HeraSys(
            {EngineType.AGGREGATOR: pool}, config, profiles=Profile()
        ) as runtime:
            futures = []
            for index, ms in enumerate([1] * 6 + [100]):
                dag = DAG()
                dag.register_nodes(
                    primitive(
                        "echo",
                        NodeOps.AGGREGATOR,
                        EngineType.AGGREGATOR,
                        ["value"],
                        ["result"],
                        ms=ms,
                    )
                )
                futures.append(
                    runtime.submit(dag, {"value": index}, query_id=f"q{index}")
                )
            assert await asyncio.gather(*futures) == [{"result": i} for i in range(7)]
        dispatch = [e for e in runtime.events if e["event"] == "dispatch"]
        first = dispatch[0]
        assert first["reserved"]
        assert (
            sum(
                share
                for share, heavy in zip(first["resource_shares"], first["heavy"])
                if heavy
            )
            >= 0.4
        )
        assert all(sum(e["resource_shares"]) == pytest.approx(1) for e in dispatch)
        assert not runtime.tasks

    asyncio.run(scenario())


def test_allocation_models_decode_before_prefill_session_capacity_is_released():
    class SessionProfile:
        def batch_time(self, nodes):
            if nodes[0].op_type == NodeOps.LLM_PREFILLING:
                return 1 if len(nodes) == 1 else 3
            return 10

    prefills, continuations, queries = [], {}, {}
    for index in range(2):
        q = f"q{index}"
        prefill = primitive(
            "prefill", NodeOps.LLM_PREFILLING, EngineType.LLM, [], ["state"], ms=1000
        )
        decode = primitive(
            "decode",
            NodeOps.LLM_DECODING,
            EngineType.LLM,
            ["state"],
            ["answer"],
            ms=10000,
        )
        task = Task(
            index, prefill, q, {}, set(), owners={(q, "prefill"): 0}, priority=1
        )
        tail = Task(
            index + 2, decode, q + "tail", {}, {index}, owners={(q, "decode"): 0}
        )
        prefills.append(task)
        continuations[index] = [tail]
        queries[q] = SimpleNamespace(
            started=0, predictions={"prefill": 1, "decode": 10}, completed=set()
        )

    def choose(future):
        return choose_allocation(
            [prefills],
            slot=0,
            capacity=1,
            max_batch_size=2,
            profiles=SessionProfile(),
            queries=queries,
            remaining={q: 11 for q in queries},
            now=0,
            balance=1,
            busy={},
            affinity=lambda _: None,
            reservation=0,
            config=replace(HeraConfig(), srtf_penalty=0, topology_penalty=0),
            eligible={0, 1},
            limit=lambda _: 2,
            continuations=future,
        )

    assert len(choose({}).tasks) == 1  # A ready-only horizon underprices tiny prefills.
    allocation = choose(continuations)
    assert len(allocation.tasks) == 2
    assert allocation.objective_s == pytest.approx(13)
    assert allocation.duration_s == pytest.approx(3)  # Commit only actual prefill.


def test_batch_profiles_include_measured_dispatch_overhead_but_service_does_not(
    tmp_path,
):
    from HeraSys.profiling import ProfileLibrary

    node = primitive(
        "echo", NodeOps.AGGREGATOR, EngineType.AGGREGATOR, ["input"], ["output"], ms=1
    )
    profiles = ProfileLibrary()
    profiles.observe(node, 0.001)
    profiles.observe_batch([node, node], 0.002, wall_s=0.012)
    profiles.freeze()
    assert profiles.predict(node) == pytest.approx(0.001)
    assert profiles.batch_time([node, node]) == pytest.approx(0.012)
    target = tmp_path / "profiles.json"
    profiles.save(target)
    assert ProfileLibrary.load(target).batch_time([node, node]) == pytest.approx(0.012)


def test_uniform_decode_cost_cache_matches_generic_rollout_for_uneven_progress():
    from HeraSys.profiling import ProfileLibrary

    tasks, tails, queries = [], {}, {}
    for i, steps in enumerate([1, 4, 2, 5]):
        q = f"q{i}"
        node = primitive(
            "decode",
            NodeOps.LLM_PARTIAL_DECODING,
            EngineType.LLM,
            ["state"],
            ["next"],
            ms=32,
            max_tokens=128,
            token_budget=16,
        )
        task = Task(i, node, q, {}, set(), owners={(q, "current"): 0}, priority=1)
        tasks.append(task)
        tails[i] = [
            Task(
                100 + i * 10 + step,
                node,
                f"{q}/{step}",
                {},
                set(),
                owners={(q, f"tail{step}"): 0},
            )
            for step in range(steps)
        ]
        queries[q] = SimpleNamespace(
            started=0,
            completed=set(),
            predictions={"current": 0.032, **{f"tail{j}": 0.032 for j in range(steps)}},
        )
    key = ProfileLibrary.key(tasks[0].node)
    profile = ProfileLibrary(
        batch_factors={key: {"1": 5, "4": 2}},
        batch_overheads={key: {"1": 0.004, "4": 0.006}},
    )
    generic = SimpleNamespace(batch_time=profile.batch_time)
    results = []
    for estimator in (profile, generic):
        results.append(
            choose_allocation(
                [tasks],
                slot=0,
                capacity=2,
                max_batch_size=4,
                profiles=estimator,
                queries=queries,
                remaining={q: sum(s.predictions.values()) for q, s in queries.items()},
                now=0,
                balance=0.6,
                busy={},
                affinity=lambda _: None,
                reservation=0,
                config=HeraConfig(),
                eligible={t.id for t in tasks},
                limit=lambda _: 4,
                continuations=tails,
            )
        )
    assert [t.id for t in results[0].tasks] == [t.id for t in results[1].tasks]
    assert results[0].objective_s == pytest.approx(results[1].objective_s, abs=1e-12)
    assert results[0].duration_s == pytest.approx(results[1].duration_s, abs=1e-12)


def test_sparse_batch_profiles_do_not_invent_faster_intermediate_kernels():
    from HeraSys.profiling import ProfileLibrary

    node = primitive(
        "decode",
        NodeOps.LLM_PARTIAL_DECODING,
        EngineType.LLM,
        ["state"],
        ["next"],
        ms=1000,
    )
    key = ProfileLibrary.key(node)
    profiles = ProfileLibrary(batch_factors={key: {"2": 0.15, "4": 0.08}})
    assert profiles.batch_time([node] * 2) == pytest.approx(0.30)
    assert profiles.batch_time([node] * 3) == pytest.approx(0.31)
    assert profiles.batch_time([node] * 4) == pytest.approx(0.32)
