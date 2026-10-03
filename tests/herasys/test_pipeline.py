import asyncio
from dataclasses import replace

import pytest

from HeraSys.dags.dag import DAG
from HeraSys.dags.node_commons import NodeOps
from HeraSys.engines.engine_types import EngineType
from HeraSys import HeraConfig, HeraSys
from HeraSys.backends import ReferenceWorker, reference_pools
from HeraSys.pipeline import pipeline, stage
from HeraSys.workflows import primitive


def query_pipeline():
    embed = primitive(
        "embed",
        NodeOps.EMBEDDING,
        EngineType.EMBEDDER,
        ["question"],
        ["query_vector"],
        role="query",
        ms=20,
    )
    search = primitive(
        "search",
        NodeOps.VECTORDB_SEARCHING,
        EngineType.VECTOR_DB,
        ["query_vector", "index"],
        ["contexts"],
        top_k=3,
        ms=10,
    )
    rerank = primitive(
        "rerank",
        NodeOps.RERANKING,
        EngineType.RERANKER,
        ["question", "contexts"],
        ["answers"],
        top_k=2,
        ms=10,
    )
    return pipeline(
        "retrieve_many",
        [
            stage(embed),
            stage(search, map_ports=["query_vector"]),
            stage(rerank, map_ports=["question", "contexts"]),
        ],
        inputs=["question", "index"],
        split_ports=["question"],
        max_splits=4,
    )


def inputs():
    documents = ["Paris France", "London England", "HeraSys workflows", "Ayo pipelines"]
    vectors = ReferenceWorker()._execute(
        {
            "op": "embedding",
            "inputs": {"texts": documents},
            "config": {},
            "outputs": ["vectors"],
        }
    )["vectors"]
    return {
        "questions": [
            "Paris",
            "London",
            "HeraSys",
            "Ayo",
            "France",
            "England",
            "workflows",
        ],
        "index": [{"text": d, "vector": v} for d, v in zip(documents, vectors)],
    }


def graph():
    source = primitive(
        "expanded_questions",
        NodeOps.AGGREGATOR,
        EngineType.AGGREGATOR,
        ["questions"],
        ["question"],
        mode="identity",
    )
    chain = query_pipeline()
    source >> chain
    dag = DAG()
    dag.register_nodes(source, chain)
    return dag


def test_ready_pipeline_preserves_order_and_overlaps_successive_stages():
    async def scenario(splits):
        pools = reference_pools(delay_s=0.002)
        for pool in pools.values():
            pool.max_batch_size = 1
        config = replace(HeraConfig(), adaptive_splitting=False, fixed_splits=splits)
        async with HeraSys(pools, config) as runtime:
            output = await runtime.submit(graph(), inputs())
        assert not runtime.tasks and not runtime.queries
        return output, runtime.events

    expected, _ = asyncio.run(scenario(1))
    actual, events = asyncio.run(scenario(4))
    assert actual == expected
    assert len(actual["answers"]) == 7
    split = next(e for e in events if e["event"] == "pipeline_split")
    assert split["splits"] == 4 and split["stages"] == 3
    first_search = min(
        e["time_s"]
        for e in events
        if e["event"] == "dispatch" and e["engine"] == str(EngineType.VECTOR_DB)
    )
    last_embed = max(
        e["time_s"]
        for e in events
        if e["event"] == "node_complete" and e["node"].endswith(":0")
    )
    assert first_search < last_embed


def test_late_fused_pipeline_survives_first_owner_cancellation():
    async def scenario():
        pools = reference_pools(delay_s=0.01)
        async with HeraSys(pools) as runtime:
            first = runtime.submit(graph(), inputs(), query_id="first")

            async def expanded():
                while not any(e["event"] == "pipeline_split" for e in runtime.events):
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(expanded(), 1)
            second = runtime.submit(graph(), inputs(), query_id="second")
            first.cancel()
            result = await asyncio.wait_for(second, 2)
            assert len(result["answers"]) == 7
        assert not runtime.tasks and not runtime.queries

    asyncio.run(scenario())


def test_bad_row_axis_fails_one_query_without_poisoning_runtime():
    async def scenario():
        broken = query_pipeline()
        broken.config["pipeline"]["split_ports"].append("index")
        bad_dag = DAG()
        bad_dag.register_nodes(broken)
        values = inputs()
        values["question"] = values.pop("questions")
        async with HeraSys(reference_pools()) as runtime:
            with pytest.raises(ValueError, match="same row count"):
                await runtime.submit(bad_dag, values)
            result = await runtime.submit(graph(), inputs())
            assert len(result["answers"]) == 7
        assert not runtime.tasks

    asyncio.run(scenario())


def test_empty_pipeline_preserves_empty_result():
    async def scenario():
        values = inputs()
        values["questions"] = []
        async with HeraSys(reference_pools()) as runtime:
            assert await runtime.submit(graph(), values) == {"answers": []}

    asyncio.run(scenario())


def test_dynamic_rag_indexing_matches_static_graph():
    from HeraSys.workflows import rag

    async def scenario():
        results = []
        for dynamic in (False, True):
            dag, values = rag(
                "Paris",
                ["Paris France", "London England", "Ayo systems"],
                splits=4,
                with_llm=False,
                dynamic_pipeline=dynamic,
            )
            async with HeraSys(reference_pools()) as runtime:
                results.append(await runtime.submit(dag, values))
        assert results[0] == results[1]
        assert any(e["event"] == "pipeline_split" for e in runtime.events)

    asyncio.run(scenario())


def test_lazy_pipeline_prediction_uses_each_stage_profile_not_aggregator():
    from HeraSys.profiling import ProfileLibrary

    chain = query_pipeline()
    profiles = ProfileLibrary()
    # Aggregator cost is intentionally huge: it must not affect the deferred
    # embedding -> retrieval -> reranking chain's remaining service estimate.
    profiles.factors[profiles.key(chain)] = 1000
    stages = chain.config["pipeline"]["stages"]
    costs = [0.04, 0.03, 0.08]
    for spec, cost in zip(stages, costs):
        node = primitive(
            spec["name"], NodeOps(spec["op"]), EngineType(spec["engine"]),
            spec["inputs"], spec["outputs"], ms=spec["relative_ms"],
            **spec["config"],
        )
        profiles.observe(node, cost)
    profiles.freeze()
    assert profiles.predict(chain) == pytest.approx(sum(costs))
