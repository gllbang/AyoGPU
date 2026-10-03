"""Real BGE/pgvector check of ready-time embedding→search→rerank splitting."""

import argparse
import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import ray

from HeraSys.engines.engine_types import EngineType
from HeraSys import HeraConfig, HeraSys, ResourcePool
from HeraSys.backends import RayWorker, reference_pools
from HeraSys.gpu import GPUWorker
from HeraSys.postgres import PgVectorWorker
from tests.herasys.test_pipeline import graph


async def main(args):
    ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False)
    actors = []
    try:
        embed = ray.remote(num_gpus=1, num_cpus=2)(GPUWorker).remote(
            "embedding", args.embedding_model
        )
        rerank = ray.remote(num_gpus=1, num_cpus=2)(GPUWorker).remote(
            "reranking", args.reranker_model
        )
        vector = ray.remote(num_cpus=1)(PgVectorWorker).remote()
        actors.extend([embed, rerank, vector])
        await asyncio.gather(*(a.ready.remote() for a in actors))
        documents = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "HeraSys schedules concurrent language model workflows.",
            "Ayo decomposes workflows into primitive operators.",
            "Embedding models map text to vectors for retrieval.",
        ]
        vectors = (
            await embed.execute_batch.remote(
                [
                    {
                        "op": "embedding",
                        "config": {"role": "document"},
                        "inputs": {"texts": documents},
                        "outputs": ["vectors"],
                    }
                ]
            )
        )[0]["output"]["vectors"]
        index = (
            await vector.execute_batch.remote(
                [
                    {
                        "op": "vectordb_ingestion",
                        "config": {},
                        "inputs": {"texts": documents, "vectors": vectors},
                        "outputs": ["index"],
                    }
                ]
            )
        )[0]["output"]["index"]
        values = {
            "questions": [
                "Paris",
                "London",
                "HeraSys",
                "Ayo",
                "France",
                "England",
                "vectors",
            ],
            "index": index,
        }
        runs = []
        for splits in [1, 2, 4]:
            pools = reference_pools()
            for kind, actor in [
                (EngineType.EMBEDDER, embed),
                (EngineType.RERANKER, rerank),
                (EngineType.VECTOR_DB, vector),
            ]:
                pools[kind] = ResourcePool(
                    [RayWorker(actor)], max_batch_size=4, batch_by_port_names=False
                )
            config = replace(
                HeraConfig(), adaptive_splitting=False, fixed_splits=splits
            )
            async with HeraSys(pools, config) as runtime:
                answer = await runtime.submit(graph(), values)
                assert len(answer["answers"]) == 7
            runs.append(
                {
                    "splits": splits,
                    "output": answer,
                    "events": [
                        e for e in runtime.events if e["event"] == "pipeline_split"
                    ],
                }
            )
        if not all(r["output"] == runs[0]["output"] for r in runs):
            raise AssertionError(
                "Pipeline partitioning changed ordered retrieval/reranking output"
            )
        output = {
            "passed": True,
            "backend": "BGE GPU + PostgreSQL/pgvector",
            "rows": 7,
            "runs": runs,
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__),
                    *sorted(Path("HeraSys").glob("*.py")),
                    Path("tests/herasys/test_pipeline.py"),
                ]
            },
        }
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"passed": True, "rows": 7, "split_factors": [1, 2, 4]}))
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-model", default="/disk_n/lsz/model/bge-large-en-v1.5"
    )
    parser.add_argument(
        "--reranker-model", default="/disk_n/lsz/model/bge-reranker-large"
    )
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
