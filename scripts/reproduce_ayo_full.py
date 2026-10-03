"""Time pinned upstream APP, GraphScheduler, EngineScheduler and native engines.

The isolated checkout must contain the recorded compatibility patch. This runner
does not import HeraSys's runtime, workers, graph adapter or FIFO dispatcher.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path


def samples(records, count, seed, rate):
    # Identical request selection to benchmarks.herasys.samples. Persist the
    # selected inputs so pairing can be independently checked by question/text.
    rng = random.Random(seed)
    repeated = set(rng.sample(range(1, count), min(count - 1, round(count * rate))))
    chosen = []
    for i in range(count):
        if i in repeated:
            index = rng.choices(
                range(len(chosen)),
                weights=[1 / (j + 1) ** 1.2 for j in range(len(chosen))],
            )[0]
            chosen.append(chosen[index])
        else:
            item = records[i % len(records)]
            chosen.append(
                {
                    "question": item["question"]
                    if i < len(records)
                    else f"{item['question']} (request {i})",
                    "documents": item["documents"],
                }
            )
    return chosen


def graph(sample, query_id, rerank, tokens, splits):
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node
    from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
    from Ayo.engines.engine_types import EngineType

    def node(name, op, engine, inputs, outputs, **config):
        return Node(
            name,
            NodeType.COMPUTE,
            engine_type=engine,
            op_type=op,
            io_schema=NodeIOSchema(
                dict.fromkeys(inputs, object), dict.fromkeys(outputs, object)
            ),
            config=config,
        )

    values, nodes, indexes = {"query": sample["question"]}, [], []
    documents = sample["documents"]
    count = min(splits, len(documents))
    for lane in range(count):
        text, vector = f"passages_{lane}", f"embeddings_{lane}"
        values[text] = documents[
            lane * len(documents) // count : (lane + 1) * len(documents) // count
        ]
        embed = node(
            f"embed_{lane}", NodeOps.EMBEDDING, EngineType.EMBEDDER, [text], [vector]
        )
        ingest = node(
            f"ingest_{lane}",
            NodeOps.VECTORDB_INGESTION,
            EngineType.VECTOR_DB,
            [text, vector],
            [f"index_{lane}"],
        )
        embed >> ingest
        nodes.extend([embed, ingest])
        indexes.append(ingest)
    query = node(
        "embed_question",
        NodeOps.EMBEDDING,
        EngineType.EMBEDDER,
        ["query"],
        ["query_embedding"],
        role="query",
    )
    search = node(
        "retrieve",
        NodeOps.VECTORDB_SEARCHING,
        EngineType.VECTOR_DB,
        ["query_embedding"],
        ["contexts"],
        top_k=12,
    )
    query >> search
    for ingest in indexes:
        ingest >> search
    nodes.extend([query, search])
    tail, context = search, "contexts"
    if rerank:
        tail = node(
            "rerank",
            NodeOps.RERANKING,
            EngineType.RERANKER,
            ["query", "contexts"],
            ["ranked_contexts"],
            top_k=8,
        )
        search >> tail
        nodes.append(tail)
        context = "ranked_contexts"
    prefill = node(
        "answer_prefill",
        NodeOps.LLM_PREFILLING,
        EngineType.LLM,
        ["query", context],
        ["prefilled"],
        llm_internal_id=query_id + "_answer",
        max_tokens=tokens,
        temperature=0.0,
        top_p=1.0,
        benchmark_prompt=True,
        prompt_template="Context: {" + context + "}\nQuestion: {query}\nAnswer:",
        context_port=context,
    )
    decode = node(
        "answer_decode",
        NodeOps.LLM_DECODING,
        EngineType.LLM,
        [],
        ["answer"],
        llm_internal_id=query_id + "_answer",
        max_tokens=tokens,
        temperature=0.0,
        top_p=1.0,
    )
    tail >> prefill >> decode
    nodes.extend([prefill, decode])
    dag = DAG(name="ayo_full_rag")
    dag.register_nodes(*nodes)
    return dag, values


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered) - 1) * p
    i = int(point)
    return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (
        point - i
    )


async def main(args):
    import ray
    from psycopg.conninfo import conninfo_to_dict

    from Ayo.app import APP
    from Ayo.configs.config import EngineConfig
    from Ayo.engines.engine_types import EngineType
    from Ayo.queries.query import Query

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    implementation_variant = (
        "ayo_upstream_full_batched"
        if args.batch_wait_s is not None
        else "ayo_upstream_full_compat"
    )
    variant = "Ayo"
    patch = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "Ayo"], cwd=args.upstream
    )
    (output / "upstream.patch").write_bytes(patch)
    subprocess.run(
        ["tar", "-czf", str(output / "source.tar.gz"), "Ayo"],
        cwd=args.upstream,
        check=True,
    )
    (output / "reproduce_ayo_full.py").write_bytes(Path(__file__).read_bytes())
    (output / "vllm_compat.py").write_bytes(
        Path(os.environ["AYO_FULL_VLLM_COMPAT"]).read_bytes()
    )
    source_hashes = {
        str(p.relative_to(args.upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(args.upstream).glob("Ayo/**/*.py"))
    }
    source_id = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    (output / "environment.json").write_text(
        json.dumps(
            {
                "arguments": {
                    **vars(args),
                    "backend": "gpu",
                    "llm_instances": 2,
                    "model_instances": 3,
                    "reranker_dtype": "float32",
                    "variants": [variant],
                },
                "source_sha256": source_hashes,
                "source_id": source_id,
                "upstream_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
                ).strip(),
                "dataset_sha256": {
                    "rag": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
                },
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "variant": variant,
                "implementation_variant": implementation_variant,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "ray",
                        "vllm",
                        "transformers",
                        "sentence-transformers",
                        "asyncpg",
                        "pgvector",
                    )
                },
                "gpus": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    records = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    ray.init(num_cpus=24, include_dashboard=False, log_to_driver=False)
    app = APP.init()
    app.query_timeout = args.timeout
    handles = {}
    summaries = []
    try:
        dsn = conninfo_to_dict(os.environ["HERASYS_PG_DSN"])
        configurations = [
            EngineConfig(
                "llm",
                EngineType.LLM,
                num_gpus=1,
                instances=2,
                model_config={
                    "model_name": args.llm_model,
                    "max_num_seqs": args.max_sessions,
                    "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.memory_fraction,
                    "max_num_batched_tokens": 8192,
                    "dtype": "half",
                },
            ),
            EngineConfig(
                "embed",
                EngineType.EMBEDDER,
                num_gpus=1 / 3,
                instances=3,
                model_config={
                    "model_name": args.embedding_model,
                    "max_batch_size": args.embedding_batch_size,
                    "benchmark_batch_wait_s": args.batch_wait_s,
                },
            ),
            EngineConfig(
                "rerank",
                EngineType.RERANKER,
                num_gpus=1 / 3,
                instances=3,
                model_config={
                    "model_name": args.reranker_model,
                    "max_batch_size": args.reranker_batch_size,
                    "benchmark_batch_wait_s": args.batch_wait_s,
                },
            ),
            EngineConfig(
                "vector",
                EngineType.VECTOR_DB,
                instances=1,
                model_config={
                    "host": dsn.get("host", "127.0.0.1"),
                    "port": int(dsn.get("port", 15432)),
                    "user": dsn["user"],
                    "password": dsn.get("password", ""),
                    "database": dsn.get("dbname", "postgres"),
                    "vector_dim": 1024,
                },
            ),
        ]
        for config in configurations:
            app.register_engine(config)
            handles[config.engine_type] = await app.engine_schedulers[
                config.engine_type
            ].engine_handles.remote()
        await app.start()

        async def execute(item, query_id):
            dag, values = graph(
                item,
                query_id,
                current_workload == "rag_rerank",
                args.max_tokens,
                args.splits,
            )
            query = Query(
                uuid=query_id,
                query_id=query_id,
                query_inputs=values,
                DAG=dag,
                timeout=args.timeout,
            )
            try:
                future = await app.submit_query(query, timeout=args.timeout)
                results = await asyncio.wait_for(future, args.timeout + 5)
                answer = results["answer_decode"]
                while isinstance(answer, ray.ObjectRef):
                    answer = await answer
                if not isinstance(answer, str):
                    raise TypeError(f"Invalid native answer {type(answer)}")
                return {"answer": answer}
            except BaseException:
                print(
                    json.dumps(
                        {
                            "event": "native_failure",
                            "query": query_id,
                            "nodes": await app.graph_scheduler.diagnostic.remote(
                                query_id
                            ),
                        }
                    ),
                    flush=True,
                )
                raise
            finally:
                await app.graph_scheduler.forget_query.remote(query_id)
                # Native per-query collections are transient; cleanup is scoped
                # to this run's UUID names inside ayo_full_reproduction.
                for actor in handles[EngineType.VECTOR_DB]:
                    await actor.cleanup_query.remote(query_id)
                ray.kill(query.query_state)

        for current_workload in args.workloads:
            for repeat in range(args.warmup):
                for count in (1, 8, args.warmup_concurrency):
                    await asyncio.gather(
                        *(
                            execute(
                                records[
                                    (repeat * args.warmup_concurrency + i)
                                    % len(records)
                                ],
                                "warm_" + uuid.uuid4().hex,
                            )
                            for i in range(count)
                        )
                    )
            print(
                json.dumps({"event": "warmup_complete", "workload": current_workload}),
                flush=True,
            )
        for repeat in range(args.repeats):
            items = samples(
                records, args.requests, args.seed + repeat, args.duplicate_rate
            )
            (output / f"inputs_r{repeat}.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in items)
            )
            for current_workload in args.workloads:
                for concurrency in args.concurrency:
                    app.query_semaphore = asyncio.Semaphore(concurrency)
                    limiter = asyncio.Semaphore(concurrency)
                    query_records, outputs = [], []
                    run_id = uuid.uuid4().hex

                    async def request(i, item):
                        async with limiter:
                            started = time.perf_counter()
                            try:
                                result = await execute(item, run_id + f"_{i}")
                                outputs.append({"query_id": f"q{i}", "output": result})
                                status = "completed"
                            except Exception as exc:
                                outputs.append(
                                    {
                                        "query_id": f"q{i}",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                                status = "failed"
                            query_records.append(
                                {
                                    "query_id": f"q{i}",
                                    "status": status,
                                    "latency_s": time.perf_counter() - started,
                                }
                            )
                            if len(outputs) % max(1, args.requests // 10) == 0:
                                print(
                                    json.dumps(
                                        {
                                            "event": "progress",
                                            "workload": current_workload,
                                            "concurrency": concurrency,
                                            "repeat": repeat,
                                            "returned": len(outputs),
                                        }
                                    ),
                                    flush=True,
                                )

                    start = time.perf_counter()
                    await asyncio.gather(
                        *(request(i, item) for i, item in enumerate(items))
                    )
                    elapsed = time.perf_counter() - start
                    latency = [
                        r["latency_s"]
                        for r in query_records
                        if r["status"] == "completed"
                    ]
                    cleanup = await asyncio.gather(
                        *(a.resource_state.remote() for a in handles[EngineType.LLM])
                    )
                    for state in cleanup:
                        state["live_states"] = int(bool(state["unfinished"]))
                    summary = {
                        "variant": variant,
                        "implementation_variant": implementation_variant,
                        "backend": "gpu",
                        "measurement_kind": "performance",
                        "dataset": args.dataset,
                        "vector_backend": "pgvector",
                        "workload": current_workload,
                        "arrival_mode": "closed_loop",
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "requests": args.requests,
                        "completed": len(latency),
                        "failed": args.requests - len(latency),
                        "elapsed_s": elapsed,
                        "qps": len(latency) / elapsed,
                        "mean_s": statistics.fmean(latency) if latency else None,
                        "p50_s": percentile(latency, 0.5),
                        "p95_s": percentile(latency, 0.95),
                        "p99_s": percentile(latency, 0.99),
                        "max_s": max(latency, default=None),
                        "source_id": source_id,
                        "config": {
                            "native_queues": True,
                            "background_decode": True,
                            "fixed_splits": args.splits,
                            "native_batch_wait_s": args.batch_wait_s,
                        },
                        "backend_cleanup": cleanup,
                        "tail_sample_warning": args.requests < 1000,
                        "paper_reproduction": False,
                    }
                    directory = (
                        output
                        / f"gpu_{current_workload}_{variant}_c{concurrency}_r{repeat}"
                    )
                    directory.mkdir()
                    (directory / "summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
                    (directory / "queries.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in query_records)
                    )
                    (directory / "outputs.jsonl").write_text(
                        "".join(
                            json.dumps(r) + "\n"
                            for r in sorted(
                                outputs, key=lambda r: int(r["query_id"][1:])
                            )
                        )
                    )
                    summaries.append(summary)
                    (output / "summary.json").write_text(
                        json.dumps(summaries, indent=2) + "\n"
                    )
                    print(json.dumps(summary), flush=True)
                    if summary["failed"] or any(
                        r["unfinished"] or r["kv_free_blocks"] != r["kv_total_blocks"]
                        for r in cleanup
                    ):
                        raise RuntimeError("Native Ayo case failed or leaked KV blocks")
    finally:
        await app.stop()
        # Call upstream engine shutdown (including scoped per-query DB cleanup)
        # before disconnecting from Ray.
        await asyncio.gather(
            *(s.shutdown.remote() for s in app.engine_schedulers.values()),
            return_exceptions=True,
        )
        ray.shutdown()


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default=str(repo / "baselines/Ayo"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset", default=str(repo / "data/msmarco_validation_full.jsonl")
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["rag", "rag_rerank"],
        default=["rag", "rag_rerank"],
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[6, 30, 100, 300])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-rate", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-concurrency", type=int, default=44)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--max-sessions", type=int, default=22)
    parser.add_argument("--max-model-len", type=int, default=1664)
    parser.add_argument("--memory-fraction", type=float, default=0.88)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument(
        "--batch-wait-s",
        type=float,
        help="Enable optional bounded cross-request native batching; separately labelled variant",
    )
    parser.add_argument("--llm-model", default="/disk_n/lsz/model/Meta-Llama-3-8B")
    parser.add_argument(
        "--embedding-model", default="/disk_n/lsz/model/bge-large-en-v1.5"
    )
    parser.add_argument(
        "--reranker-model", default="/disk_n/lsz/model/bge-reranker-large"
    )
    args = parser.parse_args()
    args.upstream = str(Path(args.upstream).resolve())
    sys.path.insert(0, args.upstream)
    # Ray workers also import the isolated checkout, never the dirty working Ayo.
    os.environ["PYTHONPATH"] = args.upstream
    os.environ["AYO_FULL_VLLM_COMPAT"] = str(repo / "HeraSys/vllm_compat.py")
    os.environ["AYO_INFO_LEVEL"] = "ERROR"
    os.chdir(args.upstream)
    args.output = str((repo / args.output).resolve())
    args.dataset = str((repo / args.dataset).resolve())
    asyncio.run(main(args))
