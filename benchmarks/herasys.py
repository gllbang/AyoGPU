"""Run reproducible, explicitly labelled reference or real GPU experiments."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from HeraSys import HeraConfig, HeraSys, policy
from HeraSys.backends import reference_pools
from HeraSys.profiling import ProfileLibrary
from HeraSys.workflows import rag, web_search

FIXTURE = [
    {
        "question": "What does HeraSys optimize?",
        "documents": [
            "HeraSys jointly schedules concurrent workflows and shares equivalent computations.",
            "Ayo represents LLM applications as directed acyclic graphs of task primitives.",
            "Retrieval augmented generation retrieves relevant passages before generating an answer.",
            "Embedding models map text into dense vectors for similarity search.",
            "Rerankers score retrieved passages against the user's question.",
            "Resource reservation gives long queries a share of execution capacity.",
            "The capital of France is Paris.",
            "Water freezes at zero degrees Celsius.",
            "GPUs perform many numerical operations in parallel.",
            "A scheduler selects ready operations whose dependencies have completed.",
            "Batched model inference groups compatible requests.",
            "A KV cache stores keys and values from transformer attention.",
        ],
    },
]


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def environment(args):
    packages = {}
    for name in [
        "torch",
        "ray",
        "vllm",
        "transformers",
        "sentence-transformers",
        "langgraph",
        "llama-index-core",
        "numpy",
        "psycopg",
        "pgvector",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if shutil.which("nvidia-smi")
        else "unavailable"
    )
    return {
        "arguments": vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "packages": packages,
        "gpus": gpu,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for base in [Path("HeraSys"), Path("benchmarks")]
            for path in sorted(base.rglob("*.py"))
        },
        "dataset_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [("rag", args.dataset), ("web", args.web_dataset)]
            if path
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
    }


def samples(records, count, seed, duplicate_rate, *, indices=None):
    if not count:
        return []
    indices = list(range(count)) if indices is None else list(indices)
    if len(indices) != count:
        raise ValueError("The sample positions must match the requested count")
    rng = random.Random(seed)
    repeated = set(
        rng.sample(range(1, count), min(count - 1, round(count * duplicate_rate)))
    )
    chosen = []
    for position, i in enumerate(indices):
        if position in repeated:
            # Zipfian hot-query selection, deterministic across all system variants.
            index = rng.choices(
                range(len(chosen)),
                weights=[1 / (j + 1) ** 1.2 for j in range(len(chosen))],
            )[0]
            chosen.append(chosen[index])
        else:
            item = records[i % len(records)]
            # Small fixture sets have unique questions so their duplicate rate is explicit.
            question = (
                item["question"]
                if i < len(records)
                else f"{item['question']} (request {i})"
            )
            chosen.append(
                {
                    "question": question,
                    "documents": item.get("documents", FIXTURE[0]["documents"]),
                }
            )
    return chosen


def workload_samples(records, web_records, count, seed, duplicate_rate, workload):
    if workload != "mixed":
        source = web_records if workload == "web_search" and web_records else records
        return samples(source, count, seed, duplicate_rate)
    chosen = [None] * count
    for offset, kind in enumerate(("rag_rerank", "rag", "web_search")):
        indices = [i for i in range(count) if mixed_workload(i) == kind]
        source = web_records if kind == "web_search" and web_records else records
        group = samples(
            source, len(indices), seed + offset, duplicate_rate, indices=indices
        )
        for index, sample in zip(indices, group):
            chosen[index] = sample
    return chosen


def mixed_workload(index):
    return "web_search" if index % 4 == 3 else ("rag" if index % 2 else "rag_rerank")


async def drive_requests(request, items, concurrency, arrival_mode):
    """Closed-loop refill or drained waves, with identical global request IDs."""
    if arrival_mode == "closed_loop":
        await asyncio.gather(*(request(i, item) for i, item in enumerate(items)))
    elif arrival_mode == "waves":
        for start in range(0, len(items), concurrency):
            await asyncio.gather(
                *(
                    request(i, items[i])
                    for i in range(start, min(start + concurrency, len(items)))
                )
            )
    else:
        raise ValueError(f"Unknown arrival mode: {arrival_mode}")


def warmup_batches(args, records, web_records):
    """Cover every requested workflow and actual corpus without result reuse."""
    kinds = list(
        dict.fromkeys(
            kind
            for workload in args.workloads
            for kind in (
                ("rag_rerank", "rag", "web_search")
                if workload == "mixed"
                else (workload,)
            )
        )
    )
    sizes = getattr(args, "profile_batch_sizes", None) or [args.warmup_concurrency]
    for kind in kinds:
        source = web_records if kind == "web_search" and web_records else records
        for repeat in range(args.warmup):
            for size in sizes:
                # Distinct inputs expose prompt-length and EOS variation; cloning
                # the first graph hides both and previously skipped Web entirely.
                yield (
                    kind,
                    [
                        source[(repeat * max(sizes) + i) % len(source)]
                        for i in range(size)
                    ],
                )


def build(workload, sample, runtime, args, index, admission_load=None):
    quantum = args.max_tokens if hasattr(runtime, "framework") else args.decode_quantum
    if workload == "mixed":
        workload = mixed_workload(index)
    if workload == "web_search":
        return web_search(
            sample["question"],
            llm_model=args.llm_model,
            max_tokens=args.max_tokens,
            quantum=quantum,
            search_provider=args.search_provider,
        )
    load = (
        admission_load
        if admission_load is not None
        else policy.load_factor(
            len(runtime.queries) + 1,
            0,
            sum(len(p.workers) for p in runtime.pools.values()),
            runtime.config,
        )
    )
    splits = policy.split_factor(load, len(sample["documents"]), runtime.config)
    dynamic_pipeline = (
        getattr(args, "pipeline_mode", "dynamic") == "dynamic"
        and not hasattr(runtime, "framework")
        and runtime.config.adaptive_splitting
    )
    if dynamic_pipeline:
        splits = runtime.config.max_splits
    return rag(
        sample["question"],
        sample["documents"],
        rerank=workload == "rag_rerank",
        splits=splits,
        max_tokens=args.max_tokens,
        quantum=quantum,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        llm_model=args.llm_model,
        dynamic_pipeline=dynamic_pipeline,
    )


async def run_case(
    pools,
    args,
    records,
    workload,
    variant,
    concurrency,
    repeat,
    profiles=None,
    web_records=None,
):
    external = variant in {"langgraph", "llamaindex", "ayo_upstream_graph"}
    config = replace(
        HeraConfig.variant("ayo_static" if external else variant),
        decode_quantum=args.decode_quantum,
        cache_entries=args.cache_entries,
        cache_ttl_s=args.cache_ttl_s,
        audit_nodes=args.audit_nodes,
        **{
            name: getattr(args, name)
            for name in (
                "alpha",
                "gamma",
                "delta",
                "rho",
                "load_scale",
                "max_batch_wait_s",
            )
            if getattr(args, name, None) is not None and not external
        },
    )
    items = workload_samples(
        records,
        web_records,
        args.requests,
        args.seed + repeat,
        args.duplicate_rate,
        workload,
    )
    identities = set()
    for index, sample in enumerate(items):
        kind = mixed_workload(index) if workload == "mixed" else workload
        identities.add(
            (
                kind,
                sample["question"],
                tuple(sample["documents"]) if kind != "web_search" else (),
            )
        )
    if variant == "ayo_upstream_graph":
        from .ayo_upstream import AyoUpstreamRuntime

        runtime = AyoUpstreamRuntime(pools, batch_wait_s=args.baseline_batch_wait_s)
    elif external:
        from .baselines import FrameworkRuntime

        runtime = FrameworkRuntime(
            pools, variant, batch_wait_s=args.baseline_batch_wait_s
        )
        config = runtime.config
    else:
        runtime = HeraSys(pools, config, profiles=profiles)
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    splits = []
    async with runtime:
        print(
            json.dumps(
                {
                    "event": "case_start",
                    "workload": workload,
                    "variant": variant,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "requests": args.requests,
                }
            ),
            flush=True,
        )

        async def request(i, sample):
            async with semaphore:
                arrived = time.perf_counter()

                def factory(load=None):
                    dag, inputs = build(
                        workload, sample, runtime, args, i, admission_load=load
                    )
                    splits.append(dag.metadata["splits"])
                    return dag, inputs

                options = dict(
                    query_id=f"q{i}",
                    timeout=args.timeout,
                    namespace=f"{workload}:{repeat}",
                    submitted_at=arrived,
                )
                try:
                    if runtime.config.adaptive_splitting and hasattr(
                        runtime, "submit_factory"
                    ):
                        answer = await runtime.submit_factory(factory, **options)
                    else:
                        dag, inputs = factory()
                        answer = await runtime.submit(dag, inputs, **options)
                    results.append({"query_id": f"q{i}", "output": answer})
                except Exception as exc:
                    results.append(
                        {"query_id": f"q{i}", "error": f"{type(exc).__name__}: {exc}"}
                    )
                if len(results) % max(1, args.requests // 10) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "case_progress",
                                "workload": workload,
                                "variant": variant,
                                "concurrency": concurrency,
                                "repeat": repeat,
                                "returned": len(results),
                                "elapsed_s": time.perf_counter() - start,
                            }
                        ),
                        flush=True,
                    )

        start = time.perf_counter()
        await drive_requests(request, items, concurrency, args.arrival_mode)
        elapsed = time.perf_counter() - start
    cleanup = []
    if args.backend == "gpu":
        cleanup = await asyncio.gather(
            *(
                worker.actor.ready.remote()
                for pool in pools.values()
                if pool.supports_state_fork
                for worker in pool.workers
            )
        )
        if any(
            state["live_states"] or state["kv_free_blocks"] != state["kv_total_blocks"]
            for state in cleanup
        ):
            raise RuntimeError(f"LLM state or KV blocks leaked after case: {cleanup}")
    completed = [r for r in runtime.query_records if r["status"] == "completed"]
    latencies = [r["latency_s"] for r in completed]
    summary = {
        "backend": args.backend,
        "measurement_kind": "diagnostic" if args.audit_nodes else "performance",
        "dataset": args.dataset or "builtin_fixture",
        "web_dataset": args.web_dataset,
        "search_provider": args.search_provider,
        "vector_backend": args.vector_backend,
        "workload": workload,
        "arrival_mode": args.arrival_mode,
        "variant": variant,
        "concurrency": concurrency,
        "repeat": repeat,
        "requests": args.requests,
        "effective_duplicate_rate": 1 - len(identities) / len(items),
        "completed": len(completed),
        "effective_concurrency_limit": min(concurrency, args.requests),
        "failed": args.requests - len(completed),
        "elapsed_s": elapsed,
        "qps": len(completed) / elapsed,
        "mean_s": statistics.fmean(latencies) if latencies else None,
        "p50_s": percentile(latencies, 0.5),
        "p95_s": percentile(latencies, 0.95),
        "p99_s": percentile(latencies, 0.99),
        "max_s": max(latencies, default=None),
        "slo_s": args.slo,
        "goodput_qps": sum(t <= args.slo for t in latencies) / elapsed,
        "mean_logical_nodes": statistics.fmean(
            r["logical_nodes"] for r in runtime.query_records
        )
        if runtime.query_records
        else None,
        "mean_splits": statistics.fmean(splits) if splits else None,
        "mean_runtime_splits": statistics.fmean(
            e["splits"] for e in runtime.events if e["event"] == "pipeline_split"
        )
        if any(e["event"] == "pipeline_split" for e in runtime.events)
        else None,
        "reused_nodes": sum(r["reused_nodes"] for r in runtime.query_records),
        "executed_nodes": sum(
            e.get("batch_size", 0) for e in runtime.events if e["event"] == "dispatch"
        ),
        "config": asdict(config),
        "tail_sample_warning": args.requests < 1000,
        "paper_reproduction": False,
        "scheduler_overhead_s": dict(getattr(runtime, "timings", {})),
        "backend_cleanup": cleanup,
    }
    summary["source_id"] = getattr(args, "source_id", None)
    name = f"{args.backend}_{workload}_{variant}_c{concurrency}_r{repeat}"
    out = Path(args.output) / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "queries.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in runtime.query_records)
    )
    (out / "outputs.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in sorted(results, key=lambda x: int(x["query_id"][1:]))
        )
    )
    (out / "trace.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in runtime.events)
    )
    if args.audit_nodes:
        (out / "node_audit.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in runtime.node_audits)
        )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "workload",
                    "variant",
                    "concurrency",
                    "completed",
                    "failed",
                    "mean_s",
                    "p99_s",
                    "qps",
                    "reused_nodes",
                ]
            }
        ),
        flush=True,
    )
    if summary["failed"]:
        raise RuntimeError(
            f"{summary['failed']} requests failed; inspect {out}/outputs.jsonl"
        )
    return summary


async def prepare_profiles(pools, args, records, web_records, profiles, *, collect):
    """Warm every workload/batch shape before collecting any cost samples.

    Cold kernel/tokenizer initialization must never become the monotone lower
    bound of a steady-state throughput curve. Calibration uses fixed graphs and
    FCFS so it does not depend on an as-yet uncalibrated HeraSys policy.
    """
    coverage = []
    rounds = getattr(args, "profile_rounds", 3) if collect else 0
    for phase in range(rounds + 1):
        measured = phase > 0
        for workload, warm_samples in warmup_batches(args, records, web_records):
            warm_config = replace(
                HeraConfig.variant("ayo_static"), trace=False
            )
            async with HeraSys(
                pools, warm_config, profiles=profiles, calibrate=measured
            ) as warm:
                futures = []
                for i, sample in enumerate(warm_samples):
                    dag, inputs = build(workload, sample, warm, args, i)
                    futures.append(warm.submit(dag, inputs, timeout=args.timeout))
                await asyncio.gather(*futures)
            row = {
                "workload": workload,
                "requests": len(warm_samples),
                "phase": "calibration" if measured else "warmup",
                "round": phase,
                "samples_collected": measured,
            }
            coverage.append(row)
            print(json.dumps({"event": "profile_batch_complete", **row}), flush=True)
    profiles.freeze()
    return coverage


async def main(args):
    if args.backend == "reference" and (
        args.vector_backend != "memory" or args.search_provider != "fixture"
    ):
        raise ValueError(
            "The reference backend supports only memory retrieval and fixture search"
        )
    if args.search_provider == "google" and any(
        not os.environ.get(key) for key in ("GOOGLE_API_KEY", "GOOGLE_CSE_ID")
    ):
        raise ValueError(
            "Google search requires GOOGLE_API_KEY and GOOGLE_CSE_ID in the environment"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_environment = environment(args)
    args.source_id = hashlib.sha256(
        json.dumps(run_environment["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    environment_path = output / "environment.json"
    if environment_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use a new output directory or --resume: {output}")
        previous_environment = json.loads(environment_path.read_text())
        ignored = {"output", "resume", "source_id"}
        previous_args = {
            k: v
            for k, v in previous_environment["arguments"].items()
            if k not in ignored
        }
        current_args = {k: v for k, v in vars(args).items() if k not in ignored}
        if (
            previous_args != current_args
            or previous_environment.get("source_sha256")
            != run_environment["source_sha256"]
            or previous_environment.get("dataset_sha256")
            != run_environment["dataset_sha256"]
        ):
            raise ValueError(
                "Resume requires identical code, datasets, and experiment arguments; use a new output directory"
            )
        environment_path = output / f"environment-resume-{time.time_ns()}.json"
    environment_path.write_text(json.dumps(run_environment, indent=2) + "\n")
    records = (
        [
            json.loads(line)
            for line in Path(args.dataset).read_text().splitlines()
            if line.strip()
        ]
        if args.dataset
        else FIXTURE
    )
    web_records = (
        [
            json.loads(line)
            for line in Path(args.web_dataset).read_text().splitlines()
            if line.strip()
        ]
        if args.web_dataset
        else None
    )
    if not records:
        raise ValueError("Dataset is empty")
    if any(
        not isinstance(record.get("question"), str)
        for record in records + (web_records or [])
    ):
        raise ValueError("Every dataset record needs a string question")
    if args.dataset and any(w != "web_search" for w in args.workloads):
        if any(
            not isinstance(record.get("documents"), list)
            or not record["documents"]
            or not all(isinstance(d, str) for d in record["documents"])
            for record in records
        ):
            raise ValueError(
                "RAG datasets must contain nonempty document lists; fixture fallback is not allowed"
            )
    actors = []
    try:
        if args.backend == "reference":
            pools = reference_pools()
        else:
            import ray

            from HeraSys.gpu import create_gpu_pools

            ray.init(include_dashboard=False, log_to_driver=False, num_cpus=16)
            pools, actors = await create_gpu_pools(
                args.embedding_model,
                args.reranker_model,
                args.llm_model,
                llm_instances=args.llm_instances,
                model_instances=args.model_instances,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                vector_backend=args.vector_backend,
                aux_gpu_share=args.aux_gpu_share,
                llm_max_sessions=args.llm_max_sessions,
                prefill_token_budget=args.prefill_token_budget,
                reranker_dtype=args.reranker_dtype,
            )
            readiness = await asyncio.gather(
                *(actor.ready.remote() for actor in actors)
            )
            (output / "backend_manifest.json").write_text(
                json.dumps(readiness, indent=2) + "\n"
            )
        # Warm kernels and model caches without preserving HeraSys's result cache.
        profile_path = args.profiles or (
            str(output / "profiles.json")
            if args.resume and (output / "profiles.json").exists()
            else None
        )
        profiles = (
            ProfileLibrary.load(profile_path) if profile_path else ProfileLibrary()
        )
        calibration_coverage = await prepare_profiles(
            pools, args, records, web_records, profiles, collect=not bool(profile_path)
        )
        profiles.save(output / "profiles.json")
        (output / "calibration_coverage.json").write_text(
            json.dumps(calibration_coverage, indent=2) + "\n"
        )
        summaries = []
        for repeat in range(args.repeats):
            for workload in args.workloads:
                for concurrency in args.concurrency:
                    variants = list(args.variants)
                    random.Random(args.seed + repeat).shuffle(variants)
                    for variant in variants:
                        case_path = (
                            output
                            / f"{args.backend}_{workload}_{variant}_c{concurrency}_r{repeat}"
                            / "summary.json"
                        )
                        if args.resume and case_path.exists():
                            previous = json.loads(case_path.read_text())
                            if (
                                previous["failed"] == 0
                                and previous["requests"] == args.requests
                            ):
                                summaries.append(previous)
                                print(
                                    f"Resumed completed case: {case_path.parent.name}",
                                    flush=True,
                                )
                                continue
                            raise ValueError(
                                f"Existing incomplete/incompatible case needs a new output directory: {case_path}"
                            )
                        summaries.append(
                            await run_case(
                                pools,
                                args,
                                records,
                                workload,
                                variant,
                                concurrency,
                                repeat,
                                profiles,
                                web_records,
                            )
                        )
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    finally:
        if args.backend == "gpu":
            import ray

            for actor in actors:
                ray.kill(actor)
            ray.shutdown()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["reference", "gpu"], default="reference")
    p.add_argument(
        "--workloads",
        nargs="+",
        choices=["rag", "rag_rerank", "web_search", "mixed"],
        default=["rag_rerank"],
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "HeraSys",
            "herasys",
            "herasys_unaged",
            "herasys_legacy",
            "no_joint_allocation",
            "no_tail_feedback",
            "no_pipeline",
            "herasys_transitive",
            "herasys_direct",
            "herasys_aged",
            "herasys_direct_aged",
            "herasys_legacy_direct_aged",
            "herasys_node_aged",
            "herasys_aged_no_fusion",
            "herasys_direct_aged_no_fusion",
            "ayo_static",
            "no_fusion",
            "fcfs",
            "srtf",
            "no_skew",
            "no_adaptive",
            "langgraph",
            "llamaindex",
            "ayo_upstream_graph",
        ],
        default=["HeraSys"],
    )
    p.add_argument("--concurrency", nargs="+", type=int, default=[6])
    p.add_argument(
        "--arrival-mode",
        choices=["closed_loop", "waves"],
        default="closed_loop",
        help="Continuously refill concurrency slots, or drain each concurrent wave before the next",
    )
    p.add_argument("--requests", type=int, default=24)
    p.add_argument(
        "--pipeline-mode", choices=["dynamic", "admission"], default="dynamic"
    )
    p.add_argument("--baseline-batch-wait-s", type=float, default=0.005)
    for name in ("alpha", "gamma", "delta", "rho", "load-scale", "max-batch-wait-s"):
        p.add_argument(
            "--" + name,
            type=float,
            help="Override the corresponding scheduler parameter; persisted in every case",
        )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duplicate-rate", type=float, default=0.1)
    p.add_argument("--cache-entries", type=int, default=4096)
    p.add_argument("--cache-ttl-s", type=float, default=300.0)
    p.add_argument("--dataset", help="JSONL records with question and documents")
    p.add_argument(
        "--web-dataset",
        help="Separate TruthfulQA question JSONL for web and mixed workflows",
    )
    p.add_argument(
        "--profiles",
        help="Previously measured profile JSON; otherwise calibrate after a separate unmeasured warmup",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--profile-rounds", type=int, default=3,
        help="Measured calibration repetitions after the unmeasured warmup pass",
    )
    p.add_argument("--warmup-concurrency", type=int, default=8)
    p.add_argument(
        "--profile-batch-sizes",
        nargs="+",
        type=int,
        help="Warmup client concurrency grid; actual physical batch sizes are recorded in profiles.json",
    )
    p.add_argument(
        "--audit-nodes",
        action="store_true",
        help="Record intermediate hashes and exact LLM prompts/tokens; diagnostic only",
    )
    p.add_argument("--vector-backend", choices=["memory", "pgvector"], default="memory")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed cases; record a separate environment manifest",
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--decode-quantum", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--slo", type=float, default=60)
    p.add_argument(
        "--search-provider", choices=["fixture", "google"], default="fixture"
    )
    p.add_argument("--embedding-model", default="/disk_n/lsz/model/bge-large-en-v1.5")
    p.add_argument("--reranker-model", default="/disk_n/lsz/model/bge-reranker-large")
    p.add_argument(
        "--reranker-dtype", choices=["float16", "float32"], default="float32"
    )
    p.add_argument("--llm-model", default="/disk_n/lsz/model/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--llm-instances", type=int, default=2)
    p.add_argument("--llm-max-sessions", type=int, default=16)
    p.add_argument("--prefill-token-budget", type=int, default=8192)
    p.add_argument("--model-instances", type=int, default=3)
    p.add_argument(
        "--aux-gpu-share",
        type=float,
        help="GPU resource fraction per embedding/reranking actor; default 1/model-instances",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if (
        args.requests < 1
        or args.repeats < 1
        or min(args.concurrency) < 1
        or not 0 <= args.duplicate_rate <= 1
        or args.cache_entries < 0
        or args.cache_ttl_s < 0
        or min(
            args.max_tokens,
            args.decode_quantum,
            args.max_model_len,
            args.llm_instances,
            args.llm_max_sessions,
            args.prefill_token_budget,
            args.model_instances,
            args.warmup_concurrency,
            args.profile_rounds,
        )
        < 1
        or args.warmup < 0
        or (args.profile_batch_sizes and min(args.profile_batch_sizes) < 1)
        or args.timeout <= 0
        or args.slo <= 0
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        raise SystemExit(
            "Requests/repeats/concurrency must be positive and duplicate rate in [0,1]"
        )
    if args.audit_nodes and any(
        v in {"langgraph", "llamaindex", "ayo_upstream_graph"} for v in args.variants
    ):
        raise SystemExit(
            "Node auditing currently supports HeraSys/shared-engine variants"
        )
    asyncio.run(main(args))
