"""Deterministic reference operators and adapters for real batched Ray engines."""

import asyncio
import hashlib
import math
import time


def first_output(request, value):
    return {request["outputs"][0]: value}


def aggregate(request):
    values = list(request["inputs"].values())
    mode = request["config"].get("mode", "identity")
    if mode == "pipeline_gather":
        return {
            output: [row for port in ports for row in request["inputs"][port]]
            for output, ports in request["config"]["gather_ports"].items()
        }
    if mode == "indexes":
        result = [part for value in values for part in value]
    elif mode == "identity":
        result = values[0]
    else:
        raise ValueError(f"Unknown aggregation mode {mode}")
    return first_output(request, result)


def memory_index(request):
    values = list(request["inputs"].values())
    documents = next(
        v for v in values if isinstance(v, list) and v and isinstance(v[0], str)
    )
    vectors = next(
        v
        for v in values
        if len(v) and (isinstance(v[0], (list, tuple)) or getattr(v, "ndim", 0) == 2)
    )
    if len(documents) != len(vectors):
        raise ValueError("Document/vector count mismatch")
    return first_output(
        request, [{"text": t, "vector": v} for t, v in zip(documents, vectors)]
    )


def memory_search(request):
    inputs, config = request["inputs"], request["config"]
    query, entries = inputs["query_vector"], inputs["index"]
    scored = [
        (
            sum(float(a) * float(b) for a, b in zip(query, entry["vector"])),
            i,
            entry["text"],
        )
        for i, entry in enumerate(entries)
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return first_output(request, [t for _, _, t in scored[: config.get("top_k", 12)]])


class ReferenceWorker:
    """CPU correctness backend. It is explicitly not an LLM or a performance model."""

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.calls = []

    async def execute_batch(self, requests):
        if any(r["config"].get("map_ports") for r in requests):
            from .pipeline import collect_mapped, expand_mapped

            expanded, layout = expand_mapped(requests)
            results = await self.execute_batch(expanded) if expanded else []
            return collect_mapped(results, layout)
        self.calls.append(requests)
        started = time.perf_counter()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        outputs = [self._execute(request) for request in requests]
        service = (time.perf_counter() - started) / len(requests)
        return [{"output": output, "service_s": service} for output in outputs]

    def _execute(self, r):
        op, inputs, config = r["op"], r["inputs"], r["config"]
        if op == "embedding":
            texts = next(iter(inputs.values()))
            single = isinstance(texts, str)
            texts = [texts] if single else texts
            vectors = []
            for text in texts:
                vector = [0.0] * 32
                for token in text.lower().split():
                    value = int.from_bytes(
                        hashlib.sha256(token.encode()).digest()[:4], "little"
                    )
                    vector[value % 32] += 1
                norm = math.sqrt(sum(x * x for x in vector)) or 1
                vectors.append([x / norm for x in vector])
            return first_output(r, vectors[0] if single else vectors)
        if op == "vectordb_ingestion":
            return memory_index(r)
        if op == "vectordb_searching":
            return memory_search(r)
        if op == "aggregator":
            return aggregate(r)
        if op == "reranking":
            question = set(inputs["question"].lower().split())
            contexts = inputs["contexts"]
            order = sorted(
                range(len(contexts)),
                key=lambda i: (
                    -len(question.intersection(contexts[i].lower().split())),
                    i,
                ),
            )
            return first_output(
                r, [contexts[i] for i in order[: config.get("top_k", 8)]]
            )
        if op == "llm_prefilling":
            question = inputs[config["question_port"]]
            return first_output(r, {"text": question, "tokens": 0, "done": False})
        if op in {"llm_parallel_decoding", "llm_decoding"}:
            state = dict(next(iter(inputs.values())))
            state["tokens"] += config["token_budget"]
            return first_output(
                r, f"[reference] {state['text']}" if config["final"] else state
            )
        if op == "web_search":
            if config["provider"] != "fixture":
                raise ValueError("ReferenceWorker only supports fixture search")
            return first_output(
                r, ["Offline search fixture: " + inputs["search_query"]]
            )
        raise ValueError(f"Unsupported reference operation: {op}")


class RayWorker:
    def __init__(self, actor, *, defer_releases=False):
        self.actor = actor
        self.defer_releases = defer_releases
        self.pending_releases = []

    async def execute_batch(self, requests):
        if self.pending_releases:
            pending, self.pending_releases = self.pending_releases, []
            try:
                return await self.actor.execute_batch.remote(requests, releases=pending)
            except BaseException:
                self.pending_releases = pending + self.pending_releases
                raise
        return await self.actor.execute_batch.remote(requests)

    async def release(self, outputs):
        # Only KV handles own backend resources. Never serialize embeddings,
        # passages, or generated strings just to call a no-op actor method.
        handles = [
            {
                key: value
                for key, value in output.items()
                if isinstance(value, dict) and "state_key" in value
            }
            for output in outputs
        ]
        handles = [output for output in handles if output]
        if self.defer_releases:
            self.pending_releases.extend(handles)
        elif handles:
            await self.actor.release.remote(handles)

    async def flush_releases(self):
        if self.pending_releases:
            pending, self.pending_releases = self.pending_releases, []
            try:
                await self.actor.release.remote(pending)
            except BaseException:
                self.pending_releases = pending + self.pending_releases
                raise


def reference_pools(instances=2, delay_s=0.0):
    from HeraSys.engines.engine_types import EngineType

    from .runtime import ResourcePool

    return {
        engine: ResourcePool(
            [ReferenceWorker(delay_s) for _ in range(instances)],
            max_batch_size=8,
            supports_state_fork=True,
        )
        for engine in [
            EngineType.EMBEDDER,
            EngineType.VECTOR_DB,
            EngineType.RERANKER,
            EngineType.AGGREGATOR,
            EngineType.LLM,
            EngineType.DUMMY,
        ]
    }
