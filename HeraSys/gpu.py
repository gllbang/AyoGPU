"""Real engines for the vendored HeraSys/vLLM API, wrapped by Ray actors.

Decode chunks are actual engine.step() boundaries. Paused groups keep their KV
blocks; a fork uses vLLM's block-manager reference counts and copy-on-write.
No background decoder runs while a logical decode node waits in HeraSys.
"""

import itertools
import time
from copy import deepcopy

from .backends import RayWorker, aggregate, first_output, memory_index, memory_search


class GPUWorker:
    def __init__(
        self,
        kind,
        model=None,
        *,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        max_batch_size=16,
        prefill_token_budget=8192,
        reranker_dtype="float32",
    ):
        import torch

        torch.set_num_threads(2)
        self.kind, self.model_name = kind, model
        self.states = {}
        self.counter = itertools.count()
        if kind == "embedding":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model, device="cuda")
            self.model.half()
        elif kind == "reranking":
            from sentence_transformers import CrossEncoder

            if reranker_dtype not in {"float16", "float32"}:
                raise ValueError("reranker_dtype must be float16 or float32")
            self.model = CrossEncoder(model, device="cuda", max_length=512)
            self.model.model.to(dtype=getattr(torch, reranker_dtype))
        elif kind == "llm":
            from .vllm_compat import install

            install()
            import vllm
            from vllm import EngineArgs, LLMEngine

            if not vllm.__version__.startswith("0.2.2"):
                raise RuntimeError(
                    "This KV-fork backend requires HeraSys's vendored vLLM 0.2.2 API"
                )
            self.engine = LLMEngine.from_engine_args(
                EngineArgs(
                    model=model,
                    max_model_len=max_model_len,
                    dtype="half",
                    max_num_seqs=max_batch_size,
                    max_num_batched_tokens=max(max_model_len, prefill_token_budget),
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=0,
                    disable_log_stats=True,
                )
            )
        elif kind not in {"vector", "aggregate", "tool"}:
            raise ValueError(f"Unknown backend kind {kind}")

    def ready(self):
        result = {
            "kind": self.kind,
            "model": self.model_name,
            "live_states": len(self.states),
        }
        if self.kind == "llm":
            result.update(
                kv_total_blocks=self.engine.cache_config.num_gpu_blocks,
                kv_free_blocks=self.engine.scheduler.block_manager.get_num_free_gpu_blocks(),
                kv_block_size=self.engine.cache_config.block_size,
                kv_watermark_blocks=self.engine.scheduler.block_manager.watermark_blocks,
                prefill_token_budget=self.engine.scheduler_config.max_num_batched_tokens,
            )
        if self.kind in {"llm", "embedding", "reranking"}:
            import os

            import torch

            result["cuda_visible_devices"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES", "all"
            )
            result["gpu_uuid"] = str(
                getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")
            )
        if self.kind in {"embedding", "reranking"}:
            module = self.model.model if self.kind == "reranking" else self.model
            result["model_dtype"] = str(next(module.parameters()).dtype)
        return result

    def _sync(self):
        if self.kind in {"embedding", "reranking", "llm"}:
            import torch

            torch.cuda.synchronize()

    def execute_batch(self, requests, *, releases=None):
        if any(r["config"].get("map_ports") for r in requests):
            from .pipeline import collect_mapped, expand_mapped

            expanded, layout = expand_mapped(requests)
            results = (
                self.execute_batch(expanded, releases=releases) if expanded else []
            )
            return collect_mapped(results, layout)
        if releases:
            self.release(releases)
        for request in requests:
            expected = request["config"].get("model")
            if expected and self.model_name and expected != self.model_name:
                raise ValueError(
                    f"Requested model {expected} differs from loaded model {self.model_name}"
                )
        self._sync()
        start = time.perf_counter()
        if self.kind == "embedding":
            all_texts, segments = [], []
            for r in requests:
                texts = next(iter(r["inputs"].values()))
                single = isinstance(texts, str)
                texts = [texts] if single else texts
                if r["config"].get("role") == "query":
                    texts = [
                        "Represent this sentence for searching relevant passages: " + t
                        for t in texts
                    ]
                segments.append((len(all_texts), len(texts), single))
                all_texts.extend(texts)
            vectors = self.model.encode(
                all_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            outputs = [
                first_output(
                    r, vectors[start] if single else vectors[start : start + size]
                )
                for r, (start, size, single) in zip(requests, segments)
            ]
        elif self.kind == "reranking":
            import torch

            pairs, segments = [], []
            for r in requests:
                contexts = r["inputs"]["contexts"]
                segments.append((len(pairs), len(contexts)))
                pairs.extend((r["inputs"]["question"], text) for text in contexts)
            scores = (
                self.model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                    # Ranking needs logits. Half-precision sigmoid collapses
                    # distinct high scores into ties and changes top-k results.
                    activation_fct=torch.nn.Identity(),
                ).tolist()
                if pairs
                else []
            )
            outputs = []
            for r, (offset, size) in zip(requests, segments):
                order = sorted(range(size), key=lambda i: (-scores[offset + i], i))
                outputs.append(
                    first_output(
                        r,
                        [
                            r["inputs"]["contexts"][i]
                            for i in order[: r["config"]["top_k"]]
                        ],
                    )
                )
        elif self.kind == "llm":
            outputs = self._llm_batch(requests)
        else:
            outputs = [self._cpu(r) for r in requests]
        self._sync()
        elapsed = time.perf_counter() - start
        # Each co-executed logical node attains the measured batch wall service time.
        results = [{"output": output, "service_s": elapsed} for output in outputs]
        if self.kind == "llm":
            for result, audit in zip(results, self._batch_audits):
                if audit is not None:
                    result["audit"] = audit
        return results

    def _cpu(self, request):
        op = request["op"]
        if op == "aggregator":
            return aggregate(request)
        if op == "vectordb_ingestion":
            return memory_index(request)
        if op == "vectordb_searching":
            return memory_search(request)
        if op == "web_search":
            if request["config"]["provider"] == "fixture":
                return first_output(
                    request,
                    ["Offline fixture for " + request["inputs"]["search_query"]],
                )
            import os

            import requests

            try:
                response = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": os.environ["GOOGLE_API_KEY"],
                        "cx": os.environ["GOOGLE_CSE_ID"],
                        "q": request["inputs"]["search_query"],
                        "num": request["config"]["top_k"],
                    },
                    timeout=20,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                status = getattr(exc.response, "status_code", "network error")
                # RequestException URLs may contain the API key in their query string.
                raise RuntimeError(f"Google Search request failed: {status}") from None
            return first_output(
                request,
                [item.get("snippet", "") for item in response.json().get("items", [])],
            )
        raise ValueError(f"Unsupported CPU operation {op}")

    def _detach(self, group):
        for queue in (
            self.engine.scheduler.running,
            self.engine.scheduler.waiting,
            self.engine.scheduler.swapped,
        ):
            if group in queue:
                queue.remove(group)

    def _free(self, group):
        if group is not None:
            self._detach(group)
            for seq in group.get_seqs():
                if not seq.is_finished():
                    self.engine.scheduler.free_seq(seq)

    def _fork(self, state):
        from vllm.sequence import SequenceGroup

        parent = state["group"]
        seqs = []
        for seq in parent.get_seqs():
            child = seq.fork(next(self.engine.seq_counter))
            self.engine.scheduler.fork_seq(seq, child)
            seqs.append(child)
        return SequenceGroup(
            f"hera_{next(self.counter)}",
            seqs,
            deepcopy(parent.sampling_params),
            time.monotonic(),
        )

    def _llm_batch(self, requests):
        from vllm import SamplingParams

        prefill = requests[0]["op"] == "llm_prefilling"
        pending, finished = {}, {}
        prepared = {}
        ordered = list(enumerate(requests))
        try:
            if prefill:
                # vLLM 0.2 checks padded tokens in waiting-queue order. Long/short
                # alternation fragments one logical batch into singleton kernels.
                # Reorder only within this committed batch, retaining original
                # indices for state handles and outputs. No extra batching wait.
                for i, request in ordered:
                    config = request["config"]
                    context = request["inputs"][config["context_port"]]
                    context = (
                        "\n".join(context)
                        if isinstance(context, list)
                        else str(context)
                    )
                    prompt = config["prompt_template"].format(
                        question=request["inputs"][config["question_port"]],
                        context=context,
                    )
                    token_ids = self.engine.tokenizer.encode(prompt)
                    limit = (
                        self.engine.model_config.max_model_len - config["max_tokens"]
                    )
                    if limit < 1:
                        raise ValueError(
                            "max_tokens must be smaller than max_model_len"
                        )
                    if len(token_ids) > limit:
                        raise ValueError(
                            f"Prompt ({len(token_ids)} tokens) exceeds input limit {limit}; truncate dataset explicitly"
                        )
                    prepared[i] = (prompt, token_ids)
                ordered.sort(key=lambda item: len(prepared[item[0]][1]))
            for i, request in ordered:
                config = request["config"]
                if prefill:
                    prompt, token_ids = prepared[i]
                    key = f"hera_{next(self.counter)}"
                    self.engine.add_request(
                        key,
                        prompt,
                        SamplingParams(
                            temperature=config.get("temperature", 0),
                            max_tokens=config["max_tokens"],
                        ),
                        prompt_token_ids=token_ids,
                    )
                    group = self.engine.scheduler.waiting[-1]
                    pending[key] = (i, group, 1, 0)
                else:
                    handle = next(iter(request["inputs"].values()))
                    state = self.states[handle["state_key"]]
                    consumed = state["consumed"] + config["token_budget"]
                    if state["done"] or len(state["token_ids"]) >= consumed:
                        finished[i] = {**state, "consumed": consumed, "group": None}
                        if not state["done"]:
                            group = self._fork(state)
                            finished[i]["group"] = group
                    else:
                        group = self._fork(state)
                        self.engine.scheduler.running.append(group)
                        pending[group.request_id] = (i, group, consumed, consumed)
            stalls = 0
            while pending:
                outputs = self.engine.step()
                stalls = stalls + 1 if not outputs else 0
                if stalls > 2:
                    free = self.engine.scheduler.block_manager.get_num_free_gpu_blocks()
                    raise RuntimeError(
                        f"vLLM cannot advance: pending={len(pending)}, "
                        f"live_states={len(self.states)}, free_kv_blocks={free}, "
                        f"waiting={len(self.engine.scheduler.waiting)}, "
                        f"running={len(self.engine.scheduler.running)}"
                    )
                for output in outputs:
                    if output.request_id not in pending:
                        continue
                    i, group, target, consumed = pending[output.request_id]
                    completion = output.outputs[0]
                    if output.finished or len(completion.token_ids) >= target:
                        self._detach(group)
                        finished[i] = {
                            "group": None if output.finished else group,
                            "done": output.finished,
                            "text": completion.text,
                            "token_ids": list(completion.token_ids),
                            "consumed": consumed,
                        }
                        del pending[output.request_id]
            results = []
            self._batch_audits = []
            for i, request in enumerate(requests):
                state = finished[i]
                audit = None
                if request["config"].get("_hera_audit"):
                    audit = {
                        "token_ids": list(state["token_ids"]),
                        "done": state["done"],
                    }
                    if prefill:
                        prompt, token_ids = prepared[i]
                        audit.update(
                            prompt=prompt,
                            prompt_token_ids=token_ids,
                        )
                self._batch_audits.append(audit)
                if not prefill and request["config"]["final"]:
                    results.append(first_output(request, state["text"]))
                    self._free(state["group"])
                else:
                    key = f"state_{next(self.counter)}"
                    self.states[key] = state
                    handle = {"state_key": key}
                    if state["done"]:
                        handle.update(
                            _hera_finished=True,
                            text=state["text"],
                            model=self.model_name,
                        )
                    results.append(first_output(request, handle))
            return results
        except Exception:
            for _, group, _, _ in pending.values():
                self._free(group)
            for state in finished.values():
                self._free(state["group"])
            raise

    def release(self, outputs):
        for output in outputs:
            for value in output.values():
                if isinstance(value, dict) and "state_key" in value:
                    state = self.states.pop(value["state_key"], None)
                    if state:
                        self._free(state["group"])


async def create_gpu_pools(
    embedding_model,
    reranker_model,
    llm_model,
    *,
    llm_instances=2,
    model_instances=3,
    max_model_len=2048,
    gpu_memory_utilization=0.8,
    vector_backend="memory",
    aux_gpu_share=None,
    llm_max_sessions=16,
    prefill_token_budget=8192,
    mixed_prefill_batches=True,
    reranker_dtype="float32",
):
    """Two full GPU LLM workers, 3 embedding/reranker workers sharing two GPUs."""
    import ray

    from HeraSys.engines.engine_types import EngineType

    from .runtime import ResourcePool

    actor_class = ray.remote(GPUWorker)
    share = 1 / model_instances if aux_gpu_share is None else aux_gpu_share
    if not 0 < share <= 1:
        raise ValueError("aux_gpu_share must be in (0, 1]")
    required = llm_instances + 2 * model_instances * share
    if required > ray.cluster_resources().get("GPU", 0) + 1e-4:
        raise ValueError(
            f"Engine layout requires {required:.4f} GPUs; fewer are visible to Ray"
        )
    pools, actors, readiness = {}, [], []
    specs = [
        (EngineType.LLM, "llm", llm_model, llm_instances, 1),
        (
            EngineType.EMBEDDER,
            "embedding",
            embedding_model,
            model_instances,
            share,
        ),
        (
            EngineType.RERANKER,
            "reranking",
            reranker_model,
            model_instances,
            share,
        ),
        (EngineType.VECTOR_DB, "vector", None, 1, 0),
        (EngineType.AGGREGATOR, "aggregate", None, 1, 0),
        (EngineType.DUMMY, "tool", None, 1, 0),
    ]
    try:
        for engine, kind, model, count, gpu_share in specs:
            workers = []
            for _ in range(count):
                if kind == "vector" and vector_backend == "pgvector":
                    from .postgres import PgVectorWorker

                    actor = ray.remote(PgVectorWorker).options(num_cpus=1).remote()
                else:
                    actor = actor_class.options(num_gpus=gpu_share, num_cpus=1).remote(
                        kind,
                        model,
                        max_model_len=max_model_len,
                        prefill_token_budget=prefill_token_budget,
                        max_batch_size=max(16, llm_max_sessions)
                        if kind == "llm"
                        else 16,
                        gpu_memory_utilization=gpu_memory_utilization,
                        reranker_dtype=reranker_dtype,
                    )
                actors.append(actor)
                readiness.append(actor.ready.remote())
                workers.append(RayWorker(actor, defer_releases=kind == "llm"))
            pools[engine] = ResourcePool(
                workers,
                max_batch_size=max(16, llm_max_sessions) if kind == "llm" else 16,
                supports_state_fork=kind == "llm",
                max_sessions=llm_max_sessions if kind == "llm" else None,
                mixed_decode_batches=kind == "llm",
                mixed_prefill_batches=kind == "llm" and mixed_prefill_batches,
                batch_by_port_names=False,
            )
        reports = await asyncio_gather_refs(readiness)
        for report in reports:
            if report["kind"] == "llm":
                blocks_per_session = (
                    max_model_len + report["kv_block_size"] - 1
                ) // report["kv_block_size"] + 2
                usable_blocks = (
                    report["kv_total_blocks"] - report["kv_watermark_blocks"]
                )
                if llm_max_sessions * blocks_per_session > usable_blocks:
                    raise ValueError(
                        "Requested LLM sessions exceed the profiled KV capacity at "
                        "max_model_len (including copy-on-write margin); reduce "
                        "--llm-max-sessions or --max-model-len"
                    )
    except BaseException:
        for actor in actors:
            ray.kill(actor)
        raise
    return pools, actors


async def asyncio_gather_refs(refs):
    import asyncio

    return await asyncio.gather(*refs)
