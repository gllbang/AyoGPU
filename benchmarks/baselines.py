"""Actual LangGraph / LlamaIndex orchestration with the shared backend engines.

These adapters use independent per-query modular execution and FIFO engine
batching. They do not call HeraSys's graph fuser or priority scheduler.
"""

import asyncio
import importlib
import time
from collections import deque
from copy import deepcopy
from dataclasses import replace

from HeraSys.dags.node_commons import NodeOps, NodeType
from HeraSys import HeraConfig
from HeraSys.semantics import digest, parameters, prefill_batch_key


class FIFOEngines:
    def __init__(self, pools, events, *, batch_wait_s=0.005):
        if batch_wait_s < 0:
            raise ValueError("Batch wait must be nonnegative")
        self.batch_wait_s = batch_wait_s
        self.pools, self.events = pools, events
        self.queues = {
            key: [asyncio.Queue() for _ in pool.workers] for key, pool in pools.items()
        }
        self.sessions = {
            key: [asyncio.Semaphore(pool.max_sessions or 100000) for _ in pool.workers]
            for key, pool in pools.items()
        }
        self.loops = []
        self.affinities = {}
        self.outputs = {}
        self.cursor = dict.fromkeys(pools, 0)
        for key, pool in pools.items():
            for slot, worker in enumerate(pool.workers):
                self.loops.append(asyncio.create_task(self._loop(key, slot, worker)))

    async def execute(self, query_id, node, inputs):
        engine = node.engine_type
        session = node.config.get("_hera_session")
        identity = (query_id, session)
        if session and identity in self.affinities:
            slot = self.affinities[identity]
        else:
            slot = self.cursor[engine] % len(self.pools[engine].workers)
            self.cursor[engine] += 1
            if session:
                await self.sessions[engine][slot].acquire()
                self.affinities[identity] = slot
        request = {
            "op": node.op_type.value
            if hasattr(node.op_type, "value")
            else node.op_type,
            "config": node.config,
            "inputs": inputs,
            "outputs": node.output_names,
        }
        key = digest(
            [
                request["op"],
                parameters(node.config),
                list(inputs) if self.pools[engine].batch_by_port_names else len(inputs),
                node.output_names
                if self.pools[engine].batch_by_port_names
                else len(node.output_names),
            ]
        )
        if (
            self.pools[engine].mixed_prefill_batches
            and node.op_type == NodeOps.LLM_PREFILLING
        ):
            key = prefill_batch_key(node.config)
        if self.pools[engine].mixed_decode_batches and node.op_type in {
            NodeOps.LLM_DECODING,
            NodeOps.LLM_PARTIAL_DECODING,
        }:
            key = digest(["decode", node.config.get("model")])
        future = asyncio.get_running_loop().create_future()
        await self.queues[engine][slot].put((key, request, future, time.perf_counter()))
        result = await future
        self.outputs.setdefault(query_id, []).append((engine, slot, result["output"]))
        worker = self.pools[engine].workers[slot]
        if hasattr(worker, "release"):
            await worker.release([inputs])
        if session and node.config.get("final"):
            self.sessions[engine][slot].release()
            self.affinities.pop(identity)
        return result

    async def cleanup(self, query_id):
        for engine, slot, output in self.outputs.pop(query_id, []):
            worker = self.pools[engine].workers[slot]
            if hasattr(worker, "release"):
                await worker.release([output])
        for (query, session), slot in list(self.affinities.items()):
            if query == query_id:
                # Session-affine operations are LLM tasks in these workloads.
                from HeraSys.engines.engine_types import EngineType

                self.sessions[EngineType.LLM][slot].release()
                del self.affinities[(query, session)]

    async def _loop(self, engine, slot, worker):
        queue, pool = self.queues[engine][slot], self.pools[engine]
        pending = deque()
        while True:
            first = pending.popleft() if pending else await queue.get()
            batch = [first]
            # The oldest request chooses the batch class. Incompatible requests
            # retain their FIFO order and original deadline while compatible
            # arrivals may join the batch during a bounded wait.
            for _ in range(len(pending)):
                item = pending.popleft()
                if item[0] == first[0] and len(batch) < pool.max_batch_size:
                    batch.append(item)
                else:
                    pending.append(item)
            deadline = first[3] + self.batch_wait_s
            while len(batch) < pool.max_batch_size:
                if not queue.empty():
                    item = queue.get_nowait()
                else:
                    delay = deadline - time.perf_counter()
                    if delay <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), delay)
                    except asyncio.TimeoutError:
                        break
                if item[0] == first[0]:
                    batch.append(item)
                else:
                    pending.append(item)
            self.events.append(
                {
                    "event": "dispatch",
                    "engine": str(engine),
                    "slot": slot,
                    "batch_size": len(batch),
                    "batch_wait_s": self.batch_wait_s,
                    "queue_wait_s": [time.perf_counter() - item[3] for item in batch],
                    "time_s": time.perf_counter(),
                }
            )
            try:
                results = await worker.execute_batch(
                    [request for _, request, _, _ in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError("Backend result count mismatch")
                for (_, _, future, _), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
                    elif hasattr(worker, "release"):
                        await worker.release([result["output"]])
            except Exception as exc:
                for _, _, future, _ in batch:
                    if not future.done():
                        future.set_exception(exc)

    async def close(self):
        for task in self.loops:
            task.cancel()
        await asyncio.gather(*self.loops, return_exceptions=True)
        await asyncio.gather(
            *(
                worker.flush_releases()
                for pool in self.pools.values()
                for worker in pool.workers
                if hasattr(worker, "flush_releases")
            )
        )


class FrameworkRuntime:
    def __init__(self, pools, framework, *, batch_wait_s=0.005):
        if framework not in {"langgraph", "llamaindex"}:
            raise ValueError(framework)
        self.framework, self.pools = framework, pools
        self.batch_wait_s = batch_wait_s
        # Exclude first import / registry construction from serving latency.
        self.framework_api = importlib.import_module(
            "langgraph.graph"
            if framework == "langgraph"
            else "llama_index.core.workflow"
        )
        self.config = replace(HeraConfig.variant("ayo_static"), fixed_splits=1)
        self.events, self.query_records = [], []
        self.queries, self.tasks = {}, set()
        self.dispatcher = None

    def submit(
        self, dag, inputs, *, query_id, timeout=300, submitted_at=None, **kwargs
    ):
        started = submitted_at or time.perf_counter()
        if self.dispatcher is None:
            self.dispatcher = FIFOEngines(
                self.pools, self.events, batch_wait_s=self.batch_wait_s
            )
        nodes = [
            n
            for n in deepcopy(dag).topological_sort()
            if n.node_type == NodeType.COMPUTE
        ]
        outputs = {}
        service = 0.0

        async def execute(index):
            nonlocal service
            node = nodes[index]
            values = {}
            for port in node.input_names:
                parents = [
                    p
                    for p in node.parents
                    if port in p.output_names and p.node_type == NodeType.COMPUTE
                ]
                values[port] = (
                    outputs[parents[0].name][port] if parents else inputs[port]
                )
            result = await self.dispatcher.execute(query_id, node, values)
            service += result["service_s"]
            outputs[node.name] = result["output"]

        async def run():
            status = "completed"
            try:
                if self.framework == "langgraph":
                    from langgraph.graph import END, START, StateGraph

                    graph = StateGraph(dict)
                    for i, node in enumerate(nodes):

                        async def step(state, index=i):
                            await execute(index)
                            return {"step": index}

                        graph.add_node(node.name, step)
                        graph.add_edge(
                            START if i == 0 else nodes[i - 1].name, node.name
                        )
                    graph.add_edge(nodes[-1].name, END)
                    await graph.compile().ainvoke(
                        {}, {"recursion_limit": len(nodes) + 5}
                    )
                else:
                    from llama_index.core.workflow import (
                        Event,
                        StartEvent,
                        StopEvent,
                        Workflow,
                        step,
                    )

                    class NextNode(Event):
                        index: int

                    class QueryWorkflow(Workflow):
                        @step
                        async def start(self, event: StartEvent) -> NextNode:
                            return NextNode(index=0)

                        @step
                        async def advance(
                            self, event: NextNode
                        ) -> NextNode | StopEvent:
                            await execute(event.index)
                            if event.index + 1 == len(nodes):
                                return StopEvent(result=True)
                            return NextNode(index=event.index + 1)

                    await QueryWorkflow(timeout=timeout).run()
                return {
                    key: value
                    for node in nodes
                    if not any(c.node_type == NodeType.COMPUTE for c in node.children)
                    for key, value in outputs[node.name].items()
                }
            except BaseException:
                status = "failed"
                raise
            finally:
                await self.dispatcher.cleanup(query_id)
                self.query_records.append(
                    {
                        "query_id": query_id,
                        "status": status,
                        "latency_s": time.perf_counter() - started,
                        "attained_service_s": service,
                        "logical_nodes": len(nodes),
                        "reused_nodes": 0,
                    }
                )
                self.queries.pop(query_id, None)

        task = asyncio.create_task(asyncio.wait_for(run(), timeout))
        self.queries[query_id] = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.dispatcher:
            await self.dispatcher.close()
