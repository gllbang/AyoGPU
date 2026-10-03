"""Original Ayo graph scheduler with an explicit shared-engine adapter.

The scheduling source is read from the immutable repository commit below. Only
payload serialization is adapted; this is labelled ayo_upstream_graph, since the
original APP frontend and original engine-specific queues are not being timed.
"""

import asyncio
import subprocess
import time
import types
from copy import deepcopy
from functools import partial

import ray

from HeraSys.dags.node_commons import NodeType
from HeraSys import HeraConfig
from HeraSys.queries.query_state import QueryStates, QueryStatus
from HeraSys.schedulers.engine_scheduler import EngineRequest

from .baselines import FIFOEngines

UPSTREAM_COMMIT = "3748fc4"


@ray.remote(num_cpus=1)
class GraphActor:
    def __init__(self, source, bridge):
        module = types.ModuleType("ayo_upstream_graph")
        source = source.replace("from Ayo.", "from HeraSys.")
        exec(
            compile(
                source, f"{UPSTREAM_COMMIT}:Ayo/schedulers/graph_scheduler.py", "exec"
            ),
            module.__dict__,
        )
        module.EngineRequest = partial(
            EngineRequest,
            stq=0,
            source_id="",
            contribution=0,
            is_long_heavy_node=False,
            downstream_urgency=0,
            priority=0,
            node_name="",
            op_type="",
        )
        # Preserve the original scheduler's readiness/queue/polling code. This
        # adapter sends a complete primitive to the shared engine pool.
        module.QueryRunner.prepare_engine_payload = lambda runner, node: node
        from HeraSys.engines.engine_types import EngineType

        self.scheduler = module.GraphScheduler.__ray_metadata__.modified_class(
            {engine: bridge for engine in EngineType}
        )

    async def submit(self, query):
        return await self.scheduler.submit_query(query)

    def ready(self):
        return True

    async def status(self, query_id):
        return await self.scheduler.get_query_status(query_id)

    async def forget(self, query_id):
        await self.scheduler.cleanup_query(query_id)
        self.scheduler.query_runners.pop(query_id, None)


@ray.remote(num_cpus=1)
class EngineBridge:
    def __init__(self, pools, batch_wait_s=0.005):
        self.pools, self.events = pools, []
        self.batch_wait_s = batch_wait_s
        self.dispatcher = None
        self.tasks = set()
        self.service = {}

    def ready(self):
        return True

    async def submit_request(self, request):
        if self.dispatcher is None:
            self.dispatcher = FIFOEngines(
                self.pools, self.events, batch_wait_s=self.batch_wait_s
            )

        async def work():
            node = request.payload
            node.op_type = node.config.pop("_hera_original_op")
            try:
                result = await self.dispatcher.execute(
                    request.query_id, node, node.input_kwargs
                )
                self.service[request.query_id] = (
                    self.service.get(request.query_id, 0) + result["service_s"]
                )
                # The upstream scheduler wraps a raw single-port result itself.
                value = result["output"][node.output_names[0]]
            except Exception as exc:
                value = exc
            await request.query.query_state.set_node_result.remote(node.name, value)

        task = asyncio.create_task(work())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def cleanup(self, query_id):
        if self.dispatcher:
            await self.dispatcher.cleanup(query_id)
        return self.service.pop(query_id, 0)

    async def finish(self):
        if self.tasks:
            await asyncio.gather(*self.tasks)
        if self.dispatcher:
            await self.dispatcher.close()
        return self.events


class AyoUpstreamRuntime:
    def __init__(self, pools, *, batch_wait_s=0.005):
        self.pools = pools
        self.batch_wait_s = batch_wait_s
        self.config = HeraConfig.variant("ayo_static")
        self.events, self.query_records, self.queries = [], [], {}
        self.bridge = self.graph = None

    async def __aenter__(self):
        if not ray.is_initialized():
            ray.init(include_dashboard=False, log_to_driver=False)
        source = subprocess.check_output(
            ["git", "show", f"{UPSTREAM_COMMIT}:Ayo/schedulers/graph_scheduler.py"],
            text=True,
        )
        self.bridge = EngineBridge.remote(self.pools, self.batch_wait_s)
        self.graph = GraphActor.remote(source, self.bridge)
        await asyncio.gather(self.bridge.ready.remote(), self.graph.ready.remote())
        return self

    def submit(
        self, dag, inputs, *, query_id, timeout=300, submitted_at=None, **kwargs
    ):
        started = submitted_at or time.perf_counter()
        dag = deepcopy(dag)
        compute = [n for n in dag.nodes if n.node_type == NodeType.COMPUTE]
        sinks = [
            n
            for n in compute
            if not any(c.node_type == NodeType.COMPUTE for c in n.children)
        ]
        for node in compute:
            node.config["_hera_original_op"] = node.op_type
            # Our engine adapter exposes completion of actual prefill/decode work,
            # so use ordinary dependency readiness, not Ayo's background-stream shortcut.
            node.op_type = "shared_engine_adapter"
        dag._query_inputs = deepcopy(inputs)
        dag.create_input_nodes()
        state = QueryStates.remote()
        query = types.SimpleNamespace(
            query_id=query_id, DAG=dag, query_state=state, status=QueryStatus.RUNNING
        )

        async def run():
            status = "completed"
            try:
                await self.graph.submit.remote(query)
                while True:
                    result = await self.graph.status.remote(query_id)
                    if result == QueryStatus.COMPLETED:
                        outputs = await state.get_node_results.remote()
                        return {n.output_names[0]: outputs[n.name] for n in sinks}
                    if result == QueryStatus.FAILED:
                        raise RuntimeError(
                            "Upstream Ayo graph execution failed; inspect Ray actor logs"
                        )
                    await asyncio.sleep(0.01)
            except BaseException:
                status = "failed"
                raise
            finally:
                service = await self.bridge.cleanup.remote(query_id)
                await self.graph.forget.remote(query_id)
                ray.kill(state)
                self.query_records.append(
                    {
                        "query_id": query_id,
                        "status": status,
                        "latency_s": time.perf_counter() - started,
                        "logical_nodes": len(compute),
                        "reused_nodes": 0,
                        "attained_service_s": service,
                    }
                )
                self.queries.pop(query_id, None)

        task = asyncio.create_task(asyncio.wait_for(run(), timeout))
        self.queries[query_id] = task
        return task

    async def __aexit__(self, *args):
        if self.queries:
            await asyncio.gather(*self.queries.values(), return_exceptions=True)
        self.events = await self.bridge.finish.remote()
        ray.kill(self.graph)
        ray.kill(self.bridge)
