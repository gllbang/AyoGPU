"""Event-driven HeraSys scheduler over HeraSys's existing DAG and Node primitives.

One physical task may have several logical owners. All dependency bindings point
to the representative, so fusion removes both duplicate dispatch and graph edges.
Backends execute homogeneous batches and report device service time separately
from queue/transport time. No model imports are needed by the scheduler.
"""

import asyncio
import itertools
import math
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from HeraSys.dags.node_commons import NodeOps, NodeType

from . import policy
from .config import HeraConfig
from .semantics import digest, parameters, prefill_batch_key, signature


@dataclass
class ResourcePool:
    workers: list
    max_batch_size: int = 8
    supports_state_fork: bool = False
    max_sessions: int | None = None
    mixed_decode_batches: bool = False
    mixed_prefill_batches: bool = False
    batch_by_port_names: bool = True

    def __post_init__(self):
        if not self.workers or self.max_batch_size < 1:
            raise ValueError("Resource pools need workers and a positive batch limit")
        if self.max_sessions is not None and self.max_sessions < 1:
            raise ValueError("max_sessions must be positive when provided")


@dataclass
class Task:
    id: int
    node: Any
    signature: str
    bindings: dict
    dependencies: set
    owners: dict = field(default_factory=dict)  # (query id, node name) -> phi
    status: str = "pending"
    output: Any = None
    slot: Any = None
    ready_at: float = 0.0
    created_at: float = field(default_factory=time.perf_counter)
    cacheable: bool = False
    heavy: bool = False
    priority: float = 0.0
    resources_released: bool = False
    batch_key: str | None = None
    namespace: str = "default"
    signature_aliases: set = field(default_factory=set)
    value_signature: str | None = None
    audit: dict | None = None
    audit_inputs: dict | None = None
    pipeline_units: list | None = None
    resource_share: float = 0.0
    expected_finish: float = 0.0


@dataclass
class QueryState:
    id: str
    future: Any
    mapping: dict
    predictions: dict
    sinks: list
    deadline: float
    started: float
    parents: dict = field(default_factory=dict)
    completed: set = field(default_factory=set)
    attained_s: float = 0.0
    affinities: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)
    reused: int = 0
    pruned: int = 0


class HeraSys:
    def __init__(self, pools, config=None, profiles=None, calibrate=False):
        self.pools = pools
        self.config = config or HeraConfig()
        self.profiles = profiles
        self.calibrate = calibrate
        self.tasks = {}
        self.queries = {}
        self.signatures = {}
        self.value_signatures = {}
        self.cache = OrderedDict()
        self.events = []
        self.query_records = []
        self.node_audits = []
        self.timings = defaultdict(float)
        self._ids = itertools.count()
        self._used_query_ids = set()
        self._busy = {engine: {} for engine in pools}
        self._executions = set()
        self._wake = asyncio.Event()
        self._loop_task = None
        self._closed = False
        self.load = 0.0
        self.balance = 0.0
        self.remaining = {}
        self.pending_admissions = 0

    def record(self, event, **fields):
        if self.config.trace:
            self.events.append(
                {"event": event, "time_s": time.perf_counter(), **fields}
            )

    def _can_reuse(self, node):
        if not self.config.fusion or node.config.get("_hera_reuse") is False:
            return False
        if node.op_type in {NodeOps.LLM_PREFILLING, NodeOps.LLM_PARTIAL_PREFILLING}:
            return (
                self.pools[node.engine_type].supports_state_fork
                and node.config.get("temperature", 0) == 0
            )
        if node.op_type == NodeOps.VECTORDB_INGESTION:
            return bool(node.config.get("_hera_immutable", False))
        if node.op_type in {
            NodeOps.LLM_DECODING,
            NodeOps.LLM_PARTIAL_DECODING,
            NodeOps.LLM_FULL_PREFILLING,
        }:
            return (
                node.config.get("_hera_deterministic", False)
                and self.pools[node.engine_type].supports_state_fork
            )
        return node.op_type in {
            NodeOps.EMBEDDING,
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.RERANKING,
            NodeOps.AGGREGATOR,
        } or bool(node.config.get("_hera_reuse", False))

    def _cache_get(self, key):
        entry = self.cache.get(key)
        if entry is None:
            return None
        expires, output = entry
        if expires <= time.perf_counter():
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return deepcopy(output)

    def _cache_put(self, task):
        if (
            not task.cacheable
            or not self.config.completed_reuse
            or not self.config.cache_entries
        ):
            return
        if any(isinstance(v, dict) and "state_key" in v for v in task.output.values()):
            return
        entry = (
            time.perf_counter() + self.config.cache_ttl_s,
            deepcopy(task.output),
        )
        keys = [task.value_signature] if task.value_signature else []
        keys += sorted(task.signature_aliases - {task.signature})
        # Preserve admission-time suffix pruning even with a one-entry cache.
        keys.append(task.signature)
        for key in keys:
            self.cache[key] = entry
            self.cache.move_to_end(key)
        while len(self.cache) > self.config.cache_entries:
            self.cache.popitem(last=False)

    def submit(
        self,
        dag,
        inputs=None,
        *,
        query_id=None,
        namespace="default",
        timeout=300,
        submitted_at=None,
    ):
        """Submit a HeraSys DAG; return an asyncio Future of the sink output mapping.

        Namespace must identify the tenant/corpus/model revision whose results can
        safely be shared. The graph and input data are snapshotted at submission.
        """
        if self._closed:
            raise RuntimeError("HeraSys is closed")
        admission_started = time.perf_counter()
        arrival = time.perf_counter() if submitted_at is None else submitted_at
        loop = asyncio.get_running_loop()
        query_id = query_id or f"q{next(self._ids)}"
        if query_id in self._used_query_ids:
            raise ValueError(f"Duplicate query id {query_id}")
        if timeout <= 0:
            raise ValueError("Query timeout must be positive")
        dag = deepcopy(dag)
        supplied = deepcopy(dag.query_inputs if inputs is None else inputs)
        topo = dag.topological_sort()
        compute = [n for n in topo if n.node_type == NodeType.COMPUTE]
        if not compute:
            raise ValueError("DAG contains no compute nodes")
        for node in compute:
            if node.engine_type not in self.pools:
                raise ValueError(f"No resource pool for {node.engine_type}")
        # Validate every binding before mutating scheduler state.
        plans, sigs, predictions, descendants, depths = {}, {}, {}, {}, {}
        for node in reversed(compute):
            children = [c for c in node.children if c.node_type == NodeType.COMPUTE]
            # HeraSys also has virtual input/output nodes. They must not add a
            # fictitious computational stage to every sink's reverse depth.
            depths[node.name] = max((depths[c.name] + 1 for c in children), default=0)
            descendants[node.name] = set(children)
            if self.config.contribution_scope == "transitive":
                for child in children:
                    descendants[node.name].update(descendants[child.name])
            predictions[node.name] = (
                self.profiles.predict(node)
                if self.profiles
                else max(float(node.relative_ms or 1.0) / 1000, 1e-6)
            )
        for node in compute:
            bindings, semantic_bindings = {}, {}
            for port in node.input_names:
                producers = [p for p in node.parents if port in p.output_names]
                if len(producers) > 1:
                    raise ValueError(
                        f"Ambiguous input {node.name}.{port}; use distinct ports"
                    )
                if producers and producers[0].node_type == NodeType.COMPUTE:
                    parent = producers[0]
                    bindings[port] = ("node", parent.name, port)
                    semantic_bindings[port] = {
                        "producer": sigs[parent.name],
                        "port": port,
                    }
                else:
                    input_values = producers[0].input_values if producers else {}
                    if port in supplied:
                        value = supplied[port]
                    elif port in input_values:
                        value = input_values[port]
                    elif node.input_kwargs.get(port) is not None:
                        value = node.input_kwargs[port]
                    else:
                        raise ValueError(f"Missing input {node.name}.{port}")
                    bindings[port] = ("value", deepcopy(value))
                    semantic_bindings[port] = {"value": value}
            parents = [p.name for p in node.parents if p.node_type == NodeType.COMPUTE]
            reuse = self._can_reuse(node)
            sig = signature(
                node, semantic_bindings, [sigs[p] for p in parents], namespace
            )
            if not reuse:
                sig = digest([sig, query_id, node.name])
            sigs[node.name] = sig
            phi = sum(
                self.config.omega_work * predictions[u.name]
                + self.config.omega_depth * depths[u.name]
                for u in descendants[node.name]
            )
            phi += self.config.omega_branch * len(descendants[node.name])
            plans[node.name] = (node, bindings, parents, reuse, phi)

        mapping = {}
        state = QueryState(
            query_id,
            loop.create_future(),
            mapping,
            predictions,
            [
                n.name
                for n in compute
                if not any(c.node_type == NodeType.COMPUTE for c in n.children)
            ],
            arrival + timeout,
            arrival,
        )
        state.sessions = {
            name: node.config.get("_hera_session") for name, (node, *_) in plans.items()
        }
        state.parents = {name: list(plan[2]) for name, plan in plans.items()}
        self.queries[query_id] = state
        self._used_query_ids.add(query_id)
        for name, (node, bindings, parents, reuse, phi) in plans.items():
            sig = sigs[name]
            # Decode signatures express semantic equality, but the corresponding
            # prefill may have been executed independently on another actor.
            # Reuse completed answers here; fuse live decodes only after actual
            # KV identity and actor affinity are known in _fuse_ready_values.
            is_decode = node.op_type in {
                NodeOps.LLM_DECODING,
                NodeOps.LLM_PARTIAL_DECODING,
            }
            existing_id = self.signatures.get(sig) if reuse and not is_decode else None
            task = self.tasks.get(existing_id)
            if task is not None and task.status in {"pending", "running"}:
                state.reused += 1
                self.record(
                    "fusion" if task.status == "pending" else "inflight_reuse",
                    query_id=query_id,
                    node=name,
                    representative=task.id,
                )
            else:
                task_id = next(self._ids)
                physical_bindings = {
                    port: (
                        ("node", mapping[binding[1]], binding[2])
                        if binding[0] == "node"
                        else binding
                    )
                    for port, binding in bindings.items()
                }
                task = Task(
                    task_id,
                    node,
                    sig,
                    physical_bindings,
                    {mapping[p] for p in parents},
                    namespace=namespace,
                    signature_aliases={sig},
                )
                # Backend state handles cannot outlive their owning engine session.
                task.cacheable = reuse and node.op_type not in {
                    NodeOps.LLM_PREFILLING,
                    NodeOps.LLM_PARTIAL_PREFILLING,
                }
                if is_decode and not node.config.get("final"):
                    task.cacheable = False
                cached = (
                    self._cache_get(sig)
                    if task.cacheable and self.config.completed_reuse
                    else None
                )
                if cached is not None:
                    task.status, task.output = "completed", cached
                    task.dependencies.clear()
                    state.reused += 1
                    self.record("cache_hit", query_id=query_id, node=name)
                self.tasks[task.id] = task
                if reuse:
                    self.signatures[sig] = task.id
            task.owners[(query_id, name)] = phi
            mapping[name] = task.id
            if task.status == "completed":
                state.completed.add(name)
        # A cached suffix needs no recomputation of ancestors that serve only it.
        required = set()
        frontier = list(state.sinks)
        while frontier:
            name = frontier.pop()
            if name in required:
                continue
            required.add(name)
            if self.tasks[mapping[name]].status != "completed":
                frontier.extend(plans[name][2])
        for name in set(mapping) - required:
            self.tasks[mapping[name]].owners.pop((query_id, name), None)
            state.completed.add(name)
            state.pruned += 1
        for name in required:
            task = self.tasks[mapping[name]]
            if task.pipeline_units is not None:
                self._attach_pipeline_owner(task, query_id, name)
        self.record(
            "query_arrival",
            query_id=query_id,
            logical_nodes=len(mapping),
            reused_nodes=state.reused,
            namespace=namespace,
        )
        state.future.add_done_callback(lambda _: self._wake.set())
        self._wake.set()
        if self._loop_task is None:
            self._loop_task = loop.create_task(self._run())
        self.timings["admission_s"] += time.perf_counter() - admission_started
        return state.future

    async def run_query(self, query):
        """Adapter for an existing HeraSys Query (its Ray state actor is not required here)."""
        return await self.submit(
            query.DAG,
            query.query_inputs,
            query_id=query.query_id,
            namespace=query.context.get("namespace", "default"),
            timeout=query.timeout,
        )

    async def submit_factory(self, factory, **options):
        """Choose graph granularity after observing the actual arrival window.

        The factory receives current load and returns (DAG, inputs). Queueing and
        construction remain inside the request's original latency/deadline.
        """
        options.setdefault("submitted_at", time.perf_counter())
        self.pending_admissions += 1
        try:
            await asyncio.sleep(self.config.fusion_window_s)
            load = policy.load_factor(
                len(self.queries) + self.pending_admissions,
                0,
                sum(len(pool.workers) for pool in self.pools.values()),
                self.config,
            )
            dag, inputs = factory(load)
        finally:
            self.pending_admissions -= 1
        return await self.submit(dag, inputs, **options)

    def _inputs(self, task):
        values = {}
        for port, binding in task.bindings.items():
            if binding[0] == "value":
                values[port] = deepcopy(binding[1])
            else:
                values[port] = deepcopy(self.tasks[binding[1]].output[binding[2]])
        return values

    def _attach_pipeline_owner(self, task, query_id, original_name):
        """Attach a late duplicate to every still-needed unit of a shared pipeline."""
        state = self.queries[query_id]
        original_parents = list(state.parents[original_name])
        names = {
            unit["suffix"]: f"{original_name}::{unit['suffix']}"
            for unit in task.pipeline_units
        }
        for unit in task.pipeline_units:
            name = names[unit["suffix"]]
            if name in state.mapping:
                raise ValueError(f"Reserved pipeline node name already exists: {name}")
            child = self.tasks[unit["task_id"]]
            state.mapping[name] = child.id
            state.predictions[name] = unit["prediction"]
            state.sessions[name] = None
            state.parents[name] = (
                [names[parent] for parent in unit["parents"]]
                if unit["parents"]
                else original_parents
            )
            child.owners[(query_id, name)] = 0.0
            if child.status == "completed":
                state.completed.add(name)
        state.parents[original_name] = [
            names[unit["suffix"]] for unit in task.pipeline_units if unit["last"]
        ]
        state.predictions[original_name] = 0.000001
        self._recompute_contributions(state)

    def _recompute_contributions(self, state):
        children = {name: set() for name in state.mapping}
        for name, parents in state.parents.items():
            for parent in parents:
                children[parent].add(name)
        depths, descendants = {}, {}

        def visit(name):
            if name in depths:
                return
            for child in children[name]:
                visit(child)
            depths[name] = max((depths[c] + 1 for c in children[name]), default=0)
            descendants[name] = set(children[name])
            if self.config.contribution_scope == "transitive":
                for child in children[name]:
                    descendants[name].update(descendants[child])

        for name in children:
            visit(name)
            task = self.tasks.get(state.mapping[name])
            if task is not None and (state.id, name) in task.owners:
                successors = descendants[name]
                task.owners[(state.id, name)] = sum(
                    self.config.omega_work * state.predictions[child]
                    + self.config.omega_depth * depths[child]
                    for child in successors
                ) + self.config.omega_branch * len(successors)

    def _expand_ready_pipelines(self, ready):
        from HeraSys.engines.engine_types import EngineType

        from .pipeline import partitions
        from .workflows import primitive

        changed = False
        for task in ready:
            spec = task.node.config.get("pipeline")
            if spec is None:
                continue
            try:
                inputs = self._inputs(task)
                size = len(inputs[spec["split_ports"][0]])
                count = policy.split_factor(self.balance, max(1, size), self.config)
                if task.heavy and self.config.adaptive_splitting:
                    count = max(count, self.config.min_heavy_splits)
                pieces = partitions(spec, inputs, count)
                # Validate all engines and owner names before changing graph state.
                for stage in spec["stages"]:
                    if EngineType(stage["engine"]) not in self.pools:
                        raise ValueError(
                            f"No pipeline resource pool for {stage['engine']}"
                        )
                for q, original in task.owners:
                    if any(
                        name.startswith(original + "::pipeline:")
                        for name in self.queries[q].mapping
                    ):
                        raise ValueError("Pipeline internal names collide with the DAG")
                units, last_units = [], []
                for lane, piece in enumerate(pieces if size else []):
                    values = {port: ("value", value) for port, value in piece.items()}
                    row_count = len(piece[spec["split_ports"][0]])
                    previous, previous_suffix = None, None
                    for step, stage in enumerate(spec["stages"]):
                        suffix = f"pipeline:{lane}:{step}"
                        node = primitive(
                            f"{task.node.name}::{suffix}",
                            NodeOps(stage["op"]),
                            EngineType(stage["engine"]),
                            stage["inputs"],
                            stage["outputs"],
                            ms=stage["relative_ms"] * row_count / size,
                            **{**stage["config"], "_hera_rows": row_count},
                        )
                        bindings = {port: values[port] for port in node.input_names}
                        semantic = {
                            port: {"value": binding[1]}
                            if binding[0] == "value"
                            else {
                                "producer": self.tasks[binding[1]].signature,
                                "port": binding[2],
                            }
                            for port, binding in bindings.items()
                        }
                        parents = [previous] if previous is not None else []
                        sig = signature(
                            node,
                            semantic,
                            [self.tasks[p].signature for p in parents],
                            task.namespace,
                        )
                        reuse = self._can_reuse(node)
                        if not reuse:
                            sig = digest([sig, task.id, suffix])
                        child = (
                            self.tasks.get(self.signatures.get(sig)) if reuse else None
                        )
                        if child is None or child.status == "failed":
                            child = Task(
                                next(self._ids),
                                node,
                                sig,
                                bindings,
                                set(parents),
                                cacheable=reuse,
                                namespace=task.namespace,
                                signature_aliases={sig},
                            )
                            cached = (
                                self._cache_get(sig)
                                if reuse and self.config.completed_reuse
                                else None
                            )
                            if cached is not None:
                                child.status, child.output = "completed", cached
                                child.dependencies.clear()
                            self.tasks[child.id] = child
                            if reuse:
                                self.signatures[sig] = child.id
                        prediction = (
                            self.profiles.predict(node)
                            if self.profiles
                            else node.relative_ms / 1000
                        )
                        units.append(
                            {
                                "suffix": suffix,
                                "task_id": child.id,
                                "prediction": prediction,
                                "parents": [previous_suffix] if previous_suffix else [],
                                "last": step == len(spec["stages"]) - 1,
                            }
                        )
                        for port in node.output_names:
                            values[port] = ("node", child.id, port)
                        previous, previous_suffix = child.id, suffix
                    last_units.append(previous)
                gather_ports = {
                    port: [f"{port}__lane_{lane}" for lane in range(len(last_units))]
                    for port in task.node.output_names
                }
                task.bindings = {
                    f"{port}__lane_{lane}": ("node", child_id, port)
                    for lane, child_id in enumerate(last_units)
                    for port in task.node.output_names
                }
                task.dependencies = set(last_units)
                task.node = primitive(
                    task.node.name,
                    NodeOps.AGGREGATOR,
                    task.node.engine_type,
                    list(task.bindings),
                    task.node.output_names,
                    ms=0.001,
                    mode="pipeline_gather",
                    gather_ports=gather_ports,
                )
                task.batch_key = None
                task.ready_at = 0.0
                task.pipeline_units = units
                for q, original in list(task.owners):
                    self._attach_pipeline_owner(task, q, original)
                self.record(
                    "pipeline_split",
                    task_id=task.id,
                    owners=[list(owner) for owner in task.owners],
                    rows=size,
                    splits=len(pieces) if size else 0,
                    stages=len(spec["stages"]),
                    unit_ids=[unit["task_id"] for unit in units],
                    load=self.load,
                )
                changed = True
            except Exception as exc:
                task.status = "failed"
                for q, _ in task.owners:
                    future = self.queries[q].future
                    if not future.done():
                        future.set_exception(exc)
                self.record("pipeline_error", task_id=task.id, error=str(exc))
                changed = True
        return changed

    def _affinity(self, task):
        slots = set()
        for q, name in task.owners:
            state = self.queries[q]
            session = state.sessions[name]
            if session in state.affinities:
                slots.add(state.affinities[session])
        if len(slots) > 1:
            raise RuntimeError(
                "A fused stateful task has conflicting engine affinities"
            )
        return next(iter(slots), None)

    def _refresh(self):
        estimates = {
            q: policy.remaining(s.predictions, s.completed, s.attained_s, self.config)
            for q, s in self.queries.items()
            if not s.future.done()
        }
        longs = policy.long_queries(estimates, self.config.eta)
        ready = []
        now = time.perf_counter()
        for task in self.tasks.values():
            if task.status != "pending" or not task.owners:
                continue
            if not all(self.tasks[d].status == "completed" for d in task.dependencies):
                continue
            owners = [(q, n) for q, n in task.owners if q in estimates]
            if not owners:
                continue
            if not task.ready_at:
                task.ready_at = now
            task.heavy = any(
                q in longs
                and self.queries[q].predictions[n]
                >= sum(self.queries[q].predictions.values())
                / len(self.queries[q].predictions)
                for q, n in owners
            )
            # Eq. 3: sum original logical contributions, never recompute on fused graph.
            phi = sum(task.owners[o] for o in owners)
            sq = min(estimates[q] for q, _ in owners)
            task.priority = policy.priority(sq, phi, task.heavy, self.config)
            ready.append(task)
        self.load = policy.load_factor(
            len(estimates),
            len(ready),
            sum(len(p.workers) for p in self.pools.values()),
            self.config,
        )
        from .allocation import balancing_weight

        self.remaining = estimates
        self.balance = balancing_weight(
            self.load,
            [now - self.queries[q].started + value for q, value in estimates.items()],
            self.config,
        )
        self.record(
            "schedule",
            load=self.load,
            balance=self.balance,
            active=len(estimates),
            ready=len(ready),
            long_queries=sorted(longs),
            remaining_s=estimates,
        )
        return ready

    def _fuse_ready_values(self, ready):
        """Resolve equivalent actual inputs after differently split graphs converge.

        Admission signatures identify structural equality. At a join, two distinct
        producer graphs can yield identical values; use those immutable values to
        recover the fusion opportunity without assuming the producers equivalent.
        """
        if not self.config.fusion or not self.config.value_fusion:
            return ready, False
        changed = False
        eligible = {
            NodeOps.RERANKING,
            NodeOps.LLM_PREFILLING,
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.LLM_PARTIAL_DECODING,
            NodeOps.LLM_DECODING,
        }
        for task in ready:
            if task.id not in self.tasks or task.node.op_type not in eligible:
                continue
            if not self._can_reuse(task.node):
                continue
            if task.value_signature is None:
                task.value_signature = digest(
                    {
                        "ready_values": True,
                        "namespace": task.namespace,
                        "op": task.node.op_type,
                        "engine": task.node.engine_type,
                        "parameters": parameters(task.node.config),
                        # Engine-local KV handle names can collide across actors.
                        "affinity": self._affinity(task),
                        "inputs": self._inputs(task),
                        "outputs": list(task.node.io_schema.output_format),
                    }
                )
            key = task.value_signature
            existing = self.tasks.get(self.value_signatures.get(key))
            if (
                existing is not None
                and existing.id != task.id
                and existing.status in {"pending", "running"}
            ):
                left, right = self._affinity(task), self._affinity(existing)
                if left is not None and right is not None and left != right:
                    continue
                # Redirect every consumer and logical owner to one physical task.
                for child in self.tasks.values():
                    for unit in child.pipeline_units or []:
                        if unit["task_id"] == task.id:
                            unit["task_id"] = existing.id
                    if task.id in child.dependencies:
                        child.dependencies.remove(task.id)
                        child.dependencies.add(existing.id)
                    for port, binding in child.bindings.items():
                        if binding[0] == "node" and binding[1] == task.id:
                            child.bindings[port] = ("node", existing.id, binding[2])
                for (query_id, name), phi in task.owners.items():
                    state = self.queries[query_id]
                    state.mapping[name] = existing.id
                    state.reused += 1
                    session = state.sessions[name]
                    if session and existing.slot is not None:
                        state.affinities[session] = existing.slot
                    existing.owners[(query_id, name)] = phi
                    self.record(
                        "value_fusion",
                        query_id=query_id,
                        node=name,
                        representative=existing.id,
                    )
                existing.signature_aliases.update(task.signature_aliases)
                for alias in task.signature_aliases:
                    self.signatures[alias] = existing.id
                del self.tasks[task.id]
                changed = True
                continue
            if existing is not None and existing.id == task.id:
                continue
            cached = (
                self._cache_get(key)
                if task.cacheable and self.config.completed_reuse
                else None
            )
            if cached is not None:
                task.output, task.status = cached, "completed"
                for query_id, name in task.owners:
                    state = self.queries[query_id]
                    state.completed.add(name)
                    state.reused += 1
                    self.record("value_cache_hit", query_id=query_id, node=name)
                self._cache_put(task)
                self._wake.set()
                changed = True
            else:
                self.value_signatures[key] = task.id
        return [
            task for task in ready if task.id in self.tasks and task.status == "pending"
        ], changed

    def _finish_terminal_decodes(self, ready):
        changed = False
        for task in ready:
            if task.node.op_type not in {
                NodeOps.LLM_DECODING,
                NodeOps.LLM_PARTIAL_DECODING,
            }:
                continue
            handle = next(iter(self._inputs(task).values()))
            if not isinstance(handle, dict) or handle.get("_hera_finished") is not True:
                continue
            model = task.node.config.get("model")
            if model and model != handle.get("model"):
                continue
            task.output = {
                task.node.output_names[0]: handle["text"]
                if task.node.config.get("final")
                else handle
            }
            task.status, task.slot = "completed", self._affinity(task)
            if self.config.audit_nodes:
                source = next(
                    (
                        self.tasks[b[1]]
                        for b in task.bindings.values()
                        if b[0] == "node"
                    ),
                    None,
                )
                task.audit = source.audit if source is not None else None
            for query_id, name in task.owners:
                state = self.queries[query_id]
                state.completed.add(name)
                state.pruned += 1
            self._cache_put(task)
            self.record(
                "terminal_decode",
                task_id=task.id,
                node=task.node.name,
                owners=[list(owner) for owner in task.owners],
                service_s=0.0,
                elapsed_s=0.0,
            )
            changed = True
        if changed:
            self._wake.set()
        return [task for task in ready if task.status == "pending"], changed

    def _batch_key(self, task):
        if task.batch_key is None:
            if (
                self.pools[task.node.engine_type].mixed_prefill_batches
                and task.node.op_type == NodeOps.LLM_PREFILLING
            ):
                task.batch_key = prefill_batch_key(task.node.config)
                return task.batch_key
            if self.pools[
                task.node.engine_type
            ].mixed_decode_batches and task.node.op_type in {
                NodeOps.LLM_DECODING,
                NodeOps.LLM_PARTIAL_DECODING,
            }:
                # GPUWorker tracks each sequence's token target and final-output
                # conversion independently. Port names and final flags are not
                # CUDA compatibility constraints for a decode step.
                task.batch_key = digest(["decode", task.node.config.get("model")])
                return task.batch_key
            task.batch_key = digest(
                [
                    task.node.op_type,
                    parameters(task.node.config),
                    list(task.node.io_schema.input_format)
                    if self.pools[task.node.engine_type].batch_by_port_names
                    else len(task.node.io_schema.input_format),
                    list(task.node.io_schema.output_format)
                    if self.pools[task.node.engine_type].batch_by_port_names
                    else len(task.node.io_schema.output_format),
                ]
            )
        return task.batch_key

    def _active_sessions(self, engine, slot):
        sessions = set()
        for task in self.tasks.values():
            if task.node.engine_type != engine:
                continue
            holds_state = (
                task.status == "completed"
                and not task.resources_released
                and isinstance(task.output, dict)
                and any(
                    isinstance(value, dict) and "state_key" in value
                    for value in task.output.values()
                )
            )
            # A final decode can complete while _retire awaits another release.
            # Its source still owns KV until the next retirement pass. Admitting
            # new prefills in that gap can temporarily double physical occupancy.
            if task.status not in {"pending", "running"} and not holds_state:
                continue
            for q, name in task.owners:
                session = self.queries[q].sessions[name]
                if session and self.queries[q].affinities.get(session) == slot:
                    sessions.add((q, session))
        return len(sessions)

    def _age_key(self, task, now):
        if self.config.max_query_wait_s is not None:
            arrived = min(self.queries[q].started for q, _ in task.owners)
            if now - arrived >= self.config.max_query_wait_s:
                return (arrived, task.created_at, task.id)
        if self.config.max_ready_wait_s is not None:
            if now - task.ready_at >= self.config.max_ready_wait_s:
                return (task.ready_at, task.created_at, task.id)
        return None

    def _dispatch(self, ready):
        if (
            self.config.joint_allocation
            and self.config.adaptive_batching
            and self.profiles is not None
            and not self.calibrate
        ):
            return self._dispatch_joint(ready)
        return self._dispatch_legacy(ready)

    def _dispatch_joint(self, ready):
        from .allocation import choose_allocation, session_continuations

        now, next_wake = time.perf_counter(), None
        continuations = session_continuations(
            self.tasks,
            self.queries,
            ready
            + [
                t
                for workers in self._busy.values()
                for batch in workers.values()
                for t in batch
            ],
        )
        for engine, pool in self.pools.items():
            busy = self._busy[engine]
            candidates = sorted(
                [task for task in ready if task.node.engine_type == engine],
                key=lambda task: (-task.priority, task.created_at, task.id),
            )
            for slot, worker in enumerate(pool.workers):
                if slot in busy:
                    continue
                sessions = (
                    self._active_sessions(engine, slot) if pool.max_sessions else 0
                )

                def limit(task):
                    if (
                        pool.max_sessions
                        and task.node.op_type == NodeOps.LLM_PREFILLING
                    ):
                        return min(pool.max_batch_size, pool.max_sessions - sessions)
                    return pool.max_batch_size

                available = [
                    t
                    for t in candidates
                    if self._affinity(t) in {None, slot} and limit(t) > 0
                ]
                if not available:
                    continue
                age_keys = {
                    t.id: key
                    for t in available
                    if (key := self._age_key(t, now)) is not None
                }
                heavy_share = sum(
                    t.resource_share
                    for batch in busy.values()
                    for t in batch
                    if t.heavy
                )
                reservation = (
                    min(
                        1.0, max(0.0, self.config.rho * len(pool.workers) - heavy_share)
                    )
                    if self.config.resource_skew
                    and any(t.heavy for t in available)
                    and not age_keys
                    else 0.0
                )
                groups = defaultdict(list)
                for task in candidates:
                    groups[self._batch_key(task)].append(task)
                for group in groups.values():
                    if age_keys:
                        group.sort(
                            key=lambda t: (
                                (0, age_keys[t.id])
                                if t.id in age_keys
                                else (1, (-t.priority, t.created_at, t.id))
                            )
                        )
                    elif reservation:
                        group.sort(key=lambda t: not t.heavy)
                eligible = set()
                ids = {t.id for t in available}
                oldest = min(age_keys, key=age_keys.get) if age_keys else None
                for group in groups.values():
                    feasible = [t for t in group if t.id in ids]
                    if not feasible or (
                        oldest is not None and feasible[0].id != oldest
                    ):
                        continue
                    head = feasible[0]
                    window = max(
                        self.config.max_batch_wait_s * self.balance,
                        self.config.fusion_window_s if self.config.fusion else 0,
                    )
                    delay = head.ready_at + window - now
                    if len(feasible) < limit(head) and delay > 0:
                        next_wake = (
                            delay if next_wake is None else min(next_wake, delay)
                        )
                        continue
                    eligible.update(t.id for t in feasible)
                # Do not block other classes on an unexpired heavy batching window.
                required = (
                    reservation
                    if any(t.heavy and t.id in eligible for t in available)
                    else 0.0
                )
                # Algorithm 1 selects a priority-queue head before searching
                # that class's batch size. Comparing unrelated classes with an
                # approximate ready-only rollout can override Eq. 7 entirely.
                heads = [
                    next((t for t in group if t.id in eligible), None)
                    for group in groups.values()
                ]
                heads = [
                    t for t in heads if t is not None and (not required or t.heavy)
                ]
                if not heads:
                    continue
                head = max(heads, key=lambda t: (t.priority, -t.id))
                eligible.intersection_update(
                    t.id for t in groups[self._batch_key(head)]
                )
                allocation = choose_allocation(
                    list(groups.values()),
                    slot=slot,
                    capacity=len(pool.workers),
                    max_batch_size=pool.max_batch_size,
                    profiles=self.profiles,
                    queries=self.queries,
                    remaining=self.remaining,
                    now=now,
                    balance=self.balance,
                    busy=busy,
                    affinity=self._affinity,
                    reservation=required,
                    config=self.config,
                    eligible=eligible,
                    limit=limit,
                    continuations=continuations,
                )
                if allocation is None:
                    continue
                batch = allocation.tasks
                busy[slot] = batch
                for task, share in zip(batch, allocation.shares):
                    candidates.remove(task)
                    task.status, task.slot = "running", slot
                    task.resource_share = share
                    task.expected_finish = now + allocation.duration_s
                    for q, name in task.owners:
                        session = self.queries[q].sessions[name]
                        if session:
                            self.queries[q].affinities[session] = slot
                self.record(
                    "dispatch",
                    engine=str(engine),
                    slot=slot,
                    task_ids=[t.id for t in batch],
                    nodes=[t.node.name for t in batch],
                    batch_size=len(batch),
                    priorities=[t.priority for t in batch],
                    contributions=[sum(t.owners.values()) for t in batch],
                    heavy=[t.heavy for t in batch],
                    reserved=bool(required),
                    aging_override=bool(age_keys),
                    ready_wait_s=[now - t.ready_at for t in batch],
                    load=self.load,
                    balance=self.balance,
                    resource_shares=allocation.shares,
                    allocation_grid="equal_batch_engine_shares",
                    predicted_batch_s=allocation.duration_s,
                    allocation_objective_s=allocation.objective_s,
                    allocation_penalty_s=allocation.penalty_s,
                )
                execution = asyncio.create_task(
                    self._execute(engine, slot, worker, batch)
                )
                self._executions.add(execution)
                execution.add_done_callback(self._executions.discard)
        return next_wake

    def _dispatch_legacy(self, ready):
        next_wake = None
        now = time.perf_counter()
        for engine, pool in self.pools.items():
            busy = self._busy[engine]
            candidates = [t for t in ready if t.node.engine_type == engine]
            candidates.sort(key=lambda t: (-t.priority, t.created_at, t.id))
            reservation = policy.reserved_slots(len(pool.workers), self.config)
            for slot, worker in enumerate(pool.workers):
                if slot in busy:
                    continue
                available = [t for t in candidates if self._affinity(t) in {None, slot}]
                sessions = (
                    self._active_sessions(engine, slot) if pool.max_sessions else 0
                )
                if pool.max_sessions and sessions >= pool.max_sessions:
                    available = [
                        t for t in available if t.node.op_type != NodeOps.LLM_PREFILLING
                    ]
                if not available:
                    continue
                heavy_running = sum(
                    any(t.heavy for t in batch) for batch in busy.values()
                )
                heavy = [t for t in available if t.heavy]
                reserve = bool(heavy and heavy_running < reservation)
                head = heavy[0] if reserve else available[0]
                age_keys = {
                    t.id: key
                    for t in available
                    if (key := self._age_key(t, now)) is not None
                }
                if age_keys:
                    # Optional App. B.2 hard priority floor: oldest feasible ready
                    # task wins even when mean+std labels no active query as long.
                    # Apply the floor to every batch position; otherwise new,
                    # high-priority tasks can consume all but the first slot.
                    available.sort(
                        key=lambda t: (
                            (0, age_keys[t.id])
                            if t.id in age_keys
                            else (1, (-t.priority, t.created_at, t.id))
                        )
                    )
                    head = available[0]
                    # Once the floor is active, all feasible aged tasks precede
                    # the heavy-node preference, including the rest of a batch.
                    reserve = False
                key = self._batch_key(head)
                compatible = [head] + [
                    t for t in available if t is not head and self._batch_key(t) == key
                ]
                if reserve:
                    # Serve ready heavy nodes first, then lend otherwise idle
                    # batch positions to compatible normal nodes.
                    compatible.sort(key=lambda t: not t.heavy)
                # Never renew a batching deadline: prevents starvation at sparse arrivals.
                wait = (
                    self.config.max_batch_wait_s * self.load
                    if self.config.adaptive_batching
                    else 0
                )
                window = max(
                    wait, self.config.fusion_window_s if self.config.fusion else 0
                )
                deadline = head.ready_at + window
                if len(compatible) < pool.max_batch_size and now < deadline:
                    delay = deadline - now
                    next_wake = delay if next_wake is None else min(next_wake, delay)
                    continue
                limit = pool.max_batch_size
                if pool.max_sessions and head.node.op_type == NodeOps.LLM_PREFILLING:
                    limit = min(limit, max(1, pool.max_sessions - sessions))
                candidates_for_batch = compatible[:limit]
                chosen = (
                    policy.batch_size(candidates_for_batch, self.profiles, self.load)
                    if self.config.adaptive_batching
                    else len(candidates_for_batch)
                )
                batch = candidates_for_batch[:chosen]
                busy[slot] = batch
                for task in batch:
                    candidates.remove(task)
                    task.status, task.slot = "running", slot
                    for q, name in task.owners:
                        session = self.queries[q].sessions[name]
                        if session:
                            self.queries[q].affinities[session] = slot
                self.record(
                    "dispatch",
                    engine=str(engine),
                    slot=slot,
                    task_ids=[t.id for t in batch],
                    nodes=[t.node.name for t in batch],
                    batch_size=len(batch),
                    priorities=[t.priority for t in batch],
                    contributions=[sum(t.owners.values()) for t in batch],
                    heavy=[t.heavy for t in batch],
                    reserved=reserve,
                    aging_override=bool(age_keys),
                    ready_wait_s=[now - t.ready_at for t in batch],
                    load=self.load,
                )
                execution = asyncio.create_task(
                    self._execute(engine, slot, worker, batch)
                )
                self._executions.add(execution)
                execution.add_done_callback(self._executions.discard)
        return next_wake

    async def _execute(self, engine, slot, worker, batch):
        started = time.perf_counter()
        try:
            payload = [
                {
                    "task_id": t.id,
                    "op": t.node.op_type.value
                    if hasattr(t.node.op_type, "value")
                    else t.node.op_type,
                    "config": {**t.node.config, "_hera_audit": True}
                    if self.config.audit_nodes
                    else t.node.config,
                    "inputs": self._inputs(t),
                    "outputs": list(t.node.io_schema.output_format),
                }
                for t in batch
            ]
            self.timings["payload_s"] += time.perf_counter() - started
            if self.config.audit_nodes:
                for task, request in zip(batch, payload):
                    task.audit_inputs = {
                        port: digest(value)
                        for port, value in request["inputs"].items()
                        if not (isinstance(value, dict) and "state_key" in value)
                    }
            results = await worker.execute_batch(payload)
            postprocess_started = time.perf_counter()
            if len(results) != len(batch):
                raise RuntimeError("Backend returned wrong batch result count")
            for task, result in zip(batch, results):
                output = result["output"]
                required = set(task.node.io_schema.output_format)
                if not isinstance(output, dict) or not required.issubset(output):
                    raise RuntimeError(f"Invalid output schema for {task.node.name}")
                rows = task.node.config.get("_hera_rows")
                if rows is not None and any(
                    isinstance(output[port], (str, bytes, dict))
                    or not hasattr(output[port], "__len__")
                    or len(output[port]) != rows
                    for port in required
                ):
                    raise ValueError(
                        f"Pipeline stage changed its row axis: {task.node.name}"
                    )
                service_s = float(result["service_s"])
                if not math.isfinite(service_s) or service_s < 0:
                    raise ValueError(
                        "Backend service time must be finite and nonnegative"
                    )
            if self.profiles and self.calibrate:
                self.profiles.observe_batch(
                    [t.node for t in batch],
                    max(float(r["service_s"]) for r in results),
                    wall_s=postprocess_started - started,
                )
            for task, result in zip(batch, results):
                task.output, task.status = result["output"], "completed"
                if self.config.audit_nodes:
                    task.audit = result.get("audit")
                service_s = float(result["service_s"])
                if self.profiles and self.calibrate:
                    self.profiles.observe(task.node, service_s)
                self._cache_put(task)
                for query_id, name in task.owners:
                    state = self.queries.get(query_id)
                    if state and name not in state.completed:
                        state.completed.add(name)
                        state.attained_s += service_s
                        session = state.sessions[name]
                        if session:
                            state.affinities[session] = slot
                self.record(
                    "node_complete",
                    task_id=task.id,
                    node=task.node.name,
                    owners=[list(o) for o in task.owners],
                    service_s=service_s,
                    elapsed_s=time.perf_counter() - started,
                )
            self.timings["postprocess_s"] += time.perf_counter() - postprocess_started
        except Exception as exc:
            affected = {q for task in batch for q, _ in task.owners}
            for task in batch:
                task.status = "failed"
            for q in affected:
                state = self.queries.get(q)
                if state and not state.future.done():
                    state.future.set_exception(exc)
            self.record(
                "batch_error",
                engine=str(engine),
                error=str(exc),
                queries=sorted(affected),
            )
        finally:
            self._busy[engine].pop(slot, None)
            self._wake.set()

    async def _retire(self):
        now = time.perf_counter()
        for q, state in list(self.queries.items()):
            if not state.future.done() and now >= state.deadline:
                state.future.set_exception(
                    TimeoutError(f"Query {q} exceeded its deadline")
                )
            if not state.future.done() and len(state.completed) == len(state.mapping):
                output = {}
                for name in state.sinks:
                    for key, value in self.tasks[state.mapping[name]].output.items():
                        if key in output:
                            raise ValueError(
                                f"Duplicate sink output {key}; use distinct output names"
                            )
                        output[key] = deepcopy(value)
                state.future.set_result(output)
            if not state.future.done():
                continue
            failed = state.future.cancelled() or state.future.exception() is not None
            record = {
                "query_id": q,
                "latency_s": now - state.started,
                "status": "failed" if failed else "completed",
                "logical_nodes": len(state.mapping),
                "reused_nodes": state.reused,
                "pruned_nodes": state.pruned,
                "attained_service_s": state.attained_s,
            }
            if self.config.audit_nodes:
                nodes = {}
                for name, task_id in state.mapping.items():
                    task = self.tasks.get(task_id)
                    if task is None:
                        continue
                    has_handle = isinstance(task.output, dict) and any(
                        isinstance(v, dict) and "state_key" in v
                        for v in task.output.values()
                    )
                    nodes[name] = {
                        "op": task.node.op_type,
                        "status": task.status,
                        "input_sha256": task.audit_inputs,
                        "output_sha256": digest(task.audit["token_ids"])
                        if has_handle and task.audit
                        else None
                        if has_handle or task.output is None
                        else digest(task.output),
                        "llm": task.audit,
                    }
                self.node_audits.append({"query_id": q, "nodes": nodes})
            self.query_records.append(record)
            self.record("query_complete", **record)
            for name, task_id in state.mapping.items():
                if task_id in self.tasks:
                    self.tasks[task_id].owners.pop((q, name), None)
            del self.queries[q]
        # Outputs held by active queries stay alive; release engine KV handles explicitly.
        releases = {}
        needed = {
            d
            for t in self.tasks.values()
            if t.status in {"pending", "running"} and t.owners
            for d in t.dependencies
        }
        needed.update(
            unit["task_id"]
            for task in self.tasks.values()
            if task.owners
            for unit in task.pipeline_units or []
        )
        for task_id, task in list(self.tasks.items()):
            has_handle = isinstance(task.output, dict) and any(
                isinstance(v, dict) and "state_key" in v for v in task.output.values()
            )
            if has_handle and task_id not in needed and not task.resources_released:
                releases.setdefault((task.node.engine_type, task.slot), []).append(
                    task.output
                )
                task.resources_released = True
            if task.owners or task.status == "running" or task_id in needed:
                continue
            if (
                task.output is not None
                and task.slot is not None
                and not task.resources_released
            ):
                releases.setdefault((task.node.engine_type, task.slot), []).append(
                    task.output
                )
            for alias in task.signature_aliases or {task.signature}:
                if self.signatures.get(alias) == task_id:
                    del self.signatures[alias]
            if self.value_signatures.get(task.value_signature) == task_id:
                del self.value_signatures[task.value_signature]
            del self.tasks[task_id]
        await asyncio.gather(
            *(
                self.pools[engine].workers[slot].release(outputs)
                for (engine, slot), outputs in releases.items()
                if hasattr(self.pools[engine].workers[slot], "release")
            )
        )

    async def _run(self):
        try:
            while not self._closed:
                self._wake.clear()
                phase_started = time.perf_counter()
                await self._retire()
                self.timings["retire_wall_s"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                ready = self._refresh()
                self.timings["refresh_s"] += time.perf_counter() - phase_started
                while True:
                    phase_started = time.perf_counter()
                    expanded = self._expand_ready_pipelines(ready)
                    self.timings["pipeline_expansion_s"] += (
                        time.perf_counter() - phase_started
                    )
                    if expanded:
                        phase_started = time.perf_counter()
                        ready = self._refresh()
                        self.timings["refresh_s"] += time.perf_counter() - phase_started
                        continue
                    phase_started = time.perf_counter()
                    ready, terminal_changed = self._finish_terminal_decodes(ready)
                    self.timings["terminal_decode_s"] += (
                        time.perf_counter() - phase_started
                    )
                    phase_started = time.perf_counter()
                    ready, changed = self._fuse_ready_values(ready)
                    self.timings["value_fusion_s"] += (
                        time.perf_counter() - phase_started
                    )
                    if not changed and not terminal_changed:
                        break
                    # Recompute Eq. 3 contributions and discover cached successors
                    # before any of the newly fused graph is dispatched.
                    phase_started = time.perf_counter()
                    ready = self._refresh()
                    self.timings["refresh_s"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                delay = self._dispatch(ready)
                self.timings["dispatch_s"] += time.perf_counter() - phase_started
                deadline = min(
                    (s.deadline for s in self.queries.values()), default=None
                )
                if deadline is not None:
                    timeout = max(0.001, deadline - time.perf_counter())
                    delay = timeout if delay is None else min(delay, timeout)
                try:
                    if delay is None:
                        await self._wake.wait()
                    else:
                        await asyncio.wait_for(self._wake.wait(), max(delay, 0.0001))
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:
            self._closed = True
            for state in self.queries.values():
                if not state.future.done():
                    state.future.set_exception(exc)
            self.record("scheduler_error", error=str(exc))
            raise

    async def close(self):
        for state in self.queries.values():
            if not state.future.done():
                state.future.cancel()
        if self._executions:
            await asyncio.gather(*list(self._executions))
        await self._retire()
        self._closed = True
        self._wake.set()
        try:
            if self._loop_task:
                await self._loop_task
        finally:
            await asyncio.gather(
                *(
                    worker.flush_releases()
                    for pool in self.pools.values()
                    for worker in pool.workers
                    if hasattr(worker, "flush_releases")
                )
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
