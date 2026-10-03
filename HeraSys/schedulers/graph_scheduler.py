import asyncio
import hashlib
import json
import math
import traceback
from collections import deque
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import ray

from HeraSys.configs.model_config import AggMode, get_aggregator_config
from HeraSys.dags.node import Node, NodeOps, NodeStatus, NodeType
from HeraSys.engines.payload_transformers import TRANSFORMER_REGISTRY, DefaultTransformer
from HeraSys.logger import GLOBAL_INFO_LEVEL, get_logger
from HeraSys.queries.query import Query
from HeraSys.queries.query_state import QueryStatus
from HeraSys.schedulers.engine_scheduler import EngineRequest

# 1 节点时间
import time

import numpy as np

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class QueryRunner:
    """A class for running a query"""

    def __init__(
        self,
        query: Query,
        config: Dict,
        engine_schedulers: Dict[str, ray.actor.ActorHandle],
    ):

        self.query = query
        self.config = config
        self.dag = query.DAG
        self.engine_schedulers = engine_schedulers

        # Task queues
        self.running = deque()  # Running nodes
        self.ready = Queue()  # Ready nodes
        self.pending = deque()  # Pending nodes

        # Result store
        self.nodes_outputs = {}

        # Add tracking for different node types
        self.input_nodes = []
        self.compute_nodes = []
        self.output_nodes = []

        logger.info(f"query.query_state in query runner: {self.query.query_state}")

    def initialize(self):
        """Initialize the runner"""
        # Get topological sort
        self.dag.topological_sort()

        # Categorize nodes by type
        for node in self.dag.topo_list:
            if node.node_type == NodeType.INPUT:
                self.input_nodes.append(node)
            elif node.node_type == NodeType.COMPUTE:
                self.compute_nodes.append(node)
            elif node.node_type == NodeType.OUTPUT:
                self.output_nodes.append(node)

        # Initialize input nodes with values
        for node in self.input_nodes:
            node.status = NodeStatus.COMPLETED
            self.nodes_outputs[node.name] = node.input_values

        # Add compute and output nodes to pending
        self.pending.extend(self.compute_nodes)
        self.pending.extend(self.output_nodes)

        # log the dag's input nodes
        logger.info(f"DAG input nodes: {self.input_nodes}")
        logger.info(f"DAG compute nodes: {self.compute_nodes}")
        logger.info(f"DAG output nodes: {self.output_nodes}")

    def prepare_engine_payload(self, node: Node) -> Any:
        """prepare the data to send to the engine"""
        if node.node_type != NodeType.COMPUTE:
            return node.input_kwargs

        # get the corresponding transformer, if not, use the default transformer
        transformer = TRANSFORMER_REGISTRY.get(node.engine_type, DefaultTransformer())

        return transformer.transform(node)

    async def check_node_ready(self, node: Node) -> bool:
        """Check if a node is ready to execute"""
        # Input nodes are always ready
        if node.node_type == NodeType.INPUT:
            return True

        elif node.op_type in [NodeOps.LLM_DECODING, NodeOps.LLM_PARTIAL_DECODING]:
            parent_op_types = [parent.op_type for parent in node.parents]
            if NodeOps.LLM_PARTIAL_DECODING in parent_op_types:
                return node.parents[
                    parent_op_types.index(NodeOps.LLM_PARTIAL_DECODING)
                ].status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
            else:
                potential_prefilling_status = [
                    parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                    for parent in node.parents
                    if parent.op_type == NodeOps.LLM_PREFILLING
                ]
                potential_full_prefilling_status = [
                    parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                    for parent in node.parents
                    if parent.op_type == NodeOps.LLM_FULL_PREFILLING
                ]

                # Check if at least one parent node is prefilling type, and all such parent nodes are ready
                has_prefilling_parents = len(potential_prefilling_status) > 0
                has_full_prefilling_parents = len(potential_full_prefilling_status) > 0

                if (has_prefilling_parents and all(potential_prefilling_status)) or (
                    has_full_prefilling_parents
                    and all(potential_full_prefilling_status)
                ):
                    return True
                else:
                    return False

        elif node.op_type == NodeOps.LLM_FULL_PREFILLING:
            llm_partial_prefilling_status = [
                parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                for parent in node.parents
                if parent.op_type == NodeOps.LLM_PARTIAL_PREFILLING
            ]
            other_parents_status = [
                parent.status in [NodeStatus.COMPLETED]
                for parent in node.parents
                if parent.op_type not in [NodeOps.LLM_PARTIAL_PREFILLING]
            ]
            if all(llm_partial_prefilling_status) and all(other_parents_status):
                return True
            else:
                return False

        else:
            # For compute and output nodes, check parent completion
            return all(parent.status == NodeStatus.COMPLETED for parent in node.parents)

    async def submit_node(self, node: Node, stq: float, is_long_heavy: bool, source_id: str, contribution: float, priority: float):
        """submit a node to the engine scheduler"""
        try:
            logger.info(f"submit node: {node.name}")
            if node.node_type == NodeType.INPUT:
                node.start_time = time.time()
                return

            if node.node_type == NodeType.OUTPUT:
                if (
                    len(node.parents) == 1
                    and node.parents[0].status == NodeStatus.COMPLETED
                ):
                    node.status = NodeStatus.COMPLETED
                    # parent_node = node.parents[0]
                    node.update_input_kwargs(self.nodes_outputs)
                    self.nodes_outputs[node.name] = node.input_kwargs
                node.start_time = time.time()
                return

            if node.op_type == NodeOps.AGGREGATOR:
                # we do the in-place aggregation here, do not need to submit to the aggregator engine which would be discarded later
                aggregator_mode = get_aggregator_config(node)["agg_mode"]

                print(f"aggregatore node {node}, agg mode {aggregator_mode}")

                if aggregator_mode == AggMode.DUMMY:
                    node.status = NodeStatus.COMPLETED
                    self.nodes_outputs[node.name] = {k: True for k in node.output_names}

                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )
                    node.start_time = time.time()
                    return
                elif aggregator_mode == AggMode.MERGE:
                    node.status = NodeStatus.COMPLETED
                    node.update_input_kwargs(self.nodes_outputs)
                    if isinstance(list(node.input_kwargs.values())[0], dict):
                        output = {}
                    elif isinstance(list(node.input_kwargs.values())[0], list):
                        output = []
                    else:
                        raise ValueError(
                            f"Unsupported input type: {type(list(node.input_kwargs.values())[0])}"
                        )

                    for k, v in node.input_kwargs.items():
                        if isinstance(v, dict):
                            output.update(v)
                        elif isinstance(v, list):
                            output.extend(v)
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")

                    self.nodes_outputs[node.name] = output
                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )
                    node.start_time = time.time()
                    return

                elif aggregator_mode == AggMode.TOP_K:
                    # assume each output is a list of dicts or tuples
                    # if is dict, {"text": "xxx", "score": 0.9}
                    # if is tuple, ("xxx", 0.9)
                    node.status = NodeStatus.COMPLETED
                    node.update_input_kwargs(self.nodes_outputs)

                    logger.debug(
                        f"aggregator: {node.name} input_kwargs: {node.input_kwargs}"
                    )
                    top_k = node.config.get("top_k", None)
                    if top_k is None:
                        raise ValueError(
                            f"Missing top_k for node {node.name} aggregator"
                        )
                    output = []
                    for k, v in node.input_kwargs.items():
                        if isinstance(v, list):
                            if isinstance(v[0], dict):
                                keys = v[0].keys()
                                if "score" not in keys:
                                    raise ValueError(
                                        f"Dict keys do not contain 'score': {keys}"
                                    )
                                score_key = "score"
                                # Assume there is only one non-"score" field is the text
                                text_keys = [key for key in keys if key != "score"]
                                if not text_keys:
                                    raise ValueError(
                                        f"No text key found in dict: {keys}"
                                    )
                                text_key = text_keys[0]

                                output.extend(
                                    (v[i][text_key], v[i][score_key])
                                    for i in range(len(v))
                                )
                            elif isinstance(v[0], tuple):
                                if isinstance(v[0][0], str):
                                    text_key = 0
                                    score_key = 1
                                else:
                                    text_key = 1
                                    score_key = 0
                                output.extend(
                                    (v[i][text_key], v[i][score_key])
                                    for i in range(len(v))
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported input type: {type(v[0])}"
                                )
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")

                    # sort the output by score
                    output.sort(key=lambda x: x[1], reverse=True)
                    # keep the top k
                    output = output[:top_k]
                    # only keep the text
                    output = [x[0] for x in output]
                    self.nodes_outputs[node.name] = output
                    logger.info(
                        f"top {top_k} aggregation output for node: {node.name} output: {output}"
                    )
                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )
                    node.start_time = time.time()
                    return
                else:
                    raise ValueError(
                        f"Unsupported aggregator mode: {get_aggregator_config(node)}"
                    )

            # Address the compute node
            node.update_input_kwargs(self.nodes_outputs)

            payload = self.prepare_engine_payload(node)

            # print(f"payload: {payload} for node: {node.name}")

            engine_type = node.engine_type
            scheduler = self.engine_schedulers.get(engine_type)

            # 2 多实例
            engine_instance_index = node.config.get("engine_instance_index")

            if scheduler:
                engine_request = EngineRequest(
                    stq=stq,
                    is_long_heavy_node=is_long_heavy,
                    source_id = source_id,
                    contribution = contribution,
                    downstream_urgency=len(node.children),
                    priority=priority,
                    node_name=node.name,
                    op_type=str(node.op_type),
                    request_id=f"{self.query.query_id}::{node.name}",  # the engine scheduler will use the request_id.split("::")[-1] to get the node_name
                    query_id=self.query.query_id,
                    query=self.query,
                    payload=payload,
                    # 2 多实例
                    engine_instance_index=engine_instance_index
                )
                engine_request.timing_profile["T0_graph_scheduler_sent"] = time.perf_counter()
                await scheduler.submit_request.remote(engine_request)
                node.start_time = time.time()
                node.status = NodeStatus.RUNNING
            else:
                raise ValueError(f"No scheduler found for engine type: {engine_type}")

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(
                f"Error submitting node {node.name}: {str(e)}\nTraceback:\n{error_trace}"
            )
            node.status = NodeStatus.FAILED
            node.error_message = f"{str(e)}\nTraceback:\n{error_trace}"
            self.dag.error_nodes.append(node)

    async def cleanup_runtime_context(self):
        """Cleanup runtime context"""
        self.input_nodes.clear()
        self.compute_nodes.clear()
        self.output_nodes.clear()
        self.running.clear()
        self.ready.queue.clear()
        self.pending.clear()


@ray.remote
class GraphScheduler:
    """Scheduler for managing multiple queries and their DAG execution"""

    # 1 节点时间
    def get_all_completed_node(self):
        return self.all_completed_node
    def get_time_test_info(self):
        return self.time_test_info
    def get_excuted_dag(self):
        return

    def __init__(self, engine_schedulers: Dict[str, ray.actor.ActorHandle]):
        self.engine_schedulers = engine_schedulers
        self.query_runners: Dict[str, QueryRunner] = {}

        #全局缓存命中
        self.cache_actor: Optional[ray.actor.ActorHandle] = None

        #查询剩余时间STq
        # 结构: {
        #    "source_id_1": {"sq_init": ..., "total_nodes": ..., "completed_nodes": ..., "is_long": False},
        #    "source_id_2": {"sq_init": ..., "total_nodes": ..., "completed_nodes": ..., "is_long": True}
        # }
        self.query_stq_map: Dict[str, Dict] = {}
        self.signature_lock = asyncio.Lock()
        self.active_signature_map: Dict[str, Dict[str, Any]] = {}
        self.completed_signature_map: Dict[str, Any] = {}

        self.priority_alpha = 1.0
        self.priority_gamma = 0.3
        self.priority_delta = 0.5
        self.priority_epsilon = 1e-6
        self.long_query_eta = 1.0

        #初始λ值
        self.lambda_value = 0.5

        # 1 节点时间
        self.all_completed_node = []
        self.time_test_info = []
        self.excuted_dag = None

        self.trace_enabled = True
        self.trace_events: List[Dict[str, Any]] = []
        self.enable_runtime_reuse = True
        self.enable_completed_reuse = True
        self.enable_dynamic_bsq = True
        self.enable_priority_phi = True
        self.enable_priority_heavy = True
        self.enable_long_skewing = True
        self.enable_adaptive_batching = True
        self.lambda_low = 0.2
        self.lambda_high = 0.9


    def _record_trace(self, event: str, **fields) -> None:
        if not self.trace_enabled:
            return
        record = {
            "ts": time.perf_counter(),
            "component": "graph_scheduler",
            "event": event,
        }
        record.update(fields)
        self.trace_events.append(record)

    def get_trace_events(self) -> List[Dict[str, Any]]:
        return list(self.trace_events)

    def clear_trace_events(self) -> None:
        self.trace_events.clear()

    async def update_runtime_config(self, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update_lambda_value(self) -> float:
        active_source_count = len(self.query_stq_map)
        ready_count = 0
        running_count = 0
        for runner in self.query_runners.values():
            ready_count += runner.ready.qsize()
            running_count += len(runner.running)
        load_score = min(1.0, (active_source_count / 8.0) + (ready_count + running_count) / 32.0)
        self.lambda_value = self.lambda_low + (self.lambda_high - self.lambda_low) * load_score
        self._record_trace('lambda_update', active_source_count=active_source_count, ready_count=ready_count, running_count=running_count, lambda_value=self.lambda_value)
        return self.lambda_value

    def get_lambda(self) -> float:
        return self.lambda_value

    #更新查询剩余时间
    def get_current_sq(self, source_id: str) -> float:
        """Return a PLAS-inspired remaining-time estimate for one logical source query."""
        t_begin = time.perf_counter()
        if source_id not in self.query_stq_map:
            result = float('inf')
            self._record_trace('remaining_time_estimate', source_id=source_id, estimate_ms=result, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

        query_data = self.query_stq_map[source_id]
        profile_total_ms = max(query_data.get("profile_total_ms", 0.0), 0.0)
        profile_done_ms = max(query_data.get("profile_done_ms", 0.0), 0.0)
        attained_service_ms = max(query_data.get("attained_service_ms", 0.0), 0.0)
        remaining_profile_ms = max(profile_total_ms - profile_done_ms, 0.0)

        if not self.enable_dynamic_bsq:
            result = max(query_data.get("sq_init", remaining_profile_ms), 0.0)
            self._record_trace('remaining_time_estimate', source_id=source_id, estimate_ms=result, dynamic=False, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

        if remaining_profile_ms <= 0:
            result = 0.0
            self._record_trace('remaining_time_estimate', source_id=source_id, estimate_ms=result, dynamic=True, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

        eps_t = 1e-6
        if profile_done_ms <= eps_t or attained_service_ms <= eps_t:
            result = remaining_profile_ms
            self._record_trace('remaining_time_estimate', source_id=source_id, estimate_ms=result, dynamic=True, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

        correction_factor = attained_service_ms / (profile_done_ms + eps_t)
        result = max(correction_factor * remaining_profile_ms, eps_t)
        self._record_trace('remaining_time_estimate', source_id=source_id, estimate_ms=result, dynamic=True, duration_ms=(time.perf_counter() - t_begin) * 1000)
        return result

    #全局缓存命中
    async def set_global_cache_actor(self, cache_actor: ray.actor.ActorHandle):
        self.cache_actor = cache_actor

    async def update_schedulers(
        self, engine_schedulers: Dict[str, ray.actor.ActorHandle]
    ) -> None:
        """Update engine schedulers

        Args:
            engine_schedulers: Dictionary of engine type to scheduler actor handle
        """
        self.engine_schedulers = engine_schedulers


    def _advance_source_progress(self, node: Node, observed_ms: float) -> None:
        t_begin = time.perf_counter()
        source_id = node.query_source
        if not source_id or source_id not in self.query_stq_map:
            return
        if getattr(node, "_stq_accounted", False):
            return

        query_data = self.query_stq_map[source_id]
        old_sq = query_data.get("sq_init", 0.0)
        old_profile_done_ms = query_data.get("profile_done_ms", 0.0)
        old_attained_service_ms = query_data.get("attained_service_ms", 0.0)

        query_data["completed_compute_nodes"] += 1
        query_data["profile_done_ms"] += max(node.relative_ms or 0.0, 0.0)
        query_data["attained_service_ms"] += max(observed_ms, node.relative_ms or 0.0, 0.0)
        query_data["sq_init"] = max(
            query_data.get("profile_total_ms", 0.0) - query_data.get("profile_done_ms", 0.0),
            0.0,
        )
        node._stq_accounted = True

        self._record_trace(
            'stq_dynamic_update',
            source_id=source_id,
            node_name=node.name,
            op_type=str(node.op_type),
            observed_ms=observed_ms,
            profile_ms=max(node.relative_ms or 0.0, 0.0),
            old_sq_ms=old_sq,
            new_sq_ms=query_data.get("sq_init", 0.0),
            old_profile_done_ms=old_profile_done_ms,
            new_profile_done_ms=query_data.get("profile_done_ms", 0.0),
            old_attained_service_ms=old_attained_service_ms,
            new_attained_service_ms=query_data.get("attained_service_ms", 0.0),
            duration_ms=(time.perf_counter() - t_begin) * 1000,
        )

        if query_data["completed_compute_nodes"] >= query_data["total_compute_nodes"]:
            logger.info(f"源查询 {source_id} 的所有计算节点已完成，从 STq 追踪字典中移除。")
            del self.query_stq_map[source_id]

    def _normalize_signature_value(self, value: Any) -> Any:
        if isinstance(value, ray.ObjectRef):
            return {"ray_object_ref": str(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(k): self._normalize_signature_value(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_signature_value(v) for v in value]
        if isinstance(value, set):
            return [self._normalize_signature_value(v) for v in sorted(value, key=lambda item: str(item))]
        return str(value)

    def _signature_config(self, node: Node) -> Dict[str, Any]:
        transient_keys = {
            "llm_internal_id",
            "engine_instance_index",
            "request_id",
            "timing_profile",
            "callback_ref",
        }
        config = node.config or {}
        return {
            key: self._normalize_signature_value(value)
            for key, value in sorted(config.items(), key=lambda item: str(item[0]))
            if key not in transient_keys
        }

    def _collect_source_inputs(self, node: Node) -> Dict[str, Any]:
        t_begin = time.perf_counter()
        source_inputs: Dict[str, Any] = {}
        queue = deque(node.parents)
        visited = {parent.name for parent in node.parents}
        traversed = 0

        while queue:
            current_node = queue.popleft()
            traversed += 1
            if current_node.node_type == NodeType.INPUT:
                if getattr(current_node, "input_values", None):
                    source_inputs.update(current_node.input_values)
                continue
            for parent in current_node.parents:
                if parent.name not in visited:
                    visited.add(parent.name)
                    queue.append(parent)
        self._record_trace('source_input_traversal', node_name=node.name, traversed_nodes=traversed, duration_ms=(time.perf_counter() - t_begin) * 1000)
        return source_inputs

    def _get_signature_seed(self, node: Node) -> str:
        existing = getattr(node, "canonical_signature", "") or getattr(node, "fingerprint", "")
        if existing:
            return existing
        if node.node_type == NodeType.INPUT:
            payload = self._normalize_signature_value(getattr(node, "input_values", {}))
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            node.fingerprint = signature
            node.canonical_signature = signature
            return signature
        return node.name

    def build_node_signature(self, node: Node) -> str:
        t_begin = time.perf_counter()
        predecessor_signatures: List[str] = []
        for parent in sorted(node.parents, key=lambda item: item.name):
            predecessor_signatures.append(self._get_signature_seed(parent))

        signature_payload = {
            "op_type": str(node.op_type),
            "engine_type": str(node.engine_type),
            "params": self._signature_config(node),
            "input_summary": self._normalize_signature_value(self._collect_source_inputs(node)),
            "pred_sigs": predecessor_signatures,
        }
        node.signature_payload = signature_payload
        canonical = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        node.canonical_signature = signature
        self._record_trace(
            'build_signature',
            node_name=node.name,
            op_type=str(node.op_type),
            engine_type=str(node.engine_type),
            signature=signature[:16],
            duration_ms=(time.perf_counter() - t_begin) * 1000,
        )
        return signature

    def should_enable_reuse(self, node: Node) -> bool:
        if node.node_type != NodeType.COMPUTE:
            return False
        deterministic_ops = {
            NodeOps.EMBEDDING,
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.RERANKING,
        }
        return node.op_type in deterministic_ops

    async def try_reuse_node(self, node: Node, query_id: str) -> Dict[str, Any]:
        t_begin = time.perf_counter()
        if not self.should_enable_reuse(node) or not self.enable_runtime_reuse:
            result = {"action": "submit", "signature": ""}
            self._record_trace('reuse_decision', node_name=node.name, query_id=query_id, action='submit', reuse_enabled=False, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

        signature = self.build_node_signature(node)
        request_id = f"{query_id}::{node.name}"

        async with self.signature_lock:
            if self.enable_completed_reuse and signature in self.completed_signature_map:
                result = {
                    "action": "completed",
                    "signature": signature,
                    "result": self.completed_signature_map[signature],
                }
                self._record_trace('reuse_decision', node_name=node.name, query_id=query_id, action='completed', signature=signature[:16], active_map_size=len(self.active_signature_map), completed_map_size=len(self.completed_signature_map), duration_ms=(time.perf_counter() - t_begin) * 1000)
                return result

            active = self.active_signature_map.get(signature)
            if active and active.get("request_id") != request_id:
                active.setdefault("waiters", []).append((query_id, node.name))
                active["reuse_count"] = active.get("reuse_count", 1) + 1
                result = {
                    "action": "join",
                    "signature": signature,
                    "representative": active.get("request_id"),
                }
                self._record_trace('reuse_decision', node_name=node.name, query_id=query_id, action='join', signature=signature[:16], active_map_size=len(self.active_signature_map), completed_map_size=len(self.completed_signature_map), duration_ms=(time.perf_counter() - t_begin) * 1000)
                return result

            self.active_signature_map[signature] = {
                "request_id": request_id,
                "query_id": query_id,
                "node_name": node.name,
                "waiters": [],
                "reuse_count": 1,
            }
            result = {"action": "submit", "signature": signature}
            self._record_trace('reuse_decision', node_name=node.name, query_id=query_id, action='submit', signature=signature[:16], active_map_size=len(self.active_signature_map), completed_map_size=len(self.completed_signature_map), duration_ms=(time.perf_counter() - t_begin) * 1000)
            return result

    async def complete_reused_nodes(self, node: Node, result: Any) -> None:
        t_begin = time.perf_counter()
        signature = getattr(node, "canonical_signature", "")
        if not signature:
            return

        async with self.signature_lock:
            active = self.active_signature_map.pop(signature, None)
            self.completed_signature_map[signature] = result

        waiter_count = 0
        if active:
            for waiter_query_id, waiter_node_name in active.get("waiters", []):
                waiter_runner = self.query_runners.get(waiter_query_id)
                if waiter_runner is None:
                    continue
                waiter_node = waiter_runner.dag.get_node(waiter_node_name)
                if waiter_node is None:
                    continue
                await waiter_runner.query.query_state.set_node_result.remote(waiter_node.name, result)
                waiter_count += 1

        self._record_trace('reuse_fanout', node_name=node.name, signature=signature[:16], waiter_count=waiter_count, duration_ms=(time.perf_counter() - t_begin) * 1000)

    async def fail_reused_nodes(self, node: Node, error: Exception) -> None:
        signature = getattr(node, "canonical_signature", "")
        if not signature:
            return

        async with self.signature_lock:
            active = self.active_signature_map.pop(signature, None)

        if not active:
            return

        for waiter_query_id, waiter_node_name in active.get("waiters", []):
            waiter_runner = self.query_runners.get(waiter_query_id)
            if waiter_runner is None:
                continue
            waiter_node = waiter_runner.dag.get_node(waiter_node_name)
            if waiter_node is None:
                continue
            waiter_node.status = NodeStatus.FAILED
            waiter_node.error_message = str(error)
            await waiter_runner.query.query_state.set_node_result.remote(waiter_node.name, error)

    def compute_joint_priority(self, node: Node, stq: float, is_long_heavy: bool) -> float:
        t_begin = time.perf_counter()
        phi_term = math.log1p(max(node.contribution or 0.0, 0.0))
        heavy_term = 1.0 if is_long_heavy else 0.0

        priority = 0.0
        srtf_term = 0.0
        if self.enable_dynamic_bsq:
            srtf_term = self.priority_alpha / (max(stq, 0.0) + self.priority_epsilon)
            priority += srtf_term
        if self.enable_priority_phi:
            priority += self.priority_gamma * phi_term
        if self.enable_priority_heavy:
            priority += self.priority_delta * heavy_term
        if priority == 0.0:
            priority = max(node.contribution or 0.0, 0.0)

        node.scheduler_priority = priority
        self._record_trace(
            'priority_compute',
            node_name=node.name,
            source_id=node.query_source,
            op_type=str(node.op_type),
            priority=priority,
            srtf_term=srtf_term,
            phi_term=phi_term if self.enable_priority_phi else 0.0,
            heavy_term=heavy_term if self.enable_priority_heavy else 0.0,
            duration_ms=(time.perf_counter() - t_begin) * 1000,
        )
        return priority

    async def submit_query(self, query: Query, config: Optional[Dict] = None) -> str:
        """Submit a new query asynchronously"""
        try:

            temp_stq_map = {}
            for node in query.DAG.nodes:
                if node.node_type != NodeType.COMPUTE:
                    continue

                raw_source_id = node.query_source or "root"
                if isinstance(raw_source_id, str) and raw_source_id.startswith(f"{query.query_id}::"):
                    source_id = raw_source_id
                else:
                    source_id = f"{query.query_id}::{raw_source_id}"
                node.query_source = source_id

                if source_id not in temp_stq_map:
                    temp_stq_map[source_id] = {
                        "sq_init": 0.0,
                        "profile_total_ms": 0.0,
                        "profile_done_ms": 0.0,
                        "attained_service_ms": 0.0,
                        "total_compute_nodes": 0,
                        "completed_compute_nodes": 0,
                        "is_long": False,
                    }

                estimated_ms = max(node.relative_ms or 0.0, 0.0)
                temp_stq_map[source_id]["sq_init"] += estimated_ms
                temp_stq_map[source_id]["profile_total_ms"] += estimated_ms
                temp_stq_map[source_id]["total_compute_nodes"] += 1

            # 3. 将计算结果存入调度器的状态追踪器
            self.query_stq_map.update(temp_stq_map)
            logger.info(f"Initialized STq map: {self.query_stq_map}")

            runner = QueryRunner(
                query=query, config=config, engine_schedulers=self.engine_schedulers
            )
            self.query_runners[query.query_id] = runner

            # Initialize and start processing
            runner.initialize()
            asyncio.create_task(self._process_dag(runner))

            self.classify_long_source_queries()
            self.update_lambda_value()

            return query.query_id

        except Exception as e:
            logger.error(f"Failed to submit query {query.query_id}: {str(e)}")
            raise

    async def _process_dag(self, runner: QueryRunner):
        """Process DAG execution asynchronously"""
        try:
            last_pending_nodes_names = []
            last_running_nodes_names = []
            while not runner.dag.check_completion():
                # check the ready nodes
                ready_nodes = [
                    node
                    for node in runner.pending
                    if await runner.check_node_ready(node)
                ]

                #同查询内按照贡献度排序
                ready_nodes.sort(key=lambda n: -n.contribution)

                # batch update the queue
                for node in ready_nodes:
                    runner.ready.put(node)
                    runner.pending.remove(node)

                # batch submit the nodes
                while not runner.ready.empty():
                    nodes_to_submit = []
                    while not runner.ready.empty():
                        nodes_to_submit.append(runner.ready.get())

                    # 1 节点时间
                    #for node in nodes_to_submit:
                    #    node.start_time = time.time()

                    # parallel submit the nodes
                    # await asyncio.gather(
                    #     *[runner.submit_node(node) for node in nodes_to_submit]
                    # )

                    tasks_to_submit = []
                    scheduled_nodes = []
                    for node in nodes_to_submit:
                        current_stq = self.get_current_sq(node.query_source)
                        is_heavy = self.check_long_heavy_node(node)
                        source_id = node.query_source
                        contribution = node.contribution or 0.0
                        priority = self.compute_joint_priority(node, current_stq, is_heavy)
                        logger.info(
                            f"节点 {node.name}: STq={current_stq:.2f}, priority={priority:.4f}, is_long_heavy_node={is_heavy}"
                        )

                        reuse_state = await self.try_reuse_node(node, runner.query.query_id)
                        action = reuse_state.get("action")
                        if action == "completed":
                            cached_result = reuse_state["result"]
                            node.status = NodeStatus.COMPLETED
                            runner.nodes_outputs[node.name] = {node.output_names[0]: cached_result}
                            await runner.query.query_state.set_node_result.remote(node.name, cached_result)
                            node.start_time = time.time()
                            node.completed_time = node.start_time
                            self._advance_source_progress(node, node.relative_ms or 0.0)
                            self.all_completed_node.append(node)
                            continue
                        if action == "join":
                            node.status = NodeStatus.RUNNING
                            node.start_time = time.time()
                            runner.running.append(node)
                            continue

                        node.canonical_signature = reuse_state.get("signature", "")
                        tasks_to_submit.append(
                            runner.submit_node(
                                node,
                                current_stq,
                                is_heavy,
                                source_id,
                                contribution,
                                priority,
                            )
                        )
                        scheduled_nodes.append(node)

                    if tasks_to_submit:
                        await asyncio.gather(*tasks_to_submit)

                    for node in scheduled_nodes:
                        if node.status == NodeStatus.RUNNING:
                            runner.running.append(node)

                    self.classify_long_source_queries()
                    self.update_lambda_value()

                # print(f"runner.running: {[node.name for node in runner.running]}")

                current_pending_nodes_names = [node.name for node in runner.pending]
                current_running_nodes_names = [node.name for node in runner.running]

                if (
                    last_pending_nodes_names != current_pending_nodes_names
                    or last_running_nodes_names != current_running_nodes_names
                ):
                    logger.info(f"pending nodes: {current_pending_nodes_names}")
                    logger.info(f"running nodes: {current_running_nodes_names}")

                    last_pending_nodes_names = current_pending_nodes_names
                    last_running_nodes_names = current_running_nodes_names

                # batch check the running nodes
                if runner.running:
                    try:
                        completed_nodes = []  # add: store the completed nodes
                        results = await asyncio.gather(
                            *[
                                asyncio.wrap_future(
                                    runner.query.query_state.get_node_result.remote(
                                        node.name
                                    ).future()
                                )
                                for node in runner.running
                            ],
                            return_exceptions=True,
                        )

                        # check the errors in the results
                        for node, result in zip(runner.running, results):
                            if isinstance(result, Exception):
                                # handle the failed nodes
                                node.status = NodeStatus.FAILED
                                node.error_message = str(result)
                                runner.dag.error_nodes.append(node)
                                await self.fail_reused_nodes(node, result)
                                logger.error(
                                    f"Node execution failed: {node.name}: {str(result)}"
                                )
                                # terminate the DAG execution immediately
                                raise Exception(
                                    f"DAG execution terminated due to node failure: {node.name} - {str(result)}"
                                )

                            elif result is not None:
                                # handle the successful nodes
                                logger.info(
                                    f"result have been received for node: {node.name} in graph scheduler"
                                )
                                runner.nodes_outputs[node.name] = {
                                    node.output_names[0]: result
                                }
                                node.status = NodeStatus.COMPLETED

                                # 1 节点时间
                                node:Node
                                node.completed_time = time.time()

                                completed_nodes.append(
                                    node
                                )  # add to the completed nodes list

                                observed_ms = max((node.completed_time - node.start_time) * 1000, node.relative_ms or 0.0)
                                self._advance_source_progress(node, observed_ms)

                                # 全局缓存命中
                                fingerprint = getattr(node, 'fingerprint', None)
                                if fingerprint and self.cache_actor is not None and not hasattr(node,'is_cached_node'):
                                    logger.info(
                                        f"正在将节点 '{node.name}' 的结果写入全局缓存，指纹: {fingerprint[:12]}...")
                                    await self.cache_actor.set.remote(fingerprint, result)

                                await self.complete_reused_nodes(node, result)
                                self.classify_long_source_queries()
                                self.update_lambda_value()

                        # remove the completed nodes
                        for node in completed_nodes:
                            runner.running.remove(node)

                            # 1 节点时间
                            # self.time_test_info.append(node.name+"complete "+str(node.start_time)+" "+str(node.complete_time))
                            self.all_completed_node.append(node)

                    except Exception as e:
                        logger.error(f"Error in DAG execution: {str(e)}")
                        # mark all the running nodes as failed
                        for node in runner.running:
                            node.status = NodeStatus.FAILED
                            node.error_message = str(e)
                            runner.dag.error_nodes.append(node)
                        runner.running.clear()
                        # set the query status to failed and raise the exception
                        runner.query.status = QueryStatus.FAILED
                        raise

                # 1 节点时间
                self.excuted_dag = runner.dag
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(
                f"DAG execution failed for query {runner.query.query_id}: {str(e)}"
            )
            # ensure all the running nodes are marked as failed
            runner.dag.error_nodes.extend(runner.running)
            for node in runner.running:
                node.status = NodeStatus.FAILED
                node.error_message = str(e)
            runner.query.status = QueryStatus.FAILED
            # raise the exception to the upper layer
            raise Exception(f"DAG execution failed: {str(e)}")
        finally:
            # only set the query status to completed when there is no error
            if not runner.dag.error_nodes:
                runner.query.status = QueryStatus.COMPLETED
            await self.cleanup_query(runner.query.query_id)

    async def get_query_status(self, query_id: str) -> QueryStatus:
        """Get query status"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            return runner.query.status
        else:
            return None

    async def cleanup_query(self, query_id: str):
        """Clean up query resources"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            await runner.cleanup_runtime_context()

    async def shutdown(self):
        """Shutdown the graph scheduler"""
        for query_id, runner in self.query_runners.items():
            await self.cleanup_query(query_id)
        self.query_runners = {}

    def classify_long_source_queries(self):
        """Dynamically classify active source queries using the current remaining-time estimate."""
        t_begin = time.perf_counter()
        source_ids = list(self.query_stq_map.keys())

        if len(source_ids) <= 1:
            for source_id in source_ids:
                if source_id in self.query_stq_map:
                    self.query_stq_map[source_id]["is_long"] = False
            self._record_trace('classify_long_queries', source_count=len(source_ids), long_count=0, threshold=None, duration_ms=(time.perf_counter() - t_begin) * 1000)
            return

        stq_values = [self.get_current_sq(source_id) for source_id in source_ids]
        mean_stq = float(np.mean(stq_values))
        std_stq = float(np.std(stq_values))
        dynamic_threshold = mean_stq + self.long_query_eta * std_stq

        long_cnt = 0
        for source_id in source_ids:
            if source_id in self.query_stq_map:
                is_long = self.get_current_sq(source_id) > dynamic_threshold
                self.query_stq_map[source_id]["is_long"] = is_long
                long_cnt += int(is_long)

        if not long_cnt and stq_values:
            max_index = stq_values.index(max(stq_values))
            max_stq_sid = source_ids[max_index]
            self.query_stq_map[max_stq_sid]["is_long"] = True
            long_cnt = 1

        self._record_trace('classify_long_queries', source_count=len(source_ids), long_count=long_cnt, threshold=dynamic_threshold, duration_ms=(time.perf_counter() - t_begin) * 1000)

    def check_long_heavy_node(self, node: Node) -> bool:
        """Mark heavy nodes conservatively inside dynamically classified long queries."""
        source_id = node.query_source
        if source_id not in self.query_stq_map:
            return False

        if not self.enable_long_skewing:
            return False

        query_data = self.query_stq_map[source_id]
        if not query_data.get("is_long", False):
            return False

        heavy_ops = {
            NodeOps.LLM_PREFILLING,
            NodeOps.LLM_PARTIAL_PREFILLING,
            NodeOps.LLM_FULL_PREFILLING,
            NodeOps.LLM_DECODING,
            NodeOps.LLM_PARTIAL_DECODING,
            NodeOps.RERANKING,
            NodeOps.VECTORDB_SEARCHING,
        }
        avg_node_ms = query_data.get("profile_total_ms", 0.0) / max(query_data.get("total_compute_nodes", 1), 1)
        return node.op_type in heavy_ops or (node.relative_ms or 0.0) >= max(50.0, avg_node_ms)

    async def check_for_imminent_nodes(self, engine_type: str) -> Dict[str, Any]:
        """
        检查是否有即将就绪的、匹配特定引擎类型的节点。
        "即将就绪"定义为：其父节点中至少有一个正在运行中。
        """
        highest_contribution = -1
        found_imminent_node = False
        node_to_arrive = None
        for runner in self.query_runners.values():
            for node in runner.pending:
                if node.engine_type != engine_type:
                    continue

                is_imminent = any(parent.status in [NodeStatus.RUNNING,NodeStatus.COMPLETED] for parent in node.parents)

                if is_imminent:
                    found_imminent_node = True
                    node_score = getattr(
                        node,
                        "scheduler_priority",
                        self.compute_joint_priority(
                            node,
                            self.get_current_sq(node.query_source),
                            self.check_long_heavy_node(node),
                        ),
                    )
                    if node_score > highest_contribution:
                        highest_contribution = node_score
                        node_to_arrive = node
        return {"imminent": found_imminent_node, "contribution": highest_contribution, "node":node_to_arrive.name if node_to_arrive else None}

