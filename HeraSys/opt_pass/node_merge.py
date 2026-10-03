import hashlib
import json
from collections import deque
from typing import Any, Dict, List

from HeraSys.dags.dag import DAG
from HeraSys.dags.node import Node, NodeType
from HeraSys.dags.node_commons import NodeOps
from HeraSys.opt_pass.base_pass import OPT_Pass


class SourceBasedMergingPass(OPT_Pass):

    def __init__(self, similarity_threshold: float = 1.0, n_gram_size: int = 5):
        super().__init__(name="source_based_merging")
        self.similarity_threshold = similarity_threshold
        self.n_gram_size = n_gram_size
        self.source_input_cache: Dict[str, Dict[str, Any]] = {}
        self.signature_to_node: Dict[str, Node] = {}

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {
                str(k): self._normalize(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        if isinstance(value, set):
            return [self._normalize(v) for v in sorted(value, key=lambda item: str(item))]
        return str(value)

    def _stable_config(self, node: Node) -> Dict[str, Any]:
        transient_keys = {
            "llm_internal_id",
            "engine_instance_index",
            "request_id",
            "timing_profile",
            "callback_ref",
        }
        return {
            key: self._normalize(value)
            for key, value in sorted((node.config or {}).items(), key=lambda item: str(item[0]))
            if key not in transient_keys
        }

    def _get_source_inputs(self, node: Node) -> Dict[str, Any]:
        if node.name in self.source_input_cache:
            return self.source_input_cache[node.name]

        source_inputs: Dict[str, Any] = {}
        queue = deque(node.parents)
        visited = {parent.name for parent in node.parents}

        while queue:
            current_node = queue.popleft()
            if current_node.node_type == NodeType.INPUT:
                if getattr(current_node, "input_values", None):
                    source_inputs.update(current_node.input_values)
                continue
            for parent in current_node.parents:
                if parent.name not in visited:
                    visited.add(parent.name)
                    queue.append(parent)

        self.source_input_cache[node.name] = source_inputs
        return source_inputs

    def _build_signature(self, node: Node) -> str:
        predecessor_signatures = []
        for parent in sorted(node.parents, key=lambda item: item.name):
            parent_sig = getattr(parent, "canonical_signature", "") or getattr(parent, "fingerprint", "") or parent.name
            predecessor_signatures.append(parent_sig)

        payload = {
            "op_type": str(node.op_type),
            "engine_type": str(node.engine_type),
            "params": self._stable_config(node),
            "input_summary": self._normalize(self._get_source_inputs(node)),
            "pred_sigs": predecessor_signatures,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        node.signature_payload = payload
        node.canonical_signature = signature
        node.fingerprint = signature
        return signature

    def _is_merge_allowed(self, node: Node) -> bool:
        mergeable_ops = {
            NodeOps.EMBEDDING,
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.RERANKING,
            NodeOps.LLM_PREFILLING,
            NodeOps.LLM_PARTIAL_PREFILLING,
        }
        return node.op_type in mergeable_ops

    def run(self, dag: DAG) -> DAG:
        self.log_optimization("开始执行公共子节点优化...")

        self.source_input_cache = {}
        self.signature_to_node = {}
        nodes_to_remove: List[Node] = []

        for node in dag.topological_sort():
            if node.node_type == NodeType.INPUT:
                payload = self._normalize(node.input_values)
                canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                node.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                node.canonical_signature = node.fingerprint
                continue

            if node.node_type != NodeType.COMPUTE:
                continue

            signature = self._build_signature(node)
            if not self._is_merge_allowed(node):
                self.signature_to_node.setdefault(signature, node)
                continue

            original_node = self.signature_to_node.get(signature)
            if original_node is None or original_node.name == node.name:
                self.signature_to_node[signature] = node
                continue

            self.log_optimization(
                f"发现等价节点: '{node.name}' -> '{original_node.name}' (signature={signature[:12]}...)"
            )

            if node.op_type in {NodeOps.LLM_PREFILLING, NodeOps.LLM_PARTIAL_PREFILLING}:
                original_id = original_node.config.get("llm_internal_id")
                redundant_id = node.config.get("llm_internal_id")
                if original_id and redundant_id:
                    if not isinstance(original_id, tuple):
                        original_node.config["llm_internal_id"] = (original_id,)
                    if redundant_id not in original_node.config["llm_internal_id"]:
                        original_node.config["llm_internal_id"] += (redundant_id,)

            for child in list(node.children):
                if child not in original_node.children:
                    original_node.add_child(child)
                if node in child.parents:
                    child.parents.remove(node)

            node.children.clear()
            nodes_to_remove.append(node)

        if nodes_to_remove:
            self.log_optimization(f"共发现 {len(nodes_to_remove)} 个可优化的冗余节点。")
            for node in nodes_to_remove:
                dag.remove_node(node.name)
        else:
            self.log_optimization("未发现可优化的公共子节点。")

        self.log_optimization("公共子节点优化完成。")
        return dag
