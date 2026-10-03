from typing import Dict, Any
import ray
from HeraSys.dags.dag import DAG
from HeraSys.dags.node import Node, NodeType
from HeraSys.opt_pass.base_pass import OPT_Pass


class GlobalCachePass(OPT_Pass):

    def __init__(self, global_cache: Dict[str, Any]):
        super().__init__(name="global_cache_pass")
        self.global_cache = global_cache
        self.nodes_to_remove = set()

    def _prune_upstream(self, node: Node):
        """递归地向上游剪枝。"""
        for parent in list(node.parents):
            # 断开父子关系
            parent.children.remove(node)
            node.parents.remove(parent)

            # 检查父节点是否还有其他子节点
            if not parent.children:
                # 如果没有，则这个父节点也没用了，将其加入待删除列表
                self.nodes_to_remove.add(parent)
                # 继续向上递归剪枝
                self._prune_upstream(parent)

    def run(self, dag: DAG) -> DAG:
        self.log_optimization("开始执行全局缓存优化...")
        self.nodes_to_remove.clear()

        if not self.global_cache:
            self.log_optimization("全局缓存为空，跳过此轮优化。")
            return dag

        # 从下游向上游遍历，最大化剪枝效果
        sorted_nodes = reversed(dag.topological_sort())

        for node in sorted_nodes:
            if node.node_type != NodeType.COMPUTE or node in self.nodes_to_remove:
                continue

            fingerprint = getattr(node, 'fingerprint', None)
            if fingerprint and fingerprint in self.global_cache:
                self.log_optimization(f"缓存命中: 节点 '{node.name}' (指纹: {fingerprint[:12]}...) 的结果已在缓存中。")

                # 1. 注入结果
                cached_result = self.global_cache[fingerprint]
                node.is_cached_node = True  # 标记，防止重复
                node.node_type = NodeType.INPUT
                node.input_values = {node.output_names[0]: cached_result}
                node.input_kwargs = {node.output_names[0]: cached_result}
                node.op_type = None
                node.engine_type = None

                # 2. 向上游递归剪枝
                self._prune_upstream(node)

        if self.nodes_to_remove:
            self.log_optimization(f"共发现 {len(self.nodes_to_remove)} 个因缓存命中而冗余的上游节点")
            for node in self.nodes_to_remove:
                for child in list(node.children):
                    child.parents.remove(node)
                for parent in list(node.parents):
                    parent.children.remove(node)
                dag.remove_node(node.name)
        else:
            self.log_optimization("未发现可利用的全局缓存。")

        self.log_optimization("全局缓存剪枝优化完成。")
        return dag