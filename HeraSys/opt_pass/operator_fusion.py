# 文件: HeraSys/opt_pass/node_fusion_pass.py (新文件)

from typing import List, Dict
from HeraSys.dags.dag import DAG
from HeraSys.dags.node import Node, NodeType, EngineType, NodeOps, NodeIOSchema
from HeraSys.opt_pass.base_pass import OPT_Pass
from HeraSys.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class NodeFusionPass(OPT_Pass):
    def __init__(self):
        super().__init__(name="node_fusion")

    def run(self, dag: DAG) -> DAG:
        self.log_optimization("开始执行[节点融合-图重写]优化...")

        fusion_patterns = [(NodeOps.VECTORDB_SEARCHING, NodeOps.RERANKING)]
        nodes_to_process = dag.topological_sort()

        for start_node in nodes_to_process:
            if start_node not in dag.nodes: continue  # 可能已经被移除了

            for start_op, next_op in fusion_patterns:
                if start_node.op_type == start_op and len(start_node.children) == 1:
                    next_node = start_node.children[0]
                    if next_node.op_type == next_op:
                        self.log_optimization(
                            f"发现可融合对: '{start_node.name}' -> '{next_node.name}'. 正在执行图重写...")

                        # 1. 创建全新的融合节点
                        fused_node_name = f"{start_node.name}_fused_with_{next_node.name}"

                        # 融合节点的输入是两个节点所有输入的并集
                        fused_input_format = {}
                        for p in start_node.parents + next_node.parents:
                            # 假设父节点的输出名就是子节点的输入key
                            fused_input_format.update(p.io_schema.output_format)

                        fused_node = Node(
                            name=fused_node_name,
                            node_type=NodeType.COMPUTE,
                            # 【核心】指定全新的引擎类型
                            engine_type=EngineType.FUSED_SEARCH_RERANK,
                            op_type=NodeOps.FUSED_SEARCH_RERANK,  # 【核心】定义全新的操作类型
                            io_schema=NodeIOSchema(
                                input_format=fused_input_format,
                                output_format=next_node.io_schema.output_format  # 输出与最后一个节点相同
                            ),
                            config=deepcopy(start_node.config)
                        )
                        fused_node.config.update(deepcopy(next_node.config))

                        # 2. 处理依赖关系
                        # a. 融合节点的父节点 = a的父节点 + b的父节点 (去重)
                        all_parents = list(set(start_node.parents + next_node.parents))
                        for parent in all_parents:
                            fused_node.add_parent(parent)
                            parent.add_child(fused_node)

                        # b. 融合节点的子节点 = b的子节点
                        for child in list(next_node.children):
                            fused_node.add_child(child)
                            child.add_parent(fused_node)

                        # 3. 从图中移除旧节点
                        dag.remove_node(start_node.name)
                        dag.remove_node(next_node.name)

                        # 4. 注册新节点
                        dag.register_nodes(fused_node)

                        # 因为修改了图，需要重新开始遍历
                        return self.run(dag)  # 递归调用以处理后续的融合机会

        self.log_optimization("节点融合优化完成。")
        return dag