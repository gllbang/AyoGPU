from HeraSys.dags.node_commons import NodeOps


PERFORMANCE_COEFFICIENTS = {
    # 非LLM节点
    NodeOps.EMBEDDING: 0.0037,

    NodeOps.VECTORDB_INGESTION: 0.01,

    NodeOps.VECTORDB_SEARCHING: 0.01,

    NodeOps.RERANKING: 0.5,

    # LLM节点
    # 假设：LLM处理/生成1个Token耗时2ms
    NodeOps.LLM_PREFILLING: 2.0,
    NodeOps.LLM_FULL_PREFILLING: 2.0,
    NodeOps.LLM_PARTIAL_PREFILLING: 2.0,
    NodeOps.LLM_DECODING: 2.0,
    NodeOps.LLM_PARTIAL_DECODING: 2.0,

    # 其他节点
    NodeOps.AGGREGATOR: 0.0001,
}


def get_estimated_time(workload: float, op_type: NodeOps) -> float:
    """
    根据节点的原始工作量和操作类型，查询性能库并返回预估执行时间（毫秒）。

    Args:
        workload: 节点的原始工作量 (来自 estimate_workloads 函数)。
        op_type: 节点的操作类型。

    Returns:
        预估的执行时间（毫秒）。
    """
    # 查找对应的性能系数，如果找不到，则默认系数为0.5
    coefficient = PERFORMANCE_COEFFICIENTS.get(op_type, 0.05)

    # 相对工作时间 = 原始工作量 * 性能系数
    return workload * coefficient