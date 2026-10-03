"""Data-parallel linear pipelines lowered only after their inputs are ready.

Stages must preserve the declared row axis. Global reductions belong after the
pipeline, not inside it. This contract makes splitting explicit and checkable.
"""

from copy import deepcopy

from HeraSys.dags.node_commons import NodeOps
from HeraSys.engines.engine_types import EngineType


def stage(node, *, map_ports=()):
    """Describe an existing primitive; map_ports lifts a scalar operator over rows."""
    if node.op_type not in {
        NodeOps.EMBEDDING,
        NodeOps.VECTORDB_INGESTION,
        NodeOps.VECTORDB_SEARCHING,
        NodeOps.RERANKING,
    }:
        raise ValueError("Parallel pipelines require stateless row-preserving stages")
    if any(port not in node.input_names for port in map_ports):
        raise ValueError("Mapped ports must be stage inputs")
    return {
        "name": node.name,
        "op": node.op_type.value,
        "engine": node.engine_type.value,
        "inputs": list(node.input_names),
        "outputs": list(node.output_names),
        "config": {
            **deepcopy(node.config),
            **({"map_ports": list(map_ports)} if map_ports else {}),
        },
        "relative_ms": node.relative_ms or 1.0,
    }


def pipeline(name, stages, *, inputs, split_ports, max_splits=4):
    """Construct a lazy pipeline node; input bindings may come from any predecessor."""
    from .workflows import primitive

    if not stages or max_splits < 1 or not split_ports:
        raise ValueError(
            "A pipeline needs stages, a row axis and a positive split limit"
        )
    if not set(split_ports).issubset(inputs):
        raise ValueError("Split ports must be pipeline inputs")
    available = set(inputs)
    produced = set()
    for spec in stages:
        if not set(spec["inputs"]).issubset(available):
            raise ValueError(f"Unbound pipeline stage: {spec['name']}")
        if produced and not set(spec["inputs"]).intersection(produced):
            raise ValueError("Pipeline stages must form one dependent linear chain")
        produced = set(spec["outputs"])
        if not produced or produced.intersection(available):
            raise ValueError("Pipeline output ports must be nonempty and distinct")
        available.update(produced)
    return primitive(
        name,
        NodeOps.AGGREGATOR,
        EngineType.AGGREGATOR,
        inputs,
        stages[-1]["outputs"],
        ms=sum(s["relative_ms"] for s in stages),
        pipeline={
            "stages": deepcopy(stages),
            "split_ports": list(split_ports),
            "max_splits": max_splits,
        },
        _hera_reuse=True,
    )


def partitions(spec, inputs, count):
    """Balanced contiguous partitions; no element duplication, loss or reordering."""
    sizes = []
    for port in spec["split_ports"]:
        value = inputs[port]
        if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__len__"):
            raise ValueError(f"Pipeline split port {port} is not a row collection")
        sizes.append(len(value))
    if len(set(sizes)) != 1:
        raise ValueError("Pipeline split ports must have the same row count")
    size = sizes[0]
    count = min(count, spec["max_splits"], max(1, size))
    return [
        {
            port: value[lane * size // count : (lane + 1) * size // count]
            if port in spec["split_ports"]
            else value
            for port, value in inputs.items()
        }
        for lane in range(count)
    ]


def expand_mapped(requests):
    """Expand mapped requests into the existing batched backend protocol."""
    expanded, layout = [], []
    for request in requests:
        ports = request["config"].get("map_ports", ())
        if not ports:
            layout.append((len(expanded), None, request))
            expanded.append(request)
            continue
        if request["op"] not in {
            "embedding",
            "vectordb_ingestion",
            "vectordb_searching",
            "reranking",
        }:
            raise ValueError("Mapped execution supports only stateless row operators")
        lengths = [len(request["inputs"][port]) for port in ports]
        if len(set(lengths)) != 1 or any(
            isinstance(request["inputs"][port], (str, bytes, dict)) for port in ports
        ):
            raise ValueError("Mapped inputs must be aligned row collections")
        count = lengths[0]
        layout.append((len(expanded), count, request))
        for row in range(count):
            expanded.append(
                {
                    **request,
                    "config": {
                        k: v for k, v in request["config"].items() if k != "map_ports"
                    },
                    "inputs": {
                        port: value[row] if port in ports else value
                        for port, value in request["inputs"].items()
                    },
                }
            )
    return expanded, layout


def collect_mapped(results, layout):
    collected = []
    for start, count, request in layout:
        if count is None:
            collected.append(results[start])
        else:
            rows = results[start : start + count]
            collected.append(
                {
                    "output": {
                        port: [r["output"][port] for r in rows]
                        for port in request["outputs"]
                    },
                    "service_s": max((r["service_s"] for r in rows), default=0.0),
                }
            )
    return collected
