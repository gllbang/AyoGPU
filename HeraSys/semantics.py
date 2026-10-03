"""Content addressing for Eq. 2. Input bindings, not node names, define identity."""

import hashlib
import json
import math
from enum import Enum


def canonical(value):
    if isinstance(value, Enum):
        return canonical(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values cannot be hashed as semantic input")
        return value
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("Semantic dictionaries require string keys")
        return ["dict", [[k, canonical(v)] for k, v in sorted(value.items())]]
    if isinstance(value, list):
        return ["list", [canonical(v) for v in value]]
    if isinstance(value, tuple):
        return ["tuple", [canonical(v) for v in value]]
    if hasattr(value, "tolist"):
        return canonical(value.tolist())
    raise TypeError(f"Unsupported semantic input: {type(value).__name__}")


def digest(value):
    encoded = json.dumps(
        canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parameters(config):
    # Private scheduler metadata and request handles do not change an operator's output.
    return {
        k: v
        for k, v in config.items()
        if not k.startswith("_hera_")
        and k
        not in {
            "llm_internal_id",
            "engine_instance_index",
            "request_id",
            "timing_profile",
        }
    }


def signature(node, bindings, predecessor_signatures, namespace):
    return digest(
        {
            "namespace": namespace,
            "op": node.op_type,
            "engine": node.engine_type,
            "parameters": parameters(node.config),
            "bindings": bindings,
            "controls": sorted(predecessor_signatures),
            "outputs": list(node.io_schema.output_format),
        }
    )


def prefill_batch_key(config):
    # Formatting is applied per request by GPUWorker. Preserve generation
    # limits/sampling parameters: mixing short rewrites with long answers
    # fragments subsequent decode batches even though the kernel accepts it.
    generation = {
        key: value
        for key, value in parameters(config).items()
        if key not in {"question_port", "context_port", "prompt_template"}
    }
    return digest(["prefill", generation])
