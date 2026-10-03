"""Offline calibration, kept separate from the Eq. 4 per-query runtime correction."""

import json
import statistics
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


@lru_cache(maxsize=256)
def _encoded_key(values):
    return json.dumps(values)


class ProfileLibrary:
    def __init__(self, factors=None, batch_factors=None, batch_overheads=None):
        self.samples = defaultdict(list)
        self.factors = factors or {}
        self.batch_samples = defaultdict(lambda: defaultdict(list))
        self.batch_factors = batch_factors or {}
        self.overhead_samples = defaultdict(lambda: defaultdict(list))
        self.batch_overheads = batch_overheads or {}
        self._parameter_cache = {}

    @staticmethod
    def key(node):
        op = node.op_type.value if hasattr(node.op_type, "value") else node.op_type
        key = [op, node.config.get("model"), node.config.get("role")]
        if op.startswith("llm_"):
            key += [node.config.get("max_tokens"), node.config.get("token_budget")]
        return _encoded_key(tuple(key))

    @staticmethod
    def legacy_key(node):
        op = node.op_type.value if hasattr(node.op_type, "value") else node.op_type
        return _encoded_key((op, node.config.get("model"), node.config.get("role")))

    def predict(self, node):
        pipeline = node.config.get("pipeline")
        if pipeline is not None:
            # The lazy wrapper is an aggregator only for graph construction.
            # Eq. 4 requires the service of its unfinished operators, not the
            # aggregator's throughput applied to all the embedding/DB work.
            return sum(
                self.predict(
                    SimpleNamespace(
                        op_type=stage["op"],
                        config=stage["config"],
                        relative_ms=stage["relative_ms"],
                    )
                )
                for stage in pipeline["stages"]
            )
        initial = max(float(node.relative_ms or 1) / 1000, 1e-6)
        factor = self.factors.get(
            self.key(node), self.factors.get(self.legacy_key(node), 1)
        )
        return max(initial * factor, 1e-6)

    def observe(self, node, service_s):
        initial = max(float(node.relative_ms or 1) / 1000, 1e-6)
        self.samples[self.key(node)].append(service_s / initial)

    def freeze(self):
        self._parameter_cache.clear()
        self.factors.update(
            {key: statistics.median(values) for key, values in self.samples.items()}
        )
        self.batch_factors.update(
            {
                key: {
                    str(size): statistics.median(values)
                    for size, values in sizes.items()
                }
                for key, sizes in self.batch_samples.items()
            }
        )
        self.batch_overheads.update(
            {
                key: {
                    str(size): statistics.median(values)
                    for size, values in sizes.items()
                }
                for key, sizes in self.overhead_samples.items()
            }
        )

    def observe_batch(self, nodes, service_s, *, wall_s=None):
        units = sum(max(float(n.relative_ms or 1) / 1000, 1e-6) for n in nodes)
        self.batch_samples[self.key(nodes[0])][len(nodes)].append(service_s / units)
        if wall_s is not None:
            self.overhead_samples[self.key(nodes[0])][len(nodes)].append(
                max(0.0, wall_s - service_s)
            )

    def batch_parameters(self, node, size):
        key = self.key(node)
        identity = (key, size)
        if identity in self._parameter_cache:
            return self._parameter_cache[identity]
        curve = self.batch_factors.get(
            key, self.batch_factors.get(self.legacy_key(node), {})
        )
        overheads = self.batch_overheads.get(key, {})
        # A robust fixed dispatch cost avoids treating a cold-start or delayed
        # response at one sparse batch size as a hardware performance feature.
        overhead = statistics.median(overheads.values()) if overheads else 0.0
        factor = None
        if curve:
            sizes, totals, largest = [], [], 0.0
            for measured, value in sorted((int(k), v) for k, v in curve.items()):
                sizes.append(measured)
                # Interpolate total batch cost per uniform work unit. Nearest
                # per-item factors create artificial dips: a size-3 batch must
                # not be predicted as 3/4 of a measured size-4 kernel merely
                # because size 4 is the nearest calibration point.
                largest = max(largest, measured * value)
                totals.append(largest)
            position = bisect_left(sizes, size)
            if position == 0:
                total = totals[0]
            elif position == len(sizes):
                total = totals[-1] * size / sizes[-1]
            else:
                lower, upper = sizes[position - 1], sizes[position]
                weight = (size - lower) / (upper - lower)
                total = totals[position - 1] * (1 - weight) + totals[position] * weight
            factor = total / size
        self._parameter_cache[identity] = factor, overhead
        return factor, overhead

    def estimate_batch(self, node, size, units):
        """Same prediction as batch_time for a homogeneous, uniform-cost batch."""
        factor, overhead = self.batch_parameters(node, size)
        if factor is None:
            factor = self.factors.get(
                self.key(node), self.factors.get(self.legacy_key(node), 1)
            )
            return overhead + size * max(1e-6, units / size * factor)
        return overhead + max(1e-6, units * factor)

    def batch_time(self, nodes):
        factor, overhead = self.batch_parameters(nodes[0], len(nodes))
        if factor is None:
            return overhead + sum(self.predict(n) for n in nodes)
        units = sum(max(float(n.relative_ms or 1) / 1000, 1e-6) for n in nodes)
        return overhead + max(1e-6, units * factor)

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "version": 4,
                    "batch_fit": "monotone_total_linear_with_fixed_overhead",
                    "factors": self.factors,
                    "batch_factors": self.batch_factors,
                    "batch_overheads": self.batch_overheads,
                    "samples_per_operator": {
                        k: len(v) for k, v in self.samples.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(
            data["factors"],
            data.get("batch_factors", {}),
            data.get("batch_overheads", {}),
        )
