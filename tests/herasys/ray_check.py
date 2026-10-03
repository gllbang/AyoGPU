"""Opt-in coordinator integration; uses real Ray transport with reference engines."""

import ray

from HeraSys.backends import reference_pools
from HeraSys.config import HeraConfig
from HeraSys.policy import split_factor
from HeraSys.ray_api import HeraSysActor
from HeraSys.workflows import rag
from benchmarks.herasys import FIXTURE


def main():
    ray.init(include_dashboard=False, log_to_driver=False, num_cpus=2)
    actor = HeraSysActor.remote(reference_pools())
    try:
        item = FIXTURE[0]
        dag, inputs = rag(item["question"], item["documents"], max_tokens=8)
        refs = [actor.execute.remote(dag, inputs, query_id=f"q{i}") for i in range(6)]
        results = ray.get(refs, timeout=60)
        assert all(result == results[0] for result in results)
        metrics = ray.get(actor.metrics.remote())
        assert sum(q["reused_nodes"] for q in metrics["queries"]) > 0

        def factory(load):
            return rag(
                item["question"],
                item["documents"],
                splits=split_factor(load, len(item["documents"]), HeraConfig()),
                max_tokens=8,
            )

        refs = [
            actor.execute_factory.remote(factory, query_id=f"factory{i}")
            for i in range(6)
        ]
        factory_results = ray.get(refs, timeout=60)
        assert all(result == results[0] for result in factory_results)
        metrics = ray.get(actor.metrics.remote())
        assert len(metrics["queries"]) == 12
        assert all(query["status"] == "completed" for query in metrics["queries"])
        ray.get(actor.shutdown.remote())
        assert ray.get(actor.metrics.remote())["physical_tasks"] == 0
        print("HERASYS_RAY_CHECK_OK: direct and factory clients, fusion, and cleanup")
    finally:
        ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()
