"""Verify a completed experiment's source snapshot, request accounting and KV cleanup."""

import argparse
import hashlib
import json
import math
import tarfile
from itertools import product
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    env = json.loads((args.directory / "environment.json").read_text())
    settings = env["arguments"]
    with tarfile.open(args.directory / "source.tar.gz") as archive:
        for name, expected in env["source_sha256"].items():
            source = archive.extractfile(name)
            if source is None or hashlib.sha256(source.read()).hexdigest() != expected:
                raise ValueError(f"Source snapshot mismatch: {name}")
    source_id = hashlib.sha256(
        json.dumps(env["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    expected_cases = set(
        product(
            settings["workloads"],
            settings["variants"],
            settings["concurrency"],
            range(settings["repeats"]),
        )
    )
    seen, total = set(), 0
    for path in sorted(args.directory.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        identity = tuple(
            summary[k] for k in ("workload", "variant", "concurrency", "repeat")
        )
        if identity in seen or identity not in expected_cases:
            raise ValueError(f"Unexpected or repeated case: {identity}")
        seen.add(identity)
        if (
            summary["source_id"] != source_id
            or summary["requests"] != settings["requests"]
        ):
            raise ValueError(f"Incompatible case metadata: {identity}")
        if summary["failed"] or summary["completed"] != summary["requests"]:
            raise ValueError(f"Failed/incomplete requests: {identity}")
        queries = list(
            map(json.loads, (path.parent / "queries.jsonl").read_text().splitlines())
        )
        outputs = list(
            map(json.loads, (path.parent / "outputs.jsonl").read_text().splitlines())
        )
        expected_ids = {f"q{i}" for i in range(settings["requests"])}
        for records in (queries, outputs):
            if (
                len(records) != len(expected_ids)
                or {row["query_id"] for row in records} != expected_ids
            ):
                raise ValueError(f"Request IDs/counts do not match: {identity}")
        if any(
            q["status"] != "completed"
            or not math.isfinite(q["latency_s"])
            or q["latency_s"] < 0
            for q in queries
        ):
            raise ValueError(f"Invalid query result: {identity}")
        if any("error" in row or "output" not in row for row in outputs):
            raise ValueError(f"Invalid output result: {identity}")
        latencies = sorted(q["latency_s"] for q in queries)
        for name, fraction in (("p50_s", 0.5), ("p95_s", 0.95), ("p99_s", 0.99)):
            location = (len(latencies) - 1) * fraction
            lower = int(location)
            value = latencies[lower] + (
                latencies[min(lower + 1, len(latencies) - 1)] - latencies[lower]
            ) * (location - lower)
            if not math.isclose(value, summary[name], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Incorrect {name}: {identity}")
        if not math.isclose(
            summary["qps"], len(queries) / summary["elapsed_s"], rel_tol=1e-10
        ):
            raise ValueError(f"Incorrect throughput: {identity}")
        if settings["backend"] == "gpu":
            cleanup = summary["backend_cleanup"]
            if len(cleanup) != settings["llm_instances"] or any(
                state["live_states"]
                or state["kv_free_blocks"] != state["kv_total_blocks"]
                for state in cleanup
            ):
                raise ValueError(f"KV state not fully released: {identity}")
        total += len(queries)
    if seen != expected_cases:
        raise ValueError(f"Missing {len(expected_cases - seen)} cases")
    manifest = args.directory / "backend_manifest.json"
    if manifest.exists():
        workers = json.loads(manifest.read_text())
        rerankers = [r for r in workers if r["kind"] == "reranking"]
        if len(rerankers) != settings["model_instances"] or any(
            r["model_dtype"] != "torch." + settings["reranker_dtype"] for r in rerankers
        ):
            raise ValueError(
                "Actual reranker precision differs from the declared setting"
            )
    result = {
        "verified": True,
        "cases": len(seen),
        "successful_requests": total,
        "source_files": len(env["source_sha256"]),
        "source_id": source_id,
        "checked": [
            "source archive hashes",
            "complete case matrix",
            "every query/output ID",
            "zero request failures",
            "latency percentiles",
            "throughput",
            "KV cleanup",
        ],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.directory / "VERIFICATION.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
