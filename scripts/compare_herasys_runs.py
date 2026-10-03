"""Paired, per-repeat comparisons from one measured experiment directory."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="ayo_static")
    args = parser.parse_args()
    runs = [
        json.loads(path.read_text())
        for path in sorted(args.directory.glob("*/summary.json"))
    ]
    baselines = {
        (row["workload"], row["concurrency"], row["repeat"]): row
        for row in runs
        if row["variant"] == args.baseline
    }
    if any(
        row.get("measurement_kind") == "diagnostic"
        or row.get("config", {}).get("audit_nodes")
        for row in runs
    ):
        raise SystemExit(
            "Diagnostic node-audit runs cannot be used for performance comparisons"
        )
    pairs, groups = [], defaultdict(list)
    for row in runs:
        key = (row["workload"], row["concurrency"], row["repeat"])
        if row["variant"] == args.baseline or key not in baselines:
            continue
        base = baselines[key]
        if row["failed"] or base["failed"] or row["requests"] != base["requests"]:
            raise SystemExit(f"Failed or unequal-size pair: {key} / {row['variant']}")
        pair = {
            "workload": row["workload"],
            "concurrency": row["concurrency"],
            "repeat": row["repeat"],
            "variant": row["variant"],
            "baseline": args.baseline,
            "p99_speedup": base["p99_s"] / row["p99_s"],
            "qps_speedup": row["qps"] / base["qps"],
            "mean_speedup": base["mean_s"] / row["mean_s"],
        }
        pairs.append(pair)
        groups[(row["workload"], row["concurrency"], row["variant"])].append(pair)
    if not pairs:
        raise SystemExit("No completed paired comparisons yet")
    lines = [
        f"# Paired comparisons against {args.baseline}",
        "",
        "Ratios above 1 favor the compared variant. Each ratio pairs the same workload, "
        "concurrency, request count, and repeat seed within this experiment. "
        "Ranges below describe observed runs, not confidence intervals. "
        "Aging remains an explicitly separate optional extension.",
        "",
        "| Workload | Concurrency | Variant | Repeats | Median P99 speedup | P99 range | "
        "Median QPS speedup | QPS range | Runs improving both |",
        "|---|---:|---|---:|---:|---|---:|---|---:|",
    ]
    for (workload, concurrency, variant), values in sorted(groups.items()):
        tails = [row["p99_speedup"] for row in values]
        qps = [row["qps_speedup"] for row in values]
        wins = sum(row["p99_speedup"] > 1 and row["qps_speedup"] > 1 for row in values)
        lines.append(
            f"| {workload} | {concurrency} | {variant} | {len(values)} | "
            f"{statistics.median(tails):.3f} | {min(tails):.3f}–{max(tails):.3f} | "
            f"{statistics.median(qps):.3f} | {min(qps):.3f}–{max(qps):.3f} | {wins}/{len(values)} |"
        )
    lines += [
        "",
        "## Individual pairs",
        "",
        "| Workload | Concurrency | Variant | Repeat | P99 speedup | QPS speedup | Mean speedup |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['workload']} | {pair['concurrency']} | {pair['variant']} | "
            f"{pair['repeat']} | {pair['p99_speedup']:.3f} | {pair['qps_speedup']:.3f} | "
            f"{pair['mean_speedup']:.3f} |"
        )
    target = args.directory / f"COMPARISON-{args.baseline}.md"
    target.write_text("\n".join(lines) + "\n")
    target.with_suffix(".json").write_text(json.dumps(pairs, indent=2) + "\n")
    print(f"{target}: {len(pairs)} paired cases")


if __name__ == "__main__":
    main()
