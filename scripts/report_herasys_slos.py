"""Compute goodput at explicit latency budgets from every recorded request."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=[5, 10, 15, 20, 30, 60]
    )
    args = parser.parse_args()
    if min(args.budgets) <= 0:
        parser.error("latency budgets must be positive")
    rows = []
    for path in sorted(args.directory.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("measurement_kind") == "diagnostic" or summary["config"].get(
            "audit_nodes"
        ):
            raise ValueError("Diagnostic timing cannot be used to evaluate goodput")
        queries = list(
            map(json.loads, (path.parent / "queries.jsonl").read_text().splitlines())
        )
        if len(queries) != summary["requests"]:
            raise ValueError(f"Incomplete request accounting: {path.parent}")
        for budget in sorted(set(args.budgets)):
            met = sum(
                q["status"] == "completed" and q["latency_s"] <= budget for q in queries
            )
            rows.append(
                {
                    "run": path.parent.name,
                    "workload": summary["workload"],
                    "variant": summary["variant"],
                    "concurrency": summary["concurrency"],
                    "repeat": summary["repeat"],
                    "slo_s": budget,
                    "met": met,
                    "requests": len(queries),
                    "fraction_met": met / len(queries),
                    "goodput_qps": met / summary["elapsed_s"],
                }
            )
    (args.directory / "slo_goodput.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Goodput under explicit latency budgets",
        "",
        "Budgets are diagnostic choices, not undocumented paper settings. A request counts only if it succeeds within the budget; the denominator is the full measured wall time. All requests, including violations and failures, remain in the accounting.",
        "",
        "| Run | Budget (s) | Requests within budget | Fraction | Goodput (QPS) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['slo_s']:g} | {row['met']}/{row['requests']} | {row['fraction_met']:.3f} | {row['goodput_qps']:.3f} |"
        )
    (args.directory / "SLO_GOODPUT.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} goodput observations written")


if __name__ == "__main__":
    main()
