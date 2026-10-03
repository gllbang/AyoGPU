"""Summarize measured runs and audit generation outputs; never import paper numbers."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--audit-baseline",
        default="ayo_static",
        help="Variant used for output comparison",
    )
    args = p.parse_args()
    root = Path(args.directory)
    runs = [
        (path.parent, json.loads(path.read_text()))
        for path in sorted(root.glob("*/summary.json"))
    ]
    if not runs:
        raise SystemExit("No completed run summaries")
    summaries = [summary for _, summary in runs]
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    grouped = defaultdict(list)
    for _, summary in runs:
        grouped[
            (summary["workload"], summary["concurrency"], summary["variant"])
        ].append(summary)
    lines = [
        "# HeraSys measured experiment report",
        "",
        "These are local validation measurements. They do not reproduce the numerical claims of Table 1.",
        "",
        "`ayo_static` is a controlled scheduling baseline on the shared engine pool. "
        "`ayo_upstream_graph` adapts the original Ayo graph scheduler to that pool; "
        "it does not measure the original APP frontend or engine queues.",
        "",
        "HeraSys uses the Appendix B.2 query-aging extension with a local 2-second threshold. "
        "Unaged controls remain separate. The recorded settings below identify each variant; "
        "names alone must not be used to merge results from different source snapshots.",
        "",
        "| Workload | Concurrency | Variant | Runs | Mean (s) | P99 (s) | QPS | Failures |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    if any(
        s.get("measurement_kind") == "diagnostic" or s["config"].get("audit_nodes")
        for s in summaries
    ):
        lines[2:2] = [
            "**Diagnostic output capture is enabled. Timings include audit overhead and must not be used as performance evidence.**",
            "",
        ]
    for (workload, concurrency, variant), values in sorted(grouped.items()):

        def formatted_mean(key, digits=4):
            measured = [v[key] for v in values if v[key] is not None]
            return f"{statistics.fmean(measured):.{digits}f}" if measured else "n/a"

        lines.append(
            f"| {workload} | {concurrency} | {variant} | {len(values)} | {formatted_mean('mean_s')} | "
            f"{formatted_mean('p99_s')} | {formatted_mean('qps', 3)} | {sum(v['failed'] for v in values)} |"
        )
    baselines = {}
    for path, summary in runs:
        if summary["variant"] == args.audit_baseline:
            baselines[
                (summary["workload"], summary["concurrency"], summary["repeat"])
            ] = {
                r["query_id"]: r.get("output")
                for r in map(
                    json.loads, (path / "outputs.jsonl").read_text().splitlines()
                )
            }
    audit = []
    for path, summary in runs:
        key = (summary["workload"], summary["concurrency"], summary["repeat"])
        if key not in baselines or summary["variant"] == args.audit_baseline:
            continue
        outputs = [
            json.loads(line)
            for line in (path / "outputs.jsonl").read_text().splitlines()
        ]
        differences = [
            r["query_id"]
            for r in outputs
            if r.get("output") != baselines[key].get(r["query_id"])
        ]
        audit.append(
            {
                "run": path.name,
                "baseline": args.audit_baseline,
                "compared": len(outputs),
                "different_outputs": len(differences),
                "query_ids": differences,
            }
        )
    (root / "output_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    settings = {
        (
            s["variant"],
            s["config"].get("contribution_scope", "not recorded"),
            s["config"].get("max_query_wait_s"),
            s["config"].get("max_ready_wait_s"),
            s.get("arrival_mode", "not recorded"),
        )
        for s in summaries
    }
    lines += [
        "",
        "| Recorded variant | Successor scope | Query aging (s) | Node aging (s) | Arrival mode |",
        "|---|---|---:|---:|---|",
    ]
    for variant, scope, query_age, node_age, arrival in sorted(settings, key=str):
        lines.append(
            f"| {variant} | {scope} | {query_age if query_age is not None else 'off'} | "
            f"{node_age if node_age is not None else 'off'} | {arrival} |"
        )
    lines += [
        "",
        f"Completed {sum(s['completed'] for s in summaries)} requests across {len(summaries)} cases; "
        f"failures: {sum(s['failed'] for s in summaries)}.",
        "",
        f"Output audit baseline: `{args.audit_baseline}`. "
        f"Compared {sum(a['compared'] for a in audit)} outputs; "
        f"{sum(a['different_outputs'] for a in audit)} differ.",
        "",
        "P99 from small samples is a diagnostic only. See each summary's `tail_sample_warning`, environment.json, "
        "profiles.json (when calibrated), raw traces, and output_audit.json. Numerical differences in batched FP16 "
        "generation may change decoded text; exact KV-fork/chunk correctness is tested separately.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        workloads = sorted({s["workload"] for s in summaries})
        variants = sorted({s["variant"] for s in summaries})
        for metric, ylabel, filename in [
            ("qps", "Measured QPS", "throughput"),
            ("p99_s", "Measured P99 latency (s)", "tail_latency"),
        ]:
            fig, axes = plt.subplots(
                1, len(workloads), figsize=(5 * len(workloads), 4), squeeze=False
            )
            for ax, workload in zip(axes[0], workloads):
                for variant in variants:
                    entries = []
                    for (w, c, v), values in grouped.items():
                        measured = [s[metric] for s in values if s[metric] is not None]
                        if w == workload and v == variant and measured:
                            entries.append((c, statistics.fmean(measured)))
                    if entries:
                        ax.plot(*zip(*sorted(entries)), marker="o", label=variant)
                ax.set(title=workload, xlabel="Concurrency limit", ylabel=ylabel)
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend(fontsize=8)
            fig.suptitle("Local validation — not the paper testbed")
            fig.tight_layout()
            fig.savefig(root / f"{filename}.png", dpi=180)
            fig.savefig(root / f"{filename}.svg")
            plt.close(fig)
    print(
        f"{root / 'REPORT.md'}: {len(runs)} cases; output differences "
        f"{sum(a['different_outputs'] for a in audit)}/{sum(a['compared'] for a in audit)}"
    )


if __name__ == "__main__":
    main()
