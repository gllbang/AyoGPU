"""Pair separately executed full-stack and shared-engine experiments safely."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(directory):
    env = json.loads((directory / "environment.json").read_text())
    settings = env["arguments"]
    dataset = Path(settings["dataset"])
    expected = env["dataset_sha256"]
    expected = expected["rag"] if isinstance(expected, dict) else expected
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Dataset changed since measurement: {directory}")
    verification_path = directory / "VERIFICATION.json"
    if not verification_path.exists():
        raise ValueError(
            f"Verify all request accounting and source archives first: {directory}"
        )
    verification = json.loads(verification_path.read_text())
    source_id = hashlib.sha256(
        json.dumps(env["source_sha256"], sort_keys=True).encode()
    ).hexdigest()
    if not verification.get("verified") or verification.get("source_id") != source_id:
        raise ValueError(f"Invalid verification or changed manifest: {directory}")
    if not env.get("gpus"):
        raise ValueError(f"GPU inventory was not recorded in this older run: {directory}")
    comparison = {
        key: settings.get(key)
        for key in (
            "seed",
            "duplicate_rate",
            "max_tokens",
            "max_model_len",
            "llm_model",
            "embedding_model",
            "reranker_model",
            "reranker_dtype",
            "llm_instances",
            "model_instances",
        )
    }
    comparison.update(
        dataset=expected,
        arrival_mode=settings.get("arrival_mode", "closed_loop"),
        llm_max_sessions=settings.get("llm_max_sessions", settings.get("max_sessions")),
        gpu_memory_utilization=settings.get(
            "gpu_memory_utilization", settings.get("memory_fraction")
        ),
        gpus=env["gpus"],
    )
    for path in sorted(directory.glob("*/summary.json")):
        row = json.loads(path.read_text())
        if row["failed"] or row.get("measurement_kind") != "performance":
            raise ValueError(f"Failed or diagnostic case: {path}")
        if row["source_id"] != source_id:
            raise ValueError(f"Case source differs from verified source: {path}")
        yield row, path.parent, {**comparison, "vector_backend": row["vector_backend"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="herasys")
    parser.add_argument("--variant-label", help="Display name; raw variant IDs remain unchanged")
    parser.add_argument("--baseline-label", help="Display name for a single explicit baseline")
    parser.add_argument(
        "--baselines",
        nargs="+",
        help="Explicit baseline variants; all input directories must still be verified",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    if args.baseline_label and (not args.baselines or len(args.baselines) != 1):
        parser.error("--baseline-label requires exactly one explicit --baselines entry")
    labels = {args.variant: args.variant_label or args.variant}
    if args.baseline_label:
        labels[args.baselines[0]] = args.baseline_label
    cases = {}
    for directory in args.directories:
        for row, path, comparison in load(directory):
            if args.baselines is not None and row["variant"] not in {
                args.variant,
                *args.baselines,
            }:
                continue
            key = (row["workload"], row["concurrency"], row["repeat"], row["variant"])
            if key in cases:
                raise ValueError(f"Overlapping runs must not be silently pooled: {key}")
            cases[key] = row, path, comparison
    pairs = []
    for (workload, concurrency, repeat, variant), (
        base,
        base_path,
        settings,
    ) in cases.items():
        if variant == args.variant:
            continue
        match = cases.get((workload, concurrency, repeat, args.variant))
        if match is None:
            continue
        ours, our_path, our_settings = match
        if settings != our_settings or base["requests"] != ours["requests"]:
            raise ValueError(f"Incompatible settings: {base_path} and {our_path}")
        baseline_outputs = {
            r["query_id"]: r["output"]
            for r in map(
                json.loads, (base_path / "outputs.jsonl").read_text().splitlines()
            )
        }
        outputs = list(
            map(json.loads, (our_path / "outputs.jsonl").read_text().splitlines())
        )
        pairs.append(
            {
                "workload": workload,
                "concurrency": concurrency,
                "repeat": repeat,
                "variant": args.variant,
                "baseline": variant,
                "system": labels[args.variant],
                "baseline_system": labels.get(variant, variant),
                "requests": ours["requests"],
                "qps_speedup": ours["qps"] / base["qps"],
                "p99_reduction_pct": 100 * (1 - ours["p99_s"] / base["p99_s"]),
                "different_outputs": sum(
                    r["output"] != baseline_outputs[r["query_id"]] for r in outputs
                ),
                "variant_case": str(our_path.resolve()),
                "baseline_case": str(base_path.resolve()),
                "variant_source": ours["source_id"],
                "baseline_source": base["source_id"],
            }
        )
    if not pairs:
        raise ValueError("No comparable pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["workload"], row["concurrency"], row["baseline"]].append(row)
    lines = [
        "# " + labels[args.variant] + " 实验结果",
        "",
        "每对使用相同模型、完整文本、生成预算、请求数、并发和重复种子。源版本逐项记录；"
        "完整 Ayo 的原生引擎与共享后端不同，单独标明。数值是本机测量，不能视为论文表格数值复现。",
        "",
        "| 系统 | 工作负载 | 并发 | baseline | 重复 | QPS 倍率中位数（范围） | P99 降幅中位数 | 双项改善 | 输出不同/配对请求 |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    if args.baselines is not None:
        lines[4:4] = ["比较基线：" + ", ".join(labels.get(v, v) for v in args.baselines) + "。", ""]
    for (workload, concurrency, baseline), rows in sorted(grouped.items()):
        qps = [r["qps_speedup"] for r in rows]
        tail = [r["p99_reduction_pct"] for r in rows]
        wins = sum(r["qps_speedup"] > 1 and r["p99_reduction_pct"] > 0 for r in rows)
        lines.append(
            f"| {labels[args.variant]} | {workload} | {concurrency} | {labels.get(baseline, baseline)} | {len(rows)} | "
            f"{statistics.median(qps):.3f} ({min(qps):.3f}–{max(qps):.3f}) | "
            f"{statistics.median(tail):.2f}% | {wins}/{len(rows)} | "
            f"{sum(r['different_outputs'] for r in rows)}/{sum(r['requests'] for r in rows)} |"
        )
    lines += [
        "",
        "范围描述实际重复之间的波动，不是置信区间。负降幅表示退步。输出差异保留在原始文件中，"
        "另需结合答案质量诊断和相同提示词重放判断；不能直接称为无损。",
        "",
    ]
    if args.table_only:
        lines = [line for line in lines if line.startswith("|")]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grouped_cases = defaultdict(list)
        for (workload, concurrency, repeat, variant), (row, _, _) in cases.items():
            grouped_cases[workload, concurrency, variant].append(row)
        workloads = sorted({key[0] for key in grouped_cases})
        variants = sorted({key[2] for key in grouped_cases})
        for metric, ylabel in [
            ("qps", "Completed requests/s"),
            ("mean_s", "Mean latency (s)"),
            ("p99_s", "P99 latency (s)"),
        ]:
            fig, axes = plt.subplots(
                1, len(workloads), figsize=(6 * len(workloads), 4.5), squeeze=False
            )
            for ax, workload in zip(axes[0], workloads):
                for variant in variants:
                    points = sorted(
                        (
                            c,
                            statistics.median(r[metric] for r in rows),
                            min(r[metric] for r in rows),
                            max(r[metric] for r in rows),
                        )
                        for (w, c, v), rows in grouped_cases.items()
                        if w == workload and v == variant
                    )
                    if points:
                        x, y, lower, upper = zip(*points)
                        ax.plot(x, y, marker="o", label=labels.get(variant, variant))
                        ax.fill_between(x, lower, upper, alpha=0.12)
                ax.set(title=workload, xlabel="Concurrency", ylabel=ylabel)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
            fig.suptitle("Local full-text measurement (median and observed range)")
            fig.tight_layout()
            fig.savefig(args.output / f"{metric}.png", dpi=180)
            fig.savefig(args.output / f"{metric}.svg")
            plt.close(fig)
    print(f"{args.output / 'REPORT.md'}: {len(pairs)} paired cases")


if __name__ == "__main__":
    main()
