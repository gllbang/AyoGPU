"""Report explicit parameter settings without inventing the paper's presets."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = defaultdict(list)
    for path in sorted(args.directory.glob("*/*/summary.json")):
        case = json.loads(path.read_text())
        if case["failed"] or not (path.parent.parent / "VERIFICATION.json").exists():
            raise ValueError(f"Unverified case: {path}")
        rows[path.parent.parent.name].append(case)
    lines = [
        "# 本地参数敏感性",
        "",
        "RAG＋Rerank，100 并发，完整段落。参数值明确列出；论文未公开 Table 3 各预设的具体取值，"
        "且该表使用含 Google 的 Mixed，因此这里不声称复现该表数值。",
        "",
        "| 配置 | α | γ | δ | ρ | 重复 | P50 (s) | P95 (s) | P99 (s) | Mean (s) | QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cases in sorted(rows.items()):
        configs = {
            tuple(c["config"][p] for p in ("alpha", "gamma", "delta", "rho"))
            for c in cases
        }
        if len(configs) != 1:
            raise ValueError(f"Mixed parameter settings in {name}")
        params = " | ".join(str(v) for v in next(iter(configs)))
        metrics = " | ".join(
            f"{statistics.median(c[k] for c in cases):.4f}"
            for k in ("p50_s", "p95_s", "p99_s", "mean_s", "qps")
        )
        lines.append(f"| {name} | {params} | {len(cases)} | {metrics} |")
    if not rows:
        raise ValueError("No verified sensitivity results")
    (args.directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.directory / "REPORT.md")


if __name__ == "__main__":
    main()
