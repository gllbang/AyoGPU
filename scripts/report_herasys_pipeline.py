"""Report measured pipeline granularity without imposing the paper's node counts."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from report_herasys_suite import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = defaultdict(list)
    settings_by_repeat = {}
    seen = set()
    for directory in args.directories:
        for case, path, settings in load(directory):
            if (
                case["workload"] != "rag_rerank"
                or case["concurrency"] not in (6, 100)
                or case["variant"]
                not in ("herasys", "no_pipeline", "ayo_upstream_full_batched")
            ):
                continue
            key = (case["concurrency"], case["variant"], case["repeat"])
            if key in seen:
                raise ValueError(f"Duplicate pipeline case: {path}")
            seen.add(key)
            pair = (case["concurrency"], case["repeat"])
            setting = {**settings, "requests": case["requests"]}
            previous = settings_by_repeat.setdefault(pair, setting)
            if setting != previous:
                raise ValueError(f"Incompatible pipeline experiment: {path}")
            rows[key[:2]].append(case)
    required = {(c, v) for c in (6, 100) for v in ("herasys", "no_pipeline")}
    if not required.issubset(rows):
        raise ValueError(f"Missing pipeline cases: {required - rows.keys()}")
    lines = [
        "# 动态流水线的实测粒度",
        "",
        "RAG＋Rerank，完整段落；各指标取重复中位数。动态模式列出实际运行时拆分路数，"
        "按真实拆分事件平均，未执行拆分的缓存命中不计入该平均；固定模式列出构图拆分路数。"
        "节点数来自所有完成请求的逻辑图统计，"
        "不会强行对齐论文的 19 节点；原生 Ayo 未记录相同口径，标为未记录。",
        "",
        "| 并发 | 配置 | 重复 | 请求/轮 | 平均逻辑节点 | 平均拆分路数 | Mean (s) | P99 (s) | QPS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (concurrency, variant), cases in sorted(rows.items()):
        node_counts = [
            c["mean_logical_nodes"]
            for c in cases
            if c.get("mean_logical_nodes") is not None
        ]
        splits = [
            c.get("mean_runtime_splits")
            if c.get("mean_runtime_splits") is not None
            else c.get("mean_splits", c["config"].get("fixed_splits"))
            for c in cases
        ]
        nodes = f"{statistics.median(node_counts):.3f}" if node_counts else "未记录"
        split = (
            f"{statistics.median(splits):.3f}"
            if all(s is not None for s in splits)
            else "未记录"
        )
        metrics = " | ".join(
            f"{statistics.median(c[k] for c in cases):.4f}"
            for k in ("mean_s", "p99_s", "qps")
        )
        requests = {c["requests"] for c in cases}
        if len(requests) != 1:
            raise ValueError("Request counts differ between repetitions")
        lines.append(
            f"| {concurrency} | {variant} | {len(cases)} | {requests.pop()} | "
            f"{nodes} | {split} | {metrics} |"
        )
    lines += [
        "",
        "完整原生 Ayo 使用独立引擎和原生图，且开启显式标注的跨请求批处理增强。"
        "归因动态拆分效果应比较 herasys 与 no_pipeline；Ayo 一栏是端到端对照。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
