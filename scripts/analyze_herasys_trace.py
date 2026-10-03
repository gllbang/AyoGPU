"""Explain long requests using recorded scheduler events, without rerunning models."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze(directory, top):
    summary = json.loads((directory / "summary.json").read_text())
    queries = [
        json.loads(line)
        for line in (directory / "queries.jsonl").read_text().splitlines()
    ]
    worst = sorted(queries, key=lambda row: row["latency_s"], reverse=True)[:top]
    selected = {row["query_id"] for row in worst}
    arrivals, dispatches = {}, {}
    stages, waits = defaultdict(list), defaultdict(list)
    predicted_batches, prediction_ratios = {}, defaultdict(list)
    active_events, long_events = defaultdict(int), defaultdict(int)
    dispatch_count = aging_count = schedule_count = no_long_count = 0
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["event"]
            if kind == "query_arrival" and event["query_id"] in selected:
                arrivals[event["query_id"]] = event["time_s"]
            elif kind == "schedule":
                if event["active"]:
                    schedule_count += 1
                    no_long_count += not event["long_queries"]
                longs = set(event["long_queries"])
                for query in selected.intersection(event["remaining_s"]):
                    active_events[query] += 1
                    long_events[query] += query in longs
            elif kind == "dispatch":
                dispatch_count += 1
                aging_count += bool(event.get("aging_override"))
                ready_waits = event.get("ready_wait_s")
                if ready_waits is not None:
                    waits[event["engine"]].extend(ready_waits)
                if "predicted_batch_s" in event:
                    predicted_batches[dispatch_count] = {
                        "remaining": set(event["task_ids"]),
                        "predicted_s": event["predicted_batch_s"],
                        "observed_s": 0.0,
                        "engine": event["engine"],
                    }
                for index, task in enumerate(event.get("task_ids", [])):
                    dispatches[task] = {
                        "batch_id": dispatch_count,
                        "dispatch_time_s": event["time_s"],
                        "engine": event["engine"],
                        "batch_size": event["batch_size"],
                        "ready_wait_s": ready_waits[index] if ready_waits else None,
                        "aging_override": event.get("aging_override"),
                    }
            elif kind in {"node_complete", "terminal_decode"}:
                dispatch = dispatches.pop(event["task_id"], None)
                if dispatch and kind == "node_complete":
                    batch_id = dispatch["batch_id"]
                    prediction = predicted_batches.get(batch_id)
                    if prediction is not None:
                        prediction["remaining"].discard(event["task_id"])
                        prediction["observed_s"] = max(
                            prediction["observed_s"], event["elapsed_s"]
                        )
                        if not prediction["remaining"]:
                            if prediction["observed_s"] > 0:
                                prediction_ratios[prediction["engine"]].append(
                                    prediction["predicted_s"] / prediction["observed_s"]
                                )
                            del predicted_batches[batch_id]
                for query, node in event["owners"]:
                    if query not in selected:
                        continue
                    start = arrivals.get(query)
                    if start is None:
                        continue
                    stages[query].append(
                        {
                            "node": node,
                            "task_id": event["task_id"],
                            "dispatch_since_arrival_s": (
                                dispatch["dispatch_time_s"] - start
                                if dispatch
                                else None
                            ),
                            "complete_since_arrival_s": event["time_s"] - start,
                            "service_s": event["service_s"],
                            "ready_wait_s": dispatch["ready_wait_s"]
                            if dispatch
                            else None,
                            "engine": dispatch["engine"] if dispatch else None,
                            "batch_size": dispatch["batch_size"] if dispatch else None,
                            "aging_override": dispatch["aging_override"]
                            if dispatch
                            else None,
                        }
                    )
    return {
        "case": directory.name,
        "summary": {
            key: summary[key]
            for key in ["variant", "concurrency", "completed", "failed", "p99_s", "qps"]
        },
        "active_schedule_events": schedule_count,
        "events_without_long_queries": no_long_count,
        "dispatch_batches": dispatch_count,
        "aging_override_batches": aging_count,
        "batch_prediction_accuracy": {
            engine: {
                "completed_batches": len(ratios),
                "median_predicted_over_observed": statistics.median(ratios),
                "fraction_within_factor_two": sum(0.5 <= r <= 2 for r in ratios)
                / len(ratios),
            }
            for engine, ratios in sorted(prediction_ratios.items())
        },
        "ready_wait_by_engine": {
            engine: {
                "physical_nodes": len(values),
                "mean_s": statistics.fmean(values),
                "p99_s": percentile(values, 0.99),
                "max_s": max(values),
            }
            for engine, values in sorted(waits.items())
        },
        "worst_queries": [
            {
                **query,
                "active_schedule_events": active_events[query["query_id"]],
                "classified_long_events": long_events[query["query_id"]],
                "stages": stages[query["query_id"]],
            }
            for query in worst
        ],
    }


def render(analyses):
    lines = [
        "# HeraSys tail-latency trace analysis",
        "",
        "Measured events only. Service time is a sum over logical nodes and may overlap; "
        "subtracting it from latency is not an exact queue-time decomposition. "
        "Ready-wait values describe physical tasks, including affinity/session-capacity "
        "waits; they are unavailable in older traces. A shared task may start before "
        "a later owner's arrival. Cache-hit nodes have no execution event. "
        "Long-query counts are event counts, not time-weighted fractions.",
        "",
    ]
    for analysis in analyses:
        lines += [
            f"## {analysis['case']}",
            "",
        ]
        if not analysis["active_schedule_events"]:
            lines += [
                "Detailed HeraSys scheduler events are unavailable in this adapter's "
                "trace; no node-wait or long-query diagnosis is inferred.",
                "",
            ]
            continue
        lines += [
            f"Aging override batches: {analysis['aging_override_batches']} / "
            f"{analysis['dispatch_batches']}. Active schedule events with no long query: "
            f"{analysis['events_without_long_queries']} / "
            f"{analysis['active_schedule_events']}.",
            "",
        ]
        if analysis["ready_wait_by_engine"]:
            lines += [
                "| Engine | Physical nodes | Mean ready wait (s) | P99 (s) | Max (s) |",
                "|---|---:|---:|---:|---:|",
            ]
            for engine, row in analysis["ready_wait_by_engine"].items():
                lines.append(
                    f"| {engine} | {row['physical_nodes']} | {row['mean_s']:.3f} | "
                    f"{row['p99_s']:.3f} | {row['max_s']:.3f} |"
                )
            lines.append("")
        if analysis["batch_prediction_accuracy"]:
            lines += [
                "Batch prediction includes measured dispatch overhead. Observed time "
                "is the maximum recorded execution elapsed time among that batch's "
                "nodes; queue waiting is excluded. Only fully completed batches with "
                "a recorded prediction are included. A ratio near one indicates a "
                "more accurate cost model, not necessarily a better scheduling policy.",
                "",
                "| Engine | Complete batches | Median predicted / observed | Within 2x |",
                "|---|---:|---:|---:|",
            ]
            for engine, row in analysis["batch_prediction_accuracy"].items():
                lines.append(
                    f"| {engine} | {row['completed_batches']} | "
                    f"{row['median_predicted_over_observed']:.3f} | "
                    f"{row['fraction_within_factor_two']:.1%} |"
                )
            lines.append("")
        for query in analysis["worst_queries"]:
            lines += [
                f"### {query['query_id']}",
                "",
                f"Runtime latency: {query['latency_s']:.3f} s; attained service: "
                f"{query['attained_service_s']:.3f} s; classified long at "
                f"{query['classified_long_events']} / "
                f"{query['active_schedule_events']} active scheduling events.",
                "",
                "| Node | Dispatch after arrival (s) | Complete after arrival (s) | "
                "Service (s) | Physical ready wait (s) |",
                "|---|---:|---:|---:|---:|",
            ]
            for stage in query["stages"]:
                values = [
                    "unavailable" if stage[key] is None else f"{stage[key]:.3f}"
                    for key in [
                        "dispatch_since_arrival_s",
                        "complete_since_arrival_s",
                        "service_s",
                        "ready_wait_s",
                    ]
                ]
                lines.append(f"| {stage['node']} | " + " | ".join(values) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Experiment root or one case directory"
    )
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    cases = (
        [args.directory]
        if (args.directory / "trace.jsonl").exists()
        else sorted(path.parent for path in args.directory.glob("*/trace.jsonl"))
    )
    if not cases:
        parser.error("No completed case traces found")
    analyses = [analyze(path, args.top) for path in cases]
    (args.directory / "latency_analysis.json").write_text(
        json.dumps(analyses, indent=2) + "\n"
    )
    target = args.directory / "LATENCY_ANALYSIS.md"
    target.write_text(render(analyses))
    print(f"{target}: analyzed {len(cases)} cases")


if __name__ == "__main__":
    main()
