"""Locate observed generation differences without assuming they are numerical noise."""

import argparse
import json
from collections import Counter
from pathlib import Path


def jsonl(path):
    return {
        row["query_id"]: row for row in map(json.loads, path.read_text().splitlines())
    }


def locate(left, right):
    """Describe the earliest observed divergence; missing cache provenance stays unknown."""
    sessions = ["rewrite", "answer"] if "rewrite_prefill" in left else ["answer"]
    details = []
    for session in sessions:
        name = f"{session}_prefill"
        a, b = left.get(name, {}).get("llm"), right.get(name, {}).get("llm")
        if not a or not b:
            return "missing_cached_provenance", details
        if a["prompt_token_ids"] != b["prompt_token_ids"]:
            changed = [
                n
                for n in ("retrieve", "rerank", "web_search")
                if n in left
                and n in right
                and left[n]["output_sha256"] != right[n]["output_sha256"]
            ]
            details.append({"node": name, "changed_upstream": changed})
            return "different_prompt", details
        if a["token_ids"] != b["token_ids"]:
            details.append(
                {
                    "node": name,
                    "left_tokens": a["token_ids"],
                    "right_tokens": b["token_ids"],
                }
            )
            return "same_prompt_prefill_difference", details
        decodes = sorted(
            (n for n in left if n.startswith(f"{session}_decode_")),
            key=lambda n: int(n.rsplit("_", 1)[1]),
        )
        for n in decodes:
            a, b = left[n].get("llm"), right.get(n, {}).get("llm")
            if a and b and a["token_ids"] != b["token_ids"]:
                common = 0
                for x, y in zip(a["token_ids"], b["token_ids"]):
                    if x != y:
                        break
                    common += 1
                details.append(
                    {
                        "node": n,
                        "common_prefix_tokens": common,
                        "left_tokens": a["token_ids"],
                        "right_tokens": b["token_ids"],
                    }
                )
                return "same_prefill_later_decode_difference", details
    return "unresolved", details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    generation_limit = json.loads((args.directory / "environment.json").read_text())[
        "arguments"
    ]["max_tokens"]
    summaries = [
        (p.parent, json.loads(p.read_text()))
        for p in sorted(args.directory.glob("*/summary.json"))
    ]
    baselines = {
        (s["workload"], s["concurrency"], s["repeat"]): path
        for path, s in summaries
        if s["variant"] == "ayo_static"
    }
    reports, probes = [], {}
    for path, summary in summaries:
        if summary["variant"] == "ayo_static":
            continue
        baseline = baselines.get(
            (summary["workload"], summary["concurrency"], summary["repeat"])
        )
        if baseline is None:
            continue
        left, right = (
            jsonl(path / "node_audit.jsonl"),
            jsonl(baseline / "node_audit.jsonl"),
        )
        output_left, output_right = (
            jsonl(path / "outputs.jsonl"),
            jsonl(baseline / "outputs.jsonl"),
        )
        counts, differences = Counter(), []
        prompt_pairs = prompt_differences = 0
        rerank_same_input_pairs = rerank_same_input_differences = 0
        for query_id, row in output_left.items():
            if row.get("error") or output_right[query_id].get("error"):
                raise ValueError(
                    "Failed requests cannot be audited as equivalent outputs"
                )
            a, b = left[query_id]["nodes"], right[query_id]["nodes"]
            for name in ("rewrite_prefill", "answer_prefill"):
                pa, pb = a.get(name, {}).get("llm"), b.get(name, {}).get("llm")
                if pa and pb:
                    prompt_pairs += 1
                    prompt_differences += int(
                        pa["prompt_token_ids"] != pb["prompt_token_ids"]
                    )
            ra, rb = a.get("rerank", {}), b.get("rerank", {})
            if ra.get("input_sha256") and ra["input_sha256"] == rb.get("input_sha256"):
                rerank_same_input_pairs += 1
                rerank_same_input_differences += int(
                    ra["output_sha256"] != rb["output_sha256"]
                )
            if row["output"] == output_right[query_id]["output"]:
                continue
            kind, details = locate(a, b)
            counts[kind] += 1
            differences.append(
                {"query_id": query_id, "classification": kind, "details": details}
            )
            for nodes in (a, b):
                for name, node in nodes.items():
                    audit = node.get("llm")
                    if name.endswith("_prefill") and audit and "prompt" in audit:
                        session = name.removesuffix("_prefill")
                        last = max(
                            (n for n in nodes if n.startswith(f"{session}_decode_")),
                            key=lambda n: int(n.rsplit("_", 1)[1]),
                        )
                        max_tokens = 16 if session == "rewrite" else generation_limit
                        # These probes are diagnostic inputs, never substituted for measured answers.
                        probes[(audit["prompt"], max_tokens)] = {
                            "prompt": audit["prompt"],
                            "max_tokens": max_tokens,
                            "example_query_id": query_id,
                            "observed_token_ids": (nodes[last].get("llm") or {}).get(
                                "token_ids"
                            ),
                        }
        reports.append(
            {
                "run": path.name,
                "compared": len(output_left),
                "different": len(differences),
                "classifications": dict(counts),
                "available_prompt_pairs": prompt_pairs,
                "different_prompt_pairs": prompt_differences,
                "rerank_same_input_pairs": rerank_same_input_pairs,
                "rerank_same_input_differences": rerank_same_input_differences,
                "differences": differences,
            }
        )
    (args.directory / "node_comparison.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    (args.directory / "replay_prompts.json").write_text(
        json.dumps(list(probes.values()), indent=2) + "\n"
    )
    lines = [
        "# Intermediate-output audit",
        "",
        "Diagnostic runs include hashing and prompt capture; do not use their timing as performance evidence.",
        "",
        "Classifications locate an observed divergence; they do not by themselves establish its cause.",
        "",
        "| Run | Compared | Different | Observed divergence counts |",
        "|---|---:|---:|---|",
    ]
    for report in reports:
        lines.append(
            f"| {report['run']} | {report['compared']} | {report['different']} | {report['classifications']} |"
        )
    lines += [
        "",
        "Exact prompts and token sequences remain in node_audit.jsonl. Replay inputs are saved in replay_prompts.json.",
        "",
        "The following counts inspect all available intermediate pairs, including equal final answers. Cached/pruned inputs without provenance are not counted as verified pairs.",
        "",
        "| Run | Available prompt pairs | Different prompts | Same-input rerank pairs | Different rerank results |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['run']} | {report['available_prompt_pairs']} | {report['different_prompt_pairs']} | "
            f"{report['rerank_same_input_pairs']} | {report['rerank_same_input_differences']} |"
        )
    (args.directory / "NODE_AUDIT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "differences"} for r in reports],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
