"""Reference-answer diagnostics on MS MARCO; not the official ranking/QA evaluator."""

import argparse
import hashlib
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path

from benchmarks.herasys import mixed_workload, workload_samples


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction, reference):
    predicted, expected = normalize(prediction).split(), normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--references", type=Path, default=Path("data/raw/msmarco_validation.parquet")
    )
    args = parser.parse_args()
    env = json.loads((args.directory / "environment.json").read_text())
    settings = env["arguments"]
    dataset = Path(settings["dataset"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != env["dataset_sha256"]["rag"]:
        raise ValueError("Dataset differs from the measured corpus")
    reference_hash = hashlib.sha256(args.references.read_bytes()).hexdigest()
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text())
    if reference_hash != metadata["source_sha256"]:
        raise ValueError("Reference source differs from the documented dataset source")
    references = {}
    for row in pq.read_table(args.references, columns=["query", "answers"]).to_pylist():
        references.setdefault(row["query"], set()).update(
            answer
            for answer in row["answers"]
            if answer and answer != "No Answer Present."
        )
    records = list(map(json.loads, dataset.read_text().splitlines()))
    web = (
        list(map(json.loads, Path(settings["web_dataset"]).read_text().splitlines()))
        if settings.get("web_dataset")
        else None
    )
    reports, details = [], []
    for path in sorted(args.directory.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["failed"]:
            raise ValueError("Refuse to silently exclude failed requests")
        items = workload_samples(
            records,
            web,
            summary["requests"],
            settings["seed"] + summary["repeat"],
            settings["duplicate_rate"],
            summary["workload"],
        )
        scores, exact, missing, skipped_web = [], [], 0, 0
        for row in map(
            json.loads, (path.parent / "outputs.jsonl").read_text().splitlines()
        ):
            index = int(row["query_id"].removeprefix("q"))
            kind = (
                mixed_workload(index)
                if summary["workload"] == "mixed"
                else summary["workload"]
            )
            if kind == "web_search":
                skipped_web += 1
                continue
            question = items[index]["question"]
            answers = references.get(question, set())
            if not answers:
                missing += 1
                continue
            prediction = row["output"]["answer"]
            f1 = max(token_f1(prediction, answer) for answer in answers)
            em = int(
                any(normalize(prediction) == normalize(answer) for answer in answers)
            )
            scores.append(f1)
            exact.append(em)
            details.append(
                {
                    "run": path.parent.name,
                    "query_id": row["query_id"],
                    "token_f1": f1,
                    "exact_match": em,
                }
            )
        reports.append(
            {
                "run": path.parent.name,
                "evaluated": len(scores),
                "missing_reference": missing,
                "excluded_web": skipped_web,
                "token_f1": statistics.fmean(scores) if scores else None,
                "exact_match": statistics.fmean(exact) if exact else None,
            }
        )
    result = {
        "metric": "Maximum normalized whitespace-token F1 / EM over original MS MARCO answers; diagnostic, not official MS MARCO metrics or semantic equivalence.",
        "reference_sha256": reference_hash,
        "runs": reports,
        "queries": details,
    }
    (args.directory / "answer_quality.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# Reference-answer diagnostic",
        "",
        result["metric"],
        "",
        "Web fixture answers are excluded. Missing/unanswerable references are counted explicitly. All available RAG predictions are evaluated, including unchanged answers.",
        "",
        "| Run | Evaluated | Missing reference | Web excluded | Token F1 | Exact match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        f1 = f"{r['token_f1']:.6f}" if r["token_f1"] is not None else "n/a"
        em = f"{r['exact_match']:.6f}" if r["exact_match"] is not None else "n/a"
        lines.append(
            f"| {r['run']} | {r['evaluated']} | {r['missing_reference']} | {r['excluded_web']} | {f1} | {em} |"
        )
    (args.directory / "ANSWER_QUALITY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
