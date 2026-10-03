"""Measure BGE ranking stability on observed mismatch questions at two precisions."""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from sentence_transformers import CrossEncoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/msmarco_validation.jsonl")
    )
    parser.add_argument("--model", default="/disk_n/lsz/model/bge-reranker-large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = {
        row["question"]: row
        for row in map(json.loads, args.dataset.read_text().splitlines())
    }
    questions = set()
    for run in json.loads((args.directory / "node_comparison.json").read_text()):
        ids = {
            row["query_id"]
            for row in run["differences"]
            if row["classification"] == "different_prompt"
        }
        for row in map(
            json.loads,
            (args.directory / run["run"] / "node_audit.jsonl").read_text().splitlines(),
        ):
            if row["query_id"] in ids:
                audit = row["nodes"].get("answer_prefill", {}).get("llm")
                if audit:
                    question = (
                        audit["prompt"]
                        .rsplit("\nQuestion: ", 1)[1]
                        .removesuffix("\nAnswer:")
                    )
                    if question in records:
                        questions.add(question)
    selected = [records[q] for q in sorted(questions)]
    if not selected:
        raise ValueError("No RAG prompt differences to replay")
    torch.set_num_threads(2)
    model = CrossEncoder(args.model, device="cuda", max_length=512)
    pairs = [(row["question"], doc) for row in selected for doc in row["documents"]]
    measurements = []
    # Native BGE logits, then evaluate both raw ranking and the old half sigmoid.
    # Evaluate full checkpoint weights before casting down. Widening a model
    # after .half() cannot recover its original float32 parameter precision.
    for precision in ("float32", "float16"):
        model.model.to(dtype=getattr(torch, precision))
        for size in (1, 32):
            scores = model.predict(
                pairs,
                batch_size=size,
                show_progress_bar=False,
                activation_fct=torch.nn.Identity(),
                convert_to_tensor=True,
            )
            sigmoid = torch.sigmoid(scores)
            offset = 0
            for row in selected:
                count = len(row["documents"])
                raw = scores[offset : offset + count].tolist()
                activated = sigmoid[offset : offset + count].tolist()
                measurements.append(
                    {
                        "question": row["question"],
                        "precision": precision,
                        "batch_size": size,
                        "logits": raw,
                        "sigmoid": activated,
                        "raw_order": sorted(range(count), key=lambda i: (-raw[i], i)),
                        "sigmoid_order": sorted(
                            range(count), key=lambda i: (-activated[i], i)
                        ),
                    }
                )
                offset += count
            print(
                json.dumps(
                    {"precision": precision, "batch_size": size, "pairs": len(pairs)}
                ),
                flush=True,
            )
    summaries = []
    for row in selected:
        found = {
            (m["precision"], m["batch_size"]): m
            for m in measurements
            if m["question"] == row["question"]
        }
        summaries.append(
            {
                "question": row["question"],
                **{
                    f"{precision}_{kind}_batch_stable": found[(precision, 1)][
                        kind + "_order"
                    ]
                    == found[(precision, 32)][kind + "_order"]
                    for precision in ("float16", "float32")
                    for kind in ("raw", "sigmoid")
                },
            }
        )
    result = {
        "scope": "Replays original document collections for observed prompt mismatch questions; original retrieval order is not reconstructed.",
        "questions": len(selected),
        "summaries": summaries,
        "measurements": measurements,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
