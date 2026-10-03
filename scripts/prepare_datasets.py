"""Fetch public evaluation data with recorded URL/hash and explicit truncation.

TruthfulQA is used only for web-search questions, never as retrieved answer text.
MS MARCO accepts the official QA JSON/JSON.GZ or Hugging Face row API format.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

SOURCES = {
    "truthfulqa": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "msmarco": "https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=list(SOURCES))
    p.add_argument(
        "--source", help="Existing raw file; otherwise download the official source"
    )
    p.add_argument(
        "--source-url", help="Provenance URL for an already downloaded source"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--passage-chars", type=int, default=400)
    p.add_argument(
        "--no-truncate", action="store_true", help="Preserve full passage text"
    )
    args = p.parse_args()
    if args.source:
        raw = Path(args.source).read_bytes()
    else:
        with urllib.request.urlopen(SOURCES[args.dataset], timeout=30) as response:
            raw = response.read()
    sha = hashlib.sha256(raw).hexdigest()
    decoded = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    rows = []
    if args.dataset == "truthfulqa":
        rows = [
            {"question": r["Question"], "category": r["Category"]}
            for r in csv.DictReader(io.StringIO(decoded.decode("utf-8")))
        ]
    else:
        if raw.startswith(b"PAR1"):
            import pyarrow.parquet as pq

            data = pq.read_table(io.BytesIO(raw)).to_pylist()
        else:
            data = json.loads(decoded)
        if isinstance(data, list):
            records = data
        elif "rows" in data:
            records = [row["row"] for row in data["rows"]]
        elif isinstance(data.get("query"), dict):
            records = [
                {"query": value, "passages": data["passages"][key]}
                for key, value in data["query"].items()
            ]
        else:
            records = data if isinstance(data, list) else [data]
        for record in records:
            passages = record["passages"]
            if isinstance(passages, dict):
                passages = passages["passage_text"]
            documents = [
                passage["passage_text"] if isinstance(passage, dict) else passage
                for passage in passages
            ]
            if not args.no_truncate:
                if args.passage_chars < 1:
                    raise ValueError(
                        "passage-chars must be positive; use --no-truncate for full text"
                    )
                documents = [text[: args.passage_chars] for text in documents]
            if documents:
                rows.append({"question": record["query"], "documents": documents})
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No usable records")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    out.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "source": args.source_url or args.source or SOURCES[args.dataset],
                "source_sha256": sha,
                "records": len(rows),
                "passage_char_limit": None if args.no_truncate else args.passage_chars,
                "selection": "first records in source order",
                "paper_exact_sample": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
