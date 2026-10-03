#!/usr/bin/env bash
# Non-Google full-text experiments. Every invocation uses a new output directory.
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [ "$#" != 2 ]; then
    echo "Usage: bash scripts/reproduce_herasys_complete.sh NEW_OUTPUT_DIR screen|matrix|paper-matrix|ablations|native" >&2
    exit 2
fi
output_dir=$1
mode=$2
herasys_python=${HERASYS_PYTHON:-$repo_dir/.venv-herasys-baselines/bin/python}
: "${HERASYS_PG_DSN:?Start scripts/start_postgres.sh and set HERASYS_PG_DSN}"
if [ -e "$output_dir" ]; then
    echo "Use a new output directory: $output_dir" >&2
    exit 2
fi
requests=${HERASYS_REQUESTS:-1000}
repeats=${HERASYS_REPEATS:-3}
dataset=${HERASYS_DATASET:-data/msmarco_validation_full.jsonl}
if [ "$mode" = paper-matrix ]; then
    # Section 4.1 retrieves twelve candidates, then reranks eight. A top_k
    # argument alone cannot implement that workload with only 2--10 passages.
    # Do not pad, duplicate, truncate or silently discard input records.
    "$herasys_python" - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
shortfalls = [
    i for i, row in enumerate(records)
    if len({text for text in row.get("documents", []) if isinstance(text, str) and text.strip()}) < 12
]
if not records or shortfalls:
    print(
        f"Paper RAG matrix requires at least 12 distinct nonempty candidate passages per request; "
        f"{len(shortfalls)}/{len(records)} records fail. Supply the intended corpus via "
        "HERASYS_DATASET. Existing data are preserved; screen remains available for diagnosis.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
fi
if [ "$mode" = native ]; then
    bash scripts/setup_ayo_full.sh
    "$herasys_python" scripts/reproduce_ayo_full.py \
        --dataset "$dataset" --requests "$requests" --repeats "$repeats" --concurrency 6 30 100 300 \
        --workloads rag rag_rerank --batch-wait-s .005 \
        --embedding-batch-size 256 --reranker-batch-size 128 --output "$output_dir"
else
    case "$mode" in
        screen)
            variants=(HeraSys herasys_unaged ayo_static fcfs srtf)
            workloads=(rag_rerank)
            concurrency=(100 300)
            repeats=${HERASYS_REPEATS:-1}
            ;;
        matrix|paper-matrix)
            variants=(HeraSys langgraph llamaindex)
            workloads=(rag rag_rerank)
            concurrency=(6 30 100 300)
            ;;
        ablations)
            variants=(HeraSys herasys_unaged no_fusion fcfs srtf no_skew no_adaptive no_joint_allocation no_tail_feedback no_pipeline ayo_static)
            workloads=(rag_rerank)
            concurrency=(100)
            repeats=${HERASYS_REPEATS:-1}
            ;;
        *) echo "Unknown mode: $mode" >&2; exit 2 ;;
    esac
    mkdir -p "$output_dir"
    tar -czf "$output_dir/source.tar.gz" HeraSys benchmarks scripts tests/herasys \
        requirements-herasys-baselines.txt pyproject.toml patches
    profile_args=()
    if [ -n "${HERASYS_PROFILES:-}" ]; then
        profile_args=(--profiles "$HERASYS_PROFILES")
    fi
    "$herasys_python" -m benchmarks.herasys \
        --backend gpu --vector-backend pgvector \
        --llm-model /disk_n/lsz/model/Meta-Llama-3-8B \
        --gpu-memory-utilization .88 --llm-max-sessions 22 --max-model-len 1664 \
        --prefill-token-budget 8192 --cache-entries 4096 --cache-ttl-s 300 \
        --dataset "$dataset" \
        --max-tokens 128 --decode-quantum 16 --warmup 2 --profile-batch-sizes 1 8 44 \
        --workloads "${workloads[@]}" --variants "${variants[@]}" \
        --concurrency "${concurrency[@]}" --requests "$requests" --repeats "$repeats" \
        --timeout 600 --output "$output_dir" "${profile_args[@]}"
    "$herasys_python" -m benchmarks.report "$output_dir" --plot
    "$herasys_python" scripts/analyze_herasys_trace.py "$output_dir"
fi
PYTHONPATH=. "$herasys_python" scripts/evaluate_herasys_answers.py "$output_dir"
"$herasys_python" scripts/verify_herasys_results.py "$output_dir"
