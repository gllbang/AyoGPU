#!/usr/bin/env bash
# Explicit local parameter perturbations; not the unpublished Table 3 presets.
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [ "$#" != 1 ] || [ -e "$1" ]; then
    echo "Usage: bash scripts/reproduce_herasys_sensitivity.sh NEW_OUTPUT_DIR" >&2
    exit 2
fi
output_dir=$1
herasys_python=${HERASYS_PYTHON:-$repo_dir/.venv-herasys-baselines/bin/python}
: "${HERASYS_PG_DSN:?Set HERASYS_PG_DSN}"
for preset in default alpha_2 gamma_0.6 delta_1_rho_0.4; do
    extra=()
    case "$preset" in
        alpha_2) extra=(--alpha 2) ;;
        gamma_0.6) extra=(--gamma .6) ;;
        delta_1_rho_0.4) extra=(--delta 1 --rho .4) ;;
    esac
    case_dir="$output_dir/$preset"
    mkdir -p "$case_dir"
    tar -czf "$case_dir/source.tar.gz" HeraSys benchmarks scripts tests/herasys \
        requirements-herasys-baselines.txt pyproject.toml patches
    profile=()
    if [ "$preset" != default ]; then
        profile=(--profiles "$output_dir/default/profiles.json")
    fi
    "$herasys_python" -m benchmarks.herasys \
        --backend gpu --vector-backend pgvector \
        --llm-model /disk_n/lsz/model/Meta-Llama-3-8B \
        --gpu-memory-utilization .88 --llm-max-sessions 22 --max-model-len 1664 \
        --prefill-token-budget 8192 --cache-entries 4096 --cache-ttl-s 300 \
        --dataset data/msmarco_validation_full.jsonl \
        --max-tokens 128 --decode-quantum 16 --warmup 2 --profile-batch-sizes 1 8 44 \
        --workloads rag_rerank --variants herasys --concurrency 100 \
        --requests 1000 --repeats "${HERASYS_REPEATS:-1}" --timeout 600 \
        --output "$case_dir" "${extra[@]}" "${profile[@]}"
    "$herasys_python" scripts/verify_herasys_results.py "$case_dir"
done
"$herasys_python" scripts/report_herasys_sensitivity.py "$output_dir"
