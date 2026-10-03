"""Replay real mismatch prompts through native vLLM and HeraSys at matched batch shapes."""

import argparse
import hashlib
import json
from pathlib import Path

from HeraSys.gpu import GPUWorker


def native(worker, prompts, budget):
    from vllm import SamplingParams

    for i, prompt in enumerate(prompts):
        worker.engine.add_request(
            f"native_{i}", prompt, SamplingParams(temperature=0, max_tokens=budget)
        )
    outputs = {}
    while worker.engine.has_unfinished_requests():
        for result in worker.engine.step():
            if result.finished:
                outputs[int(result.request_id.removeprefix("native_"))] = list(
                    result.outputs[0].token_ids
                )
    return [outputs[i] for i in range(len(prompts))]


def chunked(worker, prompts, budget, quantum):
    handles = [
        r["output"]
        for r in worker.execute_batch(
            [
                {
                    "op": "llm_prefilling",
                    "inputs": {"q": prompt, "c": ""},
                    "config": {
                        "question_port": "q",
                        "context_port": "c",
                        "prompt_template": "{question}",
                        "max_tokens": budget,
                        "temperature": 0,
                        "_hera_audit": True,
                    },
                    "outputs": ["state"],
                }
                for prompt in prompts
            ]
        )
    ]
    for start in range(0, budget, quantum):
        final = start + quantum >= budget
        results = worker.execute_batch(
            [
                {
                    "op": "llm_parallel_decoding",
                    "inputs": handle,
                    "config": {
                        "token_budget": min(quantum, budget - start),
                        "final": final,
                        "_hera_audit": True,
                    },
                    "outputs": ["answer" if final else "state"],
                }
                for handle in handles
            ]
        )
        worker.release(handles)
        handles = [r["output"] for r in results]
    return [r["audit"]["token_ids"] for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="/disk_n/lsz/model/Meta-Llama-3-8B")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()
    if args.limit < 1 or min(args.batch_sizes) < 1:
        parser.error("limit and batch sizes must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    probes = json.loads(args.probes.read_text())
    worker = GPUWorker(
        "llm",
        args.model,
        gpu_memory_utilization=0.88,
        max_model_len=1152,
        max_batch_size=max(args.batch_sizes),
    )
    # A one-token native control isolates prefill ordering from decode batching.
    # Submit the same distinct prompts in reverse length order to the wrapper,
    # and verify that it restores each original request's token/result identity.
    mapping_prompts = sorted(
        dict.fromkeys(p["prompt"] for p in probes[:8]),
        key=lambda prompt: len(worker.engine.tokenizer.encode(prompt)),
    )
    mapping_expected = dict(zip(mapping_prompts, native(worker, mapping_prompts, 1)))
    mapping_actual = chunked(worker, list(reversed(mapping_prompts)), 1, 1)
    mapping_ok = mapping_actual == [
        mapping_expected[p] for p in reversed(mapping_prompts)
    ]
    observations, failures = [], int(not mapping_ok)
    for budget in sorted({p["max_tokens"] for p in probes}):
        selected = [p for p in probes if p["max_tokens"] == budget][: args.limit]
        singleton = {}
        for size in sorted(set([1, *args.batch_sizes])):
            for start in range(0, len(selected), size):
                # Match the native and chunked kernel shapes after the backend's
                # within-batch length ordering. Cross-shape changes remain a
                # separately reported numerical diagnostic.
                items = sorted(
                    selected[start : start + size],
                    key=lambda item: len(
                        worker.engine.tokenizer.encode(item["prompt"])
                    ),
                )
                prompts = [p["prompt"] for p in items]
                expected = native(worker, prompts, budget)
                observed = chunked(worker, prompts, budget, 16)
                repeated = native(worker, prompts, budget)
                for i, (a, b, c) in enumerate(zip(expected, observed, repeated)):
                    key = hashlib.sha256(prompts[i].encode()).hexdigest()
                    if size == 1:
                        singleton[key] = a
                    row = {
                        "prompt_sha256": key,
                        "max_tokens": budget,
                        "batch_size": len(items),
                        "native_tokens": a,
                        "chunked_tokens": b,
                        "native_repeat_tokens": c,
                        "chunked_matches_native": a == b,
                        "native_repeat_matches": a == c,
                        "native_matches_singleton": a == singleton[key],
                    }
                    observations.append(row)
                    failures += int(a != b or a != c)
                print(
                    json.dumps(
                        {
                            "budget": budget,
                            "batch_size": size,
                            "completed": min(start + size, len(selected)),
                            "failures": failures,
                        }
                    ),
                    flush=True,
                )
    state = worker.ready()
    cleanup_ok = (
        state["live_states"] == 0
        and state["kv_free_blocks"] == state["kv_total_blocks"]
    )
    result = {
        "probe_source_sha256": hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        "model": args.model,
        "mapping_check_prompts": len(mapping_prompts),
        "mapping_check_ok": mapping_ok,
        "observations": observations,
        "failures": failures,
        "native_batch_shape_differences": sum(
            not r["native_matches_singleton"] for r in observations
        ),
        "cleanup_ok": cleanup_ok,
        "backend": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failures or not cleanup_ok:
        raise RuntimeError(
            f"Replay found {failures} discrepancies; cleanup_ok={cleanup_ok}"
        )
    print(
        "HERASYS_REPLAY_CHECK_OK "
        + json.dumps({k: v for k, v in result.items() if k != "observations"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
