"""Opt-in real vLLM regression: KV fork, decode equivalence and state cleanup."""

import argparse
import json
from copy import deepcopy

from HeraSys.gpu import GPUWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/disk_n/lsz/model/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.max_tokens < 4 or args.max_tokens % 4:
        parser.error("--max-tokens must be a positive multiple of four")
    quantum = args.max_tokens // 4
    worker = GPUWorker(
        "llm",
        args.model,
        gpu_memory_utilization=args.memory_fraction,
        max_model_len=max(512, args.max_tokens + 256),
    )
    prefill = {
        "op": "llm_prefilling",
        "inputs": {
            "q": "What is the capital of France?",
            "c": "France is a country in Europe.",
        },
        "config": {
            "question_port": "q",
            "context_port": "c",
            "max_tokens": args.max_tokens,
            "prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:",
        },
        "outputs": ["state"],
    }
    source = worker.execute_batch([prefill])[0]["output"]
    full = {
        "op": "llm_parallel_decoding",
        "inputs": source,
        "config": {"token_budget": args.max_tokens, "final": True},
        "outputs": ["answer"],
    }
    one_shot = worker.execute_batch([full])[0]["output"]
    current = source
    intermediate = []
    for step in range(4):
        chunk = {
            "op": "llm_parallel_decoding",
            "inputs": current,
            "config": {"token_budget": quantum, "final": step == 3},
            "outputs": ["answer" if step == 3 else "state"],
        }
        current = worker.execute_batch([chunk])[0]["output"]
        if step < 3:
            intermediate.append(current)
    assert current == one_shot, (current, one_shot)
    forks = worker.execute_batch([full, full])
    assert forks[0]["output"] == forks[1]["output"] == one_shot
    # A final decode and an intermediate quantum may share a physical batch.
    partial = {
        "op": "llm_parallel_decoding",
        "inputs": source,
        "config": {"token_budget": quantum, "final": False},
        "outputs": ["partial_state"],
    }
    mixed = worker.execute_batch([full, partial])
    assert mixed[0]["output"] == one_shot
    continued = worker.execute_batch(
        [
            {
                "op": "llm_parallel_decoding",
                "inputs": mixed[1]["output"],
                "config": {"token_budget": args.max_tokens - quantum, "final": True},
                "outputs": ["answer"],
            }
        ]
    )[0]["output"]
    assert continued == one_shot
    worker.release([mixed[1]["output"]])
    worker.release([source, *intermediate])
    assert not worker.states
    from vllm import SamplingParams

    prompt = prefill["config"]["prompt_template"].format(
        question=prefill["inputs"]["q"], context=prefill["inputs"]["c"]
    )
    worker.engine.add_request(
        "native_control",
        prompt,
        SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )
    native = None
    while worker.engine.has_unfinished_requests():
        for output in worker.engine.step():
            if output.request_id == "native_control" and output.finished:
                native = output.outputs[0].text
    assert one_shot == {"answer": native}, (one_shot, native)
    # Finishing during prefill must publish a terminal handle. This is also
    # the representation used when EOS arrives before a later decode quantum.
    short = deepcopy(prefill)
    short["config"]["max_tokens"] = 1
    terminal = worker.execute_batch([short])[0]["output"]
    handle = terminal["state"]
    assert handle["_hera_finished"] is True
    assert handle["model"] == args.model
    assert worker.states[handle["state_key"]]["group"] is None
    terminal_decode = deepcopy(full)
    terminal_decode["inputs"] = terminal
    assert worker.execute_batch([terminal_decode])[0]["output"] == {
        "answer": handle["text"]
    }
    worker.release([terminal])
    # Rewrite and answer prefills share a model but can use different ports and
    # token budgets. Compare that batch with native generation at the same shape.
    varied = []
    for index, budget in enumerate((4, 8)):
        request = deepcopy(prefill)
        request["inputs"] = {
            f"q{index}": prefill["inputs"]["q"],
            f"c{index}": prefill["inputs"]["c"],
        }
        request["config"].update(
            question_port=f"q{index}", context_port=f"c{index}", max_tokens=budget
        )
        varied.append(request)
    handles = [result["output"] for result in worker.execute_batch(varied)]
    varied_outputs = worker.execute_batch(
        [
            {
                "op": "llm_parallel_decoding",
                "inputs": handle,
                "config": {"token_budget": budget, "final": True},
                "outputs": ["answer"],
            }
            for handle, budget in zip(handles, (4, 8))
        ]
    )
    worker.release(handles)
    for index, budget in enumerate((4, 8)):
        worker.engine.add_request(
            f"native_varied_{index}",
            prompt,
            SamplingParams(temperature=0, max_tokens=budget),
        )
    native_varied = {}
    while worker.engine.has_unfinished_requests():
        for output in worker.engine.step():
            if output.finished:
                native_varied[output.request_id] = output.outputs[0].text
    assert [result["output"]["answer"] for result in varied_outputs] == [
        native_varied[f"native_varied_{index}"] for index in range(2)
    ]
    capacity = worker.ready()
    assert capacity["live_states"] == 0
    assert capacity["kv_free_blocks"] == capacity["kv_total_blocks"]
    print(
        "HERASYS_GPU_CHECK_OK "
        + json.dumps(
            {"max_tokens": args.max_tokens, "output": one_shot, "backend": capacity}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
