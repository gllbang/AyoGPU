def test_reranker_orders_logits_before_half_sigmoid_can_collapse_them():
    import torch

    from HeraSys.gpu import GPUWorker

    logits = torch.tensor([6.01171875, 6.109375], dtype=torch.float16)
    assert torch.sigmoid(logits)[0] == torch.sigmoid(logits)[1]

    class Model:
        def predict(self, pairs, *, activation_fct=None, **kwargs):
            assert len(pairs) == 2
            return (activation_fct or torch.nn.Sigmoid())(logits)

    worker = object.__new__(GPUWorker)
    worker.kind = "reranking"
    worker.model_name = "test"
    worker.model = Model()
    worker._sync = lambda: None
    result = worker.execute_batch(
        [
            {
                "op": "reranking",
                "inputs": {"question": "q", "contexts": ["lower", "higher"]},
                "config": {"top_k": 1},
                "outputs": ["contexts"],
            }
        ]
    )
    assert result[0]["output"] == {"contexts": ["higher"]}


def test_prefill_length_ordering_preserves_request_to_output_mapping(monkeypatch):
    import itertools
    import sys
    from types import SimpleNamespace

    from HeraSys.gpu import GPUWorker

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw)
    )
    queue, enqueued = [], []

    def add_request(key, prompt, sampling, *, prompt_token_ids):
        enqueued.append(prompt)
        queue.append(SimpleNamespace(request_id=key, prompt=prompt))

    def step():
        return [
            SimpleNamespace(
                request_id=g.request_id,
                finished=True,
                outputs=[SimpleNamespace(text=g.prompt, token_ids=[len(g.prompt)])],
            )
            for g in list(queue)
        ]

    worker = object.__new__(GPUWorker)
    worker.kind, worker.model_name = "llm", "test"
    worker.states, worker.counter = {}, itertools.count()
    worker._sync = lambda: None
    worker.engine = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda prompt: list(range(len(prompt)))),
        model_config=SimpleNamespace(max_model_len=64),
        scheduler=SimpleNamespace(waiting=queue, running=[], swapped=[]),
        add_request=add_request,
        step=step,
    )
    prompts = ["a long prompt", "short", "x", "another long prompt"]
    results = worker.execute_batch(
        [
            {
                "op": "llm_prefilling",
                "inputs": {"q": prompt, "c": ""},
                "config": {
                    "question_port": "q",
                    "context_port": "c",
                    "prompt_template": "{question}",
                    "max_tokens": 1,
                    "_hera_audit": True,
                },
                "outputs": ["state"],
            }
            for prompt in prompts
        ]
    )
    assert enqueued == sorted(prompts, key=len)
    assert [r["output"]["state"]["text"] for r in results] == prompts
    assert [r["audit"]["prompt"] for r in results] == prompts
    worker.release([r["output"] for r in results])
    assert not worker.states and not queue
