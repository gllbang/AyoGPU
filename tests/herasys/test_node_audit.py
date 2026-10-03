from scripts.audit_herasys_nodes import locate


def test_output_audit_distinguishes_prompt_prefill_decode_and_missing_provenance():
    from copy import deepcopy

    original = {
        "answer_prefill": {"llm": {"prompt_token_ids": [1, 2], "token_ids": [3]}},
        "answer_decode_0": {"llm": {"token_ids": [3, 4, 5]}},
        "retrieve": {"output_sha256": "retrieval-a"},
    }
    changed = deepcopy(original)
    changed["answer_prefill"]["llm"]["prompt_token_ids"] = [1, 7]
    changed["retrieve"]["output_sha256"] = "retrieval-b"
    kind, details = locate(original, changed)
    assert kind == "different_prompt"
    assert details[0]["changed_upstream"] == ["retrieve"]
    changed = deepcopy(original)
    changed["answer_prefill"]["llm"]["token_ids"] = [9]
    assert locate(original, changed)[0] == "same_prompt_prefill_difference"
    changed = deepcopy(original)
    changed["answer_decode_0"]["llm"]["token_ids"] = [3, 4, 9]
    kind, details = locate(original, changed)
    assert kind == "same_prefill_later_decode_difference"
    assert details[0]["common_prefix_tokens"] == 2
    changed["answer_prefill"]["llm"] = None
    assert locate(original, changed)[0] == "missing_cached_provenance"
