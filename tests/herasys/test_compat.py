from HeraSys.vllm_compat import llama3_frequencies, remove_attention_debug_prints


def test_llama31_rotary_frequencies_match_transformers():
    import torch
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import _compute_llama3_parameters

    config = LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        rope_theta=500000,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    base = 1 / (config.rope_theta ** (torch.arange(0, 128, 2).float() / 128))
    expected, _ = _compute_llama3_parameters(config, torch.device("cpu"))
    assert torch.allclose(
        llama3_frequencies(base, config.rope_scaling), expected, atol=1e-8
    )


def test_quiet_attention_removes_tensor_formatting_but_preserves_computation(capsys):
    class DeviceTensor:
        def __format__(self, spec):
            raise AssertionError("Tensor formatting would synchronize the device")

    def attention(value):
        print(f"b_start_loc: {value}")
        print(f"b_seq_len: {value}")
        print(f"prompt_lens: {value}")
        print(f"query length: {value}")
        return value

    quiet = remove_attention_debug_prints(attention)
    value = DeviceTensor()
    assert quiet(value) is value
    assert capsys.readouterr().out == ""
