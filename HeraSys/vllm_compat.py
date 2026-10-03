"""Process-local Llama-3.1 RoPE support for HeraSys's vLLM 0.2.2 backend.

This does not relabel Llama-3.1 as the Llama-3 model used in the paper. It allows
validation with the available weights without editing a shared Python install.
"""

import math


def remove_attention_debug_prints(function):
    """Remove four known diagnostic expressions, including CUDA tensor formatting.

    Redirecting stdout or replacing print is insufficient: the f-strings already
    materialize CUDA tensors on the CPU before print is called.
    """
    import ast
    import inspect
    import textwrap

    prefixes = {"b_start_loc: ", "b_seq_len: ", "prompt_lens: ", "query length: "}
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    removed = set()
    body = []
    for statement in definition.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "print"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.JoinedStr)
            and isinstance(call.args[0].values[0], ast.Constant)
            and call.args[0].values[0].value in prefixes
        ):
            removed.add(call.args[0].values[0].value)
        else:
            body.append(statement)
    if not removed:
        return function
    if removed != prefixes:
        raise RuntimeError("Unexpected vLLM attention debug-print layout")
    definition.body = body
    namespace = dict(function.__globals__)
    exec(
        compile(ast.fix_missing_locations(tree), inspect.getfile(function), "exec"),
        namespace,
    )
    return namespace[function.__name__]


def llama3_frequencies(inv_freq, scaling):
    import torch

    wavelength = 2 * math.pi / inv_freq
    low = scaling["original_max_position_embeddings"] / scaling["low_freq_factor"]
    high = scaling["original_max_position_embeddings"] / scaling["high_freq_factor"]
    smooth = (
        scaling["original_max_position_embeddings"] / wavelength
        - scaling["low_freq_factor"]
    ) / (scaling["high_freq_factor"] - scaling["low_freq_factor"])
    medium = (1 - smooth) * inv_freq / scaling["factor"] + smooth * inv_freq
    return torch.where(
        wavelength < high,
        inv_freq,
        torch.where(wavelength > low, inv_freq / scaling["factor"], medium),
    )


def install():
    import vllm.config as config
    from vllm.model_executor.layers import attention, rotary_embedding

    if getattr(config, "_herasys_llama3", False):
        return
    attention.PagedAttention.prefix_attention = remove_attention_debug_prints(
        attention.PagedAttention.prefix_attention
    )
    original_max_len = config._get_and_verify_max_len
    original_rope = rotary_embedding.get_rope

    def max_len(hf_config, max_model_len):
        scaling = getattr(hf_config, "rope_scaling", None)
        if scaling and scaling.get("rope_type", scaling.get("type")) == "llama3":
            supported = hf_config.max_position_embeddings
            if max_model_len is not None and max_model_len > supported:
                raise ValueError(
                    "Requested context exceeds the model's supported length"
                )
            return max_model_len or supported
        return original_max_len(hf_config, max_model_len)

    class Llama3RotaryEmbedding(rotary_embedding.RotaryEmbedding):
        def __init__(self, *args, scaling, **kwargs):
            self.scaling = scaling
            super().__init__(*args, **kwargs)

        def _compute_inv_freq(self, base):
            return llama3_frequencies(super()._compute_inv_freq(base), self.scaling)

    def get_rope(
        head_size, rotary_dim, max_position, base, is_neox_style, rope_scaling
    ):
        if (
            rope_scaling
            and rope_scaling.get("rope_type", rope_scaling.get("type")) == "llama3"
        ):
            return Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling=rope_scaling,
            )
        return original_rope(
            head_size, rotary_dim, max_position, base, is_neox_style, rope_scaling
        )

    config._get_and_verify_max_len = max_len
    rotary_embedding.get_rope = get_rope
    attention.get_rope = get_rope
    config._herasys_llama3 = True
