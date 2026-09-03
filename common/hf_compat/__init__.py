"""DeepSeek-V4 shims for the platform verl-0.9.0 image.

1. Register model_type=deepseek_v4 with AutoConfig. The image transformers
   does not know it; verl 0.9 only falls back to vLLM's get_config when
   building hf_config, and the tokenizer is loaded first.
2. Official DSv4 HF repos ship no Jinja chat_template (encoding is Python
   in encoding/encoding_dsv4.py). verl's RLHFDataset.filter still calls
   tokenizer.apply_chat_template, which then raises and drops every row.

Point actor_rollout_ref.model.external_lib at this package so
import_external_libs runs before HFModelConfig loads the tokenizer.
"""

from __future__ import annotations

# Official encoding/README.md basic chat. thinking=true → open <think> after
# Assistant; chat mode closes it immediately with </think>.
DSV4_CHAT_TEMPLATE = """\
{%- set thinking = enable_thinking if enable_thinking is defined else true -%}
{{- bos_token -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- message['content'] -}}
    {%- elif message['role'] == 'user' -%}
        {{- '<｜User｜>' + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- message['content'] + eos_token -}}
    {%- elif message['role'] == 'tool' -%}
        {{- '<tool_result>' + message['content'] + '</tool_result>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<｜Assistant｜>' -}}
    {%- if thinking -%}
        {{- '<think>' -}}
    {%- else -%}
        {{- '</think>' -}}
    {%- endif -%}
{%- endif -%}
"""


def _register_deepseek_v4() -> None:
    from transformers import AutoConfig, PretrainedConfig

    config_cls = None
    for path in (
        "vllm.transformers_utils.configs.deepseek_v4",
        "vllm.transformers_utils.configs",
    ):
        try:
            mod = __import__(path, fromlist=["DeepseekV4Config"])
            config_cls = getattr(mod, "DeepseekV4Config", None)
            if config_cls is not None:
                break
        except Exception:
            continue

    if config_cls is None:

        class DeepseekV4Config(PretrainedConfig):
            model_type = "deepseek_v4"

            def __init__(self, max_position_embeddings: int = 1_048_576, **kwargs):
                kwargs.setdefault("max_position_embeddings", max_position_embeddings)
                super().__init__(**kwargs)
                if getattr(self, "max_position_embeddings", None) is None:
                    self.max_position_embeddings = max_position_embeddings

        config_cls = DeepseekV4Config

    try:
        AutoConfig.register("deepseek_v4", config_cls, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register("deepseek_v4", config_cls)
        except Exception:
            pass


def _install_chat_template() -> None:
    """Attach a DSv4 Jinja template when HF loaded none (the official case)."""
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def from_pretrained(*args, **kwargs):
        tokenizer = original(*args, **kwargs)
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = DSV4_CHAT_TEMPLATE
        return tokenizer

    AutoTokenizer.from_pretrained = from_pretrained


def _install_modelopt_stub() -> None:
    """vllm024.dev2 里的 megatron-bridge 一 import 就拉 diffusion，要 modelopt。

    镜像没装 nvidia-modelopt。缺的只是 `is_quantized`；DSv4 GRPO 不走那条
    diffusion 转换。桩住就能进 AutoBridge。真缺量化核再换镜像。
    """
    try:
        import modelopt  # noqa: F401
        return
    except ImportError:
        pass

    import sys
    import types

    def is_quantized(_module):
        return False

    root = types.ModuleType("modelopt")
    torch_mod = types.ModuleType("modelopt.torch")
    quant = types.ModuleType("modelopt.torch.quantization")
    utils = types.ModuleType("modelopt.torch.quantization.utils")
    utils.is_quantized = is_quantized
    for name, mod in (
        ("modelopt", root),
        ("modelopt.torch", torch_mod),
        ("modelopt.torch.quantization", quant),
        ("modelopt.torch.quantization.utils", utils),
    ):
        sys.modules[name] = mod


def ray_worker_setup() -> None:
    """Ray worker_process_setup_hook.

    RewardLoopWorker 不走 model.external_lib，会自己 hf_tokenizer(DSv4)。
    每个 Ray worker 进程 import 本模块时已经完成注册。
    """
    return


_register_deepseek_v4()
_install_chat_template()
_install_modelopt_stub()
