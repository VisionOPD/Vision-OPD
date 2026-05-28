# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _config_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def resolve_custom_chat_template(config: Any) -> str | None:
    custom_chat_template = _config_get(config, "custom_chat_template", None)
    if custom_chat_template is not None:
        return custom_chat_template

    custom_chat_template_file = _config_get(config, "custom_chat_template_file", None)
    if not custom_chat_template_file:
        return None

    from verl.utils.fs import copy_to_local

    template_path = copy_to_local(
        src=os.path.expanduser(custom_chat_template_file),
        use_shm=bool(_config_get(config, "use_shm", False)),
    )
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
