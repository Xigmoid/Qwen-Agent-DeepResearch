"""Global LLM configuration loader.

The default configuration lives in ``config.json`` at the repository root.
An optional ``config.local.json`` next to it is merged on top and should be
used for secrets.  String values support ``${ENV_VAR}`` expansion.  A few
environment variables override the file for deployment convenience.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / 'config.json'
_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


class LLMConfigError(ValueError):
    """Raised when the global LLM configuration is missing or invalid."""


def config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the explicit, environment-selected, or default config path."""
    if path:
        return Path(path).expanduser().resolve()
    env_path = os.getenv('QWEN_AGENT_CONFIG')
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_env(child) for child in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ''), value)
    return value


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load ``config.json`` and merge ``config.local.json`` if present."""
    main_path = config_path(path)
    if not main_path.exists():
        raise LLMConfigError(f'Config file not found: {main_path}')

    try:
        config = json.loads(main_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as ex:
        raise LLMConfigError(f'Invalid JSON in {main_path}: {ex}') from ex
    if not isinstance(config, dict):
        raise LLMConfigError(f'Config root must be a JSON object: {main_path}')

    local_path = main_path.with_name(f'{main_path.stem}.local{main_path.suffix}')
    if local_path.exists():
        try:
            local = json.loads(local_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as ex:
            raise LLMConfigError(f'Invalid JSON in {local_path}: {ex}') from ex
        if not isinstance(local, dict):
            raise LLMConfigError(f'Local config root must be a JSON object: {local_path}')
        config = _merge(config, local)

    return _expand_env(config)


def get_llm_config(config: Optional[Dict[str, Any]] = None,
                   path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return the normalized LLM config accepted by ``qwen_agent.llm.get_chat_model``."""
    data = copy.deepcopy(config) if config is not None else load_config(path)
    llm = data.get('llm')
    if not isinstance(llm, dict):
        raise LLMConfigError('Missing object "llm" in config.json')

    llm = copy.deepcopy(llm)
    llm['model_server'] = (os.getenv('LLM_BASE_URL') or llm.get('model_server') or llm.get('base_url')
                           or '').strip()
    llm['model'] = (os.getenv('LLM_MODEL') or llm.get('model') or '').strip()
    llm['model_type'] = (os.getenv('LLM_MODEL_TYPE') or llm.get('model_type') or '').strip()

    api_key = (os.getenv('LLM_API_KEY') or os.getenv('MODELSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
               or llm.get('api_key') or '').strip()
    if not api_key:
        raise LLMConfigError(
            'Missing LLM API key. Set "llm.api_key" in config.local.json, '
            'or export LLM_API_KEY / MODELSCOPE_API_KEY.')
    if not llm.get('model_server'):
        raise LLMConfigError('Missing "llm.model_server" (base URL) in config.json.')
    if not llm.get('model'):
        raise LLMConfigError('Missing "llm.model" in config.json.')

    llm['api_key'] = api_key
    if not llm.get('model_type'):
        llm.pop('model_type', None)
    return llm


def get_agent_config(config: Optional[Dict[str, Any]] = None,
                     path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return the optional agent section from the global config."""
    data = config if config is not None else load_config(path)
    value = data.get('agent', {})
    if not isinstance(value, dict):
        raise LLMConfigError('"agent" in config.json must be an object')
    return copy.deepcopy(value)


__all__ = [
    'DEFAULT_CONFIG_PATH',
    'LLMConfigError',
    'config_path',
    'get_agent_config',
    'get_llm_config',
    'load_config',
]
