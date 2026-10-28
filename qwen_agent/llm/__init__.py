# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM backends trimmed to the OpenAI-compatible ones.

The DeepResearch pipeline talks to a tool-calling capable OpenAI-compatible
endpoint (vLLM / SGLang / OpenAI / DashScope-compatible-mode).  The Azure,
DashScope-native, OpenVINO and Transformers backends were removed together with
their third-party dependencies.  See ``DEVIATIONS.md``.
"""

import copy
from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .qwenvl_oai import QwenVLChatAtOAI


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """Instantiate an LLM object from a configuration dict.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use your own model service compatible with the OpenAI API:
              'model': 'Qwen3-32B',
              'model_server': 'http://127.0.0.1:8000/v1',
              'model_type': 'qwenvl_oai',
              'api_key': 'EMPTY',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided.

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_oai'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')


__all__ = [
    'BaseChatModel',
    'TextChatAtOAI',
    'QwenVLChatAtOAI',
    'get_chat_model',
    'ModelServiceError',
]
