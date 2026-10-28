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
"""Packaging for the trimmed Qwen-Agent fork used by the DeepResearch pipeline."""

import re

from setuptools import find_packages, setup


def get_version() -> str:
    with open('qwen_agent/__init__.py', encoding='utf-8') as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(),
            re.MULTILINE,
        ).group(1)
    return version


def read_description() -> str:
    with open('README.md', 'r', encoding='UTF-8') as f:
        return f.read()


setup(
    name='qwen-agent-deepresearch',
    version=get_version(),
    author='Qwen Team',
    author_email='tujianhong.tjh@alibaba-inc.com',
    description='A trimmed Qwen-Agent fork with a DeepResearch agent and a trajectory-synthesis pipeline.',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    keywords=['LLM', 'Agent', 'Function Calling', 'DeepResearch', 'Trajectory Synthesis'],
    packages=find_packages(include=['qwen_agent', 'qwen_agent.*']),
    package_data={
        'qwen_agent': ['utils/qwen.tiktoken'],
    },
    include_package_data=True,
    install_requires=[
        'eval_type_backport',
        'json5',
        'jsonschema',
        'openai',
        'pydantic>=2.3.0',
        'requests',
        'tiktoken',
        'dotenv',
    ],
    extras_require={
        # Optional dependencies for local-document handling (search_local_docs, PDF pages).
        'rag': [
            'charset-normalizer',
            'rank_bm25',
            'jieba',
            'snowballstemmer',
            'beautifulsoup4',
            'pdfminer.six',
            'pdfplumber',
            'python-docx',
            'python-pptx',
            'pandas',
            'tabulate',
        ],
        # Test dependencies.
        'test': ['pytest'],
    },
    url='https://github.com/QwenLM/Qwen-Agent',
)
