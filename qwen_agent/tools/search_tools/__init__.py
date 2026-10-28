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
"""Only the searchers used by the hybrid retrieval stack are kept.

``KeywordSearch`` and ``FrontPageSearch`` are the two sub-searchers listed in
``qwen_agent.settings.DEFAULT_RAG_SEARCHERS``, so both are required by
``qwen_agent.tools.retrieval``.  ``VectorSearch`` was removed -- it is not part
of the default searcher list and pulls in ``langchain``.  See ``DEVIATIONS.md``.
"""

from .front_page_search import FrontPageSearch
from .hybrid_search import HybridSearch
from .keyword_search import KeywordSearch

__all__ = [
    'HybridSearch',
    'FrontPageSearch',
    'KeywordSearch',
]
