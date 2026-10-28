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
"""Tool registry trimmed to the tools reachable from the DeepResearch agent.

Kept: the base tool primitives, the local-document parsing/retrieval stack used
by ``search_local_docs``, and the searchers that back it.  Removed: the code
interpreter (plus its 8 MB font resource), image tools, weather, web search,
keyword/vector search and the MCP manager's optional extras are untouched --
see ``DEVIATIONS.md``.
"""

from .base import TOOL_REGISTRY, BaseTool, BaseToolWithFileAccess, register_tool
from .doc_parser import DocParser
from .mcp_manager import MCPManager
from .retrieval import Retrieval
from .search_tools import FrontPageSearch, HybridSearch
from .simple_doc_parser import SimpleDocParser
from .storage import Storage

__all__ = [
    'BaseTool',
    'BaseToolWithFileAccess',
    'TOOL_REGISTRY',
    'register_tool',
    'DocParser',
    'MCPManager',
    'Retrieval',
    'FrontPageSearch',
    'HybridSearch',
    'SimpleDocParser',
    'Storage',
]
