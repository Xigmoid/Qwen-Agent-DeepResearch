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
"""The DeepResearch trajectory-synthesis pipeline.

Modules
-------
``assistant_deep_research``
    The DeepResearch agent (search / read / notes) and the four tools it
    registers.  Also runnable on its own as a headless CLI.
``annotate_questions``
    The annotation runner: runs the agent over the question sets and persists
    one trace record per question.  Resumable and process-isolated.
``convert_traces``
    Shared conversion library: trace -> OpenAI ``messages`` normalisation,
    structural validation and redaction.
``convert_annotations_to_openai``
    CLI turning the annotation JSONL into OpenAI-format SFT samples.
"""

__all__ = [
    'annotate_questions',
    'assistant_deep_research',
    'convert_annotations_to_openai',
    'convert_traces',
]
