# Deviations from upstream Qwen-Agent

This repository vendors a **trimmed fork** of [QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
(`qwen_agent/__init__.py` reports `__version__ = '0.0.34'`), with the Apache-2.0
license and the per-file copyright headers of the original preserved.

There are exactly two kinds of change: **functional** customizations that make the
DeepResearch pipeline work against an OpenAI-compatible tool-calling endpoint, and
**trimming** that removes everything the pipeline cannot reach.

Everything below was verified by diffing against a fresh clone of upstream
(`git diff` against `QwenLM/Qwen-Agent@main`), and the resulting tree was
re-verified by running the offline unit tests and every CLI entry point in a clean
virtual environment.

---

## 1. Functional customizations

These are the reason this fork exists. §1.1–§1.4 are **not optional**: without them
the agent cannot round-trip native `tool_calls` against vLLM / SGLM / OpenAI-compatible
servers.

### 1.1 `qwen_agent/llm/oai.py` — tolerate empty tool-call arguments

```diff
 Message(role=ASSISTANT, content='',
         function_call=FunctionCall(name=tc.function.name,
-                                    arguments=tc.function.arguments),
+                                    arguments=tc.function.arguments or ''),
```

Some servers emit a tool call with `arguments` omitted or `null`. The upstream code
propagates that `None` into the message schema and it later breaks the
`json.dumps(arguments or '{}')` fallback in the trace converter.

### 1.2 `qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py` — keep native function calls

```diff
 if role in (SYSTEM, USER):
     new_messages.append(Message(role=role, content=content, extra=extra))
+    continue
+
+# Preserve native tool calls (e.g. streamed by VLLM/OAI models):
+if msg.function_call:
+    new_messages.append(
+        Message(role=role, content=content, function_call=msg.function_call, extra=extra))
     continue
```

### 1.3 `qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py` — same guard

```diff
+# Preserve native tool calls (e.g. streamed by VLLM/OAI models):
+if msg.function_call:
+    new_messages.append(msg)
+    continue
```

**Why 1.2 / 1.3 matter.** When `generate_cfg.use_raw_api` is `true`, `BaseChatModel`
calls `raw_chat()`, which forwards the tools to the endpoint as native OpenAI
`tools` and returns the assistant messages carrying native `tool_calls`
(`qwen_agent/llm/base.py`, `raw_chat()`). On the *next* turn those assistant messages
are replayed into the request. Upstream's function-call prompt templates assume the
Qwen-native `<tool_call>` JSON format and would rewrite (and thereby corrupt) a
native `function_call` message. The two patches short-circuit that rewriting, so the
multi-turn tool-calling loop survives.

### 1.4 `qwen_agent/utils/llm_config.py` — **new file**, global config loader

Upstream has no such module. It provides:

| Function | Purpose |
| --- | --- |
| `load_config(path=None)` | Read `config.json`, deep-merge `config.local.json` on top, expand `${ENV_VAR}` references. |
| `get_llm_config(config=None, path=None)` | Normalise the `llm` section into what `qwen_agent.llm.get_chat_model` accepts; resolve `LLM_BASE_URL` / `LLM_MODEL` / `LLM_MODEL_TYPE` / `LLM_API_KEY` / `MODELSCOPE_API_KEY` / `OPENAI_API_KEY` overrides. |
| `get_agent_config(config=None, path=None)` | Return the optional `agent` section (e.g. `max_llm_calls`). |

The repository root is resolved as `Path(__file__).resolve().parents[2]`, so
`config.json` is looked up next to `qwen_agent/`, and `QWEN_AGENT_CONFIG` overrides
the path.

### 1.5 Repository-local install hints (3 files)

The upstream text tells users to run `pip install "qwen-agent[rag]"`. This fork is
published under a different package name, so the hint points at the local file
instead. Behaviour is identical; only the message changed.

| File | Change |
| --- | --- |
| `qwen_agent/tools/retrieval.py` | `pip install "qwen-agent[rag]"` → `pip install -r requirements-rag.txt` |
| `scripts/assistant_deep_research.py` | same |
| `tests/test_assistant_deep_research.py` | the notebook case and the local-document case were split into two tests; the local-document one is guarded by `pytest.importorskip('rank_bm25')` / `pytest.importorskip('pdfplumber')` so a minimal installation reports a skip instead of a failure |

---

## 2. Trimming

### 2.1 Package entry points rewritten

| File | Upstream exports | This repo |
| --- | --- | --- |
| `qwen_agent/__init__.py` | `Agent`, `MultiAgentHub` | `Agent` |
| `qwen_agent/agents/__init__.py` | 20 agents | `Assistant`, `FnCallAgent` |
| `qwen_agent/tools/__init__.py` | 18 tools | `BaseTool`, `BaseToolWithFileAccess`, `TOOL_REGISTRY`, `register_tool`, `DocParser`, `MCPManager`, `Retrieval`, `FrontPageSearch`, `HybridSearch`, `KeywordSearch`, `SimpleDocParser`, `Storage` |
| `qwen_agent/tools/search_tools/__init__.py` | 4 searchers | `HybridSearch`, `FrontPageSearch`, `KeywordSearch` |
| `qwen_agent/llm/__init__.py` | 10 backends | `TextChatAtOAI`, `QwenVLChatAtOAI` (+ a trimmed `get_chat_model`) |

`get_chat_model()` keeps the `model_type` branch, the `http://…` → `oai` deduction
and the `-vl` → `qwenvl_oai` deduction; the Azure / DashScope-native / OpenVINO /
Transformers branches are gone along with their backends.

> `KeywordSearch` **must** stay: `qwen_agent.settings.DEFAULT_RAG_SEARCHERS` is
> `['keyword_search', 'front_page_search']`, and `qwen_agent/tools/retrieval.py`
> resolves those names through `TOOL_REGISTRY`. Removing it breaks `search_local_docs`.

### 2.2 `qwen_agent/utils/utils.py` — lazy `numpy` / `soundfile`

Upstream imports `numpy` and `soundfile` at module level but only uses them inside
`save_audio_to_file()`. `soundfile` needs the system library `libsndfile`, so the
upstream import chain makes it a hard install-time requirement for a path the
DeepResearch pipeline never touches. The two imports were moved into the function.

### 2.3 Removed files

```
qwen_agent/gui/                                  (Gradio web UI)
qwen_agent/multi_agent_hub.py
qwen_agent/agents/article_agent.py               qwen_agent/agents/dialogue_retrieval_agent.py
qwen_agent/agents/dialogue_simulator.py          qwen_agent/agents/doc_qa/
qwen_agent/agents/group_chat.py                  qwen_agent/agents/group_chat_auto_router.py
qwen_agent/agents/group_chat_creator.py          qwen_agent/agents/human_simulator.py
qwen_agent/agents/keygen_strategies/             qwen_agent/agents/memo_assistant.py
qwen_agent/agents/react_chat.py                  qwen_agent/agents/router.py
qwen_agent/agents/tir_agent.py                   qwen_agent/agents/user_agent.py
qwen_agent/agents/virtual_memory_agent.py        qwen_agent/agents/write_from_scratch.py
qwen_agent/agents/writing/
qwen_agent/llm/azure.py                          qwen_agent/llm/openvino.py
qwen_agent/llm/qwen_dashscope.py                 qwen_agent/llm/qwenaudio_dashscope.py
qwen_agent/llm/qwenomni_oai.py                   qwen_agent/llm/qwenvl_dashscope.py
qwen_agent/llm/qwenvlo_dashscope.py              qwen_agent/llm/transformers_llm.py
qwen_agent/tools/amap_weather.py                 qwen_agent/tools/code_interpreter.py
qwen_agent/tools/extract_doc_vocabulary.py       qwen_agent/tools/image_gen.py
qwen_agent/tools/image_search.py                 qwen_agent/tools/image_zoom_in_qwen3vl.py
qwen_agent/tools/python_executor.py              qwen_agent/tools/resource/
qwen_agent/tools/web_extractor.py                qwen_agent/tools/web_search.py
qwen_agent/tools/search_tools/vector_search.py
```

`qwen_agent/tools/resource/` alone was 8.4 MB (a chart-rendering font for the code
interpreter).

> **Note.** `qwen_agent/tools/web_search.py` is upstream's Serper/DashScope search
> tool. The DeepResearch agent defines its **own** `WebSearch` class in
> `scripts/assistant_deep_research.py` that keeps the result URLs, so removing the
> upstream one changes nothing.

---

## 3. Repository layout

The pipeline was originally laid out as `tests/examples/…` with the question sets under
`tests/examples/datasets/annotation_sources/questions/`, mirroring the upstream
Qwen-Agent repository. That put the pipeline CLIs exactly where a reader expects unit
tests, so this repository reorganises it into three clearly separated layers:

| Layer | Location | Contents |
| --- | --- | --- |
| Code | `scripts/` | the DeepResearch agent and the three pipeline CLIs |
| Data | `data/questions/` | the three extracted question sets (one JSON per source dataset) |
| Tests | `tests/` | the offline unit tests |

The only code changes this required were the repository-root anchors
(`Path(__file__).resolve().parents[2]` → `parents[1]`), the default question directory
(`DEFAULT_QUESTIONS_DIR`), the default output directory of `convert_traces.py`, and the
intra-package imports (`examples.assistant_deep_research` → `scripts.assistant_deep_research`,
`tests.examples.convert_traces` → `scripts.convert_traces`). **No behaviour changed.**

---

## 4. What was *not* changed

* `qwen_agent/agents/assistant.py`, `qwen_agent/agents/fncall_agent.py`,
  `qwen_agent/agent.py`, `qwen_agent/settings.py`, `qwen_agent/llm/base.py`,
  `qwen_agent/llm/schema.py`, `qwen_agent/llm/oai.py` (apart from §1.1) and the whole
  `qwen_agent/tools/` retrieval stack are byte-identical to upstream (apart from the
  install hint in §1.5).
* `scripts/annotate_questions.py`, `scripts/convert_traces.py` and
  `scripts/convert_annotations_to_openai.py` are **verbatim copies** of the working
  implementation apart from the path anchors and imports listed in §3 — no logic was
  refactored, so the numbers produced by this repository match the original pipeline.
* `scripts/assistant_deep_research.py` differs only in the install hint (§1.5).
* `tests/test_convert_traces.py` is byte-identical apart from its import line.

---

## 5. Verification performed

| Check | Result |
| --- | --- |
| `diff -r` against a fresh upstream clone | see §1–§2 above |
| `python3 -m pytest tests/ -q` in a clean venv, core requirements only | 13 passed, 1 skipped |
| `python3 -m pytest tests/ -q` in a clean venv, with `requirements-rag.txt` | 14 passed |
| `--help` on `scripts/annotate_questions.py`, `scripts/convert_annotations_to_openai.py`, `scripts/convert_traces.py` | all OK |
| `python3 -m compileall qwen_agent scripts tests` | OK |
| `pip install -e .` then `import qwen_agent` | OK |
| End-to-end: annotate 1 question against a live tool-calling endpoint, then convert | `passed`, 0 invalid samples, 0 validation errors |
| Scan for API keys, internal IPs and local absolute paths in tracked files | clean |
