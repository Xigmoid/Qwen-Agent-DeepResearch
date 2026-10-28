#!/usr/bin/env python
"""Convert DeepResearch traces or captured LLM calls into OpenAI Chat JSONL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.assistant_deep_research import (  # noqa: E402
    DEEP_RESEARCH_SYSTEM_PROMPT,
    ReadWebpage,
    ResearchNotes,
    SearchLocalDocs,
    WebSearch,
)

TOOL_CLASSES = (WebSearch, ReadWebpage, SearchLocalDocs, ResearchNotes)
_WEEKDAYS = ('一', '二', '三', '四', '五', '六', '日')
_ABS_PATH_RE = re.compile(r'/(?:Users|home|root|private|var|tmp)/[^\s"\'\]\)>,]+')
_LOCAL_FILE_RE = re.compile(r'file:///(?:Users|home|root|private|var|tmp)/[^\s"\'\]\)>,]+')
_API_KEY_RE = re.compile(r'(?:ms-[0-9a-fA-F-]{20,}|sk-[A-Za-z0-9_-]{20,})')


def _basename(path: str) -> str:
    value = path
    if value.startswith('file://'):
        value = value[len('file://'):]
    try:
        return Path(value).name or value
    except Exception:
        return value


def _content_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get('type') == 'text' and item.get('text'):
                parts.append(str(item['text']))
            elif item.get('type') == 'image_url':
                parts.append('[Image]')
            elif item.get('type') == 'video_url':
                parts.append('[Video]')
            elif item.get('type') == 'input_audio':
                parts.append('[Audio]')
            elif item.get('text'):
                parts.append(str(item['text']))
            elif item.get('file'):
                parts.append(f'[Attached local file: {_basename(str(item["file"]))}]')
            elif item.get('image'):
                parts.append('[Image]')
            elif item.get('video'):
                parts.append('[Video]')
            elif item.get('audio'):
                parts.append('[Audio]')
        elif getattr(item, 'text', None):
            parts.append(str(item.text))
        elif getattr(item, 'file', None):
            parts.append(f'[Attached local file: {_basename(str(item.file))}]')
        elif getattr(item, 'image', None):
            parts.append('[Image]')
        elif getattr(item, 'video', None):
            parts.append('[Video]')
        elif getattr(item, 'audio', None):
            parts.append('[Audio]')
    return '\n'.join(parts)


def _sanitize_text(text: str, redact_paths: bool = True) -> str:
    if not isinstance(text, str):
        return text
    if redact_paths:
        text = _LOCAL_FILE_RE.sub('<LOCAL_FILE>', text)
        text = _ABS_PATH_RE.sub('<ABS_PATH>', text)
    return _API_KEY_RE.sub('<API_KEY>', text)


def _sanitize_value(value: Any, redact_paths: bool = True) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(child, redact_paths) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item, redact_paths) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, redact_paths)
    return value


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def _load_tools() -> List[Dict[str, Any]]:
    tools = []
    for tool_class in TOOL_CLASSES:
        tool = tool_class()
        tools.append({'type': 'function', 'function': copy.deepcopy(tool.function)})
    return tools


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    if not tools:
        return []
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
            result.append({'type': 'function', 'function': copy.deepcopy(tool['function'])})
        elif 'name' in tool:
            result.append({'type': 'function', 'function': copy.deepcopy(tool)})
    return result


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    result = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get('function') or {}
        arguments = function.get('arguments', '{}')
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        result.append({
            'id': tool_call.get('id') or f'call_{index}',
            'type': 'function',
            'function': {
                'name': function.get('name', ''),
                'arguments': arguments or '{}',
            },
        })
    return result


def _apply_reasoning(content: str, reasoning_content: str, policy: str) -> Tuple[str, str]:
    if policy == 'strip':
        return content, ''
    if policy == 'inline':
        if reasoning_content:
            content = reasoning_content + ('\n\n' if content else '') + content
        return content, ''
    return content, reasoning_content


def _system_prompt_for_trace(trace: Dict[str, Any]) -> str:
    started_at = trace.get('started_at')
    try:
        current = datetime.fromisoformat(started_at)
    except Exception:
        current = datetime.now().astimezone()
    date_str = f'当前时间：{current.year}年{current.month}月{current.day}日，星期{_WEEKDAYS[current.weekday()]}。'
    return DEEP_RESEARCH_SYSTEM_PROMPT.replace('{current_date}', date_str)


def _normalize_oai_messages(messages: List[Dict[str, Any]], reasoning_policy: str) -> List[Dict[str, Any]]:
    result = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        if role in ('system', 'user'):
            result.append({'role': role, 'content': _content_text(message.get('content'))})
        elif role == 'assistant':
            content = _content_text(message.get('content'))
            reasoning = message.get('reasoning_content') or ''
            content, reasoning = _apply_reasoning(content, reasoning, reasoning_policy)
            tool_calls = _normalize_tool_calls(message.get('tool_calls'))
            item = {'role': 'assistant', 'content': content}
            if reasoning:
                item['reasoning_content'] = reasoning
            if tool_calls:
                item['tool_calls'] = tool_calls
            if content or reasoning or tool_calls:
                result.append(item)
        elif role in ('tool', 'function'):
            call_id = message.get('tool_call_id') or message.get('id') or ''
            result.append({
                'role': 'tool',
                'tool_call_id': call_id,
                'content': _content_text(message.get('content')),
            })
    return result


def _normalize_trace_messages(messages: List[Dict[str, Any]], reasoning_policy: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    last_call_ids: List[str] = []
    used_call_ids: set = set()

    def flush() -> None:
        nonlocal pending, last_call_ids, used_call_ids
        if pending is None:
            return
        content = ''.join(pending['content'])
        reasoning = ''.join(pending['reasoning'])
        tool_calls = pending['tool_calls']
        content, reasoning = _apply_reasoning(content, reasoning, reasoning_policy)
        item = {'role': 'assistant', 'content': content}
        if reasoning:
            item['reasoning_content'] = reasoning
        if tool_calls:
            item['tool_calls'] = copy.deepcopy(tool_calls)
        if content or reasoning or tool_calls:
            result.append(item)
            if tool_calls:
                last_call_ids = [tool_call['id'] for tool_call in tool_calls]
                used_call_ids = set()
        pending = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        if role == 'assistant':
            if pending is None:
                pending = {'content': [], 'reasoning': [], 'tool_calls': []}
            content = _content_text(message.get('content'))
            if content:
                pending['content'].append(content)
            reasoning = message.get('reasoning_content') or ''
            if reasoning:
                pending['reasoning'].append(reasoning)
            function_call = message.get('function_call')
            if function_call:
                call_id = (message.get('extra') or {}).get('function_id') or f'call_{len(pending["tool_calls"])}'
                pending['tool_calls'].append({
                    'id': call_id,
                    'type': 'function',
                    'function': {
                        'name': function_call.get('name', ''),
                        'arguments': function_call.get('arguments') or '{}',
                    },
                })
        elif role in ('function', 'tool'):
            flush()
            call_id = message.get('tool_call_id') or (message.get('extra') or {}).get('function_id')
            if not call_id:
                for candidate in last_call_ids:
                    if candidate not in used_call_ids:
                        call_id = candidate
                        break
            if call_id:
                used_call_ids.add(call_id)
            result.append({
                'role': 'tool',
                'tool_call_id': call_id or '',
                'content': _content_text(message.get('content')),
            })
        elif role in ('system', 'user'):
            flush()
            result.append({'role': role, 'content': _content_text(message.get('content'))})
        else:
            flush()
            result.append({'role': role, 'content': _content_text(message.get('content'))})

    flush()
    return result


def _merge_response_messages(response_messages: List[Dict[str, Any]], reasoning_policy: str) -> Optional[Dict[str, Any]]:
    content_parts = []
    reasoning_parts = []
    tool_calls = []
    for message in response_messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        content = _content_text(message.get('content'))
        if content:
            content_parts.append(content)
        reasoning = message.get('reasoning_content') or ''
        if reasoning:
            reasoning_parts.append(reasoning)
        function_call = message.get('function_call')
        if function_call:
            call_id = (message.get('extra') or {}).get('function_id') or f'call_{len(tool_calls)}'
            tool_calls.append({
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': function_call.get('name', ''),
                    'arguments': function_call.get('arguments') or '{}',
                },
            })
    content = ''.join(content_parts)
    reasoning = ''.join(reasoning_parts)
    content, reasoning = _apply_reasoning(content, reasoning, reasoning_policy)
    item = {'role': 'assistant', 'content': content}
    if reasoning:
        item['reasoning_content'] = reasoning
    if tool_calls:
        item['tool_calls'] = tool_calls
    if not content and not reasoning and not tool_calls:
        return None
    return item


def _validate_sample(sample: Dict[str, Any]) -> List[str]:
    errors = []
    messages = sample.get('messages') or []
    tools = sample.get('tools') or []
    if not messages:
        return ['empty messages']
    if messages[0].get('role') != 'system':
        errors.append('first message is not system')
    if not any(message.get('role') == 'user' for message in messages):
        errors.append('messages contain no user message')
    if messages[-1].get('role') != 'assistant':
        errors.append('last message is not assistant')
    elif messages[-1].get('tool_calls'):
        errors.append('last assistant message still contains tool_calls')
    elif not _content_text(messages[-1].get('content')).strip():
        errors.append('last assistant message is empty')

    tool_names = set()
    for tool in tools:
        function = tool.get('function') if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not function.get('name'):
            errors.append('tool schema is missing function.name')
            continue
        tool_names.add(function['name'])

    pending = []
    seen_ids = set()
    for index, message in enumerate(messages):
        role = message.get('role')
        if role == 'assistant':
            for tool_call in message.get('tool_calls') or []:
                call_id = tool_call.get('id')
                if not call_id:
                    errors.append(f'message {index}: tool_call is missing id')
                elif call_id in seen_ids:
                    errors.append(f'message {index}: duplicate tool_call id {call_id}')
                else:
                    seen_ids.add(call_id)
                    pending.append(call_id)
                function = tool_call.get('function') or {}
                name = function.get('name')
                if not name:
                    errors.append(f'message {index}: tool_call is missing function.name')
                elif name not in tool_names:
                    errors.append(f'message {index}: tool {name} is not in tools')
                arguments = function.get('arguments', '{}')
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments or '{}')
                    except Exception as ex:
                        errors.append(f'message {index}: invalid tool arguments: {ex}')
                else:
                    errors.append(f'message {index}: tool arguments must be a JSON string')
        elif role == 'tool':
            call_id = message.get('tool_call_id')
            if not call_id:
                errors.append(f'message {index}: tool message is missing tool_call_id')
            elif call_id not in pending:
                errors.append(f'message {index}: tool_call_id {call_id} has no matching tool_call')
            else:
                pending.remove(call_id)
            if not isinstance(message.get('content'), str):
                errors.append(f'message {index}: tool content must be a string')
        elif role in ('system', 'user'):
            if not isinstance(message.get('content'), str):
                errors.append(f'message {index}: {role} content must be a string')
        else:
            errors.append(f'message {index}: unsupported role {role}')

    if pending:
        errors.append(f'unanswered tool calls: {pending}')
    return errors


def _build_sample(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    sample = {
        'messages': messages,
        'tools': tools,
        'metadata': metadata,
    }
    return sample


def _sample_metadata(trace: Optional[Dict[str, Any]], record: Optional[Dict[str, Any]], source: str,
                     source_file: str) -> Dict[str, Any]:
    messages = trace.get('messages') if trace else []
    query = trace.get('query', '') if trace else ''
    if not query and messages:
        for message in messages:
            if isinstance(message, dict) and message.get('role') == 'user':
                query = _content_text(message.get('content'))
                break
    if not query and record and record.get('messages_oai'):
        for message in record['messages_oai']:
            if isinstance(message, dict) and message.get('role') == 'user':
                query = _content_text(message.get('content'))
                break
    metadata = {
        'id': (trace or {}).get('id') or (record or {}).get('conversation_id', ''),
        'query': query,
        'source': source,
        'source_file': source_file,
        'status': (trace or {}).get('status', ''),
        'local_file': _basename((trace or {}).get('local_file', '')),
        'started_at': (trace or {}).get('started_at', ''),
        'duration_seconds': (trace or {}).get('duration_seconds'),
        'model': (record or {}).get('model', ''),
        'conversation_id': (record or {}).get('conversation_id', ''),
        'call_index': (record or {}).get('call_index'),
        'truncated': bool((record or {}).get('truncated', False)),
        'truncation': (record or {}).get('truncation'),
        'warnings': [],
    }
    return metadata


def _add_sample_summary(metadata: Dict[str, Any], messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> None:
    tool_calls = []
    for message in messages:
        tool_calls.extend(message.get('tool_calls') or [])
    metadata['message_count'] = len(messages)
    metadata['tool_count'] = len(tools)
    metadata['tool_call_count'] = len(tool_calls)
    metadata['reasoning_chars'] = sum(len(message.get('reasoning_content', '')) for message in messages)
    metadata['final_answer_chars'] = len(messages[-1].get('content', '')) if messages else 0
    metadata['tools_sha256'] = _hash_json(tools)
    metadata['system_prompt_sha256'] = _hash_text(messages[0].get('content', '')) if messages else ''


def _convert_trace(trace: Dict[str, Any], tools: List[Dict[str, Any]], reasoning_policy: str,
                   skip_local_files: bool, redact_paths: bool) -> Tuple[Optional[Dict[str, Any]], str]:
    if trace.get('status') not in ('passed', 'failed', 'timeout', 'error'):
        return None, 'unknown trace status'
    if skip_local_files and trace.get('local_file'):
        return None, 'local file trace skipped'
    raw_messages = trace.get('messages')
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, 'trace has no messages'
    system_prompt = trace.get('system_prompt') or _system_prompt_for_trace(trace)
    if raw_messages[0].get('role') != 'system':
        raw_messages = [{'role': 'system', 'content': system_prompt}] + raw_messages
    messages = _normalize_trace_messages(raw_messages, reasoning_policy)
    if not messages:
        return None, 'trace normalization produced no messages'
    sample_tools = _normalize_tools(trace.get('tools') or tools)
    metadata = _sample_metadata(trace, None, 'trace', trace.get('id', ''))
    if trace.get('local_file'):
        metadata['warnings'].append('trace has no auto-RAG system message; local file content is only in tool results')
    _add_sample_summary(metadata, messages, sample_tools)
    sample = _build_sample(messages, sample_tools, metadata)
    errors = _validate_sample(sample)
    if errors:
        metadata['validation_errors'] = errors
    if redact_paths:
        sample = _sanitize_value(sample)
    return sample, ''


def _convert_llm_calls(records: List[Dict[str, Any]], trace: Optional[Dict[str, Any]], tools: List[Dict[str, Any]],
                       reasoning_policy: str, skip_truncated: bool,
                       redact_paths: bool) -> Tuple[Optional[Dict[str, Any]], str]:
    if not records:
        return None, 'no LLM call records'
    records = sorted(records, key=lambda item: item.get('call_index', 0))
    record = records[-1]
    if record.get('status') != 'ok':
        return None, f'last LLM call status is {record.get("status")}'
    if record.get('truncated') and skip_truncated:
        return None, 'last LLM call was truncated'
    raw_messages = record.get('messages_oai')
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, 'last LLM call has no OpenAI messages'
    response = _merge_response_messages(record.get('response_messages') or [], reasoning_policy)
    if response is None:
        return None, 'last LLM call has no assistant response'
    if response.get('tool_calls'):
        return None, 'last LLM call ended in a tool call'
    if not response.get('content', '').strip():
        return None, 'last LLM call has no final answer content'
    messages = _normalize_oai_messages(raw_messages, reasoning_policy)
    if messages and messages[0].get('role') != 'system':
        system_prompt = (trace or {}).get('system_prompt') or _system_prompt_for_trace(trace or {})
        messages.insert(0, {'role': 'system', 'content': system_prompt})
    if not messages:
        return None, 'OpenAI normalization produced no messages'
    messages.append(response)
    sample_tools = _normalize_tools(record.get('tools') or (trace or {}).get('tools') or tools)
    metadata = _sample_metadata(trace, record, 'llm_calls', record.get('conversation_id', ''))
    _add_sample_summary(metadata, messages, sample_tools)
    sample = _build_sample(messages, sample_tools, metadata)
    errors = _validate_sample(sample)
    if errors:
        metadata['validation_errors'] = errors
    if redact_paths:
        sample = _sanitize_value(sample)
    return sample, ''


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _is_trace_file(path: Path) -> bool:
    if path.name in ('summary.json', 'conversion_report.json'):
        return False
    if path.suffix != '.json':
        return False
    try:
        data = _read_json(path)
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get('messages'), list)


def _collect_inputs(input_path: Path) -> List[Tuple[str, Optional[Path], Optional[Path]]]:
    if input_path.is_file():
        if input_path.suffix == '.jsonl':
            conversation_id = input_path.name.replace('.llm_calls.jsonl', '').replace('.jsonl', '')
            return [(conversation_id, input_path, None)]
        if input_path.suffix == '.json':
            return [(input_path.stem, None, input_path)]
        raise SystemExit(f'Unsupported input file: {input_path}')
    traces = {path.stem: path for path in input_path.glob('*.json') if _is_trace_file(path)}
    calls = {path.name[:-len('.llm_calls.jsonl')]: path for path in input_path.glob('*.llm_calls.jsonl')}
    case_ids = sorted(set(traces) | set(calls))
    return [(case_id, calls.get(case_id), traces.get(case_id)) for case_id in case_ids]


def _split_samples(samples: List[Dict[str, Any]], val_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_ratio <= 0 or len(samples) < 2:
        return samples, []
    ordered = sorted(samples, key=lambda sample: _hash_json(sample))
    val_count = max(1, int(round(len(ordered) * val_ratio)))
    val_count = min(val_count, len(ordered) - 1)
    return ordered[:-val_count], ordered[-val_count:]


def _write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as fout:
        for sample in samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert DeepResearch traces to OpenAI Chat JSONL.')
    parser.add_argument('--input', required=True, help='Trace directory, trace JSON, or .llm_calls.jsonl file.')
    parser.add_argument('--output', default=str(Path(__file__).with_name('datasets') / 'deep_research_openai'))
    parser.add_argument('--source', choices=['auto', 'llm_calls', 'trace'], default='auto')
    parser.add_argument('--reasoning', choices=['keep', 'strip', 'inline'], default='keep')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--include-failed', action='store_true')
    parser.add_argument('--skip-local-files', action='store_true')
    parser.add_argument('--skip-truncated', action='store_true')
    parser.add_argument('--no-redact-paths', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    redact_paths = not args.no_redact_paths
    tools = _load_tools()
    samples = []
    skipped = []
    invalid = []
    duplicates = []
    seen_hashes = set()
    counts = Counter()

    for case_id, calls_path, trace_path in _collect_inputs(input_path):
        counts['cases'] += 1
        trace = None
        if trace_path is not None:
            try:
                trace = _read_json(trace_path)
            except Exception as ex:
                skipped.append({'id': case_id, 'reason': f'trace read error: {ex}'})
                continue
            if trace.get('status') not in ('passed', 'failed', 'timeout', 'error'):
                skipped.append({'id': case_id, 'reason': 'unknown trace status'})
                continue
            if trace.get('status') != 'passed' and not args.include_failed:
                skipped.append({'id': case_id, 'reason': f'trace status is {trace.get("status")}'})
                continue

        sample = None
        reason = ''
        if calls_path is not None and args.source in ('auto', 'llm_calls'):
            try:
                records = _read_jsonl(calls_path)
            except Exception as ex:
                records = []
                reason = f'LLM call read error: {ex}'
            counts['llm_call_files'] += 1
            sample, reason = _convert_llm_calls(records, trace, tools, args.reasoning, args.skip_truncated,
                                                redact_paths)
            if sample is not None:
                counts['llm_call_samples'] += 1

        if sample is None and trace is not None and args.source in ('auto', 'trace'):
            sample, trace_reason = _convert_trace(trace, tools, args.reasoning, args.skip_local_files, redact_paths)
            if sample is not None:
                counts['trace_samples'] += 1
            elif not reason:
                reason = trace_reason

        if sample is None:
            skipped.append({'id': case_id, 'reason': reason or 'no usable trace or LLM call'})
            continue

        validation_errors = _validate_sample(sample)
        if validation_errors:
            sample['metadata']['validation_errors'] = validation_errors
            invalid.append({'id': case_id, 'errors': validation_errors})
            if args.strict:
                continue

        sample_hash = _hash_json({'messages': sample['messages'], 'tools': sample['tools']})
        if sample_hash in seen_hashes:
            duplicates.append(case_id)
            continue
        seen_hashes.add(sample_hash)
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    train_samples, val_samples = _split_samples(samples, args.val_ratio)
    report = {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'input': str(input_path),
        'output': str(output_dir),
        'reasoning_policy': args.reasoning,
        'counts': {
            **dict(counts),
            'samples': len(samples),
            'train': len(train_samples),
            'val': len(val_samples),
            'skipped': len(skipped),
            'invalid': len(invalid),
            'duplicates': len(duplicates),
        },
        'samples': [
            {
                'id': sample['metadata'].get('id'),
                'source': sample['metadata'].get('source'),
                'messages': len(sample['messages']),
                'tools': len(sample['tools']),
                'tool_calls': sample['metadata'].get('tool_call_count'),
                'final_answer_chars': sample['metadata'].get('final_answer_chars'),
                'truncated': sample['metadata'].get('truncated'),
                'warnings': sample['metadata'].get('warnings', []),
                'validation_errors': sample['metadata'].get('validation_errors', []),
            }
            for sample in samples
        ],
        'skipped': skipped,
        'invalid': invalid,
        'duplicates': duplicates,
    }

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / 'train.jsonl', train_samples)
        _write_jsonl(output_dir / 'val.jsonl', val_samples)
        (output_dir / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Samples: {len(samples)} (train {len(train_samples)}, val {len(val_samples)})')
    print(f'Skipped: {len(skipped)}, invalid: {len(invalid)}, duplicates: {len(duplicates)}')
    if skipped:
        for item in skipped[:5]:
            print(f'  skip {item["id"]}: {item["reason"]}')
    if invalid:
        for item in invalid[:5]:
            print(f'  invalid {item["id"]}: {item["errors"][:2]}')
    if args.dry_run:
        print('Dry run: no files written.')
    else:
        print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
