#!/usr/bin/env python
"""Run the DeepResearch agent over question JSON files and persist annotations.

The runner is resumable: completed question ids are read from the output JSONL
files and skipped on the next invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_agent.agents.fncall_agent as fncall_agent  # noqa: E402
from examples.assistant_deep_research import init_agent_service  # noqa: E402

DEFAULT_QUESTIONS_DIR = ROOT / 'tests/examples/datasets/annotation_sources/questions'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs/deep_research_annotations'
SUCCESS_STATUSES = {'passed'}


class QuestionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise QuestionTimeout('question timed out')


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return message
    return message.model_dump()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get('text'):
                parts.append(str(item['text']))
        elif getattr(item, 'text', None):
            parts.append(str(item.text))
    return '\n'.join(parts)


def _extract_final(generated: List[Any]) -> str:
    for msg in reversed(generated):
        role = msg.get('role') if isinstance(msg, dict) else msg.role
        function_call = msg.get('function_call') if isinstance(msg, dict) else msg.function_call
        if role == 'assistant' and not function_call:
            content = msg.get('content') if isinstance(msg, dict) else msg.content
            text = _content_text(content).strip()
            if text:
                return text
    return ''


def _tool_summary(generated: List[Any]) -> Dict[str, Any]:
    counts = Counter()
    calls = []
    read_urls = []
    errors = []
    for msg in generated:
        if isinstance(msg, dict):
            role = msg.get('role')
            function_call = msg.get('function_call')
            name = msg.get('name')
            content = msg.get('content')
        else:
            role = msg.role
            function_call = msg.function_call
            name = msg.name
            content = msg.content
        if role == 'assistant' and function_call:
            tool_name = function_call.get('name') if isinstance(function_call, dict) else function_call.name
            arguments = function_call.get('arguments') if isinstance(function_call, dict) else function_call.arguments
            counts[tool_name] += 1
            try:
                parsed = json.loads(arguments or '{}')
            except Exception:
                parsed = {}
            calls.append({'tool': tool_name, 'arguments': parsed})
            if tool_name == 'read_webpage' and isinstance(parsed.get('url'), str):
                read_urls.append(parsed['url'])
        elif role == 'function':
            text = _content_text(content)
            if text.startswith(('An error occurred when calling tool', 'Failed to fetch',
                                'Local document retrieval failed', 'No supported local files')):
                errors.append({'tool': name, 'error': text[:800]})
    return {'counts': dict(counts), 'calls': calls, 'read_urls': read_urls, 'errors': errors}


def _question_id(source: str, index: int, question: str) -> str:
    digest = hashlib.sha256(question.encode('utf-8')).hexdigest()[:10]
    return f'{source}-{index:06d}-{digest}'


def _load_questions(questions_dir: Path, source: str) -> List[Dict[str, Any]]:
    path = questions_dir / f'{source}.json'
    if not path.exists():
        raise FileNotFoundError(path)
    questions = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(questions, list) or not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError(f'Expected a non-empty JSON array of strings: {path}')
    return [{'id': _question_id(source, i, question), 'source': source, 'index': i, 'question': question}
            for i, question in enumerate(questions, 1)]


def _completed_ids(path: Path, require_messages: bool = False) -> set:
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('status') in SUCCESS_STATUSES:
                if require_messages and not record.get('messages'):
                    continue
                completed.add(record.get('id'))
    return completed


def _run_question(item: Dict[str, Any], config_path: Optional[str], timeout_seconds: int,
                  save_trace: bool, max_llm_calls: int = 0) -> Dict[str, Any]:
    started_at = _now()
    started = time.time()
    generated = []
    error = ''
    error_trace = ''
    status = 'failed'
    bot = None
    if timeout_seconds > 0 and hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        bot = init_agent_service(config_path=config_path)
        if max_llm_calls > 0:
            fncall_agent.MAX_LLM_CALL_PER_RUN = max_llm_calls
        for chunk in bot.run(messages=[{'role': 'user', 'content': item['question']}]):
            generated = chunk
        status = 'passed'
    except QuestionTimeout as ex:
        status = 'timeout'
        error = str(ex)
    except Exception as ex:
        status = 'error'
        error = f'{type(ex).__name__}: {ex}'
        error_trace = traceback.format_exc()
    finally:
        if timeout_seconds > 0 and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

    final_answer = _extract_final(generated)
    if status == 'passed' and not final_answer:
        status = 'empty'
    tools = _tool_summary(generated)
    record = {
        'id': item['id'],
        'source': item['source'],
        'index': item['index'],
        'question': item['question'],
        'status': status,
        'started_at': started_at,
        'finished_at': _now(),
        'duration_seconds': round(time.time() - started, 2),
        'final_answer': final_answer,
        'tool_summary': tools,
        'error': error,
    }
    if error_trace:
        record['error_trace'] = error_trace
    if save_trace:
        record['messages'] = [{'role': 'user', 'content': item['question']}]
        record['messages'].extend(_to_dict(message) for message in generated)
    return record


def _timeout_record(item: Dict[str, Any], started_at: str, started: float, timeout_seconds: int,
                    message: str = 'question timed out') -> Dict[str, Any]:
    return {
        'id': item['id'],
        'source': item['source'],
        'index': item['index'],
        'question': item['question'],
        'status': 'timeout',
        'started_at': started_at,
        'finished_at': _now(),
        'duration_seconds': round(time.time() - started, 2),
        'final_answer': '',
        'tool_summary': {'counts': {}, 'calls': [], 'read_urls': [], 'errors': []},
        'error': message,
        'timeout_seconds': timeout_seconds,
    }


def _run_question_subprocess(item: Dict[str, Any], output_dir: Path, config_path: Optional[str],
                             timeout_seconds: int, save_trace: bool, max_llm_calls: int) -> Dict[str, Any]:
    """Run one question in a child process so a blocked LLM call can be killed."""
    started_at = _now()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='annotation-', dir=output_dir) as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / 'input.json'
        output_path = tmp_dir / 'output.json'
        input_path.write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
        cmd = [
            sys.executable, str(Path(__file__).resolve()), '--worker',
            '--worker-input', str(input_path),
            '--worker-output', str(output_path),
            '--timeout-seconds', '0',
            '--max-llm-calls', str(max_llm_calls),
        ]
        if config_path:
            cmd.extend(['--config', config_path])
        if save_trace:
            cmd.append('--save-trace')
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as ex:
            stdout = (ex.stdout or '') if isinstance(ex.stdout, str) else ''
            stderr = (ex.stderr or '') if isinstance(ex.stderr, str) else ''
            detail = (stdout + '\n' + stderr).strip()
            return _timeout_record(item, started_at, started, timeout_seconds,
                                   f'question timed out after {timeout_seconds}s' +
                                   (f'\n{detail[-2000:]}' if detail else ''))
        if not output_path.exists():
            detail = (completed.stdout + '\n' + completed.stderr).strip()
            record = _timeout_record(item, started_at, started, timeout_seconds,
                                     f'worker exited without output (code {completed.returncode})')
            record['status'] = 'error'
            record['error'] = f'worker exited without output (code {completed.returncode})' + (
                f'\n{detail[-2000:]}' if detail else '')
            return record
        try:
            record = json.loads(output_path.read_text(encoding='utf-8'))
        except Exception as ex:
            record = _timeout_record(item, started_at, started, timeout_seconds,
                                     f'could not read worker output: {ex}')
            record['status'] = 'error'
            return record
        record['duration_seconds'] = round(time.time() - started, 2)
        return record


def _write_summary(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    by_source = defaultdict(Counter)
    durations = defaultdict(float)
    for record in records:
        by_source[record['source']][record['status']] += 1
        durations[record['source']] += float(record.get('duration_seconds') or 0)
    summary = {
        'finished_at': _now(),
        'total': len(records),
        'sources': {source: dict(counts) for source, counts in by_source.items()},
        'duration_seconds': {source: round(value, 2) for source, value in durations.items()},
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Annotate question sets with the DeepResearch agent.')
    parser.add_argument('--questions-dir', default=str(DEFAULT_QUESTIONS_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--source', action='append', default=[],
                        help='Source JSON basename without .json; repeatable. Default: all files.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only run the first N questions per source (0 means all).')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry questions that previously ended in failed/timeout/error.')
    parser.add_argument('--save-trace', action='store_true',
                        help='Include the full generated message trace in each JSONL record.')
    parser.add_argument('--config', default='', help='Path to a global config JSON.')
    parser.add_argument('--max-llm-calls', type=int, default=0,
                        help='Override the configured LLM call budget for each question (0 keeps config).')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of concurrent question worker processes.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-input', default='', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', default='', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        item = json.loads(Path(args.worker_input).read_text(encoding='utf-8'))
        record = _run_question(item, args.config or None, 0, args.save_trace, args.max_llm_calls)
        Path(args.worker_output).write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
        return

    questions_dir = Path(args.questions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = args.source or sorted(path.stem for path in questions_dir.glob('*.json'))
    if not sources:
        raise SystemExit(f'No question JSON files found in {questions_dir}')

    all_records = []
    for source in sources:
        questions = _load_questions(questions_dir, source)
        if args.limit:
            questions = questions[:args.limit]
        output_path = output_dir / f'{source}.jsonl'
        completed = set() if args.retry_failed else _completed_ids(output_path, require_messages=args.save_trace)
        pending = [item for item in questions if item['id'] not in completed]
        print(f'[{source}] {len(questions)} questions, {len(completed)} completed, {len(pending)} pending',
              flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(_run_question_subprocess, item, output_dir, args.config or None,
                                args.timeout_seconds, args.save_trace, args.max_llm_calls): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as ex:
                    record = _timeout_record(item, _now(), time.time(), args.timeout_seconds,
                                             f'parent worker failed: {type(ex).__name__}: {ex}')
                    record['status'] = 'error'
                with output_path.open('a', encoding='utf-8') as fout:
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'[{source}] {item["id"]} {record["status"]} in {record["duration_seconds"]}s, '
                      f'tools={record["tool_summary"]["counts"]}', flush=True)
                all_records.append(record)

    records = []
    for source in sources:
        path = output_dir / f'{source}.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as fin:
                for line in fin:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    _write_summary(output_dir, records)
    print(f'Annotations saved to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
