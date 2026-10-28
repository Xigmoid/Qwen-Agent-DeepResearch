#!/usr/bin/env python
"""Convert saved annotation traces into an OpenAI-compatible chat JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.examples.convert_traces import (  # noqa: E402
    _convert_trace,
    _load_tools,
    _validate_sample,
)

SUCCESS_STATUSES = {'passed'}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _convert_record(record: Dict[str, Any], source: str, tools: List[Dict[str, Any]],
                    reasoning: str) -> Tuple[Dict[str, Any], str]:
    trace = {
        'id': record.get('id', ''),
        'query': record.get('question', ''),
        'status': record.get('status'),
        'started_at': record.get('started_at', ''),
        'duration_seconds': record.get('duration_seconds'),
        'messages': record.get('messages'),
    }
    sample, reason = _convert_trace(trace, tools, reasoning, False, True)
    if sample is None:
        return None, reason or 'trace conversion failed'
    sample['id'] = record.get('id', '')
    sample['metadata']['annotation_source'] = source
    sample['metadata']['annotation_index'] = record.get('index')
    return sample, ''


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert DeepResearch annotation JSONL traces to one OpenAI-compatible JSON file.')
    parser.add_argument('--input-dir', default=str(ROOT / 'outputs/deep_research_annotations'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/openai_messages/annotations.json'))
    parser.add_argument('--reasoning', choices=['keep', 'strip', 'inline'], default='keep')
    parser.add_argument('--include-failed', action='store_true')
    parser.add_argument('--allow-invalid', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    if not input_dir.exists():
        raise SystemExit(f'Input directory not found: {input_dir}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tools = _load_tools()
    counts = Counter()
    skipped = []
    invalid = []
    samples_by_id: Dict[str, Dict[str, Any]] = {}

    for path in sorted(input_dir.glob('*.jsonl')):
        for record in _read_jsonl(path):
            counts['records'] += 1
            status = record.get('status')
            if status not in SUCCESS_STATUSES and not args.include_failed:
                skipped.append({'id': record.get('id'), 'source': path.stem, 'reason': f'status={status}'})
                continue
            messages = record.get('messages')
            if not isinstance(messages, list) or not messages:
                skipped.append({'id': record.get('id'), 'source': path.stem, 'reason': 'no saved messages'})
                continue
            sample, reason = _convert_record(record, path.stem, tools, args.reasoning)
            if sample is None:
                skipped.append({'id': record.get('id'), 'source': path.stem, 'reason': reason})
                continue
            errors = _validate_sample(sample)
            if errors and not args.allow_invalid:
                invalid.append({'id': record.get('id'), 'source': path.stem, 'errors': errors})
                continue
            if errors:
                sample['metadata']['validation_errors'] = errors
            sample['metadata']['source_file'] = path.name
            samples_by_id[sample['id']] = sample
            counts['samples'] += 1

    samples = sorted(samples_by_id.values(),
                     key=lambda item: (item['metadata'].get('annotation_source', ''),
                                       item['metadata'].get('annotation_index') or 0))
    if args.max_samples:
        samples = samples[:args.max_samples]

    result = {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'input_dir': str(input_dir),
        'output': str(output_path),
        'reasoning_policy': args.reasoning,
        'counts': {
            'records': counts['records'],
            'samples': len(samples),
            'unique_ids': len(samples_by_id),
            'skipped': len(skipped),
            'invalid': len(invalid),
        },
        'samples': samples,
        'skipped': skipped,
        'invalid': invalid,
    }
    kwargs = {'ensure_ascii': False, 'indent': 2} if args.pretty else {'ensure_ascii': False}
    output_path.write_text(json.dumps(result, **kwargs) + '\n', encoding='utf-8')
    print(f'Samples: {len(samples)}, records: {counts["records"]}, '
          f'skipped: {len(skipped)}, invalid: {len(invalid)}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
