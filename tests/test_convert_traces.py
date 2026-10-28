import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import convert_traces as ct  # noqa: E402


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the web.',
            'parameters': {
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
            },
        },
    }]


def _trace(messages, status='passed', local_file=''):
    return {
        'id': 'demo',
        'query': 'demo query',
        'status': status,
        'started_at': '2026-10-23T12:00:00+08:00',
        'local_file': local_file,
        'messages': messages,
    }


def _sample(messages, tools=None):
    return {'messages': messages, 'tools': tools if tools is not None else _tools(), 'metadata': {}}


def test_normalize_trace_merges_assistant_fragments_and_maps_tool_results():
    trace = _trace([
        {'role': 'user', 'content': 'find it'},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'need a search', 'name': 'DeepResearch'},
        {
            'role': 'assistant',
            'content': '',
            'function_call': {'name': 'web_search', 'arguments': '{"query": "x"}'},
            'extra': {'function_id': 'call_1'},
            'name': 'DeepResearch',
        },
        {'role': 'function', 'name': 'web_search', 'content': '[{"title": "x"}]', 'extra': {'function_id': 'call_1'}},
        {'role': 'assistant', 'content': 'final answer', 'name': 'DeepResearch'},
    ])
    sample, reason = ct._convert_trace(trace, _tools(), 'keep', False, True)
    assert reason == ''
    assert not ct._validate_sample(sample)
    assert sample['messages'][0]['role'] == 'system'
    assert sample['messages'][1]['role'] == 'user'
    assistant = sample['messages'][2]
    assert assistant['reasoning_content'] == 'need a search'
    assert assistant['tool_calls'][0]['id'] == 'call_1'
    assert sample['messages'][3]['role'] == 'tool'
    assert sample['messages'][3]['tool_call_id'] == 'call_1'
    assert sample['messages'][-1]['content'] == 'final answer'


def test_validate_sample_rejects_unanswered_tool_call():
    sample = _sample([
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'find it'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'web_search', 'arguments': '{"query": "x"}'},
            }],
        },
        {'role': 'assistant', 'content': 'final answer'},
    ])
    errors = ct._validate_sample(sample)
    assert any('unanswered tool calls' in error for error in errors)


def test_normalize_oai_messages_maps_id_to_tool_call_id():
    messages = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'find it'},
        {
            'role': 'assistant',
            'content': None,
            'reasoning_content': 'think',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'web_search', 'arguments': '{"query": "x"}'},
            }],
        },
        {'role': 'tool', 'id': 'call_1', 'content': [{'type': 'text', 'text': 'result'}]},
        {'role': 'assistant', 'content': 'final answer'},
    ]
    result = ct._normalize_oai_messages(messages, 'keep')
    assert result[2]['tool_calls'][0]['id'] == 'call_1'
    assert result[3]['role'] == 'tool'
    assert result[3]['tool_call_id'] == 'call_1'
    assert result[3]['content'] == 'result'


def test_convert_llm_calls_uses_last_request_and_response():
    records = [
        {
            'conversation_id': 'demo',
            'call_index': 0,
            'status': 'ok',
            'messages_oai': [
                {'role': 'system', 'content': 'system'},
                {'role': 'user', 'content': 'find it'},
            ],
            'response_messages': [{'role': 'assistant', 'content': 'draft'}],
            'tools': _tools(),
            'truncated': False,
        },
        {
            'conversation_id': 'demo',
            'call_index': 1,
            'status': 'ok',
            'messages_oai': [
                {'role': 'system', 'content': 'system'},
                {'role': 'user', 'content': 'find it'},
                {'role': 'assistant', 'content': 'draft'},
            ],
            'response_messages': [
                {'role': 'assistant', 'content': '', 'reasoning_content': 'final think'},
                {'role': 'assistant', 'content': 'final answer'},
            ],
            'tools': _tools(),
            'truncated': False,
        },
    ]
    sample, reason = ct._convert_llm_calls(records, _trace([]), _tools(), 'keep', False, True)
    assert reason == ''
    assert not ct._validate_sample(sample)
    assert sample['metadata']['source'] == 'llm_calls'
    assert sample['metadata']['call_index'] == 1
    assert sample['messages'][-1]['content'] == 'final answer'
    assert sample['messages'][-1]['reasoning_content'] == 'final think'


def test_convert_llm_calls_rejects_tool_call_ending():
    records = [{
        'conversation_id': 'demo',
        'call_index': 0,
        'status': 'ok',
        'messages_oai': [
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'find it'},
        ],
        'response_messages': [{
            'role': 'assistant',
            'content': '',
            'function_call': {'name': 'web_search', 'arguments': '{"query": "x"}'},
            'extra': {'function_id': 'call_1'},
        }],
        'tools': _tools(),
        'truncated': False,
    }]
    sample, reason = ct._convert_llm_calls(records, _trace([]), _tools(), 'keep', False, True)
    assert sample is None
    assert 'tool call' in reason


def test_reasoning_strip_and_inline():
    trace = _trace([
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'think'},
        {'role': 'assistant', 'content': 'answer'},
    ])
    stripped, _ = ct._convert_trace(trace, _tools(), 'strip', False, True)
    assert 'reasoning_content' not in stripped['messages'][-1]
    assert stripped['messages'][-1]['content'] == 'answer'

    inlined, _ = ct._convert_trace(trace, _tools(), 'inline', False, True)
    assert 'reasoning_content' not in inlined['messages'][-1]
    assert inlined['messages'][-1]['content'] == 'think\n\nanswer'


def test_content_text_uses_basename_for_local_files():
    text = ct._content_text([
        {'file': '/Users/example/private/project.md'},
        {'text': 'summarize it'},
    ])
    assert text == '[Attached local file: project.md]\nsummarize it'


def test_validate_sample_rejects_tool_call_without_tools():
    sample = _sample([
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'q'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'web_search', 'arguments': '{}'},
            }],
        },
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'result'},
        {'role': 'assistant', 'content': 'answer'},
    ], tools=[])
    errors = ct._validate_sample(sample)
    assert any('is not in tools' in error for error in errors)
