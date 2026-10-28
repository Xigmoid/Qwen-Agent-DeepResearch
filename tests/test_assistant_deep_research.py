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

import json
from unittest import mock

import pytest

from scripts.assistant_deep_research import (
    ReadWebpage,
    ResearchNotes,
    SearchLocalDocs,
    WebSearch,
    _HTMLTextExtractor,
    _is_public_http_url,
)


class FakeResponse:

    def __init__(self, text='', url='', content_type='text/html', payload=None):
        self.text = text
        self.content = text.encode('utf-8')
        self.url = url
        self.headers = {'Content-Type': content_type}
        self.encoding = 'utf-8'
        self.apparent_encoding = 'utf-8'
        self.status_code = 200
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def reset_web_search_state():
    WebSearch._cache.clear()
    WebSearch._last_bing_call = 0.0
    yield
    WebSearch._cache.clear()
    WebSearch._last_bing_call = 0.0


def test_html_extractor_keeps_article_and_drops_page_chrome():
    html = ('<html><head><title>Demo</title></head><body>'
            '<nav>NAV_ONLY</nav>'
            '<div class="article"><h1>Title</h1><p>Body text.</p>'
            '<div class="sidebar">SIDEBAR_ONLY</div></div>'
            '<form><p>Form kept.</p></form>'
            '<button>BUTTON_ONLY</button>'
            '</body></html>')
    parser = _HTMLTextExtractor()
    parser.feed(html)
    text = parser.get_text()
    assert 'Title' in text
    assert 'Body text.' in text
    assert 'Form kept.' in text
    assert 'NAV_ONLY' not in text
    assert 'SIDEBAR_ONLY' not in text
    assert 'BUTTON_ONLY' not in text


def test_url_guard_allows_public_http_and_blocks_private_targets():
    assert _is_public_http_url('https://example.com/report')
    assert _is_public_http_url('http://example.com/report')
    assert not _is_public_http_url('http://127.0.0.1/report')
    assert not _is_public_http_url('http://192.168.1.10/report')
    assert not _is_public_http_url('http://169.254.169.254/latest/meta-data')
    assert not _is_public_http_url('ftp://example.com/report')


def test_web_search_uses_serper_or_bing_fallback(monkeypatch):
    post = mock.Mock(return_value=FakeResponse(payload={
        'organic': [{
            'title': 'Serper result',
            'link': 'https://example.com/a',
            'snippet': 'Serper snippet',
            'date': '2026-01-01',
        }]
    }))
    monkeypatch.setattr('scripts.assistant_deep_research.requests.post', post)
    monkeypatch.setenv('SERPER_API_KEY', 'test-key')
    monkeypatch.delenv('SERPER_URL', raising=False)

    serper = json.loads(WebSearch().call({'query': 'qwen agent', 'num_results': 3}))
    assert serper[0]['provider'] == 'serper'
    assert serper[0]['url'] == 'https://example.com/a'
    WebSearch().call({'query': 'qwen agent', 'num_results': 3})
    assert post.call_count == 1

    monkeypatch.delenv('SERPER_API_KEY')
    rss = ('<rss><channel>'
           '<item><title>Bing result</title><link>https://example.com/b</link>'
           '<description>Bing snippet</description><pubDate>2026-01-02</pubDate></item>'
           '<item><title>Bing internal</title><link>https://cn.bing.com/search?q=x</link></item>'
           '</channel></rss>')
    get = mock.Mock(return_value=FakeResponse(text=rss, content_type='application/rss+xml'))
    monkeypatch.setattr('scripts.assistant_deep_research.requests.get', get)

    bing = json.loads(WebSearch().call({'query': 'qwen agent', 'num_results': 3}))
    assert len(bing) == 1
    assert bing[0]['provider'] == 'bing_rss'
    assert bing[0]['title'] == 'Bing result'
    assert bing[0]['snippet'] == 'Bing snippet'
    assert bing[0]['date'] == '2026-01-02'


def test_read_webpage_follows_redirect_and_paginates(monkeypatch):
    html = ('<html><body><nav>NAV_ONLY</nav><h1>Hello</h1><p>World!</p>'
            '<button>BUTTON_ONLY</button></body></html>')
    response = FakeResponse(text=html, url='https://final.example/page')
    get = mock.Mock(return_value=response)
    monkeypatch.setattr('scripts.assistant_deep_research.requests.get', get)

    out = ReadWebpage().call({'url': 'https://start.example/page', 'max_chars': 1000})
    assert out.startswith('URL: https://final.example/page')
    assert 'Hello' in out
    assert 'World!' in out
    assert 'NAV_ONLY' not in out
    assert 'BUTTON_ONLY' not in out

    page = ReadWebpage().call({'url': 'https://start.example/page', 'start': 5, 'max_chars': 1000})
    assert 'Characters 5-' in page
    assert 'World' in page


def test_research_notes_crud():
    notes = ResearchNotes()
    saved = notes.call({'action': 'add', 'content': 'Qwen3.5 was released in 2026.', 'source': 'https://example.com'})
    assert 'Saved note #1' in saved
    note = json.loads(notes.call({'action': 'read', 'note_id': 1}))
    assert note['content'].startswith('Qwen3.5')
    listed = json.loads(notes.call({'action': 'list'}))
    assert listed[0]['id'] == 1
    assert 'cleared' in notes.call({'action': 'clear'}).lower()


def test_local_document_retrieval(tmp_path):
    # Deviation from upstream: the local-document stack needs the optional RAG
    # dependencies, so this case is skipped on a minimal installation instead
    # of failing.  Install them with `pip install -r requirements-rag.txt`.
    # See DEVIATIONS.md.
    pytest.importorskip('rank_bm25')
    pytest.importorskip('pdfplumber')

    path = tmp_path / 'brief.md'
    path.write_text('# Brief\n\nAurora uses BM25 retrieval and requires three sources.\n', encoding='utf-8')
    out = SearchLocalDocs().call({'query': 'BM25 retrieval'}, files=[f'file://{path}'])
    assert 'BM25' in out
    assert str(path) in out
