"""Public background research/cache adapters. Importing performs no I/O."""

import json
import os
from pathlib import Path
import threading

from .agent import IntelligenceAgent
from .extraction import OllamaFactExtractor
from .sources import FootballDataSource, GdeltNewsSource, JsonEndpointSource, OpenMeteoSource, RssSource

_default = None
_default_lock = threading.Lock()


def build_agent_from_environment():
    """No installs/credentials modifications. Optional sources use existing config."""
    sources, configuration_errors = [], []
    config_file = os.environ.get('FOOTBALL_INTELLIGENCE_SOURCES')
    if config_file:
        try:
            config = json.loads(Path(config_file).read_text(encoding='utf-8'))
            for item in config.get('sources', [])[:12]:
                if item.get('type') == 'rss':
                    sources.append(RssSource(item['name'], item['url_template']))
                elif item.get('type') == 'json':
                    sources.append(JsonEndpointSource(item['name'], item['url_template'],
                                                      categories=item.get('categories', ('injuries', 'lineup', 'news', 'weather', 'schedule', 'h2h'))))
        except (OSError, ValueError, TypeError, KeyError):
            # No partially trusted configuration or network fallback on parse failure.
            sources = []
            configuration_errors.append({'tool': 'configuration', 'error': 'invalid_sources_file'})
    key = os.environ.get('FOOTBALL_API_KEY')
    if key:
        sources += [FootballDataSource(key, 'lineup'), FootballDataSource(key, 'schedule', 'home'),
                    FootballDataSource(key, 'schedule', 'away'), FootballDataSource(key, 'h2h')]
    if os.environ.get('FOOTBALL_INTELLIGENCE_GDELT', '1') != '0':
        sources.append(GdeltNewsSource())
    if os.environ.get('FOOTBALL_INTELLIGENCE_WEATHER', '1') != '0':
        sources.append(OpenMeteoSource())
    extractor = None
    if os.environ.get('FOOTBALL_OLLAMA_MODEL'):
        try:
            extractor = OllamaFactExtractor(os.environ['FOOTBALL_OLLAMA_MODEL'],
                                            base_url=os.environ.get('FOOTBALL_OLLAMA_URL', 'http://127.0.0.1:11434'))
        except ValueError:
            configuration_errors.append({'tool': 'configuration', 'error': 'invalid_local_ollama_endpoint'})
    agent = IntelligenceAgent(cache_dir=os.environ.get('FOOTBALL_INTELLIGENCE_CACHE_DIR'), sources=sources, extractor=extractor)
    agent.configuration_errors = configuration_errors
    return agent


def _agent():
    global _default
    with _default_lock:
        if _default is None:
            _default = build_agent_from_environment()
    return _default


def research_match_context(match, *, as_of=None, findings=(), force=False):
    return _agent().research_match_context(match, as_of=as_of, findings=findings, force=force)


def get_cached_context(match_id, *, as_of, kickoff=None, max_age_seconds=1800):
    return _agent().get_cached_context(match_id, as_of=as_of, kickoff=kickoff, max_age_seconds=max_age_seconds)


def submit_research(match, *, findings=(), force=False):
    return _agent().submit_research(match, findings=findings, force=force)


def get_intelligence_status():
    return _agent().get_status()


__all__ = ['IntelligenceAgent', 'research_match_context', 'get_cached_context', 'submit_research',
           'OllamaFactExtractor', 'JsonEndpointSource', 'RssSource', 'FootballDataSource',
           'GdeltNewsSource', 'OpenMeteoSource', 'get_intelligence_status']
