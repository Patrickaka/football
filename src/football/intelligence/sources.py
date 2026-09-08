"""Read-only adapters. Sources are explicit configuration, never model decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from urllib.parse import quote, urlencode, urlsplit
import xml.etree.ElementTree as ET

from ...domain.sports.football.match_context import CATEGORIES, timestamp
from .transport import ReadOnlyHTTP


def _now():
    return datetime.now(timezone.utc).isoformat()


def query_for(match, category):
    keywords = {'injuries': 'injury suspension 伤停', 'lineup': 'confirmed starting lineup 首发',
                'news': 'team news', 'weather': 'stadium weather forecast',
                'schedule': 'recent fixtures rest days', 'h2h': 'head to head results'}
    home = match.get('home_search_name') or match.get('home', '')
    away = match.get('away_search_name') or match.get('away', '')
    return f'{home} {away} {keywords[category]}'


class JsonEndpointSource:
    """A configured JSON endpoint returning {findings: [...]} or a findings list.

    Placeholders: match_id, home, away, category, query, as_of. Values are URL
    encoded. Each finding must provide explicit published_at; legacy ts alone
    is intentionally not interpreted as a publication timestamp.
    """
    def __init__(self, name, url_template, *, categories=CATEGORIES, allowed_hosts=None, transport=None):
        self.name, self.url_template = name, url_template
        self.categories = tuple(c for c in categories if c in CATEGORIES)
        self.transport = transport or ReadOnlyHTTP(allowed_hosts or [urlsplit(url_template).hostname])

    def fetch(self, match, category, *, as_of, timeout):
        values = dict(match, category=category, query=query_for(match, category), as_of=as_of.isoformat())
        url = self.url_template.format_map({k: quote(str(v), safe='') for k, v in values.items()})
        response = self.transport.get_json(url, timeout=timeout)
        rows = response.get('findings', []) if isinstance(response, dict) else response
        if not isinstance(rows, list):
            raise ValueError('configured endpoint must return a findings list')
        collected = _now()
        return [dict(row, collected_at=collected, source=row.get('source') or self.name)
                for row in rows[:30] if isinstance(row, dict)]


class RssSource:
    """Free configurable RSS/Atom source; publication metadata remains auditable."""
    categories = ('injuries', 'lineup', 'news')

    def __init__(self, name, url_template, *, transport=None):
        self.name, self.url_template = name, url_template
        self.transport = transport or ReadOnlyHTTP([urlsplit(url_template).hostname])

    def fetch(self, match, category, *, as_of, timeout):
        url = self.url_template.format(query=quote(query_for(match, category), safe=''),
                                       home=quote(str(match.get('home', '')), safe=''),
                                       away=quote(str(match.get('away', '')), safe=''))
        root = ET.fromstring(self.transport.request(url, timeout=timeout))
        rows = list(root.findall('.//item')) or list(root.findall('.//{http://www.w3.org/2005/Atom}entry'))
        findings, collected = [], _now()
        for row in rows[:50]:
            values = {child.tag.rsplit('}', 1)[-1]: child for child in row}
            def text(key):
                return ''.join(values[key].itertext()).strip() if key in values else ''
            snippet = f"{text('title')} {text('description') or text('summary')}".strip()
            names = [match.get(key) for key in ('home', 'away', 'home_search_name', 'away_search_name')]
            if not any(name and str(name).casefold() in snippet.casefold() for name in names):
                continue
            published = text('pubDate') or text('published')
            try:
                published = parsedate_to_datetime(published).isoformat() if published and not timestamp(published) else published
            except (TypeError, ValueError):
                published = None
            link = text('link') or (values['link'].get('href') if 'link' in values else '')
            findings.append({'source': self.name, 'url': link, 'published_at': published,
                             'collected_at': collected, 'snippet': snippet[:16_000],
                             'category': 'news', 'team': 'match', 'confirmation_status': 'reported',
                             'data': {'headline': text('title')[:500]}})
        return findings


class GdeltNewsSource:
    """Optional free discovery. seendate is NOT a publisher publication time."""
    name = 'gdelt'
    categories = ('injuries', 'lineup', 'news', 'schedule', 'h2h')

    def __init__(self, *, transport=None):
        self.transport = transport or ReadOnlyHTTP(['api.gdeltproject.org'])

    def fetch(self, match, category, *, as_of, timeout):
        home = str(match.get('home_search_name') or match.get('home', '')).replace('"', '')
        away = str(match.get('away_search_name') or match.get('away', '')).replace('"', '')
        terms = {'injuries': '(injury OR injured OR suspension)', 'lineup': '(lineup OR starting)',
                 'schedule': '(fixture OR schedule OR rest)', 'h2h': '"head to head"', 'news': ''}
        teams = f'"{home}" "{away}"' if category == 'h2h' else f'("{home}" OR "{away}")'
        params = {'query': f'{teams} {terms[category]}'.strip(), 'mode': 'artlist', 'format': 'json',
                  'maxrecords': 8, 'timespan': '3d', 'sort': 'datedesc'}
        url = 'https://api.gdeltproject.org/api/v2/doc/doc?' + urlencode(params)
        response = self.transport.get_json(url, timeout=timeout)
        return [{'source': self.name, 'url': row.get('url'), 'published_at': None,
                 'collected_at': _now(), 'snippet': row.get('title', ''),
                 'category': 'news', 'team': 'match', 'confirmation_status': 'reported',
                 'data': {'headline': row.get('title', '')}}
                for row in response.get('articles', [])[:8] if isinstance(row, dict)]


class OpenMeteoSource:
    """Forecast values only. Hourly valid time is never passed off as issue time."""
    name = 'open-meteo'
    categories = ('weather',)

    def __init__(self, *, transport=None):
        self.transport = transport or ReadOnlyHTTP(['api.open-meteo.com'])

    def fetch(self, match, category, *, as_of, timeout):
        lat, lon = match.get('venue_latitude'), match.get('venue_longitude')
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return []
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return []
        kickoff = timestamp(match.get('kickoff') or match.get('time') or match.get('match_time'))
        if kickoff is None or kickoff <= as_of:
            return []
        url = 'https://api.open-meteo.com/v1/forecast?' + urlencode({
            'latitude': lat, 'longitude': lon, 'hourly': 'temperature_2m,precipitation,wind_speed_10m',
            'timezone': 'UTC', 'start_date': kickoff.date().isoformat(), 'end_date': kickoff.date().isoformat()})
        response = self.transport.get_json(url, timeout=timeout)
        hourly = response.get('hourly') or {}
        target = kickoff.replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M')
        if target not in hourly.get('time', []):
            return []
        index = hourly['time'].index(target)
        data = {'forecast_for': target + ':00+00:00'}
        for source, dest in (('temperature_2m', 'temperature_celsius'), ('precipitation', 'precipitation_mm'),
                             ('wind_speed_10m', 'wind_kmh')):
            values = hourly.get(source) or []
            if index < len(values) and values[index] is not None:
                data[dest] = values[index]
        # This endpoint does not expose a forecast issue timestamp. Keep its
        # useful observation but do not promote it into verified model features.
        return [{'source': self.name, 'url': url, 'published_at': None, 'collected_at': _now(),
                 'category': 'weather', 'team': 'match', 'data': data, 'confirmation_status': 'reported'}]


class FootballDataSource:
    """Optional existing football-data.org provider, with explicit provider IDs.

    A local/竞彩 match ID is never assumed to be the vendor match ID. One fetch
    performs one HTTP request; create separate schedule adapters for each team.
    """
    def __init__(self, api_key, category='lineup', team='home', *, transport=None):
        self.name = f'football-data:{category}:{team}'
        self.categories = (category,)
        self.category, self.team, self.api_key = category, team, api_key
        self.transport = transport or ReadOnlyHTTP(['api.football-data.org'])

    def fetch(self, match, category, *, as_of, timeout):
        ids = match.get('provider_ids') or {}
        mid = ids.get('football_data')
        tid = ids.get(f'football_data_{self.team}')
        hid, aid = ids.get('football_data_home'), ids.get('football_data_away')
        if category == 'schedule':
            if not str(tid or '').isdigit():
                return []
            path = f'teams/{tid}/matches?status=FINISHED&limit=20&dateTo={as_of.date().isoformat()}'
        else:
            if not str(mid or '').isdigit():
                return []
            path = f'matches/{mid}' + ('/head2head?limit=20' if category == 'h2h' else '')
        url = 'https://api.football-data.org/v4/' + path
        response = self.transport.get_json(url, timeout=timeout, headers={'X-Auth-Token': self.api_key})
        collected = _now()

        def fact(team, data, published):
            return {'source': 'football-data.org', 'url': url, 'published_at': published,
                    'collected_at': collected, 'category': category, 'team': team,
                    'data': data, 'confirmation_status': 'confirmed'}

        if category == 'lineup':
            kickoff = timestamp(match.get('kickoff') or match.get('time') or match.get('match_time'))
            if (timestamp(response.get('utcDate')) != kickoff
                    or response.get('status') not in ('SCHEDULED', 'TIMED')):
                return []
            results = []
            for team in ('home', 'away'):
                players = (response.get(team + 'Team') or {}).get('lineup') or []
                if len(players) == 11:
                    results.append(fact(team, {'players': [p.get('name') for p in players]}, response.get('lastUpdated')))
            return results
        rows = [row for row in response.get('matches', []) if isinstance(row, dict)
                and row.get('status') == 'FINISHED' and timestamp(row.get('utcDate'))
                and timestamp(row['utcDate']) < as_of and timestamp(row.get('lastUpdated'))
                and timestamp(row['lastUpdated']) <= as_of]
        if category == 'schedule':
            rows = [row for row in rows if str(tid) in (str((row.get('homeTeam') or {}).get('id')),
                                                       str((row.get('awayTeam') or {}).get('id')))]
            if not rows:
                return []
            previous = max(rows, key=lambda row: timestamp(row['utcDate']))
            return [fact(self.team, {'previous_kickoff': previous['utcDate']}, previous['lastUpdated'])]
        if not hid or not aid:
            return []
        counts, goals, relevant = [0, 0, 0], 0, []
        for row in rows:
            rh, ra = str((row.get('homeTeam') or {}).get('id')), str((row.get('awayTeam') or {}).get('id'))
            if {rh, ra} != {str(hid), str(aid)}:
                continue
            score = (row.get('score') or {}).get('fullTime') or {}
            h, a = score.get('home'), score.get('away')
            if not isinstance(h, int) or not isinstance(a, int) or h < 0 or a < 0:
                continue
            home_score, away_score = (h, a) if rh == str(hid) else (a, h)
            counts[0 if home_score > away_score else 1 if home_score == away_score else 2] += 1
            goals += h + a
            relevant.append(row)
        if not relevant:
            return []
        return [fact('match', {'games': len(relevant), 'home_wins': counts[0], 'draws': counts[1],
                              'away_wins': counts[2], 'avg_goals': goals / len(relevant),
                              'most_recent_match_at': max(timestamp(r['utcDate']) for r in relevant).isoformat()},
                     max(timestamp(r['lastUpdated']) for r in relevant).isoformat())]
