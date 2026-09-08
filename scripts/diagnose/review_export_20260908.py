"""Read-only, reproducible review of a football prediction export."""
import collections
import json
import math
import sys
from pathlib import Path


def norm(p):
    if not p or not all(k in p for k in 'HDA'):
        return None
    s = sum(p[k] for k in 'HDA')
    return {k: p[k] / s for k in 'HDA'} if s > 0 else None


def metrics(pairs):
    n = len(pairs)
    if not n:
        return {'n': 0}
    hits = sum(max(p, key=p.get) == y for p, y in pairs)
    return dict(n=n, hits=hits, accuracy=hits/n,
                confidence=sum(max(p.values()) for p, y in pairs)/n,
                brier=sum(sum((p[k]-(y == k))**2 for k in 'HDA') for p,y in pairs)/n,
                logloss=-sum(math.log(max(p[y],1e-12)) for p,y in pairs)/n)


def main():
    data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    records = data['records']
    settled = [r for r in records if r.get('settled') and r.get('actual_score')]
    valid = [r for r in settled if norm(r.get('predicted_1x2')) and r.get('actual_result') in 'HDA']
    pairs = lambda rs: [(norm(r['predicted_1x2']),r['actual_result']) for r in rs]
    out = {'records':len(records),'unique_ids':len({r['match_id'] for r in records}), 'settled':len(settled), 'model':metrics(pairs(valid))}
    out['versions'] = {v:metrics(pairs([r for r in valid if (r.get('model_version') or 'legacy') == v])) for v in sorted({r.get('model_version') or 'legacy' for r in valid})}
    out['leagues'] = {v:metrics(pairs([r for r in valid if r.get('league') == v])) for v,n in collections.Counter(r.get('league') for r in valid).most_common() if n >= 15}
    out['confidence_bins'] = {f'{lo}-{hi}':metrics([(p,y) for p,y in pairs(valid) if lo<=max(p.values())<hi]) for lo,hi in [(0,.45),(.45,.55),(.55,.65),(.65,.75),(.75,1.01)]}
    out['thresholds'] = {str(t):metrics([(p,y) for p,y in pairs(valid) if max(p.values())>=t and sorted(p.values(),reverse=True)[0]-sorted(p.values(),reverse=True)[1]>=.1]) for t in [.55,.60,.65,.70,.75]}
    out['outcomes'] = {k:{'actual':sum(y==k for p,y in pairs(valid)), 'predicted':sum(max(p,key=p.get)==k for p,y in pairs(valid)), 'correct':sum(y==k and max(p,key=p.get)==k for p,y in pairs(valid)), 'mean_probability':sum(p[k] for p,y in pairs(valid))/len(valid)} for k in 'HDA'}
    out['score']={str(k):{'hits':sum(r['actual_score'] in sorted(r['predicted_scores'],key=r['predicted_scores'].get,reverse=True)[:k] for r in settled), 'expected':sum(sum(sorted(r['predicted_scores'].values(),reverse=True)[:k]) for r in settled)/len(settled)} for k in [1,3,5]}
    for source in ['odds_snapshot','last_prematch_odds_snapshot','closing_odds_snapshot']:
        matched=[]
        for r in valid:
            raw=((r.get(source) or {}).get('euro') or {}).get('close') or {}
            market=norm({a:raw[b] for a,b in [('H','home'),('D','draw'),('A','away')] if b in raw})
            if market: matched.append((r,market))
        out[source]={'model':metrics(pairs([r for r,m in matched])), 'market':metrics([(m,r['actual_result']) for r,m in matched])}
        disagree=[(r,m) for r,m in matched if max(m,key=m.get)!=max(r['predicted_1x2'],key=r['predicted_1x2'].get)]
        out[source]['disagreement']={'model':metrics(pairs([r for r,m in disagree])), 'market':metrics([(m,r['actual_result']) for r,m in disagree])}
    out['quality']={'missing_match_time':sum(not r.get('match_time') for r in settled), 'missing_result_quality':sum(not r.get('result_quality') for r in settled), 'result_grades':dict(collections.Counter((r.get('result_quality') or {}).get('grade','missing') for r in settled)), 'ml_available':sum(bool(r.get('ml_available')) for r in settled)}
    checks=collections.Counter(); evidence_n=0
    for r in settled:
        ev=(r.get('professional_snapshot') or {}).get('evidence') or {}
        if ev:
            evidence_n+=1
            checks.update(c['key'] for c in ev.get('checks',[]) if c.get('available'))
    out['evidence']={'n':evidence_n,'available':dict(checks)}
    out['paired_layers']={}
    for layer in ['T-24h','T-6h','T-1h','T-15min']:
        rs=[r for r in settled if (r.get('time_layers') or {}).get(layer)]
        out['paired_layers'][layer]={'n':len(rs), 'early_hits':sum(r['actual_score']==max(r['time_layers'][layer],key=r['time_layers'][layer].get) for r in rs), 'final_hits':sum(r['actual_score']==max(r['predicted_scores'],key=r['predicted_scores'].get) for r in rs)}
    timeline=[s for r in settled for s in r.get('market_timeline',[])]
    out['timeline']={'n':len(timeline),'prematch_true':sum(s.get('is_prematch') is True for s in timeline),'missing_seconds':sum(s.get('seconds_to_kickoff') is None for s in timeline)}
    out['goals']={}
    for name in ['matrix','distribution_dict']:
        rows=[]
        for r in settled:
            gc=r.get('goal_count') or {}
            if not gc.get(name): continue
            if name=='matrix':
                dist=collections.defaultdict(float)
                for i,row in enumerate(gc[name]):
                    for j,p in enumerate(row): dist[i+j]+=p
            else: dist={int(k):v for k,v in gc[name].items()}
            actual=sum(map(int,r['actual_score'].split('-')))
            rows.append((sum(k*p for k,p in dist.items())/sum(dist.values()),actual,max(dist,key=dist.get)==actual))
        if rows: out['goals'][name]={'n':len(rows),'expected_mean':sum(x for x,y,h in rows)/len(rows),'actual_mean':sum(y for x,y,h in rows)/len(rows),'mae':sum(abs(x-y) for x,y,h in rows)/len(rows),'exact_hits':sum(h for x,y,h in rows)}
    output=Path(sys.argv[2]); output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
