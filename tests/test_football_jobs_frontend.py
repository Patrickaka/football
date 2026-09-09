"""Execute the actual football job loader with Node, fake fetch and a fake clock."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


HTML = Path('web/index.html').read_text(encoding='utf-8')


def between(start, end):
    return HTML[HTML.index(start):HTML.index(end, HTML.index(start))]


SOURCE = '\n'.join([
    "let allMatches=[], allPredictions=[], footballAnalysisProgress=null, footballAnalysisFailures=[];",
    "let footballAnalysisNotice='', footballLoadGeneration=0, footballLoadSession=null;",
    "let activeTab='football', footballSort='time', footballPage=1;",
    between('async function fetchJson(path, options = {})', 'async function fetchJsonWithTimeout'),
    between('async function loadMatches()', 'let footballProfessionalStatus'),
    between('function setFootballSort(sort)', 'function buildPredictQuery'),
    between('function buildBatchMatch(m)', 'function setFootballQualityFilter'),
])

HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
function fixture(handler) {
  let now=0, nextTimer=0;
  const timers=new Map(), requests=[], renders=[], notices=[];
  const context=vm.createContext({
    console, AbortController, DOMException, URLSearchParams,
    Date: class extends Date { static now() { return now; } },
    setTimeout(callback, ms) { const id=++nextTimer; timers.set(id,{at:now+ms,callback}); return id; },
    clearTimeout(id) { timers.delete(id); },
    apiPrefix() { return ''; },
    loading(text) { notices.push(text); },
    showFootballLoadError(text) { notices.push(text); },
    renderFootballProfessionalStatus() {}, loadFootballProfessionalStatus() {},
    renderFootballPredictions(rows) {
      context.renderedRows=rows;
      vm.runInContext('allPredictions=renderedRows',context);
      renders.push(JSON.parse(JSON.stringify(rows)));
    },
    fetch(path, options) {
      requests.push({path,options});
      return new Promise((resolve,reject) => {
        const aborted=()=>reject(new DOMException('aborted','AbortError'));
        options.signal?.addEventListener('abort',aborted,{once:true});
        Promise.resolve().then(()=>handler(path,options,requests.length)).then(value=>{
          options.signal?.removeEventListener('abort',aborted);
          const status=value.httpStatus || 200;
          resolve({ok:status>=200&&status<300,status,
                   text:async()=>value.body || JSON.stringify(value)});
        },reject);
      });
    },
  });
  vm.runInContext(SOURCE,context);
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  async function advance(ms) {
    const target=now+ms;
    await flush();
    while (true) {
      const next=[...timers].filter(([,item])=>item.at<=target)
        .sort((a,b)=>a[1].at-b[1].at || a[0]-b[0])[0];
      if (!next) break;
      now=next[1].at; timers.delete(next[0]); next[1].callback(); await flush();
    }
    now=target; await flush();
  }
  return {context,requests,renders,notices,advance,flush,
    run:code=>vm.runInContext(code,context),
    read:code=>JSON.parse(JSON.stringify(vm.runInContext(code,context))),
  };
}
const match=id=>({match_id:String(id),home:'H'+id,away:'A'+id});
const entry=(id,revision,status='completed')=>({match_id:String(id),index:999,revision,status,
  ...(status==='completed'?{result:{marker:String(id)}}:{error:'slow match'})});
const job=(status,revision,results=[],job_id='job-1')=>({success:true,result:{
  job_id,status,revision,results,total:40,completed:results.length,succeeded:results.length,failed:0}});
"""


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for executed frontend tests')
class FootballJobsFrontend(unittest.TestCase):
    def execute(self, scenario):
        script = 'const SOURCE=' + json.dumps(SOURCE) + ';\n' + HARNESS
        script += '\n(async()=>{\n' + scenario + '\n})().catch(error=>{console.error(error);process.exitCode=1;});'
        result = subprocess.run([shutil.which('node'), '-'], input=script, text=True,
                                encoding='utf-8', capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_forty_matches_incrementally_render_without_index_cross_wiring(self):
        self.execute(r"""
let polls=0;
const f=fixture((path,options)=>{
  if (path.endsWith('/start')) {
    assert.equal(JSON.parse(options.body).matches.length,40);
    assert.equal(JSON.parse(options.body).force_refresh,false);
    return job('queued',0);
  }
  polls++;
  if (polls===1) return job('running',1,[entry(40,1)]);
  if (polls===2) return job('running',2,[entry(2,2),entry(1,2)]);
  return job('completed',3,Array.from({length:37},(_,i)=>entry(i+3,3)));
});
f.context.matches=Array.from({length:40},(_,i)=>match(i+1));
f.run('allMatches=matches');
const work=f.run('loadAllPredictions()');
await f.advance(1400);
assert.deepEqual(f.renders.at(-1).map(row=>row.match.match_id),['40']);
assert.equal(f.read('footballAnalysisProgress.completed'),1);
await f.advance(2000); await work;
assert.equal(f.read('allPredictions.length'),40);
for (const row of f.read('allPredictions')) assert.equal(row.match.match_id,row.result.marker);
assert.equal(f.requests.filter(row=>row.path.endsWith('/start')).length,1);
assert.equal(f.requests.length,4);
assert.ok(f.requests[2].path.includes('after_revision=1'));
assert.equal(f.read('footballAnalysisProgress'),null);
""")

    def test_warmed_non_contiguous_slots_and_terminal_timeout_remain_independent(self):
        self.execute(r"""
const f=fixture((path,options)=>{
  if (path.endsWith('/start')) {
    assert.deepEqual(JSON.parse(options.body).matches.map(row=>row.match_id),['1','3']);
    return job('running',1,[entry(3,1)]);
  }
  return job('completed',2,[entry(1,2,'timed_out')]);
});
f.context.matches=[match(1),match(2),match(3)]; f.run('allMatches=matches');
const work=f.run("loadAllPredictions({ready:[{match_id:'2',result:{marker:'2'}}]})");
await f.advance(400);
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['2','3']);
await f.advance(1000); await work;
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['2','3']);
assert.equal(f.read('footballAnalysisFailures')[0].match.match_id,'1');
""")

    def test_new_generation_aborts_old_response_and_tab_cancel_preserves_results(self):
        self.execute(r"""
let release, starts=0;
const f=fixture(path=>{
  if (path.endsWith('/start') && ++starts===1) return new Promise(resolve=>release=resolve);
  return job('completed',1,[entry(2,1)],'new-job');
});
f.context.matches=[match(1)]; f.run('allMatches=matches');
const first=f.run('loadAllPredictions()'); await f.flush();
f.context.matches=[match(2)]; f.run('allMatches=matches');
const second=f.run('loadAllPredictions()'); await f.flush(); await second;
assert.equal(f.requests[0].options.signal.aborted,true);
release(job('completed',1,[entry(1,1)],'old-job')); await f.flush(); await first;
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['2']);
f.run("cancelFootballLoad(); activeTab='kl8'");
await f.advance(2000);
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['2']);
""")

    def test_504_retries_status_only_then_keeps_completed_matches(self):
        self.execute(r"""
const f=fixture(path=>path.endsWith('/start') ? job('running',1,[entry(1,1)])
  : {httpStatus:504,body:'<h1>Gateway Timeout</h1>'});
f.context.matches=[match(1),match(2)]; f.run('allMatches=matches');
const work=f.run('loadAllPredictions()'); await f.advance(5000); await work;
assert.equal(f.requests.length,4);
assert.ok(f.requests.slice(1).every(row=>row.path.includes('/batch/status?')));
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['1']);
assert.equal(f.read('footballAnalysisFailures.length'),0);
assert.ok(f.read('footballAnalysisNotice').includes('继续/刷新'));
assert.ok(!f.requests.some(row=>row.path.includes('/api/predict?')||row.path.includes('/bff/')));
""")

    def test_sort_is_local_and_404_requires_backend_upgrade(self):
        self.execute(r"""
const f=fixture(()=>({httpStatus:404,body:'missing'}));
f.context.matches=[match(1)]; f.run('allMatches=matches');
const work=f.run('loadAllPredictions()'); await f.flush(); await work;
assert.equal(f.requests.length,1);
assert.ok(f.read('footballAnalysisNotice').includes('升级并重启后端'));
f.run("setFootballSort('league')");
assert.equal(f.requests.length,1); assert.equal(f.read('footballSort'),'league');
""")

    def test_expired_job_404_preserves_error_metadata_and_does_not_request_upgrade(self):
        self.execute(r"""
const gone={httpStatus:404,body:JSON.stringify({code:'job_not_found',error:'任务已过期或服务已重启'})};
const metadata=fixture(()=>gone);
await assert.rejects(metadata.run("fetchJson('/api/predict/batch/status?job_id=gone')"),error=>{
  assert.equal(error.httpStatus,404); assert.equal(error.code,'job_not_found');
  assert.ok(error.message.includes('任务已过期')); return true;
});
const f=fixture(path=>path.endsWith('/start') ? job('running',1,[entry(1,1)]) : gone);
f.context.matches=[match(1),match(2)]; f.run('allMatches=matches');
const work=f.run('loadAllPredictions()'); await f.advance(5000); await work;
assert.equal(f.requests.length,2);
assert.ok(f.requests[1].path.includes('/batch/status?'));
assert.ok(f.read('footballAnalysisNotice').includes('任务已过期或服务已重启'));
assert.ok(f.read('footballAnalysisNotice').includes('刷新'));
assert.ok(!f.read('footballAnalysisNotice').includes('升级并重启后端'));
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['1']);
""")

    def test_partial_refresh_and_network_failure_preserve_all_unchanged_previous_results(self):
        self.execute(r"""
const f=fixture(path=>path.endsWith('/start') ? job('running',1,[entry(1,1)])
  : {httpStatus:504,body:'<h1>Gateway Timeout</h1>'});
f.context.matches=Array.from({length:40},(_,i)=>match(i+1));
f.context.previous=f.context.matches.map(value=>({match:value,result:{marker:'old-'+value.match_id}}));
f.run('allMatches=matches; allPredictions=previous');
const work=f.run('loadAllPredictions()'); await f.advance(400);
assert.equal(f.read('allPredictions.length'),40);
assert.equal(f.read('allPredictions')[0].result.marker,'1');
assert.equal(f.read('allPredictions')[39].result.marker,'old-40');
assert.equal(f.read('footballAnalysisProgress.completed'),1);
const submitted=JSON.parse(f.requests[0].options.body).matches;
assert.equal(submitted.length,40); // Display fallbacks cannot count as new completions.
await f.advance(5000); await work;
assert.equal(f.read('allPredictions.length'),40);
assert.equal(f.read('allPredictions')[0].result.marker,'1');
assert.equal(f.read('allPredictions')[39].result.marker,'old-40');
assert.equal(f.read('footballAnalysisProgress'),null);
assert.ok(f.read('footballAnalysisNotice').includes('已保留完成结果'));
""")

    def test_previous_display_fallback_requires_the_entire_match_context_to_agree(self):
        self.execute(r"""
const f=fixture(path=>path.endsWith('/start') ? job('running',1)
  : {httpStatus:504,body:'<h1>Gateway Timeout</h1>'});
const original=Array.from({length:5},(_,i)=>({...match(i+1),time:'09-12 20:00',
  lottery_handicap:-1,lottery_spf_odds:{home:1.8,draw:3,away:4}}));
f.context.previous=original.map(value=>({match:value,result:{marker:'old-'+value.match_id}}));
const current=JSON.parse(JSON.stringify(original));
current[1].time='09-13 20:00';
current[2].lottery_handicap=-2;
current[3].lottery_spf_odds.home=2.1;
// The same full JSON object with a different key order remains eligible.
current[4]=Object.fromEntries(Object.entries(current[4]).reverse());
current[4].lottery_spf_odds=Object.fromEntries(Object.entries(current[4].lottery_spf_odds).reverse());
f.context.matches=current; f.run('allMatches=matches; allPredictions=previous');
const work=f.run('loadAllPredictions()'); await f.advance(400);
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['1','5']);
assert.equal(f.read('footballAnalysisProgress.completed'),0);
assert.equal(JSON.parse(f.requests[0].options.body).matches.length,5);
await f.advance(5000); await work;
assert.deepEqual(f.read('allPredictions').map(row=>row.match.match_id),['1','5']);
assert.deepEqual(f.read('allPredictions').map(row=>row.result.marker),['old-1','old-5']);
""")

    def test_initial_bff_has_30_second_timeout_and_preserves_previous_success(self):
        self.execute(r"""
const f=fixture(()=>new Promise(()=>{}));
f.context.matches=[match(1)]; f.run("allMatches=matches; allPredictions=[{match:matches[0],result:{marker:'old'}}]");
const work=f.run('loadMatches()');
await f.advance(29999); assert.equal(f.requests[0].options.signal.aborted,false);
await f.advance(1); await work;
assert.equal(f.requests[0].options.signal.aborted,true);
assert.equal(f.read('allPredictions')[0].result.marker,'old');
assert.ok(f.read('footballAnalysisNotice').includes('已保留完成结果'));
""")

    def test_job_requests_timeout_in_ten_seconds_without_long_prediction_fallback(self):
        self.execute(r"""
const f=fixture(()=>new Promise(()=>{}));
f.context.matches=[match(1)]; f.run('allMatches=matches');
const work=f.run('loadAllPredictions()');
await f.advance(10000); assert.equal(f.requests[0].options.signal.aborted,true);
await f.advance(23000); await work;
assert.equal(f.requests.length,3);
assert.ok(f.requests.every(row=>row.path.endsWith('/batch/start')));
assert.ok(f.read('footballAnalysisNotice').includes('10秒'));
""")

    def test_twenty_minute_wait_limit_and_more_than_eighty_match_partition(self):
        self.execute(r"""
const endless=fixture(()=>job('running',1));
endless.context.matches=[match(1)]; endless.run('allMatches=matches');
const waiting=endless.run('loadAllPredictions()');
await endless.advance(20*60*1000); await waiting;
assert.ok(endless.read('footballAnalysisNotice').includes('20分钟'));
assert.equal(endless.read('footballAnalysisFailures.length'),0);
const sizes=[];
const f=fixture((path,options)=>{
  const submitted=JSON.parse(options.body).matches;
  sizes.push(submitted.length);
  return job('completed',1,submitted.map(row=>entry(row.match_id,1)),'job-'+sizes.length);
});
f.context.matches=Array.from({length:85},(_,i)=>match(i+1)); f.run('allMatches=matches');
await f.run('loadAllPredictions()');
assert.deepEqual(sizes,[80,5]); assert.equal(f.read('allPredictions.length'),85);
""")

    def test_switching_modules_explicitly_cancels_only_browser_wait(self):
        switch = between('function switchTab(tab)', "document.querySelectorAll('[data-tab]').forEach")
        self.assertIn("if (tab !== 'football') cancelFootballLoad();", switch)
        self.assertNotIn('/cancel', SOURCE)
