"""Wait at most 180 seconds for ComfyUI without submitting GPU work."""
import argparse, json, time, urllib.request
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8188')
p.add_argument('--timeout',type=float,default=180)
a=p.parse_args()
if a.timeout <= 0: p.error('timeout must be positive')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline=time.monotonic()+a.timeout
while time.monotonic()<deadline:
    try:
        stats=json.load(opener.open(a.url+'/system_stats',timeout=3))
        assert any(d.get('type')=='xpu' for d in stats.get('devices',[])), 'No XPU in system_stats'
        queue=json.load(opener.open(a.url+'/queue',timeout=3))
        print(json.dumps({'system_stats':stats,'queue':queue},indent=2))
        if queue.get('queue_running') or queue.get('queue_pending'):
            raise SystemExit('Service ready but queue busy. Do not submit another job.')
        print('ComfyUI ready; XPU visible and queue empty.')
        break
    except (OSError,ValueError,AssertionError) as e:
        print('Waiting:',e,flush=True)
        time.sleep(min(3,max(0,deadline-time.monotonic())))
else: raise SystemExit('Readiness timeout; inspect container and GPU logs.')
