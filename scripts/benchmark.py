import json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import PolicyRequest,evaluate_local
request=PolicyRequest(details={'score':0.8},conditions=[{'key':'score','analysis_type':'local','condition_type':'greater','threshold':0.5}])
samples=[]
for _ in range(10000):
    start=time.perf_counter_ns();assert evaluate_local(request)['allowed'];samples.append((time.perf_counter_ns()-start)/1000)
samples.sort();print(json.dumps({'workload':'local policy evaluation','iterations':10000,'p50Microseconds':samples[5000],'p95Microseconds':samples[9500],'productionQualification':False}))
