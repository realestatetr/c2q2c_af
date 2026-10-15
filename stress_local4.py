import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2q2c_peaked_mirror_attack as pm
# usage: python stress_local4.py [output.json]
OUT = sys.argv[1] if len(sys.argv) > 1 else 'local4_stress.json'
res = []
N, depth = 16, 6
for K in (24, 48):
    for sd in range(4):
        inst = pm.make_instance(N, depth, K, seed=9000 + 100 * K + sd)
        row = dict(N=N, K=K, seed=sd)
        for name, kw in (("local4", dict(greedy=True, cands="local4")),
                         ("exhaustive", dict(greedy=True, cands="all")),
                         ("naive", dict(greedy=False))):
            try:
                row[name] = pm.attack_mirror(inst, 128, **kw)
            except Exception as e:
                row[name] = dict(success=False, max_bond=128, capped=True, time=float('nan'), evals=0, error=type(e).__name__)
            r = row[name]
            print(f"K={K} seed={sd} {name:<10} ok={r['success']!s:5} bond={r['max_bond']:4d} evals={r.get('evals',0):5d} t={r['time']:.1f}s", flush=True)
        res.append(row)
        json.dump(res, open(OUT, 'w'), indent=1)
print("DONE", flush=True)
