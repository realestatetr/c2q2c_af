#!/usr/bin/env python3
"""Compare unswapping candidate strategies: naive (none) vs exhaustive vs local-4-wire.
Reports success, max bond, candidates evaluated, wall time, over several seeds."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import c2q2c_peaked_mirror_attack as pm

def run(N, depth, Ks, seeds, chi=128):
    out = []
    for K in Ks:
        for sd in range(seeds):
            inst = pm.make_instance(N, depth, K, seed=5000 + 100 * K + sd)
            row = dict(N=N, K=K, seed=sd)
            for name, kw in (("naive", dict(greedy=False)),
                             ("exhaustive", dict(greedy=True, cands="all")),
                             ("local4", dict(greedy=True, cands="local4"))):
                row[name] = pm.attack_mirror(inst, chi, **kw)
            out.append(row)
    return out

if __name__ == "__main__":
    res = []
    for N in (16, 20):
        res += run(N, 6, (4, 12), 5)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "unswap_candidates.json", "w"), indent=1)
    print("N  K  method      success  median bond  median evals  median time(s)")
    for N in (16, 20):
        for K in (4, 12):
            R = [r for r in res if r["N"] == N and r["K"] == K]
            for m in ("naive", "exhaustive", "local4"):
                ok = sum(r[m]["success"] for r in R)
                print(f"{N:2d} {K:2d}  {m:<10}  {ok}/{len(R)}     {int(np.median([r[m]['max_bond'] for r in R])):4d}"
                      f"        {int(np.median([r[m].get('evals', 0) for r in R])):5d}       "
                      f"{np.median([r[m]['time'] for r in R]):6.1f}")
