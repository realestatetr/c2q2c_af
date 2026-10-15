#!/usr/bin/env python3
"""Resume helper for stress_local4.py: runs only the (K, seed) rows that are missing from DONE.json, using the same
instances (N=16, 6 layers, seed 9000 + 100*K + seed) and the same three attackers, and writes them to OUT.json.

Usage: python stress_local4_resume.py DONE.json OUT.json

DONE.json : rows that already exist; these (K, seed) pairs are skipped and not recomputed.
OUT.json  : contains ONLY the rows produced by this run, not a merged data set. Combine it with DONE.json yourself if you need
            the complete result, and do not publish OUT.json as if it were the full experiment.
`seed` in every row is the index 0-3 inside a K setting; the instance is generated with seed 9000 + 100*K + seed.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2q2c_peaked_mirror_attack as pm

if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help") else 2)
done_rows = json.load(open(sys.argv[1]))
if any(("K" not in r or "seed" not in r) for r in done_rows):
    sys.exit("DONE.json: a row lacks K or seed")
pairs = [(r["K"], r["seed"]) for r in done_rows]
if len(pairs) != len(set(pairs)):
    sys.exit("DONE.json: duplicate (K, seed) rows")
SCOPE = {(K, sd) for K in (24, 48) for sd in range(4)}
if set(pairs) - SCOPE:
    sys.exit(f"DONE.json: rows outside the experiment grid (K in 24,48; seed 0-3): {sorted(set(pairs) - SCOPE)}")
done = set(pairs)
expected = len(SCOPE - done)  # rows this run must produce
res = []
for K in (24, 48):
    for sd in range(4):
        if (K, sd) in done:
            continue
        inst = pm.make_instance(16, 6, K, seed=9000 + 100 * K + sd)
        row = dict(N=16, K=K, seed=sd)
        for name, kw in (("local4", dict(greedy=True, cands="local4")), ("exhaustive", dict(greedy=True, cands="all")),
                         ("naive", dict(greedy=False))):
            row[name] = pm.attack_mirror(inst, 128, **kw)
        res.append(row)
        json.dump(res, open(sys.argv[2] + ".tmp", "w"))
        os.replace(sys.argv[2] + ".tmp", sys.argv[2])  # atomic: an interrupted run never leaves a half-written OUT.json
        print(f"done K={K} seed={sd}", flush=True)
if len(res) != expected:
    sys.exit(f"ERROR: wrote {len(res)} rows; expected {expected}")
print(f"PASS: wrote {len(res)}/{expected} rows to {sys.argv[2]}; OUT.json holds only these rows")
