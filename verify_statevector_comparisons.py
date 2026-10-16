#!/usr/bin/env python3
"""Exact-statevector check of the instances used by compare_unswap_candidates.py (N in {16,20}, K in {4,12}, 5 seeds,
instance seed 5000 + 100*K + s) and stress_local4.py (N=16, K in {24,48}, 4 rows, instance seed 9000 + 100*K + s).
Those two scripts do not check their instances themselves; the generator is the same one that is checked in the sweeps.

Usage: python verify_statevector_comparisons.py [OUT.json]    (about 1-2 minutes; N=20 needs ~16 MB per statevector)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2q2c_peaked_mirror_attack as pm


def check(source, index, N, K, seed):
    inst = pm.make_instance(N, 6, K, seed=seed)
    peak, dmax = pm.statevector_peak(inst)
    return dict(source=source, index=index, N=N, K=K, instance_seed=seed,
                peak_ok=bool(np.array_equal(peak, inst["s_phys"])), delta_exact=float(dmax), delta_design=float(inst["delta_design"]))


rows = []
for N in (16, 20):
    for K in (4, 12):
        for sd in range(5):
            rows.append(check("compare_unswap_candidates", sd, N, K, 5000 + 100 * K + sd))
for K in (24, 48):
    for sd in range(4):
        rows.append(check("stress_local4", sd, 16, K, 9000 + 100 * K + sd))
ok = all(r["peak_ok"] and abs(r["delta_exact"] - r["delta_design"]) < 1e-9 for r in rows)
out = sys.argv[1] if len(sys.argv) > 1 else "statevector_check_comparisons.json"
json.dump(rows, open(out, "w"), indent=1)
print(f"{sum(r['peak_ok'] for r in rows)}/{len(rows)} instances: designed peak == exact peak; "
      f"max |delta_exact - delta_design| = {max(abs(r['delta_exact'] - r['delta_design']) for r in rows):.1e}; "
      f"{'PASS' if ok else 'FAIL'} -> {out}")
sys.exit(0 if ok else 1)
