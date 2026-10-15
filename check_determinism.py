#!/usr/bin/env python3
"""Compare a re-run of the peaked-attack sweep with the stored reference JSON (all fields except wall time).

Usage: python check_determinism.py reference.json rerun1.json [rerun2.json ...]
Rows are matched on (K, seed). Exit code 1 on any mismatch.
"""
import json, sys

FIELDS = ("success", "max_bond", "capped", "frac_correct", "unswaps")
ref = json.load(open(sys.argv[1]))
rr = {(r["K"], r["seed"]): r for r in ref["rows"]}
n_ok = n_bad = 0
for f in sys.argv[2:]:
    J = json.load(open(f))
    for r in J["rows"]:
        k = (r["K"], r["seed"]); q = rr.get(k)
        if q is None:
            print("no reference row for", k); n_bad += 1; continue
        same = abs(r["delta_exact"] - q["delta_exact"]) < 1e-12
        for att in ("naive", "greedy"):
            for fld in FIELDS:
                if fld in q[att] and r[att].get(fld) != q[att][fld]:
                    same = False; print("MISMATCH", k, att, fld, q[att][fld], "->", r[att].get(fld))
        n_ok += same; n_bad += (not same)
    if J.get("direct") and ref.get("direct"):
        a = [d["chi_break"] for d in J["direct"]]; b = [d["chi_break"] for d in ref["direct"]]
        print("direct chi_break rerun", a, "reference", b, "OK" if a == b else "MISMATCH")
        n_bad += (a != b)
print(f"{n_ok} rows identical (success, max_bond, capped, frac_correct, unswaps, peak weight), {n_bad} mismatches")
sys.exit(1 if n_bad else 0)
