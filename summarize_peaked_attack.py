#!/usr/bin/env python3
"""Summarise peaked_attack_N*.json files: one row per (N, K, attack).

Usage: python summarize_peaked_attack.py out.md file1.json file2.json ...

Reported per row: number of seeds, success rate with a Wilson 95% interval, median and max bond
dimension reached, fraction of runs that hit the bond cap, median wall time, and whether every
instance passed the exact-statevector check (the generator asserts this before attacking).
"""
import json
import sys

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


out, files = sys.argv[1], sys.argv[2:]
lines = ["| N | K | attack | seeds | success (95% CI) | median bond | max bond | hit cap | median time (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
direct_lines = []
for f in files:
    D = json.load(open(f))
    N, cap = D["args"]["N"], D["args"]["chi_mirror"]
    rows = D["rows"]
    for K in sorted({r["K"] for r in rows}):
        R = [r for r in rows if r["K"] == K]
        for attack in ("naive", "greedy"):
            ok = sum(bool(r[attack]["success"]) for r in R)
            lo, hi = wilson(ok, len(R))
            bonds = [r[attack]["max_bond"] for r in R]
            capped = sum(bool(r[attack]["capped"]) for r in R)
            times = [r[attack]["time"] for r in R if r[attack]["time"] == r[attack]["time"]]
            lines.append(f"| {N} | {K} | {attack} | {len(R)} | {ok}/{len(R)} ({lo:.2f}-{hi:.2f}) | "
                         f"{int(np.median(bonds))} | {max(bonds)} | {capped}/{len(R)} | {np.median(times):.1f} |")
    exact = all(abs(r["delta_exact"] - r["delta_design"]) < 1e-9 for r in rows)
    chis = [d["chi_break"] for d in D["direct"]]
    direct_lines.append(f"- N={N}: {len(rows)} instances, bond cap {cap}, exact-statevector peak check passed for all "
                        f"instances (peak weight = design 0.100): {exact}. Direct MPS chi needed per seed: {chis} "
                        f"(ladder {D['direct'][0]['ladder']}; None = not reached).")
md = "\n".join(lines) + "\n\n" + "\n".join(direct_lines) + "\n"
open(out, "w").write(md)
print(md)
