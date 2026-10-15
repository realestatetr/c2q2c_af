#!/usr/bin/env python3
"""
Recompute every number quoted in the application text directly from the JSON artifacts and compare it
with the quoted value. Exit code 1 if any claim fails. Run from the directory holding the JSON files:

    python verify_claims.py [directory]

Each line: [PASS/FAIL] claim | computed | quoted | source file
"""
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
load = lambda f: json.load(open(os.path.join(D, f)))
results = []


def check(claim, computed, ok, quoted, src):
    results.append((bool(ok), claim, computed, quoted, src))


def med(x):
    return float(np.median(x))


# ---------------------------------------------------------------- peaked attack family (N = 16, 20, 24)
rows = {}
direct = {}
for n in (16, 20, 24):
    j = load(f"peaked_attack_N{n}.json")
    rows[n], direct[n] = j["rows"], j["direct"]
allrows = [r for n in rows for r in rows[n]]
src = "peaked_attack_N16/20/24.json"
check("instances checked against exact statevector", len(allrows), len(allrows) == 87, "87", src)
check("peak weight equals design (|delta_exact - 0.1| < 1e-9) for all", max(abs(r["delta_exact"] - 0.1) for r in allrows),
      all(abs(r["delta_exact"] - 0.1) < 1e-9 for r in allrows), "0.100 in every instance", src)
exp_naive12 = {16: (2, 5), 20: (1, 6), 24: (0, 4)}
for n, (k, t) in exp_naive12.items():
    R = [r for r in rows[n] if r["K"] == 12]
    got = (sum(r["naive"]["success"] for r in R), len(R))
    check(f"naive mirror attack success at K=12, N={n}", f"{got[0]}/{got[1]}", got == (k, t), f"{k}/{t}", src)
check("greedy unswapping success over all instances", f"{sum(r['greedy']['success'] for r in allrows)}/{len(allrows)}",
      all(r["greedy"]["success"] for r in allrows), "100%", src)
check("greedy max bond over all instances", max(r["greedy"]["max_bond"] for r in allrows),
      all(r["greedy"]["max_bond"] == 8 for r in allrows), "8 for all tested K", src)
check("naive median max bond at K=0 (all N)", med([r["naive"]["max_bond"] for r in allrows if r["K"] == 0]),
      med([r["naive"]["max_bond"] for r in allrows if r["K"] == 0]) == 8, "8", src)
check("naive median max bond at K=4, N=16 (bond cap = 128)", med([r["naive"]["max_bond"] for r in rows[16] if r["K"] == 4]),
      med([r["naive"]["max_bond"] for r in rows[16] if r["K"] == 4]) == 128, "reaches the cap by K~4", src)
exp_direct = {16: [16, 16], 20: [4, 32, 64], 24: [32, 64]}
for n, e in exp_direct.items():
    got = sorted(d["chi_break"] for d in direct[n])
    check(f"direct MPS chi needed per seed, N={n}", got, got == e, str(e), src)

# ---------------------------------------------------------------- unswapping candidate comparison
a = load("unswap_candidates.json")
b = load("local4_stress.json")
rows_u = a + b
src = "unswap_candidates.json + local4_stress.json"
for m, e in (("naive", 14), ("exhaustive", 28), ("local4", 28)):
    got = sum(r[m]["success"] for r in rows_u)
    check(f"{m} successes over all {len(rows_u)} comparison instances", f"{got}/{len(rows_u)}", got == e and len(rows_u) == 28, f"{e}/28", src)
hi = [r for r in rows_u if r["K"] >= 12]
for m, e in (("naive", 4), ("exhaustive", 18), ("local4", 18)):
    got = sum(r[m]["success"] for r in hi)
    check(f"{m} successes for K>=12", f"{got}/{len(hi)}", got == e and len(hi) == 18, f"{e}/18", src)
check("stress K=24,48: local4 / exhaustive / naive successes",
      (sum(r["local4"]["success"] for r in b), sum(r["exhaustive"]["success"] for r in b), sum(r["naive"]["success"] for r in b)),
      (sum(r["local4"]["success"] for r in b), sum(r["exhaustive"]["success"] for r in b), sum(r["naive"]["success"] for r in b)) == (8, 8, 1),
      "8/8, 8/8, 1/8", src)
check("stress median candidates: local4 / exhaustive", (med([r["local4"]["evals"] for r in b]), med([r["exhaustive"]["evals"] for r in b])),
      (med([r["local4"]["evals"] for r in b]), med([r["exhaustive"]["evals"] for r in b])) == (19.0, 2280.0), "19 vs 2280", src)
for n, k, e in ((16, 12, (10, 1200)), (20, 12, (12, 2280))):
    R = [r for r in a if r["N"] == n and r["K"] == k]
    got = (med([r["local4"]["evals"] for r in R]), med([r["exhaustive"]["evals"] for r in R]))
    check(f"median candidates local4 vs exhaustive, N={n}, K={k}", got, got == e, f"{e[0]} vs {e[1]}", src)

# ---------------------------------------------------------------- patch re-synthesis feasibility
pj = load("patch_resynthesis.json")
R = pj["results"]
src = "patch_resynthesis.json"
nontriv = [r for r in R if r["n_gates"] == 3 and not r["trivial"] and r["genuine"] == r["targets"]]
check("non-trivial genuine 3-gate patterns", [r["pattern"] for r in nontriv],
      sorted(map(str, [r["pattern"] for r in nontriv])) == sorted(map(str, [[[0, 1], [0, 2], [0, 1]], [[1, 2], [0, 2], [1, 2]]])),
      "(0,1)(0,2)(0,1) and (1,2)(0,2)(1,2)", src)
check("success on random targets for both patterns", [f"{r['success']}/{r['targets']}" for r in nontriv],
      all(r["success"] == 8 and r["targets"] == 8 for r in nontriv), "8/8 each", src)
check("max operator-norm distance to target", max(r["max_op_dist"] for r in nontriv),
      max(r["max_op_dist"] for r in nontriv) < 1e-9, "< 1e-9 (reported ~4e-10)", src)
ent = [p["min_entangling"] for r in nontriv for p in r["per_target"]]
check("min normalised entangling strength over all gates of all solutions", (round(min(ent), 2), round(max(ent), 2)),
      min(ent) > 0.05 and round(min(ent), 2) == 0.44 and round(max(ent), 2) == 0.97, "0.44 - 0.97 (threshold 0.05)", src)
two = [r for r in R if r["n_gates"] == 2 and not r["trivial"]]
check("2-gate patterns on other wires: successes", sum(r["success"] for r in two), all(r["success"] == 0 for r in two), "0/8 each (8 patterns)", src)

# ---------------------------------------------------------------- patch attack experiment
A = load("patch_attack_K0.json")
B = load("patch_attack_Kswap.json")
src = "patch_attack_K0.json / patch_attack_Kswap.json"
Ps = sorted({r["P"] for r in A})
check("patches actually used per requested level", sorted({(r["P_req"], r["P"]) for r in A}), Ps == [0, 2, 4, 6], "P in {0,2,4,6}", src)
for P in Ps:
    Rr = [r for r in A if r["P"] == P]
    ob = med([r["oracle"]["max_bond"] for r in Rr])
    check(f"oracle-aligned median max bond, P={P}", ob, ob == (8 if P == 0 else 16), "8 (P=0) / 16 (P>0)", src)
    if P:
        nb = med([r["naive"]["max_bond"] for r in Rr])
        check(f"naive index pairing median max bond, P={P}", nb, nb == 128, "128 (cap)", src)
ev = [med([r["search_full"]["evals"] for r in A if r["P"] == P]) for P in Ps]
check("search (full move set) median candidates for P=0,2,4,6", ev, ev == [48.0, 58.0, 68.0, 78.0], "48, 58, 68, 78 (+5 per patch)", src)
evl = [med([r["search_limited"]["evals"] for r in A if r["P"] == P]) for P in Ps]
check("search (small move set) median candidates for P=0,2,4,6", evl, evl == [48.0, 56.0, 64.0, 72.0], "48, 56, 64, 72", src)
for k in ("oracle", "search_full", "search_limited", "naive"):
    got = sum(r[k]["success"] for r in A)
    check(f"{k}: peak recovered in all K=0 patch runs", f"{got}/{len(A)}", got == len(A), "success is not discriminating", src)
B4 = [r for r in B if r["P"] == 4]
check("K=4 swaps + 4 patches, oracle alignment, local4 success", f"{sum(r['oracle_local4']['success'] for r in B4)}/{len(B4)}",
      all(r["oracle_local4"]["success"] for r in B4) and len(B4) == 5, "5/5", src)
check("same, local4 median bond and candidate range",
      (med([r["oracle_local4"]["max_bond"] for r in B4]), min(r["oracle_local4"]["evals"] for r in B4), max(r["oracle_local4"]["evals"] for r in B4)),
      med([r["oracle_local4"]["max_bond"] for r in B4]) == 16 and 4 <= min(r["oracle_local4"]["evals"] for r in B4)
      and max(r["oracle_local4"]["evals"] for r in B4) <= 6, "bond 16; 4-6 candidates", src)
check("same, exhaustive median candidates", med([r["oracle_all"]["evals"] for r in B4]), med([r["oracle_all"]["evals"] for r in B4]) == 480, "480", src)

# ---------------------------------------------------------------- single-qubit benchmark and VQE harness
sq = load("single_qubit_report.json")
src = "single_qubit_report.json"
hmin = min(min(sq["equality"][e]["pvalue_holm"]) for e in ("angle", "c2q2c"))
check("Holm-adjusted p-values (22 tests)", round(hmin, 3), hmin >= 0.05, ">= 0.05", src)
neg = sq["negative_control_pvalues"]
check("negative control p-values (adversary off by +0.005)", [f"{x:.1e}" for x in neg], max(neg) <= 2.1e-8, "<= 2e-8", src)
vq = load("report.json")
src = "report.json"
rates = {c["name"]: c["pass_rate"] for c in vq["conditions"]}
check("VQE pass rate: small init exact / shots", (rates["exact | init=0.3"], rates["shots | init=0.3"]),
      rates["exact | init=0.3"] == 1.0 and rates["shots | init=0.3"] == 1.0, "20/20 and 20/20", src)
check("VQE pass rate: init=pi exact / shots", (rates["exact | init=pi (original)"], rates["shots | init=pi (original)"]),
      rates["exact | init=pi (original)"] == 0.15 and rates["shots | init=pi (original)"] == 0.10, "15% and 10%", src)


# ---------------------------------------------------------------- gap-patch experiments (swaps + non-adjacent patches)
PIN = {"K0/oracle": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 0, "evals_min": 0, "evals_max": 0, "unswaps_med": 0, "wire_max": 0}, "K0/oracle_local4": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 0, "evals_min": 0, "evals_max": 0, "unswaps_med": 0, "wire_max": 0}, "K0/search_commute": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 57, "evals_min": 48, "evals_max": 62, "unswaps_med": 0, "wire_max": 0, "aligned_strict": 8}, "K0/commute_local4": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 210, "evals_min": 148, "evals_max": 221, "unswaps_med": 4, "wire_max": 4, "aligned_strict": 0}, "K4/oracle": {"n": 8, "ok": 8, "peak": 7, "identity": 0, "forced0": 8, "bond_med": 64, "bond_max": 64, "evals_med": 0, "evals_min": 0, "evals_max": 0, "unswaps_med": 0, "wire_max": 0}, "K4/oracle_local4": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 5, "evals_min": 2, "evals_max": 8, "unswaps_med": 4, "wire_max": 3}, "K4/search_commute": {"n": 8, "ok": 8, "peak": 5, "identity": 0, "forced0": 0, "bond_med": 64, "bond_max": 64, "evals_med": 183, "evals_min": 112, "evals_max": 300, "unswaps_med": 0, "wire_max": 0, "aligned_strict": 0}, "K4/commute_local4": {"n": 8, "ok": 8, "peak": 8, "identity": 8, "forced0": 8, "bond_med": 16, "bond_max": 16, "evals_med": 151, "evals_min": 100, "evals_max": 239, "unswaps_med": 8, "wire_max": 4, "aligned_strict": 0}, "K12/oracle_local4": {"n": 4, "ok": 4, "peak": 4, "identity": 4, "forced0": 4, "bond_med": 16, "bond_max": 16, "evals_med": 11, "evals_min": 8, "evals_max": 12, "unswaps_med": 10, "wire_max": 3}, "K12/commute_local4": {"n": 4, "ok": 4, "peak": 4, "identity": 4, "forced0": 4, "bond_med": 16, "bond_max": 16, "evals_med": 176, "evals_min": 142, "evals_max": 189, "unswaps_med": 13, "wire_max": 4, "aligned_strict": 0}}
def _gstats(K, att):
    R = load(f"gap_patch_unswap_K{K}.json"); ok = [r for r in R if r[att].get("status", "OK") == "OK"]
    g = lambda f: [r[att].get(f, 0) for r in R]
    d = dict(n=len(R), ok=len(ok), peak=sum(bool(r[att].get("peak_verified", r[att]["success"])) for r in ok),
             identity=sum(r[att]["final_profile_excess"] <= 0.5 for r in ok), forced0=sum(r[att].get("forced", 0) == 0 for r in ok),
             bond_med=int(med(g("max_bond"))), bond_max=int(max(g("max_bond"))), evals_med=int(med(g("evals"))),
             evals_min=int(min(g("evals"))), evals_max=int(max(g("evals"))), unswaps_med=int(med(g("unswaps"))),
             wire_max=int(max(g("max_wire_set"))))
    if "alignment_correct" in R[0][att]:
        d["aligned_strict"] = sum(bool(r[att]["alignment_correct"]) for r in R)
    return d
for key, pinned in PIN.items():
    K, att = key.split("/"); K = int(K[1:])
    got = _gstats(K, att)
    check(f"gap experiment K={K}, {att}: peak / operator collapsed / forced=0 / timeouts / bond med,max / evals min,med,max / unswaps / max wire set",
          (got["peak"], got["identity"], got["forced0"], got["n"] - got["ok"], got["bond_med"], got["bond_max"], got["evals_min"],
           got["evals_med"], got["evals_max"], got["unswaps_med"], got["wire_max"]),
          got == pinned, "pinned from the stored run", f"gap_patch_unswap_K{K}.json")
for K, n in ((0, 8), (4, 8), (12, 4)):
    R = load(f"gap_patch_unswap_K{K}.json"); P0 = R[0]["params"]
    check(f"K={K} file: instance ids, K, patches, gaps, patch error, statevector check",
          ([r["seed"] for r in R], sorted({r["K"] for r in R}), sorted({r["P"] for r in R}), min(min(r["gaps"]) for r in R), max(r["patch_err_max"] for r in R)),
          [r["seed"] for r in R] == list(range(61000, 61000 + n)) and {r["K"] for r in R} == {K} and {r["P"] for r in R} == {2}
          and all(min(r["gaps"]) >= 1 for r in R) and all(r["patch_err_max"] < 1e-11 for r in R) and all(r["statevector_peak_ok"] for r in R),
          f"seeds 61000..{60999 + n}, K={K}, P=2, every gap >= 1, patch error < 1e-11", f"gap_patch_unswap_K{K}.json")
    check(f"K={K} file: recorded search settings and one script hash for all rows",
          (P0["chi"], P0["eval_limit"], P0["commute_max_seeds"], P0["commute_max_total_group_size"], P0["cutoff"], P0["halves_boundary_known"]),
          all(r["params"]["script_sha256"] == P0["script_sha256"] for r in R) and (P0["chi"], P0["eval_limit"], P0["commute_max_seeds"],
          P0["commute_max_total_group_size"], P0["cutoff"], P0["halves_boundary_known"]) == (64, 2000, 8, 10, 1e-10, True),
          "chi 64, eval_limit 2000, max_seeds 8, max_total 10, cutoff 1e-10, halves boundary known", f"gap_patch_unswap_K{K}.json")
L = load("gap_patch_K0.json")
src = "gap_patch_K0.json (legacy: direct calls, wall-time limit 90 s; 3 of 8 rows re-run through the CLI)"
check("legacy gap file: oracle bond / peak", (sorted({r["oracle"]["max_bond"] for r in L}), sum(r["oracle"]["success"] for r in L)),
      sorted({r["oracle"]["max_bond"] for r in L}) == [16] and all(r["oracle"]["success"] for r in L), "16; 8/8", src)
for k, e_al, e_to in (("search_small", 0, 0), ("search_le3", 0, 1), ("search_wide", 4, 3)):
    got = (sum(bool(r[k]["alignment_correct"]) for r in L), sum(r[k]["status"] == "TIMEOUT" for r in L))
    check(f"legacy gap file: {k} correct alignment / timeouts of 8", got, got == (e_al, e_to), f"{e_al}/8 aligned, {e_to} timeouts", src)


# ---------------------------------------------------------------- derived statements used in the application text
src = "unswap_candidates.json, local4_stress.json"
ua, ust = load("unswap_candidates.json"), load("local4_stress.json")
ratios = []
for n in (16, 20):
    for k in (4, 12):
        R = [r for r in ua if r["N"] == n and r["K"] == k]
        ratios.append(med([r["exhaustive"]["evals"] for r in R]) / med([r["local4"]["evals"] for r in R]))
ratios.append(med([r["exhaustive"]["evals"] for r in ust]) / med([r["local4"]["evals"] for r in ust]))
check("median candidates exhaustive / local4 over the 4 comparison cells and the stress test (min, max)",
      (round(min(ratios), 1), round(max(ratios), 1)), min(ratios) > 100, "more than 100x fewer", src)
K0 = load("gap_patch_unswap_K0.json")
t = max(r["search_commute"]["time"] for r in K0)
check("search_commute, K=0: slowest of 8 instances (stored run, seconds)", round(t, 2), t < 1.0, "well under one second", "gap_patch_unswap_K0.json")
from collections import Counter
cnt = []
for n in (16, 20, 24):
    cnt += list(Counter(r["K"] for r in rows[n]).values())
cnt += list(Counter((r["N"], r["K"]) for r in ua).values()) + list(Counter(r["K"] for r in ust).values())
cnt += [len(load(f"gap_patch_unswap_K{K}.json")) for K in (0, 4, 12)] + [pj["config"]["targets"]]
check("instances per setting (peaked sweeps, comparisons, stress, gap files, re-synthesis targets): min, max", (min(cnt), max(cnt)),
      min(cnt) >= 4 and max(cnt) <= 8, "4-8 per setting", "all JSON files")

# ---------------------------------------------------------------- report
w = max(len(r[1]) for r in results)
fails = 0
for ok, claim, computed, quoted, src in results:
    fails += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {claim:<{w}} | computed: {computed} | quoted: {quoted} | {src}")
print(f"\n{len(results) - fails}/{len(results)} claims reproduced from the artifacts")
sys.exit(1 if fails else 0)
