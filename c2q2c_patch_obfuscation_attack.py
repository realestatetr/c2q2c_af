#!/usr/bin/env python3
"""
Does re-synthesized-patch obfuscation make the mirror attack harder?  (controlled comparison)

Toy family (same caveats as c2q2c_peaked_mirror_attack.py): identity block U |> U^dagger built from a
SEQUENCE of random two-qubit gates (random wire pair per gate), optional fused swap transformations,
single-qubit R/P layers, gauge scrambling.  New here: PATCHES in the first half only. A patch picks
two adjacent gates G1(u,s), G2(s,v) that share exactly one wire and replaces them by three gates on
different wire pairs ((u,s),(u,v),(u,s)) or ((s,v),(u,v),(s,v)) found numerically (exact to ~1e-12).

The unitary is unchanged, so the identity block is still an identity, but the two halves no longer
have the same gate count (m+P vs m gates), so index pairing breaks. Three questions are separated:

  naive    : index pairing (k-th gate from each side).  Fails by MISALIGNMENT, not by hardness.
  oracle   : absorbs the true patch blocks together (attacker knows patch boundaries).
             Cost here = hardness of the operator once alignment is known.
  search   : no oracle. Tries absorbing (r right, l left) gate groups from a move set and keeps
             the first group that returns the operator bond profile to the identity baseline.
             search_full contains the true patch shape (2 right, 3 left); search_limited does not.

Unswapping variants (none / exhaustive / local four-wire) are reused with the oracle alignment.
Every instance is verified against an exact statevector before any attack is run.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

import c2q2c_peaked_mirror_attack as pm
import patch_resynthesis_feasibility as pr

SWAPM = pm.SWAPM
PATTERNS = [((0, 1), (0, 2), (0, 1)), ((1, 2), (0, 2), (1, 2))]  # the two genuine re-synthesis patterns


# ----------------------------------------------------------------------------
# Generator with patches
# ----------------------------------------------------------------------------
def make_patched_instance(N, m, K, P, seed, gauge=True, delta=0.1, min_gap=0, max_gap=0):
    rng = np.random.default_rng(seed)
    logical = []
    for _ in range(m):
        a, b = rng.choice(N, size=2, replace=False)
        logical.append((int(a), int(b), pm.haar(rng, 4)))
    mirror = [(a, b, G.conj().T) for (a, b, G) in reversed(logical)]
    seq = logical + mirror

    swap_at = set(rng.choice(2 * m, size=K, replace=False).tolist()) if K > 0 else set()
    pi = list(range(N))
    phys = []
    for t, (a, b, G) in enumerate(seq):
        pa, pb = pi[a], pi[b]
        if t in swap_at:
            phys.append([pa, pb, SWAPM @ G])
            pi[a], pi[b] = pb, pa
        else:
            phys.append([pa, pb, G.copy()])

    # ---- patches in the first half ----
    # (t1, t2): G1 = gate t1 on (u,s), G2 = gate t2 on (s,v) share exactly one wire s, and no gate strictly
    # between them touches {u, s, v}. min_gap = max_gap = 0 gives adjacent patches only (original behaviour);
    # min_gap >= 1 forces gate(s) between G1 and G2. The patch is placed at the slot of G2 (G1 commutes forward).
    pair_list = []
    for t1 in range(m):
        for t2 in range(t1 + 1 + min_gap, min(m, t1 + 2 + max_gap)):
            w1, w2 = {phys[t1][0], phys[t1][1]}, {phys[t2][0], phys[t2][1]}
            if len(w1 & w2) != 1:
                continue
            w3 = w1 | w2
            if any({phys[t][0], phys[t][1]} & w3 for t in range(t1 + 1, t2)):
                continue
            pair_list.append((t1, t2))
    rng.shuffle(pair_list)
    chosen, used, patch_err = [], set(), []
    for (t1, t2) in pair_list:
        if len(chosen) == P:
            break
        if any(x in used for x in range(t1, t2 + 1)):
            continue
        a1, b1, G1 = phys[t1]
        a2, b2, G2 = phys[t2]
        s = ({a1, b1} & {a2, b2}).pop()
        u = a1 if b1 == s else b1
        v = a2 if b2 == s else b2
        M1 = G1 if (a1, b1) == (u, s) else SWAPM @ G1 @ SWAPM
        M2 = G2 if (a2, b2) == (s, v) else SWAPM @ G2 @ SWAPM
        T = pr.embed(M2, (1, 2)) @ pr.embed(M1, (0, 1))
        pat = PATTERNS[int(rng.integers(0, 2))]
        r = pr.resynth(T, pat, rng, restarts=30, sweeps=900, dist_tol=1e-12)
        if not r["ok"]:
            continue
        loc = [u, s, v]
        new = [(loc[p_], loc[q_], G) for G, (p_, q_) in zip(r["gates"], pat)]
        chosen.append((t1, t2, new))
        patch_err.append(float(r["fro"]))
        used.update(range(t1, t2 + 1))
    patch_at = {t2: (t1, new) for t1, t2, new in chosen}
    removed = {t1 for t1, t2, new in chosen}
    gaps = [t2 - t1 - 1 for t1, t2, new in chosen]

    first, left_idx = [], {}
    patch_blocks = {}
    for t in range(m):
        if t in removed:
            continue
        if t in patch_at:
            base = len(first)
            first += [[a, b, G] for (a, b, G) in patch_at[t][1]]
            patch_blocks[t] = list(range(base, base + 3))
        else:
            left_idx[t] = len(first)
            first.append(phys[t])
    m1 = len(first)
    gates = first + phys[m:]

    if gauge:
        last = {}
        for t, (pa, pb, G) in enumerate(gates):
            for role, w in enumerate((pa, pb)):
                if w in last:
                    t1, r1 = last[w]
                    u_ = pm.haar(rng, 2)
                    gates[t1][2] = pm.kron_role(u_.conj().T, r1) @ gates[t1][2]
                    gates[t][2] = gates[t][2] @ pm.kron_role(u_, role)
                last[w] = (t, role)

    theta = rng.uniform(0.3, 2.8, size=N)
    psi_in = [pm.ry(theta[w]) @ np.array([1, 0], dtype=complex) for w in range(N)]
    s_log = rng.integers(0, 2, size=N)
    p_q = delta ** (1.0 / N)
    eta = 2 * np.arccos(np.sqrt(p_q))
    P1, s_phys = [None] * N, np.zeros(N, dtype=int)
    for q in range(N):
        w = pi[q]
        uq = pm.ry(eta - theta[q])
        P1[w] = pm.X1 @ uq if s_log[q] else uq
        s_phys[w] = s_log[q]

    # ---- alignment information for the oracle attacker ----
    steps = []
    for t in range(m - 1, -1, -1):
        if t in removed:
            continue  # consumed together with its patch (G1 is absorbed ahead of the gates between)
        if t in patch_at:
            t1 = patch_at[t][0]
            steps.append(dict(right=[m1 + (m - 1 - t), m1 + (m - 1 - t1)], left=list(reversed(patch_blocks[t]))))
        else:
            steps.append(dict(right=[m1 + (m - 1 - t)], left=[left_idx[t]]))
    blocks, t = [], m - 1  # contiguous gate groups whose absorption returns the operator to the identity profile
    while t >= 0:
        if t in patch_at:
            g = t - patch_at[t][0] - 1
            blocks.append((g + 2, g + 3))
            t = patch_at[t][0] - 1
        else:
            blocks.append((1, 1))
            t -= 1
    return dict(N=N, m=m, m1=m1, K=K, P_requested=P, P=len(patch_blocks), seed=seed, gates=gates,
                psi_in=psi_in, P1=P1, s_phys=s_phys, delta_design=float(p_q ** N), pi=pi, steps=steps, gaps=gaps, blocks=blocks, patch_err=patch_err)


def naive_steps(inst):
    m, m1 = inst["m"], inst["m1"]
    st = [dict(right=[m1 + k], left=[m1 - 1 - k]) for k in range(m)]
    st += [dict(right=[], left=[m1 - 1 - k]) for k in range(m, m1)]
    return st


# ----------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------
def _finish(inst, mps, fmap, t0, extra):
    N = inst["N"]
    excess = float(pm.log_profile(mps) - N)  # 0 if the absorbed operator has collapsed to an identity-like profile
    out = []
    for w in range(N):
        tin = np.tensordot(mps.T[2 * w + 1], inst["psi_in"][w], axes=([1], [0]))
        out.append(np.tensordot(mps.T[2 * w], tin, axes=([2], [0])))
    for x in range(N):
        s = fmap[x]
        out[s] = np.tensordot(inst["P1"][x], out[s], axes=([1], [1])).transpose(1, 0, 2)
    bits_site = pm.marginal_bits(out)
    bits = np.array([bits_site[fmap[x]] for x in range(N)])
    res = dict(success=bool(np.array_equal(bits, inst["s_phys"])), max_bond=mps.max_bond, capped=mps.capped,
               time=time.perf_counter() - t0, frac_correct=float(np.mean(bits == inst["s_phys"])))
    res.update(extra)
    res["final_profile_excess"] = excess
    return res


def attack_steps(inst, steps, chi=128, cutoff=1e-10, unswap=None):
    """Absorb right/left gate groups step by step. unswap: None | 'all' | 'local4'."""
    t0 = time.perf_counter()
    N, gates = inst["N"], inst["gates"]
    mps = pm.choi_identity(N)
    base, fmap, evals, nsw, max_wires = float(N), list(range(N)), 0, 0, 0
    for st in steps:
        used = set()
        for gi in st["right"]:
            a, b, G = gates[gi]
            mps.apply_gate(2 * fmap[a], 2 * fmap[b], G, chi, cutoff)
            used |= {fmap[a], fmap[b]}
        for gi in st["left"]:
            a, b, G = gates[gi]
            mps.apply_gate(2 * a + 1, 2 * b + 1, G.T, chi, cutoff)
            used |= {a, b}
        if unswap and pm.log_profile(mps) > base + 0.5:
            for _ in range(6):
                cur = pm.log_profile(mps)
                if cur <= base + 0.5:
                    break
                ws = sorted(used)
                max_wires = max(max_wires, len(ws))
                pairs = ([(u, v) for i, u in enumerate(ws) for v in ws[i + 1:]] if unswap == "local4"
                         else [(u, v) for u in range(N) for v in range(u + 1, N)])
                best, best_uv = cur, None
                for u, v in pairs:
                    trial = mps.copy()
                    trial.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                    evals += 1
                    val = pm.log_profile(trial)
                    if val < best - 0.5:
                        best, best_uv = val, (u, v)
                if best_uv is None:
                    break
                u, v = best_uv
                mps.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                fmap = [v if s == u else (u if s == v else s) for s in fmap]
                nsw += 1
    return _finish(inst, mps, fmap, t0, dict(evals=evals, unswaps=nsw, max_wire_set=max_wires))


MOVES_FULL = sorted([(r, l) for r in range(0, 4) for l in range(0, 4) if r + l >= 2 and (r, l) not in ((0, 2), (0, 3), (2, 0), (3, 0))],
                    key=lambda x: (x[0] + x[1], x[0]))
MOVES_FULL = [(1, 1)] + [mv for mv in MOVES_FULL if mv != (1, 1)]
MOVES_LIMITED = [mv for mv in MOVES_FULL if mv in ((1, 1), (1, 2), (2, 1), (2, 2))]  # moves <= 2 per side
MOVES_LE3 = MOVES_FULL  # moves <= 3 per side (named 'search_full' in the first adjacent-patch experiment)
MOVES_WIDE = [(1, 1)] + sorted([(r, l) for r in range(1, 8) for l in range(1, 8) if (r, l) != (1, 1)],
                               key=lambda x: (x[0] + x[1], x[0]))


def attack_search(inst, moves, chi=128, cutoff=1e-10, chi_trial=None, time_limit=None, eval_limit=None):
    """No alignment knowledge: accept the first (r right, l left) group that restores the identity profile.
    Returns status 'OK' or 'TIMEOUT' (never counted as failure); `timeout_reason` is 'time' or 'evals'.
    The budget is checked at the top of every loop iteration and before every trial, but a single trial or a
    forced step is not interrupted, so time_limit is a soft limit; eval_limit is the reproducible budget.
    `ends` lists the (right, left) counts after each accepted move."""
    t0 = time.perf_counter()
    N, gates, m, m1 = inst["N"], inst["gates"], inst["m"], inst["m1"]
    mps = pm.choi_identity(N)
    base, ri, li, evals, forced, ends = float(N), 0, 0, 0, 0, []
    ct = chi_trial or chi

    def absorb(target, r, l, ri0, li0, cap):
        for k in range(r):
            a, b, G = gates[m1 + ri0 + k]
            target.apply_gate(2 * a, 2 * b, G, cap, cutoff)
        for k in range(l):
            a, b, G = gates[m1 - 1 - (li0 + k)]
            target.apply_gate(2 * a + 1, 2 * b + 1, G.T, cap, cutoff)

    def reason():
        if eval_limit is not None and evals >= eval_limit:
            return "evals"
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            return "time"
        return None

    def timeout(why):
        return dict(status="TIMEOUT", timeout_reason=why, success=None, peak_verified=None, max_bond=mps.max_bond,
                    capped=mps.capped, time=time.perf_counter() - t0, evals=evals, forced=forced, ends=ends,
                    progress=(ri, li))

    while ri < m or li < m1:
        if reason():
            return timeout(reason())
        accepted = False
        for r, l in moves:
            if reason():
                return timeout(reason())
            if ri + r > m or li + l > m1:
                continue
            trial = mps.copy()
            absorb(trial, r, l, ri, li, ct)
            evals += 1
            if pm.log_profile(trial) <= base + 0.5:
                mps, ri, li, accepted = trial, ri + r, li + l, True
                ends.append((ri, li))
                break
        if not accepted:  # forced progress, stays misaligned
            r, l = (1 if ri < m else 0), (1 if li < m1 else 0)
            absorb(mps, r, l, ri, li, chi)
            ri, li, forced = ri + r, li + l, forced + 1
    res = _finish(inst, mps, list(range(N)), t0, dict(evals=evals, forced=forced))
    res.update(status="OK", peak_verified=res["success"], ends=ends)
    return res


def alignment_correct(inst, res):
    """True if no forced steps were needed and every accepted move ends on a true block boundary."""
    if res.get("status") != "OK" or res["forced"] > 0:
        return False
    cum, b = [], (0, 0)
    for dr, dl in inst["blocks"]:
        b = (b[0] + dr, b[1] + dl)
        cum.append(b)
    return set(map(tuple, res["ends"])) <= set(cum) and tuple(res["ends"][-1]) == (inst["m"], inst["m1"])


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
def run(N, m, Ps, seeds, K_swap, out):
    rows = []
    for P in Ps:
        for sd in range(seeds):
            inst = make_patched_instance(N, m, 0, P, seed=31000 + 100 * P + sd)
            peak, dmax = pm.statevector_peak(inst)
            assert np.array_equal(peak, inst["s_phys"]) and abs(dmax - inst["delta_design"]) < 1e-6, "generator bug"
            row = dict(N=N, m=m, K=0, P_req=P, P=inst["P"], seed=sd, m1=inst["m1"])
            row["naive"] = attack_steps(inst, naive_steps(inst))
            row["oracle"] = attack_steps(inst, inst["steps"])
            row["search_full"] = attack_search(inst, MOVES_FULL)
            row["search_limited"] = attack_search(inst, MOVES_LIMITED)
            rows.append(row)
            print(f"K=0 P={inst['P']} seed={sd}: " + " | ".join(
                f"{k} ok={row[k]['success']!s:5} bond={row[k]['max_bond']:3d} t={row[k]['time']:.1f}s"
                for k in ("naive", "oracle", "search_full", "search_limited")), flush=True)
            json.dump(rows, open(os.path.join(out, "patch_attack_K0.json"), "w"))
    rows2 = []
    for P in sorted({p for p in Ps if p in (0, 4)}):
        for sd in range(seeds):
            inst = make_patched_instance(N, m, K_swap, P, seed=51000 + 100 * P + sd)
            peak, dmax = pm.statevector_peak(inst)
            assert np.array_equal(peak, inst["s_phys"]) and abs(dmax - inst["delta_design"]) < 1e-6, "generator bug"
            row = dict(N=N, m=m, K=K_swap, P_req=P, P=inst["P"], seed=sd, m1=inst["m1"])
            for name, uns in (("oracle_none", None), ("oracle_all", "all"), ("oracle_local4", "local4")):
                row[name] = attack_steps(inst, inst["steps"], unswap=uns)
            rows2.append(row)
            print(f"K={K_swap} P={inst['P']} seed={sd}: " + " | ".join(
                f"{k} ok={row[k]['success']!s:5} bond={row[k]['max_bond']:3d} evals={row[k].get('evals', 0)}"
                for k in ("oracle_none", "oracle_all", "oracle_local4")), flush=True)
            json.dump(rows2, open(os.path.join(out, "patch_attack_Kswap.json"), "w"))
    return rows, rows2


def attack_commute(inst, chi=64, cutoff=1e-10, time_limit=None, eval_limit=None, max_total=10, max_seeds=8,
                   unswap=None):
    """Blind, commutation-aware alignment (no patch boundaries are given; the split between the two halves is).

    Right gates (second half, time order) and left gates (first half, reverse time order) each form a
    dependency DAG: two gates depend on each other iff they share a physical wire. A gate is AVAILABLE if no
    earlier remaining gate on its side shares a wire with it (it then commutes with everything still unabsorbed
    on that side, so absorbing it next is exact). Candidate groups are grown breadth-first from pairs of
    available gates with overlapping wires, adding only available gates that touch the current wire union.
    A group is tried when its right wire set (mapped through the current output relabeling `fmap`) and its left
    wire set are equal (unswap=None) or differ in at most two wires (unswap='local4'). It is accepted when
    absorbing it returns the operator bond profile to the identity baseline; with unswap='local4', if the profile
    stays above baseline, transpositions of output wires among the group's wires are tried greedily (up to 3) and
    `fmap` is updated. Groups inside a trial are absorbed alternately to keep the transient bond small.
    Budgets: eval_limit counts every MPS candidate evaluation (including unswap trials) and is the reproducible
    budget; time_limit is a soft limit (single trials are not interrupted). Timeouts are never counted as failures.
    """
    from collections import deque
    t0 = time.perf_counter()
    N, gates, m, m1 = inst["N"], inst["gates"], inst["m"], inst["m1"]
    phys = lambda g: frozenset((gates[g][0], gates[g][1]))
    remR, remL = list(range(m1, m1 + m)), list(range(m1 - 1, -1, -1))
    mps = pm.choi_identity(N)
    base, evals, forced, nsw, groups, max_wires = float(N), 0, 0, 0, [], 0
    fmap = list(range(N))

    def site(g, fm):
        return frozenset((fm[gates[g][0]], fm[gates[g][1]]))

    def avail(rem):
        out, blocked = [], set()
        for g in rem:
            w = phys(g)
            if not (w & blocked):
                out.append(g)
            blocked |= w
        return out

    def absorb(target, Rc, Lc, fm):
        for i in range(max(len(Rc), len(Lc))):
            if i < len(Rc):
                a, b, G = gates[Rc[i]]
                target.apply_gate(2 * fm[a], 2 * fm[b], G, chi, cutoff)
            if i < len(Lc):
                a, b, G = gates[Lc[i]]
                target.apply_gate(2 * a + 1, 2 * b + 1, G.T, chi, cutoff)

    def reason():
        if eval_limit is not None and evals >= eval_limit:
            return "evals"
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            return "time"
        return None

    def timeout(why):
        return dict(status="TIMEOUT", timeout_reason=why, success=None, peak_verified=None, max_bond=mps.max_bond,
                    capped=mps.capped, time=time.perf_counter() - t0, evals=evals, forced=forced, unswaps=nsw,
                    groups=groups, remaining=(len(remR), len(remL)))

    while remR or remL:
        if reason():
            return timeout(reason())
        aR, aL = avail(remR), avail(remL)
        found = None
        if aR and aL:
            queue, seen = deque(), set()
            for a in aR[:max_seeds]:
                for b in aL[:max_seeds]:
                    if site(a, fmap) & phys(b):
                        queue.append(((a,), (b,)))
            while queue and found is None:
                Rc, Lc = queue.popleft()
                key = (frozenset(Rc), frozenset(Lc))
                if key in seen:
                    continue
                seen.add(key)
                SR = frozenset().union(*[site(g, fmap) for g in Rc])
                SL = frozenset().union(*[phys(g) for g in Lc])
                if len(SR ^ SL) <= (2 if unswap else 0):
                    if reason():
                        return timeout(reason())
                    trial = mps.copy()
                    absorb(trial, Rc, Lc, fmap)
                    evals += 1
                    cur, fm, k_sw = pm.log_profile(trial), list(fmap), 0
                    if cur > base + 0.5 and unswap:
                        ws = sorted(SR | SL)
                        max_wires = max(max_wires, len(ws))
                        pairs = [(u, v) for i, u in enumerate(ws) for v in ws[i + 1:]]
                        for _ in range(3):
                            best, best_uv = cur, None
                            for u, v in pairs:
                                if reason():
                                    return timeout(reason())
                                t2 = trial.copy()
                                t2.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                                evals += 1
                                val = pm.log_profile(t2)
                                if val < best - 0.5:
                                    best, best_uv = val, (u, v)
                            if best_uv is None:
                                break
                            u, v = best_uv
                            trial.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                            fm = [v if x == u else (u if x == v else x) for x in fm]
                            cur, k_sw = best, k_sw + 1
                            if cur <= base + 0.5:
                                break
                    if cur <= base + 0.5:
                        found = (Rc, Lc, trial, fm, k_sw)
                        break
                if len(Rc) + len(Lc) >= max_total:
                    continue
                U = SR | SL
                restR = [g for g in remR if g not in Rc]
                restL = [g for g in remL if g not in Lc]
                for g in avail(restR):
                    if site(g, fmap) & U:
                        queue.append((Rc + (g,), Lc))
                for g in avail(restL):
                    if phys(g) & U:
                        queue.append((Rc, Lc + (g,)))
        if found is not None:
            Rc, Lc, mps, fmap, k_sw = found
            nsw += k_sw
            groups.append([sorted(Rc), sorted(Lc)])
        else:  # forced progress (stays misaligned)
            Rc = (aR[0],) if aR else ()
            Lc = (aL[0],) if aL else ()
            absorb(mps, Rc, Lc, fmap)
            forced += 1
            groups.append([sorted(Rc), sorted(Lc), "forced"])
        remR = [g for g in remR if g not in Rc]
        remL = [g for g in remL if g not in Lc]
    res = _finish(inst, mps, fmap, t0, dict(evals=evals, forced=forced, unswaps=nsw, max_wire_set=max_wires))
    res.update(status="OK", peak_verified=res["success"], groups=groups)
    return res


def alignment_correct_commute(inst, res):
    """True if no forced steps and the accepted groups are exactly the true (right, left) gate groups."""
    if res.get("status") != "OK" or res["forced"] > 0:
        return False
    true = {(frozenset(st["right"]), frozenset(st["left"])) for st in inst["steps"]}
    got = {(frozenset(g[0]), frozenset(g[1])) for g in res["groups"]}
    return got == true


def run_params(N, m, P, K_swap, seed0, chi, chi_trial, time_limit, eval_limit):
    """Every setting that can change an attack result, recorded in each JSON row."""
    import hashlib
    import inspect
    cs = inspect.signature(attack_commute).parameters
    return dict(
        script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        N=N, m=m, P=P, K_swap=K_swap, seed0=seed0, chi=chi, chi_trial=chi_trial, time_limit_soft=time_limit,
        eval_limit=eval_limit, cutoff=cs["cutoff"].default, commute_max_seeds=cs["max_seeds"].default,
        commute_max_total_group_size=cs["max_total"].default,
        moves=dict(search_small=MOVES_LIMITED, search_le3=MOVES_LE3, search_wide="all (r,l) with 1<=r,l<=7"),
        unswap_rule=("'local4' = greedy transpositions of OUTPUT wires among the wire union of the group just absorbed "
                     "(oracle steps: up to 6 rounds; commute search: up to 3 rounds); the union is not limited to four "
                     "wires, the largest one actually used is reported as max_wire_set. The commute search additionally "
                     "tries groups whose right/left wire sets differ in up to two wires when unswapping is enabled."),
        halves_boundary_known=True)


ALL_ATTACKERS = ("oracle", "search_small", "search_le3", "search_wide", "search_commute")
UNSWAP_ATTACKERS = ("oracle", "oracle_local4", "search_commute", "commute_local4")


def run_gap(N, m, P, n_valid, out, time_limit=90, chi=64, chi_trial=64, seed0=61000, max_tries=60,
            attackers=ALL_ATTACKERS, fname="gap_patch_K0.json", eval_limit=None, K_swap=0):
    """n_valid instances with exactly P realised patches (>= 1 gate between G1 and G2 for every patch) and
    K_swap fused swap transformations. Raises if fewer than n_valid valid instances are found.
    Same seed => same logical gates for every K_swap; K_swap only changes where swaps are fused, so the
    physical circuit (and the patch placement) differs between K values."""
    rows, tries = [], 0
    while len(rows) < n_valid and tries < max_tries:
        sd = seed0 + tries
        tries += 1
        inst = make_patched_instance(N, m, K_swap, P, seed=sd, min_gap=1, max_gap=6)
        if inst["P"] != P:
            print(f"seed {sd}: realised P={inst['P']} != {P}, skipped", flush=True)
            continue
        peak, dmax = pm.statevector_peak(inst)
        assert np.array_equal(peak, inst["s_phys"]) and abs(dmax - inst["delta_design"]) < 1e-6, "generator bug"
        row = dict(N=N, m=m, K=K_swap, P=P, seed=sd, m1=inst["m1"], gaps=inst["gaps"], patch_err_max=max(inst["patch_err"]),
                   statevector_peak_ok=True, chi=chi, chi_trial=chi_trial, time_limit=time_limit, eval_limit=eval_limit)
        row["params"] = run_params(N, m, P, K_swap, seed0, chi, chi_trial, time_limit, eval_limit)
        for name in attackers:
            if name in ("oracle", "oracle_local4"):
                r = attack_steps(inst, inst["steps"], chi=chi, unswap="local4" if name == "oracle_local4" else None)
            elif name in ("search_commute", "commute_local4"):
                r = attack_commute(inst, chi=chi, time_limit=time_limit, eval_limit=eval_limit,
                                   unswap="local4" if name == "commute_local4" else None)
                r["alignment_correct"] = alignment_correct_commute(inst, r)
            else:
                mv = {"search_small": MOVES_LIMITED, "search_le3": MOVES_LE3, "search_wide": MOVES_WIDE}[name]
                r = attack_search(inst, mv, chi=chi, chi_trial=chi_trial, time_limit=time_limit, eval_limit=eval_limit)
                r["alignment_correct"] = alignment_correct(inst, r)
            row[name] = r
        rows.append(row)
        desc = " | ".join(
            f"{k} {row[k].get('status', 'OK')}{'/' + row[k]['timeout_reason'] if row[k].get('timeout_reason') else ''} "
            f"peak={row[k].get('peak_verified', row[k]['success'])} aligned={row[k].get('alignment_correct', '-')} "
            f"forced={row[k].get('forced', '-')} evals={row[k].get('evals', '-')} bond={row[k]['max_bond']} t={row[k]['time']:.0f}s"
            for k in attackers)
        print(f"seed {sd} K={K_swap} gaps={inst['gaps']} patch_err={row['patch_err_max']:.1e}: {desc}", flush=True)
        json.dump(rows, open(os.path.join(out, fname), "w"))
    if len(rows) < n_valid:
        raise RuntimeError(f"only {len(rows)} valid instances with P={P} found in {tries} tries (wanted {n_valid})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--m", type=int, default=48)
    ap.add_argument("--Ps", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--K-swap", type=int, default=4)
    ap.add_argument("--out", default=".")
    ap.add_argument("--gap", action="store_true",
                    help="run the non-adjacent (gap) patch experiment instead (--Ps is ignored; use --gap-patches)")
    ap.add_argument("--gap-patches", type=int, default=2, help="patches per instance in the gap experiment")
    ap.add_argument("--gap-K", type=int, default=0, help="fused swap transformations per instance in the gap experiment "
                    "(separate from --K-swap, whose default 4 belongs to the adjacent-patch experiment)")
    ap.add_argument("--seed0", type=int, default=61000)
    ap.add_argument("--attackers", default=",".join(ALL_ATTACKERS))
    ap.add_argument("--fname", default="gap_patch_K0.json")
    ap.add_argument("--time-limit", type=float, default=90)
    ap.add_argument("--eval-limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.gap:
        run_gap(a.N, a.m, a.gap_patches, a.seeds, a.out, time_limit=a.time_limit, seed0=a.seed0,
                attackers=tuple(a.attackers.split(",")), fname=a.fname, eval_limit=a.eval_limit, K_swap=a.gap_K)
    else:
        run(a.N, a.m, a.Ps, a.seeds, a.K_swap, a.out)
