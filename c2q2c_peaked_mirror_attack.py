#!/usr/bin/env python3
"""
Peaked-circuit generator + mirror-cancellation MPO attack (attack-cost vs obfuscation strength)
===============================================================================================

SCOPE -- read this first
------------------------
This is a SIMPLIFIED, small-scale (12-24 qubit) toy family inspired by the published
"HQAP" construction (arXiv:2510.25838) and by the idea of a mirror-aware tensor-network
attack (arXiv:2604.21908). It is NOT a reproduction of either paper, and it does NOT touch
the BlueQubit challenge circuits. Differences from the real construction:

  * R (input) and P (peaking) layers are single-qubit layers, not trained entangling
    circuits. The peak weight is tuned by P's angles. This isolates the identity block.
  * The identity block is U |> U^dagger with ALL-TO-ALL random matchings, as in the paper.
  * Obfuscation knobs implemented here are only
        - "swap transformations": a SWAP fused into a gate, with all later wires relabeled
          (exact: the overall unitary is unchanged up to a final wire permutation);
        - "gauge scrambling": random single-qubit unitaries inserted on every wire segment and
          absorbed into the neighbouring gates (exact; hides gate inverses).
    The paper's patch-retraining ("sweeping"/"masking" with lossy approximations) is NOT
    implemented. Results therefore say nothing about those defences.

Circuit:  (product state psi) -> U -> U^dagger [with swaps + gauge] -> P (1-qubit) -> measure
Peak bitstring is known to the generator and verified against an exact statevector.

Attacks (all classical, tensor-network based):
  direct   : forward MPS simulation of the whole circuit with bond cap chi.
  naive    : mirror-cancellation. The circuit's operator is built as an MPS on 2N sites
             (Choi picture, interleaved output/input legs), absorbing gate pairs outward from
             the middle. For an exact mirror, every pair cancels and the bond stays minimal.
  greedy   : naive + "unswapping": after each pair, if the operator bond profile grew,
             try transpositions of output wires and keep the one that reduces the bond
             profile (our own reading of the idea; not the authors' code).
             cands="all"    : every transposition (exhaustive baseline)
             cands="local4" : only transpositions among the 4 wires of the pair just absorbed

Cost metric: maximum bond dimension reached ("max_bond") and wall time, plus success =
exact recovery of the peak bitstring from single-qubit marginals.
"""

import argparse
import copy
import json
import os
import time

import numpy as np

SWAPM = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


# ----------------------------------------------------------------------------
# Random unitaries and small helpers
# ----------------------------------------------------------------------------
def haar(rng, d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


X1 = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_role(u, role):
    return np.kron(u, I2) if role == 0 else np.kron(I2, u)


# ----------------------------------------------------------------------------
# Instance generator
# ----------------------------------------------------------------------------
def make_instance(N, depth, K, seed, gauge=True, delta=0.1):
    """Returns dict with physical gate list, input product state, output 1q layer, peak."""
    rng = np.random.default_rng(seed)
    logical = []
    for _ in range(depth):
        perm = rng.permutation(N)
        for i in range(N // 2):
            logical.append((int(perm[2 * i]), int(perm[2 * i + 1]), haar(rng, 4)))
    m = len(logical)
    mirror = [(a, b, G.conj().T) for (a, b, G) in reversed(logical)]
    seq = logical + mirror  # logical identity block (length 2m)

    # --- swap transformations (fused into gates), only inside the identity block ---
    swap_at = set(rng.choice(2 * m, size=K, replace=False).tolist()) if K > 0 else set()
    pi = list(range(N))  # logical -> physical
    phys = []
    for t, (a, b, G) in enumerate(seq):
        pa, pb = pi[a], pi[b]
        if t in swap_at:
            phys.append([pa, pb, SWAPM @ G])  # G then SWAP
            pi[a], pi[b] = pb, pa
        else:
            phys.append([pa, pb, G.copy()])

    # --- gauge scrambling: random 1q unitary on every wire segment, absorbed exactly ---
    if gauge:
        last = {}
        for t, (pa, pb, G) in enumerate(phys):
            for role, w in enumerate((pa, pb)):
                if w in last:
                    t1, r1 = last[w]
                    u = haar(rng, 2)
                    phys[t1][2] = kron_role(u.conj().T, r1) @ phys[t1][2]  # post u^dagger
                    phys[t][2] = phys[t][2] @ kron_role(u, role)  # pre u
                last[w] = (t, role)

    # --- product input state (R layer) and peaking layer (P) ---
    theta = rng.uniform(0.3, 2.8, size=N)
    psi_in = [ry(theta[w]) @ np.array([1, 0], dtype=complex) for w in range(N)]  # on physical wire w at t=0
    s_log = rng.integers(0, 2, size=N)
    p_q = delta ** (1.0 / N)
    eta = 2 * np.arccos(np.sqrt(p_q))
    P1 = [None] * N  # on physical wires after the block
    s_phys = np.zeros(N, dtype=int)
    for q in range(N):
        w = pi[q]
        u = ry(eta - theta[q])
        if s_log[q]:
            u = X1 @ u
        P1[w] = u
        s_phys[w] = s_log[q]
    return dict(N=N, m=m, K=K, seed=seed, gates=phys, psi_in=psi_in, P1=P1,
                s_phys=s_phys, delta_design=float(p_q ** N), pi=pi)


# ----------------------------------------------------------------------------
# Exact statevector (ground truth, N <= ~22)
# ----------------------------------------------------------------------------
def statevector_peak(inst):
    N = inst["N"]
    psi = np.array(1.0 + 0j)
    for w in range(N):
        psi = np.tensordot(psi, inst["psi_in"][w], axes=0) if w else inst["psi_in"][w].copy()
    psi = psi.reshape((2,) * N)
    for a, b, G in inst["gates"]:
        out = np.tensordot(G.reshape(2, 2, 2, 2), psi, axes=([2, 3], [a, b]))
        psi = np.moveaxis(out, [0, 1], [a, b])
    for w in range(N):
        out = np.tensordot(inst["P1"][w], psi, axes=([1], [w]))
        psi = np.moveaxis(out, 0, w)
    prob = np.abs(psi) ** 2
    idx = np.unravel_index(int(np.argmax(prob)), prob.shape)
    return np.array(idx, dtype=int), float(prob.max())


# ----------------------------------------------------------------------------
# Minimal MPS engine (mixed canonical form, long-range gates by swap chains)
# ----------------------------------------------------------------------------
class MPS:
    def __init__(self, tensors, center):
        self.T = tensors
        self.c = center
        self.max_bond = max(max(t.shape[0], t.shape[2]) for t in tensors)
        self.capped = False

    def copy(self):
        o = MPS([t.copy() for t in self.T], self.c)
        o.max_bond, o.capped = self.max_bond, self.capped
        return o

    def bonds(self):
        return [self.T[i].shape[2] for i in range(len(self.T) - 1)]

    def move_center(self, k):
        while self.c < k:
            l, d, r = self.T[self.c].shape
            q, rr = np.linalg.qr(self.T[self.c].reshape(l * d, r))
            self.T[self.c] = q.reshape(l, d, q.shape[1])
            self.T[self.c + 1] = np.tensordot(rr, self.T[self.c + 1], axes=([1], [0]))
            self.c += 1
        while self.c > k:
            l, d, r = self.T[self.c].shape
            q, rr = np.linalg.qr(self.T[self.c].reshape(l, d * r).T)  # (d*r, k), (k, l)
            self.T[self.c] = q.T.reshape(q.shape[1], d, r)
            self.T[self.c - 1] = np.tensordot(self.T[self.c - 1], rr.T, axes=([2], [0]))
            self.c -= 1

    def ensure(self, i):
        if self.c < i:
            self.move_center(i)
        elif self.c > i + 1:
            self.move_center(i + 1)

    def apply2(self, i, G, chi, cutoff, to="right"):
        self.ensure(i)
        A, B = self.T[i], self.T[i + 1]
        theta = np.tensordot(A, B, axes=([2], [0]))  # l, s, t, r
        theta = np.tensordot(G.reshape(2, 2, 2, 2), theta, axes=([2, 3], [1, 2]))  # o1,o2,l,r
        theta = theta.transpose(2, 0, 1, 3)
        l, _, _, r = theta.shape
        mat = theta.reshape(l * 2, 2 * r)
        try:
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:  # rare, ill-conditioned after heavy truncation
            from scipy.linalg import svd as _svd
            u, s, vh = _svd(mat, full_matrices=False, lapack_driver="gesvd")
        keep = int(np.sum(s > cutoff * s[0])) if s[0] > 0 else 1
        keep = max(1, keep)
        if keep > chi:
            keep = chi
            self.capped = True
        u, s, vh = u[:, :keep], s[:keep], vh[:keep]
        if to == "right":
            self.T[i] = u.reshape(l, 2, keep)
            self.T[i + 1] = (s[:, None] * vh).reshape(keep, 2, r)
            self.c = i + 1
        else:
            self.T[i] = (u * s[None, :]).reshape(l, 2, keep)
            self.T[i + 1] = vh.reshape(keep, 2, r)
            self.c = i
        if keep > self.max_bond:
            self.max_bond = keep

    def apply_gate(self, i, j, G, chi, cutoff):
        """G acts on ordered (site i, site j); any i != j."""
        if i > j:
            i, j = j, i
            G = SWAPM @ G @ SWAPM
        if j == i + 1:
            self.apply2(i, G, chi, cutoff, "right")
            return
        for k in range(j - 1, i, -1):
            self.apply2(k, SWAPM, chi, cutoff, "left")
        self.apply2(i, G, chi, cutoff, "right")
        for k in range(i + 1, j):
            self.apply2(k, SWAPM, chi, cutoff, "right")


def product_mps(vectors):
    return MPS([v.reshape(1, 2, 1).astype(complex) for v in vectors], 0)


def marginal_bits(tensors):
    """Single-qubit marginal argmax of an (unnormalised) MPS given as a list of (l,2,r)."""
    T = [t.copy() for t in tensors]
    n = len(T)
    for k in range(n - 1, 0, -1):
        l, d, r = T[k].shape
        q, rr = np.linalg.qr(T[k].reshape(l, d * r).T)
        T[k] = q.T.reshape(q.shape[1], d, r)
        T[k - 1] = np.tensordot(T[k - 1], rr.T, axes=([2], [0]))
    bits = np.zeros(n, dtype=int)
    for k in range(n):
        p = np.sum(np.abs(T[k]) ** 2, axis=(0, 2))
        bits[k] = int(np.argmax(p))
        if k < n - 1:
            l, d, r = T[k].shape
            q, rr = np.linalg.qr(T[k].reshape(l * d, r))
            T[k] = q.reshape(l, d, q.shape[1])
            T[k + 1] = np.tensordot(rr, T[k + 1], axes=([1], [0]))
    return bits


# ----------------------------------------------------------------------------
# Attack 1: direct forward MPS simulation
# ----------------------------------------------------------------------------
def attack_direct(inst, chi, cutoff=1e-12):
    t0 = time.perf_counter()
    N = inst["N"]
    mps = product_mps(inst["psi_in"])
    for a, b, G in inst["gates"]:
        mps.apply_gate(a, b, G, chi, cutoff)
    for w in range(N):
        mps.move_center(w)
        mps.T[w] = np.tensordot(inst["P1"][w], mps.T[w], axes=([1], [1])).transpose(1, 0, 2)
    bits = marginal_bits(mps.T)
    ok = bool(np.array_equal(bits, inst["s_phys"]))
    return dict(success=ok, max_bond=mps.max_bond, capped=mps.capped, time=time.perf_counter() - t0,
                frac_correct=float(np.mean(bits == inst["s_phys"])))


# ----------------------------------------------------------------------------
# Attack 2/3: mirror-cancellation MPO (Choi picture), optional greedy unswapping
# ----------------------------------------------------------------------------
def choi_identity(N):
    T = []
    s = 1 / np.sqrt(2)
    for _ in range(N):
        a = np.zeros((1, 2, 2), dtype=complex)
        a[0, 0, 0] = a[0, 1, 1] = 1.0
        b = np.zeros((2, 2, 1), dtype=complex)
        b[0, 0, 0] = b[1, 1, 0] = s
        T += [a, b]
    return MPS(T, 2 * N - 1)


def log_profile(mps):
    return float(sum(np.log2(d) for d in mps.bonds()))


def attack_mirror(inst, chi, cutoff=1e-10, greedy=False, trace=False, cands="all"):
    t0 = time.perf_counter()
    N, m = inst["N"], inst["m"]
    gates = inst["gates"]
    mps = choi_identity(N)
    base = float(N)  # log2-bond profile of the identity operator
    fmap = list(range(N))  # physical output wire -> site index (virtual relabeling)
    prof, swaps_applied, evals = [], 0, 0
    for k in range(m):
        a_w, a_x, Ga = gates[m + k]  # after the middle -> output legs
        mps.apply_gate(2 * fmap[a_w], 2 * fmap[a_x], Ga, chi, cutoff)
        b_w, b_x, Gb = gates[m - 1 - k]  # before the middle -> input legs (transpose)
        mps.apply_gate(2 * b_w + 1, 2 * b_x + 1, Gb.T, chi, cutoff)
        if greedy and log_profile(mps) > base + 0.5:
            for _ in range(4):
                cur = log_profile(mps)
                if cur <= base + 0.5:
                    break
                best, best_uv = cur, None
                if cands == "local4":  # only the 4 wires of the pair just absorbed
                    ws = sorted({fmap[a_w], fmap[a_x], b_w, b_x})
                    pairs = [(u, v) for i, u in enumerate(ws) for v in ws[i + 1:]]
                else:  # exhaustive: every transposition of output wires
                    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
                for u, v in pairs:
                    trial = mps.copy()
                    trial.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                    evals += 1
                    val = log_profile(trial)
                    if val < best - 0.5:
                        best, best_uv = val, (u, v)
                if best_uv is None:
                    break
                u, v = best_uv
                mps.apply_gate(2 * u, 2 * v, SWAPM, chi, cutoff)
                fmap = [v if s == u else (u if s == v else s) for s in fmap]
                swaps_applied += 1
        if trace:
            prof.append(log_profile(mps) - base)
    # project the input legs on the product state psi_in, giving an MPS over output sites
    out = []
    for w in range(N):
        tin = np.tensordot(mps.T[2 * w + 1], inst["psi_in"][w], axes=([1], [0]))  # (d, r)
        out.append(np.tensordot(mps.T[2 * w], tin, axes=([2], [0])))  # (l, y, r)
    out_t = out
    # apply P (1-qubit) on physical wire x, which lives at site fmap[x]
    for x in range(N):
        s = fmap[x]
        out_t[s] = np.tensordot(inst["P1"][x], out_t[s], axes=([1], [1])).transpose(1, 0, 2)
    bits_site = marginal_bits(out_t)
    bits = np.array([bits_site[fmap[x]] for x in range(N)])
    ok = bool(np.array_equal(bits, inst["s_phys"]))
    res = dict(success=ok, max_bond=mps.max_bond, capped=mps.capped, time=time.perf_counter() - t0,
               frac_correct=float(np.mean(bits == inst["s_phys"])), unswaps=swaps_applied, evals=evals)
    if trace:
        res["profile"] = prof
    return res


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
def run_sweep(N, depth, Ks, seeds, chi_mirror, chi_ladder, direct_seeds, verbose=True):
    rows = []
    for K in Ks:
        for sd in range(seeds):
            inst = make_instance(N, depth, K, seed=1000 * K + sd)
            peak, dmax = statevector_peak(inst)
            assert np.array_equal(peak, inst["s_phys"]), "generator bug: designed peak != exact peak"
            row = dict(N=N, depth=depth, K=K, seed=sd, delta_exact=dmax, delta_design=inst["delta_design"])
            for name, gflag in (("naive", False), ("greedy", True)):
                try:
                    row[name] = attack_mirror(inst, chi_mirror, greedy=gflag)
                except Exception as exc:  # numerical failure counts as a failed attack
                    row[name] = dict(success=False, max_bond=chi_mirror, capped=True, time=float("nan"),
                                     frac_correct=0.0, unswaps=0, error=type(exc).__name__)
            rows.append(row)
            if verbose:
                print(f"  K={K:2d} seed={sd}: naive ok={row['naive']['success']!s:5} bond={row['naive']['max_bond']:4d} "
                      f"t={row['naive']['time']:6.1f}s | greedy ok={row['greedy']['success']!s:5} "
                      f"bond={row['greedy']['max_bond']:4d} t={row['greedy']['time']:6.1f}s "
                      f"unswaps={row['greedy']['unswaps']}", flush=True)
    direct = []
    for sd in range(direct_seeds):
        inst = make_instance(N, depth, 0, seed=777 + sd)
        found = None
        for chi in chi_ladder:
            r = attack_direct(inst, chi)
            if verbose:
                print(f"  direct seed={sd} chi={chi}: ok={r['success']} frac={r['frac_correct']:.2f} t={r['time']:.1f}s", flush=True)
            if r["success"]:
                found = chi
                break
        direct.append(dict(seed=sd, chi_break=found, ladder=list(chi_ladder)))
    return rows, direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--Ks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--chi-mirror", type=int, default=128)
    ap.add_argument("--chi-ladder", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--direct-seeds", type=int, default=1)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"N={args.N} depth={args.depth} (gates/half={args.depth * args.N // 2}) Ks={args.Ks} seeds={args.seeds}")
    rows, direct = run_sweep(args.N, args.depth, args.Ks, args.seeds, args.chi_mirror,
                             args.chi_ladder, args.direct_seeds)
    with open(os.path.join(args.out, f"peaked_attack_N{args.N}.json"), "w") as f:
        json.dump(dict(args=vars(args), rows=rows, direct=direct), f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
