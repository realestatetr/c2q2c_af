#!/usr/bin/env python3
"""
Feasibility note for WP1 (patch obfuscation): can a 2-gate 3-qubit patch be rewritten exactly
as a 3-gate circuit with a DIFFERENT wire structure?   (single script: scan + JSON + figure)

Target patch:  T = G2(wires 1,2) . G1(wires 0,1)       (G1 first), G1, G2 Haar-random 4x4.
Ansatz:        three 4x4 gates on a chosen ordered triple of wire pairs from {(0,1),(0,2),(1,2)}.
Method:        alternating polar updates. Maximise Re tr(T^dagger A) over each gate in turn; the
               optimal gate is V U^dagger from the SVD of the 4x4 environment matrix.
Success:       phase-aligned Frobenius distance ||A e^{-i phi} - T||_F < DIST_TOL (default 1e-9)
               within the restart budget. (An infidelity criterion cannot resolve distances below
               ~1e-8, so it is only logged as a diagnostic.)

Controls built into the same run:
  * Patterns that contain the original pair sequence (0,1) then (1,2) as a subsequence can
    "succeed" trivially by setting the third gate to the identity. They are flagged `trivial`
    and are NOT evidence of re-synthesis.
  * Every success is checked for genuinely entangling gates (normalised second operator-Schmidt
    value of EVERY gate above ENT_THRESHOLD), so identity padding is not counted as `genuine`.
  * Two-gate patterns on other wire pairs are scanned as negative controls.

Outputs (same run): <out>/patch_resynthesis.json and <out>/patch_resynthesis_convergence.png
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

PAIRS = [(0, 1), (0, 2), (1, 2)]
I8 = np.eye(8, dtype=complex)
ENT_THRESHOLD = 0.05


def haar(rng, d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def embed(G, wires):
    p, q = wires
    r = 3 - p - q
    out, inn = "abc", "def"
    sub = f"{out[p]}{out[q]}{inn[p]}{inn[q]},{out[r]}{inn[r]}->abcdef"
    return np.einsum(sub, G.reshape(2, 2, 2, 2), np.eye(2)).reshape(8, 8)


def env(X, wires):
    """Y[(ip iq),(op oq)] = sum_r X[(ip iq r),(op oq r)] so that tr((G x I) X) = tr(G Y)."""
    p, q = wires
    r = 3 - p - q
    li, lo = "abc", "def"
    ax = [None] * 6
    for w in range(3):
        ax[w] = li[w]
        ax[3 + w] = lo[w]
    ax[3 + r] = ax[r]
    sub = "".join(ax) + "->" + ax[p] + ax[q] + ax[3 + p] + ax[3 + q]
    return np.einsum(sub, X.reshape(2, 2, 2, 2, 2, 2)).reshape(4, 4)


def product(gates, pattern):
    A = I8
    for G, w in zip(gates, pattern):
        A = embed(G, w) @ A
    return A


def phase_aligned(T, A):
    ph = np.trace(T.conj().T @ A)
    ph = ph / abs(ph)
    return ph, np.linalg.norm(A / ph - T), np.linalg.norm(A / ph - T, 2)  # phase, Frobenius, operator norm


def entangling_strength(G):
    s = np.linalg.svd(G.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4), compute_uv=False)
    return float(s[1] / s[0])


def resynth(T, pattern, rng, restarts=12, sweeps=400, dist_tol=1e-9):
    Td = T.conj().T
    last_trace = []
    for rs in range(restarts):
        gates = [haar(rng, 4) for _ in pattern]
        trace = []
        for _ in range(sweeps):
            for k, w in enumerate(pattern):
                pre, post = I8, I8
                for j in range(k):
                    pre = embed(gates[j], pattern[j]) @ pre
                for j in range(len(pattern) - 1, k, -1):
                    post = post @ embed(gates[j], pattern[j])
                u, _, vh = np.linalg.svd(env(pre @ Td @ post, w))
                gates[k] = vh.conj().T @ u.conj().T
            A = product(gates, pattern)
            ph, dfro, dop = phase_aligned(T, A)
            trace.append(float(1 - abs(np.trace(Td @ A)) / 8))
            if dfro < dist_tol:
                gates[0] = gates[0] / ph  # absorb the global phase so the patch equals T exactly
                return dict(ok=True, restarts=rs + 1, sweeps=len(trace), gates=gates, trace=trace,
                            fro=float(dfro), op=float(dop))
        last_trace = trace
    return dict(ok=False, restarts=restarts, sweeps=sweeps, gates=None, trace=last_trace, fro=None, op=None)


def contains_original(pattern):
    for i, a in enumerate(pattern):
        if a == (0, 1) and (1, 2) in pattern[i + 1:]:
            return True
    return False


def make_figure(results, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def find(pat):
        return next(r for r in results if r["n_gates"] == 3 and r["pattern"] == [list(p) for p in pat])

    show = [((0, 1), (0, 2), (0, 1)), ((1, 2), (0, 2), (1, 2)), ((0, 2), (1, 2), (0, 2))]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, pat in zip(axs, show):
        r = find(pat)
        for t in r["per_target"]:
            ax.semilogy(np.maximum(t["trace"], 1e-17), lw=1)
        ops = [t["op"] for t in r["per_target"] if t["ok"]]
        extra = f", max op-norm dist {max(ops):.1e}" if ops else ""
        ax.set_title(f"{pat}\nconverged {r['success']}/{r['targets']} (dist tol {args.dist_tol:g}){extra}", fontsize=9)
        ax.set_xlabel("sweep (last restart shown for failures)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("infidelity 1 - |tr(T†A)|/8  (diagnostic only)")
    fig.suptitle("Exact re-synthesis of a 2-gate 3-qubit patch as 3 gates on different wire pairs "
                 f"({args.targets} random targets; success decided by phase-aligned distance)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--sweeps", type=int, default=400)
    ap.add_argument("--dist-tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    targets = []
    for _ in range(args.targets):
        g1, g2 = haar(rng, 4), haar(rng, 4)
        targets.append(embed(g2, (1, 2)) @ embed(g1, (0, 1)))

    results = []
    for n_gates in (3, 2):
        for pat in itertools.product(PAIRS, repeat=n_gates):
            t0 = time.perf_counter()
            runs = [resynth(T, pat, rng, args.restarts, args.sweeps, args.dist_tol) for T in targets]
            per_target = []
            for r in runs:
                ent = min(entangling_strength(G) for G in r["gates"]) if r["ok"] else None
                per_target.append(dict(ok=r["ok"], restarts=r["restarts"], sweeps=r["sweeps"], fro=r["fro"],
                                       op=r["op"], min_entangling=ent, trace=r["trace"]))
            ok = [p for p in per_target if p["ok"]]
            genuine = [p for p in ok if p["min_entangling"] > ENT_THRESHOLD]
            results.append(dict(
                n_gates=n_gates, pattern=[list(p) for p in pat], trivial=contains_original(pat),
                success=len(ok), genuine=len(genuine), targets=len(targets),
                median_restarts=float(np.median([p["restarts"] for p in ok])) if ok else None,
                median_sweeps=float(np.median([p["sweeps"] for p in ok])) if ok else None,
                max_op_dist=max(p["op"] for p in ok) if ok else None,
                time_s=time.perf_counter() - t0, per_target=per_target))
            r = results[-1]
            print(f"{n_gates} gates {str(pat):<42} trivial={r['trivial']!s:5} success {r['success']}/{r['targets']} "
                  f"genuine {r['genuine']}/{r['targets']} max_op_dist={r['max_op_dist']} t={r['time_s']:.1f}s", flush=True)

    make_figure(results, args, os.path.join(args.out, "patch_resynthesis_convergence.png"))
    # keep traces only for the three patterns shown in the figure, to keep the JSON small
    keep = {((0, 1), (0, 2), (0, 1)), ((1, 2), (0, 2), (1, 2)), ((0, 2), (1, 2), (0, 2))}
    for r in results:
        if not (r["n_gates"] == 3 and tuple(tuple(p) for p in r["pattern"]) in keep):
            for p in r["per_target"]:
                p["trace"] = []
    json.dump(dict(config=vars(args), results=results), open(os.path.join(args.out, "patch_resynthesis.json"), "w"))

    nontriv = [r for r in results if r["n_gates"] == 3 and not r["trivial"] and r["genuine"] == r["targets"]]
    print("\nNon-trivial genuine successes:", [r["pattern"] for r in nontriv])
    print("2-gate controls with success:", [r["pattern"] for r in results if r["n_gates"] == 2 and r["success"]])


if __name__ == "__main__":
    main()
