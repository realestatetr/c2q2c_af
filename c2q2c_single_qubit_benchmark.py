#!/usr/bin/env python3
"""
C2Q2C single-qubit benchmark with a classical adversary
=======================================================

Compares two ways of loading a classical parameter p in [0,1] into one qubit:

  ANGLE   : theta = pi * p            -> P(1) = sin^2(pi p / 2),  decode p = (2/pi) asin(sqrt(P))
  C2Q2C   : theta = 2 asin(sqrt(p))   -> P(1) = p,                decode p = P

Both circuits are Ry(theta)|0> followed by a Z-basis measurement.

What this script establishes (and what it does not)
---------------------------------------------------
1. CIRCUIT CHECK      The simulated circuits produce the intended P(1).
2. ADVERSARY          A classical Binomial(N, P) sampler reproduces the circuit's
                      shot-count distribution (two-sample chi-square, Holm-corrected,
                      with a negative control to show the test has power).
                      => no quantum advantage at the single-qubit level; the quantum
                      step is classically simulable in O(1) per shot.
3. ACCURACY vs p      RMSE(p) for both encodings against theory:
                        ANGLE : 1/(pi sqrt(N))            (variance-stabilised, flat in p)
                        C2Q2C : sqrt(p(1-p)/N)            (Cramer-Rao bound)
                      Fisher information per shot: ANGLE = pi^2, C2Q2C = 1/(p(1-p)).
                      C2Q2C has LOWER variance only where p(1-p) < 1/pi^2,
                      i.e. p < ~0.114 or p > ~0.886.
4. UNIFORM AVERAGE    Averaged over p ~ U(0,1), RMSE_C2Q2C / RMSE_ANGLE -> pi/sqrt(6) = 1.2825.

Limits: this runs on a noiseless simulator. It validates the pipeline and the
statistics. It does not test hardware noise, and it makes no advantage claim.
"""

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

import pennylane as qml  # noqa: E402

ENCODINGS = ("angle", "c2q2c")
LABEL = {"angle": "Angle  (θ = πp)", "c2q2c": "C2Q2C  (θ = 2·asin√p)"}
COLOR = {"angle": "#1f77b4", "c2q2c": "#d62728"}
P_CROSS = (1 - np.sqrt(1 - 4 / np.pi**2)) / 2  # p(1-p) = 1/pi^2  (~0.1144)


# ----------------------------------------------------------------------------
# Encoding / decoding / theory
# ----------------------------------------------------------------------------
def theta_of(p, enc):
    p = np.asarray(p, dtype=float)
    return np.pi * p if enc == "angle" else 2 * np.arcsin(np.sqrt(p))


def target_prob(p, enc):
    """The P(1) each encoding is *supposed* to produce."""
    p = np.asarray(p, dtype=float)
    return np.sin(np.pi * p / 2) ** 2 if enc == "angle" else p


def decode(k, n, enc):
    ph = np.clip(np.asarray(k, dtype=float) / n, 0.0, 1.0)
    return (2 / np.pi) * np.arcsin(np.sqrt(ph)) if enc == "angle" else ph


def rmse_theory(p, n, enc):
    p = np.asarray(p, dtype=float)
    if enc == "angle":
        return np.full_like(p, 1 / (np.pi * np.sqrt(n)))
    return np.sqrt(p * (1 - p) / n)


def fisher_per_shot(p, enc):
    p = np.asarray(p, dtype=float)
    if enc == "angle":
        return np.full_like(p, np.pi**2)
    with np.errstate(divide="ignore"):
        return 1.0 / (p * (1 - p))


# ----------------------------------------------------------------------------
# Quantum side (PennyLane; swap the device name for another backend)
# ----------------------------------------------------------------------------
class QubitRunner:
    def __init__(self, device_name="default.qubit", seed=0):
        self.device_name = device_name
        self.dev = qml.device(device_name, wires=1, seed=seed)

        @qml.qnode(self.dev)
        def probs_qnode(theta):
            qml.RY(theta, wires=0)
            return qml.probs(wires=0)

        @qml.qnode(self.dev)
        def sample_qnode(theta):
            qml.RY(theta, wires=0)
            return qml.sample(wires=0)

        self._probs = probs_qnode
        self._sample = sample_qnode

    def prob1(self, theta):
        return float(self._probs(theta)[1])

    def sample_bits(self, theta, shots):
        out = qml.set_shots(self._sample, shots=shots)(theta)
        return np.asarray(out).reshape(-1)


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def chi2_homogeneity(a, b, max_bins=12):
    """Two-sample chi-square on quantile-binned discrete counts. Returns p-value."""
    pooled = np.concatenate([a, b])
    if np.ptp(pooled) == 0:
        return 1.0  # both samples identical constants
    qs = np.unique(np.quantile(pooled, np.linspace(0, 1, max_bins + 1)))
    edges = np.concatenate([[-np.inf], qs[1:-1] + 0.5, [np.inf]])
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    table = np.vstack([ha, hb])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def empirical_rmse(p, n, enc, reps, rng):
    """RMSE of the decoded estimate via the Binomial adversary (vectorised)."""
    p = np.asarray(p, dtype=float)
    k = rng.binomial(n, target_prob(p, enc)[:, None], size=(len(p), reps))
    est = decode(k, n, enc)
    return np.sqrt(np.mean((est - p[:, None]) ** 2, axis=1))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="C2Q2C single-qubit benchmark")
    ap.add_argument("--device", default="default.qubit")
    ap.add_argument("--shots", type=int, default=1024, help="shots per repetition (N)")
    ap.add_argument("--reps", type=int, default=1000, help="repetitions for the equality test")
    ap.add_argument("--sweep-reps", type=int, default=20000, help="repetitions per p for RMSE sweeps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    runner = QubitRunner(args.device, seed=args.seed)
    N = args.shots
    report = {"device": args.device, "N": N, "reps": args.reps, "p_cross": float(P_CROSS)}

    print("=== C2Q2C SINGLE-QUBIT BENCHMARK (classical adversary) ===")
    print(f"device={args.device}  N={N}  reps={args.reps}  crossover p*={P_CROSS:.4f}\n")

    # ---- 1. Circuit check -------------------------------------------------
    grid_check = np.linspace(0, 1, 201)
    max_err = {}
    for enc in ENCODINGS:
        errs = [abs(runner.prob1(float(theta_of(p, enc))) - float(target_prob(p, enc)))
                for p in grid_check]
        max_err[enc] = float(max(errs))
    report["circuit_max_abs_prob_error"] = max_err
    print("[1] Circuit check: max |P_circuit(1) - P_target(1)| over 201 p-values")
    for enc in ENCODINGS:
        print(f"    {enc:<6} {max_err[enc]:.2e}")

    # ---- 2. Distribution equality: circuit vs Binomial adversary ----------
    p_grid = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
    eq = {enc: {"p": p_grid.tolist(), "pvalue": [], "rmse_circuit": []} for enc in ENCODINGS}
    for enc in ENCODINGS:
        for p in p_grid:
            th = float(theta_of(p, enc))
            bits = runner.sample_bits(th, N * args.reps).reshape(args.reps, N)
            k_circ = bits.sum(axis=1)
            k_adv = rng.binomial(N, float(target_prob(p, enc)), size=args.reps)
            eq[enc]["pvalue"].append(chi2_homogeneity(k_circ, k_adv))
            est = decode(k_circ, N, enc)
            eq[enc]["rmse_circuit"].append(float(np.sqrt(np.mean((est - p) ** 2))))
        eq[enc]["pvalue_holm"] = holm_adjust(eq[enc]["pvalue"]).tolist()

    # Negative control: adversary with a deliberately wrong probability (+0.005)
    neg = []
    for p in (0.2, 0.5, 0.8):
        th = float(theta_of(p, "c2q2c"))
        k_circ = runner.sample_bits(th, N * args.reps).reshape(args.reps, N).sum(axis=1)
        k_bad = rng.binomial(N, min(p + 0.005, 1.0), size=args.reps)
        neg.append(chi2_homogeneity(k_circ, k_bad))
    report["equality"] = eq
    report["negative_control_pvalues"] = neg

    print("\n[2] Distribution equality, circuit vs Binomial adversary (Holm-adjusted p-values)")
    for enc in ENCODINGS:
        adj = np.array(eq[enc]["pvalue_holm"])
        print(f"    {enc:<6} min adj-p = {adj.min():.3f}   rejected at 0.05: {int((adj < 0.05).sum())}/{len(adj)}")
    print(f"    negative control (adversary off by +0.005): p = {[f'{x:.1e}' for x in neg]}  (small = test has power)")

    # ---- 3. RMSE vs p ------------------------------------------------------
    p_fine = np.linspace(0.0, 1.0, 101)
    rmse_emp = {enc: empirical_rmse(p_fine, N, enc, args.sweep_reps, rng) for enc in ENCODINGS}
    rmse_th = {enc: rmse_theory(p_fine, N, enc) for enc in ENCODINGS}
    wins = p_fine[(rmse_emp["c2q2c"] < rmse_emp["angle"])]
    report["c2q2c_empirical_win_region"] = [float(wins.min()), float(wins.max())] if len(wins) else None

    # ---- 4. Uniform-p average ---------------------------------------------
    M = 400_000
    p_u = rng.uniform(0, 1, M)
    sq = {}
    for enc in ENCODINGS:
        k = rng.binomial(N, target_prob(p_u, enc))
        sq[enc] = (decode(k, N, enc) - p_u) ** 2
    ratio = np.sqrt(sq["c2q2c"].mean() / sq["angle"].mean())
    boots = []
    for _ in range(300):
        idx = rng.integers(0, M, M)
        boots.append(np.sqrt(sq["c2q2c"][idx].mean() / sq["angle"][idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    report["uniform_ratio"] = {"measured": float(ratio), "ci95": [float(lo), float(hi)],
                               "theory_pi_over_sqrt6": float(np.pi / np.sqrt(6))}
    print(f"\n[3] Uniform p average, N={N}: RMSE(C2Q2C)/RMSE(angle) = {ratio:.4f} "
          f"(95% CI {lo:.4f}-{hi:.4f}); theory pi/sqrt(6) = {np.pi/np.sqrt(6):.4f}")

    # Finite-N convergence of the ratio toward the asymptotic pi/sqrt(6)
    conv = {}
    for n_c in (64, 256, 1024, 4096, 16384, 65536):
        p_c = rng.uniform(0, 1, 1_000_000)
        e_a = decode(rng.binomial(n_c, target_prob(p_c, "angle")), n_c, "angle") - p_c
        e_c = decode(rng.binomial(n_c, target_prob(p_c, "c2q2c")), n_c, "c2q2c") - p_c
        conv[n_c] = float(np.sqrt(np.mean(e_c**2) / np.mean(e_a**2)))
    report["uniform_ratio_vs_N"] = conv
    print("    ratio vs N (-> pi/sqrt(6) asymptotically): "
          + "  ".join(f"N={k}: {v:.4f}" for k, v in conv.items()))

    tails = (p_fine < P_CROSS) | (p_fine > 1 - P_CROSS)
    for enc in ENCODINGS:
        report[f"mean_rmse_tails_{enc}"] = float(rmse_emp[enc][tails].mean())
        report[f"mean_rmse_middle_{enc}"] = float(rmse_emp[enc][~tails].mean())
    print(f"    tail region p<{P_CROSS:.3f} or p>{1-P_CROSS:.3f}: mean RMSE  "
          f"angle={report['mean_rmse_tails_angle']:.5f}  c2q2c={report['mean_rmse_tails_c2q2c']:.5f}")
    print(f"    middle region:                     mean RMSE  "
          f"angle={report['mean_rmse_middle_angle']:.5f}  c2q2c={report['mean_rmse_middle_c2q2c']:.5f}")

    # ---- 5. Shot scaling ---------------------------------------------------
    shot_list = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    p_probe = (0.05, 0.5)
    scaling = {}
    for pp in p_probe:
        for enc in ENCODINGS:
            scaling[f"{enc}@{pp}"] = [float(empirical_rmse(np.array([pp]), int(n), enc, 20000, rng)[0])
                                      for n in shot_list]
    report["shot_scaling"] = {"shots": shot_list.tolist(), **scaling}

    # ---- Verdict -----------------------------------------------------------
    equal_ok = all(min(eq[e]["pvalue_holm"]) >= 0.05 for e in ENCODINGS)
    neg_ok = all(x < 1e-3 for x in neg)
    report["verdict"] = {
        "classical_adversary_reproduces_distribution": bool(equal_ok),
        "negative_control_detected": bool(neg_ok),
        "advantage_claim": "REJECTED (single-qubit sampling is classically simulable in O(1) per shot)"
        if equal_ok and neg_ok else "UNVERIFIED (equality test inconclusive)",
        "accuracy_claim": "C2Q2C is NOT more accurate on average; it has lower variance only for "
                          f"p<{P_CROSS:.3f} or p>{1-P_CROSS:.3f}",
    }
    print("\n[VERDICT]")
    for k_, v in report["verdict"].items():
        print(f"    {k_}: {v}")

    # ---- Figure ------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axs[0, 0]
    for enc in ENCODINGS:
        ax.plot(p_fine, rmse_th[enc], "--", color=COLOR[enc], lw=1.5, label=f"{enc} theory")
        ax.plot(p_fine, rmse_emp[enc], "-", color=COLOR[enc], lw=1.0, alpha=0.8, label=f"{enc} Binomial adversary")
        ax.plot(p_grid, eq[enc]["rmse_circuit"], "o", color=COLOR[enc], ms=5, mfc="none",
                label=f"{enc} circuit samples")
    ax.axvspan(0, P_CROSS, color="red", alpha=0.07)
    ax.axvspan(1 - P_CROSS, 1, color="red", alpha=0.07)
    ax.set_title(f"(a) RMSE vs p   (N={N}; shaded: C2Q2C lower variance)")
    ax.set_xlabel("p")
    ax.set_ylabel("RMSE of recovered p")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axs[0, 1]
    pf = np.linspace(0.002, 0.998, 500)
    for enc in ENCODINGS:
        ax.plot(pf, fisher_per_shot(pf, enc), color=COLOR[enc], lw=2, label=LABEL[enc])
    ax.axvline(P_CROSS, color="k", ls=":", lw=1)
    ax.axvline(1 - P_CROSS, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(3, 1000)
    ax.set_title(f"(b) Fisher information per shot   (crossover p={P_CROSS:.3f}, {1-P_CROSS:.3f})")
    ax.set_xlabel("p")
    ax.set_ylabel("I(p)  (higher = lower variance)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[1, 0]
    styles = {0.05: "-", 0.5: "--"}
    for pp in p_probe:
        for enc in ENCODINGS:
            ax.loglog(shot_list, scaling[f"{enc}@{pp}"], styles[pp], color=COLOR[enc], marker="o", ms=3,
                      label=f"{enc}, p={pp}")
    ref = scaling["angle@0.5"][0] * np.sqrt(shot_list[0] / shot_list)
    ax.loglog(shot_list, ref, ":", color="gray", lw=1.5, label="∝ 1/√N")
    ax.set_title("(c) Shot scaling (Binomial adversary)")
    ax.set_xlabel("shots N")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    ax = axs[1, 1]
    x = np.arange(len(p_grid))
    w = 0.38
    for i, enc in enumerate(ENCODINGS):
        ax.bar(x + (i - 0.5) * w, eq[enc]["pvalue"], w, color=COLOR[enc], label=enc)
    ax.axhline(0.05, color="k", ls="--", lw=1, label="α = 0.05")
    ax.axhline(0.05 / len(p_grid), color="gray", ls=":", lw=1.2, label=f"Bonferroni α/{len(p_grid)}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in p_grid], fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.set_title("(d) Circuit vs Binomial adversary: raw chi-square p-values\n"
                 "(above threshold = indistinguishable; Holm-adjusted all ≥ 0.05)")
    ax.set_xlabel("p")
    ax.set_ylabel("p-value")
    ax.legend(fontsize=8, ncol=4, loc="upper center")

    fig.suptitle("C2Q2C vs angle encoding: single qubit, noiseless simulator, classical adversary", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.out, "single_qubit_benchmark.png")
    fig.savefig(png, dpi=150)

    path = os.path.join(args.out, "single_qubit_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {png}\nWrote {path}")


if __name__ == "__main__":
    main()
