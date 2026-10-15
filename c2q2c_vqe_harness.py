#!/usr/bin/env python3
"""
C2Q2C VQE Validation Harness
============================
Classical -> Quantum -> Classical pipeline smoke test with a built-in
classical adversary (exact diagonalization) and a multi-seed robustness sweep.

SCOPE / HONESTY NOTE
--------------------
* The Hamiltonian below is a 2-qubit MODEL Hamiltonian (arbitrary units).
  It is NOT a Na-K molecular Hamiltonian and its energies are not Hartree.
* A 2-qubit problem is trivially classically solvable. This harness validates
  the PIPELINE (encoding, execution, optimization, verification, reporting);
  it makes no claim of quantum advantage.

What it does
------------
1. Builds the model Hamiltonian and solves it exactly (classical baseline).
2. Runs a VQE with a hardware-efficient ansatz over many random seeds,
   under several conditions (exact statevector / finite shots, small / large
   initial angles).
3. Re-evaluates every final parameter set noiselessly, so reported errors are
   properties of the parameters, not of a lucky/unlucky shot sample.
4. Classifies each run (ground state / parity local minimum / other) and
   assigns a PASS / FAIL status against a tolerance.
5. Writes a convergence plot (PNG) and a machine-readable report (JSON).

Usage
-----
    python c2q2c_vqe_harness.py                       # default.qubit, 20 seeds
    python c2q2c_vqe_harness.py --seeds 50 --shots 2048
    python c2q2c_vqe_harness.py --device qiskit.aer   # needs pennylane-qiskit
    python c2q2c_vqe_harness.py --out results/        # output directory

Requires: pennylane, matplotlib (pennylane-qiskit only for qiskit devices).
For real IBM hardware, use the access route provided by the program
(IBM Quantum / BlueQubit) and pass the resulting PennyLane device name.
"""

import argparse
import json
import os
import time

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pennylane as qml  # noqa: E402
from pennylane import numpy as np  # noqa: E402

NUM_QUBITS = 2
TOLERANCE = 1.6e-3  # pass threshold on |E_final_noiseless - E_exact| (model units)

# ----------------------------------------------------------------------------
# 1. Model Hamiltonian + classical baseline
# ----------------------------------------------------------------------------
COEFFS = [-0.8, 0.35, 0.12, 0.25]


def build_hamiltonian():
    return qml.Hamiltonian(
        COEFFS,
        [
            qml.Identity(0),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliX(0) @ qml.PauliX(1),
        ],
    )


def exact_baseline(H):
    """Full diagonalization: spectrum, ground energy, Hartree-Fock-like |11> energy."""
    Hm = qml.matrix(H, wire_order=list(range(NUM_QUBITS)))
    evals = onp.linalg.eigvalsh(Hm)
    return {
        "spectrum": [float(x) for x in evals],
        "E_exact": float(evals[0]),
        "E_ref_state_11": float(Hm[3, 3].real),  # <11|H|11>
    }


# ----------------------------------------------------------------------------
# 2. Quantum side: ansatz + QNode factory
# ----------------------------------------------------------------------------
def make_qnode(H, device_name, shots):
    dev = qml.device(device_name, wires=NUM_QUBITS)

    def circuit(params):
        qml.BasisState(onp.array([1, 1]), wires=range(NUM_QUBITS))
        for i in range(NUM_QUBITS):
            qml.RX(params[0, i], wires=i)
            qml.RY(params[1, i], wires=i)
        qml.CNOT(wires=[0, 1])
        for i in range(NUM_QUBITS):
            qml.RZ(params[2, i], wires=i)
        return qml.expval(H)

    qnode = qml.QNode(circuit, dev, interface="autograd")
    if shots is not None:
        if hasattr(qml, "set_shots"):
            qnode = qml.set_shots(qnode, shots=shots)
        else:  # older PennyLane
            dev = qml.device(device_name, wires=NUM_QUBITS, shots=shots)
            qnode = qml.QNode(circuit, dev, interface="autograd")
    return qnode


# ----------------------------------------------------------------------------
# 3. Classical optimization loop (one run)
# ----------------------------------------------------------------------------
def run_one(qnode_opt, qnode_exact, seed, init_scale, steps, stepsize):
    rng = onp.random.default_rng(seed)
    params = np.array(rng.normal(0, init_scale, size=(3, NUM_QUBITS)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)
    trace = []
    for _ in range(steps):
        params, cost = opt.step_and_cost(qnode_opt, params)
        trace.append(float(cost))
    # Noiseless re-evaluation of the final parameters
    e_final = float(qnode_exact(np.array(params, requires_grad=False)))
    return trace, e_final, onp.array(params).tolist()


def classify(e_final, baseline, tol):
    spec = baseline["spectrum"]
    if abs(e_final - spec[0]) <= tol:
        return "ground_state"
    if abs(e_final - spec[1]) <= 5e-2:
        return "parity_local_minimum"  # trapped near the 2nd eigenvalue
    return "other"


# ----------------------------------------------------------------------------
# 4. Sweep + report
# ----------------------------------------------------------------------------
CONDITIONS = [
    # name, shots (None = exact), init_scale, stepsize, steps
    {"name": "exact | init=0.3", "shots": None, "init_scale": 0.3, "stepsize": 0.1, "steps": 100},
    {"name": "exact | init=pi (original)", "shots": None, "init_scale": float(onp.pi), "stepsize": 0.08, "steps": 41},
    {"name": "shots | init=0.3", "shots": "SHOTS", "init_scale": 0.3, "stepsize": 0.1, "steps": 100},
    {"name": "shots | init=pi (original)", "shots": "SHOTS", "init_scale": float(onp.pi), "stepsize": 0.08, "steps": 41},
]


def main():
    ap = argparse.ArgumentParser(description="C2Q2C VQE validation harness")
    ap.add_argument("--device", default="default.qubit")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    H = build_hamiltonian()
    base = exact_baseline(H)
    print("=== C2Q2C VQE VALIDATION HARNESS ===")
    print(f"Device: {args.device} | seeds: {args.seeds} | shots (noisy cond.): {args.shots}")
    print(f"Exact spectrum: {[round(x, 6) for x in base['spectrum']]}")
    print(f"E_exact = {base['E_exact']:.6f} | <11|H|11> = {base['E_ref_state_11']:.6f}\n")

    qnode_exact = make_qnode(H, args.device, None)
    report = {"baseline": base, "tolerance": args.tol, "device": args.device,
              "shots": args.shots, "conditions": []}
    traces_by_cond = {}

    t0 = time.time()
    for cond in CONDITIONS:
        shots = args.shots if cond["shots"] == "SHOTS" else None
        qnode_opt = qnode_exact if shots is None else make_qnode(H, args.device, shots)
        runs = []
        for seed in range(args.seeds):
            trace, e_final, params = run_one(
                qnode_opt, qnode_exact, seed, cond["init_scale"], cond["steps"], cond["stepsize"]
            )
            err = e_final - base["E_exact"]
            runs.append({
                "seed": seed, "E_final_noiseless": e_final, "error": err,
                "class": classify(e_final, base, args.tol),
                "status": "PASS" if abs(err) <= args.tol else "FAIL",
                "params": params, "trace": trace,
            })
        errs = onp.array([r["error"] for r in runs])
        n_pass = sum(r["status"] == "PASS" for r in runs)
        n_trap = sum(r["class"] == "parity_local_minimum" for r in runs)
        summary = {
            "name": cond["name"], "shots": shots, "init_scale": cond["init_scale"],
            "steps": cond["steps"], "stepsize": cond["stepsize"],
            "pass_rate": n_pass / args.seeds, "parity_trap_rate": n_trap / args.seeds,
            "median_error": float(onp.median(errs)), "worst_error": float(errs.max()),
            "best_error": float(errs.min()), "runs": runs,
        }
        report["conditions"].append(summary)
        traces_by_cond[cond["name"]] = [r["trace"] for r in runs]
        print(f"[{cond['name']:<28}] pass {n_pass:>3}/{args.seeds} "
              f"| trapped {n_trap:>3} | median err {onp.median(errs):+.2e} "
              f"| worst err {errs.max():+.2e}")

    # Overall verdict: a condition is only trustworthy if ALL seeds pass.
    for s in report["conditions"]:
        s["verdict"] = "PASS" if s["pass_rate"] == 1.0 else "FAIL"
    report["runtime_s"] = time.time() - t0

    # ---- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, cond in zip(axes.ravel(), CONDITIONS):
        traces = traces_by_cond[cond["name"]]
        for tr in traces:
            ax.plot(tr, lw=0.8, alpha=0.5)
        ax.axhline(base["E_exact"], color="k", ls="--", lw=1.2, label=f"exact {base['E_exact']:.4f}")
        ax.axhline(base["spectrum"][1], color="r", ls=":", lw=1.0,
                   label=f"parity trap {base['spectrum'][1]:.4f}")
        s = next(c for c in report["conditions"] if c["name"] == cond["name"])
        ax.set_title(f"{cond['name']}  (pass {s['pass_rate']*100:.0f}%)", fontsize=10)
        ax.set_xlabel("iteration")
        ax.set_ylabel("optimizer-visible energy (model units)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("C2Q2C VQE convergence across seeds vs exact diagonalization", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.out, "convergence.png")
    fig.savefig(png, dpi=150)

    # ---- Report --------------------------------------------------------------
    path = os.path.join(args.out, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote: {png}\nWrote: {path}")
    print("\nVERDICT per condition (PASS only if all seeds reach tolerance):")
    for s in report["conditions"]:
        print(f"  {s['name']:<30} {s['verdict']}")
    print("\nNOTE: model Hamiltonian, 2 qubits, classically exact -> pipeline validation only.")


if __name__ == "__main__":
    main()
