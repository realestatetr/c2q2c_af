#!/usr/bin/env python3
"""Plot attack cost vs number of swap transformations from peaked_attack_N*.json."""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "peaked_attack_cost.png"
D = json.load(open(path))
rows, direct, args = D["rows"], D["direct"], D["args"]
N, cap = args["N"], args["chi_mirror"]
Ks = sorted({r["K"] for r in rows})


def col(attack, key, K):
    return np.array([r[attack][key] for r in rows if r["K"] == K], dtype=float)


fig, axs = plt.subplots(1, 3, figsize=(17, 5))

# (a) bond dimension vs K
ax = axs[0]
for attack, color, label in (("naive", "#d62728", "naive mirror-MPO"),
                             ("greedy", "#2ca02c", "mirror-MPO + greedy unswapping")):
    med = [np.median(col(attack, "max_bond", K)) for K in Ks]
    lo = [np.percentile(col(attack, "max_bond", K), 25) for K in Ks]
    hi = [np.percentile(col(attack, "max_bond", K), 75) for K in Ks]
    ax.plot(Ks, med, "o-", color=color, label=label)
    ax.fill_between(Ks, lo, hi, color=color, alpha=0.15)
    for K in Ks:
        pts = col(attack, "max_bond", K)
        ax.scatter([K] * len(pts), pts, s=10, color=color, alpha=0.4)
chis = [d["chi_break"] for d in direct if d["chi_break"] is not None]
if chis:
    ax.axhline(np.median(chis), color="gray", ls="--", label=f"direct MPS needs chi ≈ {int(np.median(chis))}")
ax.axhline(2 ** (N // 2), color="k", ls=":", lw=1, label=f"state saturation 2^(N/2) = {2 ** (N // 2)}")
ax.axhline(cap, color="red", ls=":", lw=1, alpha=0.6)
ax.text(Ks[-1], cap * 1.08, f"bond cap {cap}", ha="right", fontsize=8, color="red")
ax.set_yscale("log", base=2)
ax.set_xlabel("number of swap transformations K (obfuscation strength)")
ax.set_ylabel("max bond dimension reached")
ax.set_title(f"(a) Attack cost vs obfuscation  (N={N}, {args['depth']} layers/half)")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3, which="both")

# (b) success rate
ax = axs[1]
for attack, color in (("naive", "#d62728"), ("greedy", "#2ca02c")):
    rate = [np.mean(col(attack, "success", K)) for K in Ks]
    ax.plot(Ks, rate, "o-", color=color, label=attack)
capped = [np.mean(col("naive", "capped", K)) for K in Ks]
ax.bar(Ks, capped, width=0.5, color="#d62728", alpha=0.2, label="naive: fraction hitting bond cap")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("K")
ax.set_ylabel("fraction of instances")
ax.set_title("(b) Peak recovery rate and cap hits")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) wall time
ax = axs[2]
for attack, color in (("naive", "#d62728"), ("greedy", "#2ca02c")):
    med = [np.median(col(attack, "time", K)) for K in Ks]
    ax.plot(Ks, med, "o-", color=color, label=attack)
ax.set_yscale("log")
ax.set_xlabel("K")
ax.set_ylabel("median wall time (s)")
ax.set_title("(c) Wall time")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle("Toy peaked-circuit family (swap + gauge obfuscation only; single-qubit R/P layers): "
             "attack cost vs obfuscation strength", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
