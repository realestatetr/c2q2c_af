#!/usr/bin/env python3
"""Plot attack cost vs number of patches from patch_attack_K0.json (and K>0 json if given)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open(sys.argv[1])); out = sys.argv[2]
Ps = sorted({r["P"] for r in A})
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
style = {"naive": ("#d62728", "naive index pairing (misaligned)"), "oracle": ("#1f77b4", "oracle-aligned (knows patch blocks)"),
         "search_full": ("#2ca02c", "search-aligned, full move set"), "search_limited": ("#9467bd", "search-aligned, small move set")}
ax = axs[0]
for k, (c, lab) in style.items():
    med = [np.median([r[k]["max_bond"] for r in A if r["P"] == P]) for P in Ps]
    ax.plot(Ps, med, "o-", color=c, label=lab)
ax.axhline(8, color="gray", ls=":", label="identity baseline (8)")
ax.set_yscale("log", base=2); ax.set_xlabel("number of patches P"); ax.set_ylabel("max bond dimension (median of 5)")
ax.set_title("(a) Attack cost vs patches (K=0 swaps)"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
ax = axs[1]
for k in ("search_full", "search_limited"):
    c, lab = style[k]
    med = [np.median([r[k]["evals"] for r in A if r["P"] == P]) for P in Ps]
    ax.plot(Ps, med, "o-", color=c, label=lab)
ax.set_xlabel("number of patches P"); ax.set_ylabel("candidate groups evaluated")
ax.set_title("(b) Cost of finding the alignment"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axs[2]
for k, (c, lab) in style.items():
    med = [np.median([r[k]["time"] for r in A if r["P"] == P]) for P in Ps]
    ax.plot(Ps, med, "o-", color=c, label=lab)
ax.set_yscale("log"); ax.set_xlabel("number of patches P"); ax.set_ylabel("median wall time (s)")
ax.set_title("(c) Wall time"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
fig.suptitle("Re-synthesized patches in the toy family (N=16, 48 gates/half, 5 seeds; peak recovered in every run)")
fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
