# C2Q2C-AF: adversarial classical simulation toolkit (preliminary, small scale)

Everything here is a **simulator-only, small-scale** study (1-24 qubits). It makes **no quantum-advantage claim**, and it does
not touch the BlueQubit challenge circuits. The peaked-circuit family is a simplified cousin of the published construction:
single-qubit R/P layers, swap transformations fused into gates, gauge scrambling, optional 2->3-gate patch re-synthesis.
Every attack's cost is a bond dimension / candidate count / wall time; peak recovery alone is not used as the criterion.

## Environment used
Python 3.12.3, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8, PennyLane 0.45.1 (only for the VQE and single-qubit scripts).
Single CPU core. Wall times are machine dependent; all other fields are deterministic given the seeds.

## Artifacts, commands, seeds
| artifact | command (run in this directory, `--out DIR` for a clean directory) | seeds / parameters |
|---|---|---|
| `report.json`, `convergence.png` | `python c2q2c_vqe_harness.py --seeds 20 --out DIR` | seeds 0..19 |
| `single_qubit_report.json`, `single_qubit_benchmark.png` | `python c2q2c_single_qubit_benchmark.py --out DIR` | seed 7, N=1024, 1000 reps |
| `peaked_attack_N16.json` | `python c2q2c_peaked_mirror_attack.py --N 16 --depth 6 --Ks 0 1 2 3 4 5 6 8 12 --seeds 5 --chi-mirror 128 --chi-ladder 16 32 64 128 256 --direct-seeds 2 --out DIR` | instance seed 1000*K+s |
| `peaked_attack_N20.json` | `... --N 20 --depth 6 --Ks 0 2 4 8 12 --seeds 6 --chi-mirror 128 --chi-ladder 4 8 16 32 64 --direct-seeds 3` | |
| `peaked_attack_N24.json` | `... --N 24 --depth 6 --Ks 0 4 12 --seeds 4 --chi-mirror 128 --chi-ladder 4 8 16 32 64 --direct-seeds 2` | |
| `peaked_attack_cost_N*.png` | `python plot_peaked_attack.py peaked_attack_N16.json peaked_attack_cost_N16.png` (same for N20, N24) | |
| `scaling_summary.md` | `python summarize_peaked_attack.py scaling_summary.md peaked_attack_N16.json peaked_attack_N20.json peaked_attack_N24.json` | |
| `unswap_candidates.json` | `python compare_unswap_candidates.py unswap_candidates.json` | N in {16,20}, K in {4,12}, 5 seeds |
| `statevector_check_comparisons.json` | `python verify_statevector_comparisons.py statevector_check_comparisons.json` | the 28 instances of `unswap_candidates.json` (20) and `local4_stress.json` (8) |
| `local4_stress.json` | `python stress_local4.py local4_stress.json` (resume missing rows: `python stress_local4_resume.py DONE.json OUT.json`; **`OUT.json` holds only the newly produced rows, not a merged data set**) | N=16, K in {24,48}, 4 rows per K; row field `seed` is the index 0-3, the instance seed is `9000 + 100*K + seed` |
| `patch_resynthesis.json`, `patch_resynthesis_convergence.png` | `python patch_resynthesis_feasibility.py --out DIR` | seed 2024, 8 targets, distance tol 1e-9 |
| `patch_attack_K0.json`, `patch_attack_Kswap.json`, `patch_attack_cost.png` | `python c2q2c_patch_obfuscation_attack.py --N 16 --m 48 --Ps 0 2 4 6 --seeds 5 --K-swap 4 --out DIR` then `python plot_patch_attack.py patch_attack_K0.json patch_attack_cost.png` | adjacent patches |
| `gap_patch_unswap_K0.json` | `python c2q2c_patch_obfuscation_attack.py --gap --gap-K 0 --N 16 --m 48 --seeds 8 --attackers oracle,oracle_local4,search_commute,commute_local4 --eval-limit 2000 --time-limit 120 --fname gap_patch_unswap_K0.json --out DIR` | seeds 61000..61007, 2 gap patches |
| `gap_patch_unswap_K4.json` | same with `--gap-K 4`, preferably `--time-limit 600` (stored from two chunks `--seeds 4 --seed0 61000` and `--seeds 4 --seed0 61004`) | slow rows: seeds 61002, 61005 (`search_commute`) |
| `gap_patch_unswap_K12.json` | same with `--gap-K 12 --seeds 4 --attackers oracle_local4,commute_local4` | seeds 61000..61003 |
| `gap_patch_K0.json` (**legacy**) | produced by direct `run_gap` calls with `search_small,search_le3,search_wide` and a 90 s wall limit (no eval limit) | seeds 61000..61007 |

`--gap` ignores `--Ps` and `--K-swap`; use `--gap-patches` (default 2) and `--gap-K` (default 0).

## Verification
* `python verify_claims.py .` recomputes every number quoted in the application from the JSON files (71 checks; exit code 1 on any failure).
* `python check_rerun.py REFERENCE.json RERUN.json [RERUN2.json ...]` is a strict, two-way reproducibility check (`--selftest` runs 25 built-in
  cases, each of which must be caught or, where stated, accepted). It rejects rows without identity fields (N, K, seed; or n_gates, pattern),
  and reports duplicate keys, reference rows missing from the re-runs (unless `--subset`), re-run rows without a reference, missing or extra
  fields at any depth (extra fields only if named in `--allow-extra`), type changes, a `config`/`args` block that is missing or different,
  and top-level blocks (e.g. `direct`) missing from a single re-run. Wall-clock fields and traces are ignored. A `--subset` result is
  partial coverage and must not be described as a full reproduction.
* Deliberately changed settings: `--ignore NAME[,NAME]` ignores those keys at any depth (row fields and `params`), `--ignore-config` only the
  top-level `config`/`args` block; it does **not** hide a row-level field such as `time_limit` (covered by a self-test). The K=4 comparison
  was run as `python check_rerun.py gap_patch_unswap_K4.json RERUN.json --ignore time_limit,time_limit_soft,seed0`; without the ignore list the
  checker reports exactly 9 issues (3 rows x those three settings) and no result field. A `TIMEOUT` versus `OK` difference is never ignored.
* `python check_determinism.py` is an older helper for the peaked sweep (matches on (K, seed)).
* `python verify_statevector_comparisons.py statevector_check_comparisons.json` checks the 28 instances of the unswap comparison and the
  stress test against exact statevectors (28/28; max |peak-weight error| 3.1e-15). `compare_unswap_candidates.py` and `stress_local4.py`
  do not check their instances themselves. The peaked sweeps (87 instances = 45 for N=16, 30 for N=20, 12 for N=24) and the gap-patch and
  adjacent-patch experiments check every instance inside their own scripts.
* `SHA256SUMS` holds the hashes of every file in this directory (`sha256sum -c SHA256SUMS`).

## Re-run status (strict `check_rerun.py`, final code)
| file | re-run coverage | result |
|---|---|---|
| `peaked_attack_N16.json` | 45/45 rows (three chunks) | 0 issues (`evals` allowed: field added after the reference run) |
| `peaked_attack_N20.json` | 9/30 rows | rows match; the `direct` baseline was not re-run (flagged by the checker) |
| `peaked_attack_N24.json` | 1/12 rows | rows match; the `direct` baseline was not re-run |
| `local4_stress.json` | 8/8 rows: 6 by `stress_local4.py` (cut off by the sandbox time limit), the last 2 by an equivalent loop with the same seeds that was not saved at the time; that loop is now shipped as `stress_local4_resume.py` and was re-run for one row (K=48, seed 3) against the reference with 0 issues | 0 issues |
| `gap_patch_unswap_K0.json`, `..._K12.json` | 8/8 and 4/4 rows, independent fresh runs | 0 issues (the stored files are themselves re-runs of earlier runs) |
| `gap_patch_unswap_K4.json` | 8/8 rows, fresh runs merged from three chunks | 0 issues after ignoring `time_limit`, `time_limit_soft`, `seed0` (deliberately changed, see below) |
| `gap_patch_K0.json` (legacy) | 3/8 rows | 0 issues (later-added fields allowed); the other rows include wall-time TIMEOUTs |
| adjacent-patch attack | 9 oracle rows regression-checked after a generator edit | not a full re-run |

**Determinism caveat.** Results are deterministic given the seeds, except where the *soft wall-clock limit* binds. In the K=4 gap
experiment the `search_commute` run (no unswapping, many forced steps at the bond cap) took 111 s on seed 61005 and 45 s on seed 61002
against a 120 s soft limit; on a slower run it hit the limit and returned `TIMEOUT`. Reproduce K=4 with `--time-limit 600`
(the fresh comparison above used 600 s for seeds 61005-61007). The candidate budget `eval_limit` (2000) never bound.

## NOT independently re-run
the full re-synthesis scan (a re-run timed out in the sandbox; the stored file comes from one run of the final script), the full adjacent-patch
experiment, `unswap_candidates.json`, the remaining N=20 and N=24 rows and their direct baselines, and the VQE and single-qubit outputs (original runs).

## Search settings that limit what a failure means
* Bond cap `chi` = 64 (gap experiments), 128 (peaked sweeps); SVD cutoff 1e-10. A run that hits the cap is a lower bound on the needed bond.
* Commutation-aware search: at most 8 seed gates per side (`max_seeds`) and groups of at most 10 gates (`max_total`). A failure there
  means the search policy did not find the alignment, not that none exists.
* `local4` unswapping = greedy transpositions of output wires among the wires of the group just absorbed; the largest wire set actually
  used is stored as `max_wire_set` (2-3 for oracle alignment, up to 4 for the commute search in these runs). The commute search also tries
  groups whose right and left wire sets differ by up to two wires when unswapping is on.
* Candidate budget `eval_limit` = 2000 (counts every MPS candidate evaluation, unswap trials included); `time_limit` is a soft limit.
  No gap run hit either limit.
* All attackers know where the two halves of the identity block meet. `oracle*` additionally know the patch boundaries.
* `alignment_correct` (exact match with the true gate groups) is meaningful only without unswapping; with unswapping the attacker may
  pair gates differently and still collapse the operator (`final_profile_excess` <= 0.5).

## Known limitations
N <= 24 (gap experiments N=16), 4-8 instances per setting, single re-synthesis family (2->3 gates on the same three wires),
no lossy patches, no entangling R/P layers, no unknown-middle attacker. Results are feasibility evidence, not evidence of
hardness or of any advantage at scale.
