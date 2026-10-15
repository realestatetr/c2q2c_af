#!/usr/bin/env python3
"""Strict reproducibility check: compare a stored reference JSON with one or more re-run JSON files.

Usage:
  python check_rerun.py REF.json RERUN1.json [RERUN2.json ...] [options]
  python check_rerun.py --selftest

Rows are keyed by (N, K, seed, P_req) for row lists / {"rows": [...]} files and by (n_gates, pattern) for
{"results": [...]} files (patch re-synthesis). Checked, in both directions:
  * duplicate keys inside any file or across re-run files                -> error
  * reference rows missing from the re-runs                               -> error (unless --subset)
  * re-run rows with no reference row                                     -> error
  * keys present in the reference but not in the re-run (any depth)       -> error
  * keys present in the re-run but not in the reference (any depth)       -> error (unless named in --allow-extra)
  * type changes (bool/str/list/dict), list length changes                -> error
  * numbers: int and float are compared numerically with --rel-tol / --abs-tol
  * top-level 'config' / 'args' blocks are compared too (the output path 'out' is ignored; --ignore-config names keys
    that may differ, e.g. for chunked re-runs); a top-level 'direct' block is compared when a single re-run file is given
Wall-clock fields (time, time_s, gen_time) and diagnostic traces ('trace') are ignored by default (--ignore adds more).
Exit code 0 only if there is no issue. --subset allows the re-run to cover only part of the reference; the
coverage is then printed, and a subset result must not be described as a full reproduction.
"""
import argparse, json, math, os, sys, tempfile

DEFAULT_IGNORE = {"time", "time_s", "gen_time", "trace"}
META_IGNORE = {"out"}


ID_FIELDS = ("N", "K", "seed")
OPT_FIELDS = ("P_req", "P")


def row_key(r, errors, where):
    missing = [k for k in ID_FIELDS if k not in r]
    if missing:
        errors.append(f"{where}: row lacks identity field(s) {missing}")
        return None
    return tuple(r.get(k) for k in ID_FIELDS + OPT_FIELDS)


def pattern_key(r, errors, where):
    if "n_gates" not in r or "pattern" not in r:
        errors.append(f"{where}: row lacks n_gates/pattern")
        return None
    return (r["n_gates"], str(r["pattern"]))


def load_rows(path, errors):
    j = json.load(open(path))
    meta, top_other = None, {}
    if isinstance(j, dict) and "results" in j:
        rows, keyf = j["results"], pattern_key
        meta = j.get("config")
    elif isinstance(j, dict) and "rows" in j:
        rows, keyf, meta = j["rows"], row_key, j.get("args")
        top_other = {k: v for k, v in j.items() if k not in ("rows", "args")}
    elif isinstance(j, list):
        rows, keyf = j, row_key
    else:
        errors.append(f"{path}: unsupported JSON layout")
        return {}, None, {}
    out = {}
    for r in rows:
        k = keyf(r, errors, path)
        if k is None:
            continue
        if k in out:
            errors.append(f"{path}: duplicate key {k!r}")
        out[k] = r
    return out, meta, top_other


def compare(a, b, path, errors, ctx):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in ctx["ignore"]:
                continue
            if k not in b:
                errors.append(f"{path}/{k}: missing in re-run")
            else:
                compare(a[k], b[k], f"{path}/{k}", errors, ctx)
        for k in b:
            if k not in a and k not in ctx["ignore"] and k not in ctx["allow"]:
                errors.append(f"{path}/{k}: extra key in re-run")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(f"{path}: list length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", errors, ctx)
        return
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(a) and num(b):
        if not (a == b or math.isclose(a, b, rel_tol=ctx["rel"], abs_tol=ctx["abs"]) or (a != a and b != b)):
            errors.append(f"{path}: {a!r} vs {b!r}")
        return
    if type(a) is not type(b) or a != b:
        errors.append(f"{path}: {a!r} vs {b!r}")


def run_check(ref_path, rerun_paths, subset=False, allow=(), ignore=(), rel=1e-6, abs_=1e-12, verbose=True, config_ignore=()):
    errors = []
    ctx = dict(ignore=DEFAULT_IGNORE | set(ignore), allow=set(allow), rel=rel, abs=abs_)
    ref, ref_meta, ref_top = load_rows(ref_path, errors)
    merged, metas, tops = {}, [], []
    for p in rerun_paths:
        cur, meta, top = load_rows(p, errors)
        metas.append((p, meta)); tops.append((p, top))
        for k, r in cur.items():
            if k in merged:
                errors.append(f"duplicate re-run key across files: {k!r}")
            merged[k] = r
    for k in sorted(set(ref) - set(merged), key=repr):
        if not subset:
            errors.append(f"{k!r}: reference row missing from re-runs")
    for k in sorted(set(merged) - set(ref), key=repr):
        errors.append(f"{k!r}: re-run row has no reference row")
    matched = 0
    for k in sorted(set(ref) & set(merged), key=repr):
        before = len(errors)
        compare(ref[k], merged[k], repr(k), errors, ctx)
        matched += (len(errors) == before)
    cctx = dict(ctx, ignore=ctx["ignore"] | META_IGNORE | set(config_ignore))
    for p, meta in metas:  # parameters of every re-run file must equal the reference's, except explicitly ignored keys
        name = os.path.basename(p)
        if ref_meta is not None and meta is None:
            errors.append(f"{name}: config block missing in re-run")
        elif ref_meta is None and meta is not None:
            errors.append(f"{name}: config block present only in re-run")
        elif ref_meta is not None:
            sub = []
            compare(ref_meta, meta, "config", sub, cctx)
            errors += [f"{name}: {e}" for e in sub]
    for p, top in tops:
        if len(rerun_paths) == 1:
            for k, v in ref_top.items():
                if k not in top:
                    errors.append(f"{os.path.basename(p)}: top-level block '{k}' missing in re-run")
                else:
                    compare(v, top[k], f"top/{k}", errors, ctx)
    covered = len(set(ref) & set(merged))
    if verbose:
        print(f"reference rows {len(ref)} | re-run rows {len(merged)} | coverage {covered}/{len(ref)}"
              f"{' (SUBSET allowed)' if subset and covered < len(ref) else ''} | fully matching {matched} | issues {len(errors)}")
        for e in errors[:30]:
            print("  ISSUE:", e)
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
    return errors, covered, len(ref)


def selftest():
    d = tempfile.mkdtemp()
    base = [dict(N=16, K=k, seed=s, a=1, b=[1.0, 2.0], c=dict(x=True, time=5.0)) for k in (0, 4) for s in (0, 1)]
    w = lambda name, obj: (json.dump(obj, open(os.path.join(d, name), "w")), os.path.join(d, name))[1]
    ref = w("ref.json", base)
    cases = []
    mut = lambda f: [f(dict(r, c=dict(r["c"]), b=list(r["b"]))) or r for r in base]
    cases.append(("identical (time differs, int vs float)", [dict(r, c=dict(x=True, time=9.0), a=1.0) for r in base], {}, True))
    cases.append(("float inside tolerance", [dict(r, b=[1.0 + 1e-9, 2.0]) for r in base], {}, True))
    cases.append(("missing reference row", base[:-1], {}, False))
    cases.append(("missing row allowed with --subset", base[:-1], dict(subset=True), True))
    cases.append(("duplicate row in re-run", base + [base[0]], {}, False))
    cases.append(("extra row in re-run", base + [dict(base[0], seed=9)], {}, False))
    cases.append(("extra key in re-run", [dict(r, extra=1) for r in base], {}, False))
    cases.append(("extra key allowed", [dict(r, extra=1) for r in base], dict(allow=["extra"]), True))
    cases.append(("missing key in re-run", [{k: v for k, v in r.items() if k != "a"} for r in base], {}, False))
    cases.append(("value change", [dict(r, a=2) for r in base], {}, False))
    cases.append(("bool vs int", [dict(r, c=dict(x=1, time=5.0)) for r in base], {}, False))
    cases.append(("list length change", [dict(r, b=[1.0]) for r in base], {}, False))
    ok = True
    for name, rr, kw, expect_pass in cases:
        errs, _, _ = run_check(ref, [w("rr.json", rr)], verbose=False, **kw)
        good = (len(errs) == 0) == expect_pass
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {'no issue' if not errs else errs[0]}")
    # config blocks and cross-file duplicates
    refd = w("refd.json", dict(args=dict(N=16, out="/a", seeds=5), rows=base, direct=[]))
    errs, _, _ = run_check(refd, [w("rd.json", dict(args=dict(N=16, out="/b", seeds=5), rows=base, direct=[]))], verbose=False)
    good = not errs; ok &= good; print(f"  [{'PASS' if good else 'FAIL'}] config equal except output path: {'no issue' if not errs else errs[0]}")
    errs, _, _ = run_check(refd, [w("rd2.json", dict(args=dict(N=20, out="/b", seeds=5), rows=base, direct=[]))], verbose=False)
    good = bool(errs); ok &= good; print(f"  [{'PASS' if good else 'FAIL'}] changed config is caught: {errs[0] if errs else 'NOT CAUGHT'}")
    errs, _, _ = run_check(ref, [w("c1.json", base[:2]), w("c2.json", base[1:])], verbose=False)
    good = any("duplicate" in e for e in errs); ok &= good; print(f"  [{'PASS' if good else 'FAIL'}] duplicate across re-run chunks is caught")
    errs, cov, n = run_check(ref, [w("c1.json", base[:2]), w("c2.json", base[2:])], verbose=False)
    good = not errs and cov == n; ok &= good; print(f"  [{'PASS' if good else 'FAIL'}] disjoint chunks cover the reference completely")
    def case(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'PASS' if good else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    noid = [{k: v for k, v in r.items() if k != "seed"} for r in base]
    errs, _, _ = run_check(ref, [w("noid.json", noid)], verbose=False)
    case("row without identity field is rejected", any("identity" in e for e in errs))
    errs, _, _ = run_check(refd, [w("nometa.json", dict(rows=base, direct=[]))], verbose=False)
    case("config block missing in re-run is caught", any("config block missing" in e for e in errs))
    errs, _, _ = run_check(refd, [w("nodirect.json", dict(args=dict(N=16, out="/b", seeds=5), rows=base))], verbose=False)
    case("top-level block missing in re-run is caught", any("top-level block" in e for e in errs))
    pr = lambda pats: dict(config=dict(seed=1), results=[dict(n_gates=3, pattern=p, success=1) for p in pats])
    rr0 = w("pr_ref.json", pr([[[0, 1]], [[0, 2]]]))
    errs, _, _ = run_check(rr0, [w("pr_dup.json", pr([[[0, 1]], [[0, 1]]]))], verbose=False)
    case("duplicate (n_gates, pattern) in a re-synthesis file is caught", any("duplicate" in e for e in errs))
    errs, _, _ = run_check(rr0, [w("pr_ok.json", pr([[[0, 1]], [[0, 2]]]))], verbose=False)
    case("distinct patterns pass", not errs)
    # row-level settings (like time_limit) versus status
    rows120 = [dict(r, time_limit=120, att=dict(status="OK", v=1)) for r in base]
    rows600 = [dict(r, time_limit=600, att=dict(status="OK", v=1)) for r in base]
    rowsTO = [dict(r, time_limit=600, att=dict(status="TIMEOUT", v=1)) for r in base]
    r120 = w("r120.json", rows120)
    errs, _, _ = run_check(r120, [w("r600.json", rows600)], verbose=False)
    case("a changed row-level setting is reported by default", bool(errs))
    errs, _, _ = run_check(r120, [w("r600.json", rows600)], verbose=False, config_ignore=["time_limit"])
    case("--ignore-config does NOT hide a row-level field (it only covers config/args)", bool(errs))
    errs, _, _ = run_check(r120, [w("r600.json", rows600)], verbose=False, ignore=["time_limit"])
    case("--ignore time_limit accepts the deliberate change", not errs)
    errs, _, _ = run_check(r120, [w("rTO.json", rowsTO)], verbose=False, ignore=["time_limit"])
    case("TIMEOUT versus OK is still caught when time_limit is ignored", any("status" in e for e in errs))
    print("selftest:", "ALL PASSED" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--allow-extra", default="", help="comma-separated key names allowed to appear only in the re-run")
    ap.add_argument("--ignore", default="", help="comma-separated additional key names to ignore")
    ap.add_argument("--ignore-config", default="", help="comma-separated config/args keys that may differ (e.g. chunked re-runs: Ks,chi_ladder,direct_seeds)")
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("--abs-tol", type=float, default=1e-12)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if len(a.files) < 2:
        ap.error("need REF.json and at least one RERUN.json")
    errs, cov, n = run_check(a.files[0], a.files[1:], a.subset, [x for x in a.allow_extra.split(",") if x],
                             [x for x in a.ignore.split(",") if x], a.rel_tol, a.abs_tol, config_ignore=[x for x in a.ignore_config.split(",") if x])
    sys.exit(1 if errs else 0)
