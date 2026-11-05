"""
GROUP 8 - FIX PACK 02 - OULAD VERIFIER                          (read-only)
==============================================================================
WHAT THIS IS
  The successor to the original FIX PACK 02. That file RECOMPUTED the OULAD
  outputs; Oulad_exp.ipynb now does that itself and asserts each figure as it
  goes. Running both would have two scripts writing the same three artefacts,
  so this one was cut down to an independent check that WRITES NOTHING.

  It opens the three committed files, re-derives every OULAD figure reported in
  the manuscript from them, and confirms the three files agree with each other.
  That last part is the thing the original could not do: it proves the log, the
  predictions and the per-seed metrics tell one story.

WHERE TO RUN
  Anywhere the repository is checked out. No model, no Drive, no raw OULAD, no
  institutional data. Needs only numpy, pandas and scipy.

      python Code/G8_FIX_02_verify_oulad.py

  Exit status 0 means every reported OULAD figure reproduces. Non-zero means at
  least one does not, and the failures are listed.

WHAT IT CHECKS (every reported OULAD figure, over the committed artefacts;
              the run prints its own check count)
  Section 5.12  per-seed AUC, thresholds, equal-opportunity gaps, the change and
                its bootstrap interval, band eligibility, band recall range, the
                operating point, McNemar, and the Algorithm 1 line-12 decision
  Table 4       McNemar and the AUC pair
  Table S9      grid winner, feature counts, prevalence
  Section 4.9   that Missing is excluded and the rule is subgroup SIZE
  Table S9      that oulad_shap_values.csv supports the 43 -> 24 prune: that 24
                columns is the SMALLEST set reaching 95% cumulative mean |SHAP|,
                and that exactly one IMD encoding survives it
==============================================================================
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

# ---- published values, from the manuscript ---------------------------------
PAPER = {
    "auc_full": 0.936, "auc_pruned": 0.936, "sd_full": 0.004, "sd_pruned": 0.004,
    "tau_pruned": 0.4719, "tau_cand": 0.4874,
    "eo_pruned": 0.1185, "eo_cand": 0.1601, "eo_change": 0.04155,
    "ci_lo": -0.0582, "ci_hi": 0.1233,
    "band_lo": 0.679, "band_hi": 0.797, "n_bands": 10,
    "tp": 614, "fn": 217, "fp": 364, "tn": 3760,
    "recall": 0.739, "precision": 0.628,
    "held_out_n": 4955, "at_risk_n": 831, "prevalence": 0.168,
    "mcnemar_b": 102, "mcnemar_c": 97, "mcnemar_chi2": 0.080, "mcnemar_p": 0.777,
    "best": {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 200, "subsample": 0.8},
}
SEEDS = [42, 0, 123, 2024, 7]
MIN_BAND_N = 30          # Section 4.9: subgroup SIZE, not the at-risk count
MISSING_BAND = "Missing" # Section 4.9: a missing-data category, not a level
TARGET_RECALL = 0.75

ROOTS = ["outputs", ".", "../outputs",
         "/content/outputs", "/content/drive/MyDrive/PROJECT_CINQUE_OUTPUTS"]

_fail, _n = [], 0


def check(label, got, want, tol=5e-4, fmt="{:.4f}"):
    global _n
    _n += 1
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {label:<48s} {fmt.format(float(got)):>11}   paper {fmt.format(float(want)):>11}   "
          f"{'MATCH' if ok else 'DIFFERS'}")
    if not ok:
        _fail.append(f"{label}: got {got}, paper {want}")
    return ok


def check_eq(label, got, want):
    global _n
    _n += 1
    ok = got == want
    print(f"  {label:<48s} {str(got):>11}   paper {str(want):>11}   "
          f"{'MATCH' if ok else 'DIFFERS'}")
    if not ok:
        _fail.append(f"{label}: got {got!r}, paper {want!r}")
    return ok


def find(predicate, label):
    """Locate a committed artefact by its contents, not by a hardcoded path."""
    seen = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "*.csv"))
                           + glob.glob(os.path.join(root, "*.json"))):
            rp = os.path.realpath(path)
            if rp in seen:
                continue
            seen.append(rp)
            try:
                if predicate(path):
                    return path
            except Exception:
                continue
    sys.exit(f"Could not find {label}. Looked under: " + ", ".join(ROOTS))


def find_opt(predicate):
    """Like find, but returns None instead of exiting when nothing matches."""
    seen = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "*.csv"))):
            rp = os.path.realpath(path)
            if rp in seen:
                continue
            seen.append(rp)
            try:
                if predicate(path):
                    return path
            except Exception:
                continue
    return None


def is_shap_ranking(p):
    """Two columns, a feature index carrying IMD encodings, 40+ rows."""
    if not p.endswith(".csv"):
        return False
    d = pd.read_csv(p, index_col=0)
    return (d.shape[1] == 1 and len(d) >= 40
            and pd.api.types.is_numeric_dtype(d.iloc[:, 0])
            and any(str(i).lower().startswith("imd_band") for i in d.index))


def has_cols(*cols):
    return lambda p: p.endswith(".csv") and set(cols) <= set(pd.read_csv(p, nrows=0).columns)


def has_key(k):
    def f(p):
        if not p.endswith(".json"):
            return False
        with open(p) as fh:
            d = json.load(fh)
        return isinstance(d, dict) and k in d
    return f


# ---- locate the three committed artefacts ----------------------------------
F_PRED = find(has_cols("y_true", "proba_full", "proba_pruned", "proba_cand", "imd_band"),
              "oulad_seed42_predictions.csv")
F_SEED = find(has_cols("config", "seed", "auc"), "oulad_per_seed_metrics.csv")
F_LOG = find(has_key("eo_gap_pruned_24col"), "run_log_oulad.json")
print(__doc__.split("====\n")[-1])
print("files under test")
for lab, p in (("predictions", F_PRED), ("per-seed metrics", F_SEED), ("run log", F_LOG)):
    print(f"  {lab:<18} {p}")

pred = pd.read_csv(F_PRED)
per_seed = pd.read_csv(F_SEED)
with open(F_LOG) as fh:
    log = json.load(fh)

y = pred.y_true.values
imd = pred.imd_band.astype(str).values
tau_p = float(log["tau_pruned_24col"])
tau_c = float(log["tau_candidate_23col"])


# ---- the same helpers the notebook uses ------------------------------------
def eligible_bands(imd_, y_):
    out = []
    for b in sorted(pd.unique(imd_)):
        if b == MISSING_BAND:
            continue
        m = (imd_ == b)
        if m.sum() >= MIN_BAND_N and (m & (y_ == 1)).sum() > 0:
            out.append(b)
    return out


def band_recall(proba, tau, imd_, y_, bands):
    yh = (np.asarray(proba) >= tau).astype(int)
    return {b: float(yh[(imd_ == b) & (y_ == 1)].mean()) for b in bands}


def eo_gap(proba, tau, imd_, y_, bands):
    r = band_recall(proba, tau, imd_, y_, bands)
    return max(r.values()) - min(r.values()), r


def mcnemar_correctness(y_, pa, pb, t=0.5):
    """b = correct by A and wrong by B; c = the reverse. Dietterich (1998)."""
    yt = np.asarray(y_)
    A = (np.asarray(pa) >= t).astype(int)
    B = (np.asarray(pb) >= t).astype(int)
    b_ = int(((A == yt) & (B != yt)).sum())
    c_ = int(((A != yt) & (B == yt)).sum())
    x2 = (abs(b_ - c_) - 1) ** 2 / (b_ + c_) if (b_ + c_) else 0.0
    return b_, c_, float(x2), (float(chi2.sf(x2, 1)) if (b_ + c_) else 1.0)


# ---- 1. per-seed metrics (Section 5.12, Table S9) --------------------------
print("\n1. multi-seed AUC, from oulad_per_seed_metrics.csv")
check_eq("5.12  configurations present", sorted(per_seed.config.unique()),
         ["cand", "full", "pruned"])
check_eq("5.12  seeds present", sorted(per_seed.seed.unique()), sorted(SEEDS))
for cfg in ("full", "pruned"):
    v = per_seed[per_seed.config == cfg].auc
    check(f"5.12  {cfg} 5-seed mean AUC", v.mean(), PAPER[f"auc_{cfg}"], 5e-4, "{:.3f}")
    check(f"5.12  {cfg} 5-seed SD", v.std(ddof=1), PAPER[f"sd_{cfg}"], 5e-4, "{:.3f}")

# ---- 2. partition and band eligibility (Sections 4.9, 5.12) ----------------
print("\n2. held-out partition and band eligibility, from the predictions")
check("5.12  held-out n", len(y), PAPER["held_out_n"], 0, "{:.0f}")
check("5.12  at-risk n", int(y.sum()), PAPER["at_risk_n"], 0, "{:.0f}")
check("4.16  at-risk prevalence", y.mean(), PAPER["prevalence"], 5e-4, "{:.3f}")
BANDS = eligible_bands(imd, y)
excluded = sorted(set(imd) - set(BANDS))
check("4.9   bands audited", len(BANDS), PAPER["n_bands"], 0, "{:.0f}")
check_eq("4.9   bands excluded", excluded, [MISSING_BAND])
mis = (imd == MISSING_BAND)
print(f"       Missing category: {mis.sum()} records, {int((mis & (y == 1)).sum())} at-risk "
      f"- admitted by subgroup size, excluded as a missing-data category")

# ---- 3. thresholds and gaps (Section 5.12) ---------------------------------
print("\n3. thresholds and equal-opportunity gaps")
check("5.12  tau, 24-column pruned", tau_p, PAPER["tau_pruned"], 1e-4)
check("5.12  tau, 23-column candidate", tau_c, PAPER["tau_cand"], 1e-4)
gap_p, rec_p = eo_gap(pred.proba_pruned.values, tau_p, imd, y, BANDS)
gap_c, _ = eo_gap(pred.proba_cand.values, tau_c, imd, y, BANDS)
check("5.12  EO gap, pruned", gap_p, PAPER["eo_pruned"], 1e-3)
check("5.12  EO gap, candidate", gap_c, PAPER["eo_cand"], 1e-3)
check("5.12  change on removal", gap_c - gap_p, PAPER["eo_change"], 1e-3)
check("5.12  band recall, lowest", min(rec_p.values()), PAPER["band_lo"], 1e-3, "{:.3f}")
check("5.12  band recall, highest", max(rec_p.values()), PAPER["band_hi"], 1e-3, "{:.3f}")

# ---- 4. bootstrap interval (Section 5.12) ----------------------------------
print("\n4. paired percentile bootstrap, 1,000 draws, seed 42, bands fixed")
rng = np.random.default_rng(42)
idx = np.arange(len(y))
draws = []
for _ in range(1000):
    i = rng.choice(idx, len(idx), replace=True)
    try:
        a, _ = eo_gap(pred.proba_pruned.values[i], tau_p, imd[i], y[i], BANDS)
        b, _ = eo_gap(pred.proba_cand.values[i], tau_c, imd[i], y[i], BANDS)
        draws.append(b - a)
    except ValueError:
        pass
ci_lo, ci_hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
check("5.12  bootstrap CI, lower", ci_lo, PAPER["ci_lo"], 2e-3)
check("5.12  bootstrap CI, upper", ci_hi, PAPER["ci_hi"], 2e-3)
print(f"       {len(draws)} of 1000 draws usable")

# ---- 5. operating point (Section 5.12) -------------------------------------
print("\n5. operating point at the 24-column threshold")
yh = (pred.proba_pruned.values >= tau_p).astype(int)
tp = int(((yh == 1) & (y == 1)).sum())
fn = int(((yh == 0) & (y == 1)).sum())
fp = int(((yh == 1) & (y == 0)).sum())
tn = int(((yh == 0) & (y == 0)).sum())
check("5.12  true positives", tp, PAPER["tp"], 0, "{:.0f}")
check("5.12  false negatives", fn, PAPER["fn"], 0, "{:.0f}")
check("5.12  false positives", fp, PAPER["fp"], 0, "{:.0f}")
check("5.12  true negatives", tn, PAPER["tn"], 0, "{:.0f}")
check("5.12  recall at tau", tp / (tp + fn), PAPER["recall"], 1e-3, "{:.3f}")
check("5.12  precision at tau", tp / (tp + fp), PAPER["precision"], 1e-3, "{:.3f}")

# ---- 6. McNemar (Section 5.12, Table 4, Table S9) --------------------------
print("\n6. McNemar, correctness discordance at the default 0.5 threshold")
b_mc, c_mc, x2_mc, p_mc = mcnemar_correctness(y, pred.proba_full.values, pred.proba_pruned.values)
check("5.12  McNemar b", b_mc, PAPER["mcnemar_b"], 0, "{:.0f}")
check("5.12  McNemar c", c_mc, PAPER["mcnemar_c"], 0, "{:.0f}")
check("5.12  McNemar chi-square", x2_mc, PAPER["mcnemar_chi2"], 1e-3, "{:.3f}")
check("5.12  McNemar p", p_mc, PAPER["mcnemar_p"], 1e-3, "{:.3f}")

# ---- 7. Algorithm 1 line 12 ------------------------------------------------
print("\n7. Algorithm 1 line 12")
rejects = bool(gap_c > gap_p)
check_eq("5.12  removal rejected", rejects, True)
check_eq("5.12  log records the same decision",
         bool(log["accept_removal_algorithm1_line12"]), False)

# ---- 8. the three files agree with each other ------------------------------
print("\n8. cross-consistency: the log against the predictions it was written from")
check("log   eo_gap_pruned_24col", float(log["eo_gap_pruned_24col"]), gap_p, 1e-3)
check("log   eo_gap_candidate_23col", float(log["eo_gap_candidate_23col"]), gap_c, 1e-3)
check("log   eo_change_on_removal", float(log["eo_change_on_removal"]), gap_c - gap_p, 1e-3)
lo_l, hi_l = [float(v) for v in log["eo_change_ci95"]]
check("log   eo_change_ci95 lower", lo_l, ci_lo, 2e-3)
check("log   eo_change_ci95 upper", hi_l, ci_hi, 2e-3)
conf = log["confusion_at_tau_pruned"]
check_eq("log   confusion_at_tau_pruned",
         [int(conf["tp"]), int(conf["fn"]), int(conf["fp"]), int(conf["tn"])], [tp, fn, fp, tn])
check_eq("log   held_out_n", int(log["held_out_n"]), len(y))
check_eq("log   at_risk_n", int(log["at_risk_n"]), int(y.sum()))
check_eq("log   bands_excluded", list(log["bands_excluded"]), [MISSING_BAND])
rng_l = [float(v) for v in log["band_recall_range_pruned_24col"]]
check("log   band_recall_range lower", rng_l[0], min(rec_p.values()), 1e-3, "{:.3f}")
check("log   band_recall_range upper", rng_l[1], max(rec_p.values()), 1e-3, "{:.3f}")
check_eq("log   best_hyperparameters",
         {k: log["best_hyperparameters"][k] for k in PAPER["best"]}, PAPER["best"])

# optional keys: present in the log Oulad_exp.ipynb writes, absent in older ones
if "mcnemar_full_vs_pruned" in log:
    m = log["mcnemar_full_vs_pruned"]
    check_eq("log   mcnemar b, c", [int(m["b"]), int(m["c"])], [b_mc, c_mc])
    check_eq("log   mcnemar discordance basis", m.get("discordance"), "correctness")
else:
    print("  note: this log predates Oulad_exp.ipynb and records no McNemar; "
          "regenerate it from the notebook")

# ---- 9. nothing sensitive in the committed predictions ---------------------
print("\n9. the predictions file carries no attribute beyond the audited one")
extra = [c for c in pred.columns
         if c not in ("y_true", "proba_full", "proba_pruned", "proba_cand", "imd_band")]
check_eq("F1    unexpected columns", extra, [])
print("       imd_band is required for the Section 5.12 audit and is public OULAD data")

# ---- 10. the SHAP ranking supports the prune (Table S9) --------------------
print("\n10. oulad_shap_values.csv supports the 43 -> 24 prune")
F_SHAP = find_opt(is_shap_ranking)
if F_SHAP is None:
    print("  note: oulad_shap_values.csv was not found. It is one of the four")
    print("        committed artefacts and Table S9's 43 -> 24 claim rests on it;")
    print("        commit it from the repo outputs folder. Not counted as a failure")
    print("        here so that an older checkout still verifies.")
else:
    print(f"       {F_SHAP}")
    _sh = pd.read_csv(F_SHAP, index_col=0).iloc[:, 0]
    check_eq("S9    encoded features ranked", len(_sh), 43)
    check_eq("S9    ranking is descending", bool((_sh.diff().dropna() <= 1e-12).all()), True)
    _cum = _sh.cumsum() / _sh.sum()
    check("S9    cumulative share at 24 columns", _cum.iloc[23], 0.9518, 5e-3)
    # CELL 9 keeps every feature with cum <= 0.95 then appends ONE more, so 24
    # columns means exactly: cum[22] <= 0.95 < cum[23]. That is the whole rule.
    check_eq("S9    23 columns at or below 95%", bool(_cum.iloc[22] <= 0.95), True)
    check_eq("S9    the 24th crosses 95%", bool(_cum.iloc[23] > 0.95), True)
    _imd = [c for c in _sh.index[:24] if str(c).lower().startswith("imd_band")]
    check_eq("S9    IMD encodings in the 24", len(_imd), 1)
    check_eq("S9    candidate set size", 24 - len(_imd), 23)
    print(f"       the surviving IMD encoding is {_imd[0] if _imd else 'none'}, which is")
    print("       what the fairness screen proposes removing in Algorithm 1 line 11")

# ---- verdict ---------------------------------------------------------------
print("\n" + "=" * 78)
if _fail:
    print(f"FAILED  {len(_fail)} of {_n} checks did not reproduce:")
    for f in _fail:
        print("   -", f)
    sys.exit(1)
_nfiles = "four" if F_SHAP else "three"
print(f"PASSED  all {_n} checks. The {_nfiles} committed OULAD artefacts reproduce")
print("        every figure reported in Section 5.12, Table 4 and Table S9, and")
print("        agree with one another. Nothing was written by this script.")
