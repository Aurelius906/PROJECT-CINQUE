"""
GROUP 8 - FIX PACK 04 - EDUCATOR STUDY OUTPUTS AND CODING RELIABILITY
==============================================================================
INPUTS (keep them OUT of git; only the outputs are committed)
  data_local/responses.csv     one row per analysed participant: participant
                               code, group (A/B), T1-T8, C1-C4, A1-A4, U1-U4,
                               E1-E4, s1..s5_appropriate/_trust/_confidence,
                               s1..s5_quality (consensus 0-2 scores)
  data_local/coding_rounds.csv one row per case-response: item_id, round
                               (cal, r1, r2), coder (1, 2), score (0-2)

WHAT IT FIXES
  5.10  names the two boilerplate participants -> outputs/educator_exclusions.json
  4.10  bootstrap CIs for r (200,000 resamples within condition), 90% CI on the
        raw intervention-quality difference, Fisher exact test for Student 5
  C6    the reliability contradiction (96.4% identical rounds vs kappa .40 -> .57)
        Prints exactly what the paper must say.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

os.makedirs("outputs", exist_ok=True)

# ===== STEP 1 - name the two excluded participants (EDIT THIS LINE) ==========
EXCLUDED = ["P07", "P09"]          # facilitator-issued codes, both Group A
# These must stay identical to EXCLUDE_BOILERPLATE in analyze_trust_study.py.
CONDITION = "predictions_only"     # Group A. Change ONLY if your records say otherwise.
# Writes outputs/educator_exclusions.json with all seven keys, byte-for-byte as
# committed. Re-running this script must not truncate the committed artefact.
json.dump({
    "excluded_participant_ids": EXCLUDED,
    "excluded_from_condition": CONDITION,
    "reason": "open-text responses identified as duplicated boilerplate",
    "identified": "post hoc, after reading the responses (Section 4.10)",
    "effect": "n falls from 19 and 20 to 17 and 20",
    "retest": {"trust_U": 277.0, "trust_p": 0.001,
               "trust_r": 0.63, "intervention_quality_p": 0.364},
    "reported_in": "Section 5.10",
}, open("outputs/educator_exclusions.json", "w"), indent=2)
# EXCLUDE_BOILERPLATE in analyze_trust_study.py already carries the same two codes.

d = pd.read_csv("data_local/responses.csv"); d.columns = [c.strip() for c in d.columns]
for c in ["T1", "T2", "T3", "T4"]:
    d[c] = 6 - d[c]                                  # reverse-code ONCE (Section 4.9)
d["Trust"] = d[[f"T{i}" for i in range(1, 9)]].mean(axis=1)
d["IQ"] = d[[f"s{i}_quality" for i in range(1, 6)]].mean(axis=1)
A, B = d[d.group.str.upper() == "A"], d[d.group.str.upper() == "B"]


def r_rb(b, a):
    return 2 * stats.mannwhitneyu(b, a, alternative="two-sided").statistic / (len(a) * len(b)) - 1


def boot_ci(b, a, n=200_000, seed=42):
    rng = np.random.default_rng(seed); b, a = np.asarray(b), np.asarray(a); out = np.empty(n)
    for i in range(n):
        out[i] = r_rb(rng.choice(b, len(b)), rng.choice(a, len(a)))
    return np.percentile(out, [2.5, 97.5])

# NOTE: 200,000 resamples take a few minutes per outcome. That is expected.
for col, want in [("Trust", "[.31, .85]"), ("IQ", "[-.15, .56]")]:
    lo, hi = boot_ci(B[col].dropna(), A[col].dropna())
    print(f"4.10 {col}: r = {r_rb(B[col].dropna(), A[col].dropna()):+.2f}, 95% CI [{lo:.2f}, {hi:.2f}]  PAPER {want}")

diff = B.IQ.mean() - A.IQ.mean()
se = np.sqrt(A.IQ.var(ddof=1) / len(A) + B.IQ.var(ddof=1) / len(B))
dfw = se ** 4 / ((A.IQ.var(ddof=1) / len(A)) ** 2 / (len(A) - 1) + (B.IQ.var(ddof=1) / len(B)) ** 2 / (len(B) - 1))
t90 = stats.t.ppf(0.95, dfw)
print(f"6.3  raw IQ difference {diff:+.2f}, 90% CI [{diff - t90*se:+.2f}, {diff + t90*se:+.2f}]  PAPER [-0.07, +0.33]")
print("     Student 5 tutoring 16/20 vs 11/19: Fisher p =",
      round(stats.fisher_exact([[16, 4], [11, 8]])[1], 3), " PAPER .176")

# ===== STEP 2 - coding reliability: resolves contradiction C6 ================
cr = pd.read_csv("data_local/coding_rounds.csv")
w = cr.pivot_table(index="item_id", columns=["round", "coder"], values="score")
for rnd in ["cal", "r1", "r2"]:
    a, b = w[(rnd, 1)], w[(rnd, 2)]; ok = a.notna() & b.notna()
    print(f"round {rnd}: n={ok.sum()}, exact agreement {(a[ok] == b[ok]).mean():.3f}, "
          f"linear kappa {cohen_kappa_score(a[ok], b[ok], weights='linear'):.2f}, "
          f"quadratic kappa {cohen_kappa_score(a[ok], b[ok], weights='quadratic'):.2f}")
for coder in [1, 2]:
    a, b = w[("r1", coder)], w[("r2", coder)]; ok = a.notna() & b.notna()
    print(f"coder {coder}: r1 vs r2 identical on {(a[ok] == b[ok]).mean():.3f} of items "
          f"({int((a[ok] != b[ok]).sum())} items changed)")
print("""
READ THE OUTPUT LIKE THIS:
  - If coder 1 changed MANY items between r1 and r2, then '96.4% identical' is
    wrong as written. Replace it with the per-coder figures printed above, and
    state that coder 1 re-scored independently under the frozen rubric.
  - If '96.4%' compares the two CONSENSUS score sets (after adjudication), say
    exactly that: 'Consensus scores from the two post-freeze rounds were 96.4%
    identical.' Then report the per-coder agreement as well.
  - Put exact agreement for every round into Table S6, not only for round 2.
""")


# ===== STEP 3 - group-level aggregates for the acceptance gate (F2) =========
# outputs/educator_aggregates.csv lets the gate verify Sections 4.9, 4.12, 5.1,
# 5.8 and 5.9 on a clone that does not hold data_local/responses.csv. It carries
# statistics only: no participant codes, no free text, and no role or years at
# row level. The logic below mirrors acceptance_gate.py block 5 exactly, so the
# committed aggregates and a local gate run cannot disagree.
import re as _re
from scipy.stats import mannwhitneyu as _mwu, chi2_contingency as _chi2c, spearmanr as _spr


def _cronbach(frame):
    frame = frame.dropna()
    k = frame.shape[1]
    if k < 2 or len(frame) < 2:
        return float("nan")
    v = frame.var(axis=0, ddof=1).sum()
    t = frame.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - v / t)) if t else float("nan")


_ALL8 = [f"T{i}" for i in range(1, 9)]
_raw = pd.read_csv("data_local/responses.csv")
_raw.columns = [c.strip() for c in _raw.columns]

_rows = []


def _put(section, measure, group, statistic, value):
    if value is None or (isinstance(value, float) and value != value):
        return
    _rows.append(dict(section=section, measure=measure, group=group or "",
                      statistic=statistic, value=round(float(value), 6)))


# ---- 4.9 scale reliability -------------------------------------------------
_put("4.9", "trust_alpha_raw", "", "alpha", _cronbach(_raw[_ALL8]))
_put("4.9", "trust_alpha_recoded", "", "alpha", _cronbach(d[_ALL8]))
for _pre in ("C", "A", "U", "E"):
    _cols = [f"{_pre}{i}" for i in range(1, 5) if f"{_pre}{i}" in d]
    _put("4.9", f"alpha_{_pre}", "", "alpha", _cronbach(d[_cols]))

# ---- 5.8 trust -------------------------------------------------------------
_tA, _tB = A["Trust"].dropna(), B["Trust"].dropna()
_put("5.8", "trust", "A", "mean", _tA.mean())
_put("5.8", "trust", "B", "mean", _tB.mean())
_put("5.8", "trust", "A", "n", len(_tA))
_put("5.8", "trust", "B", "n", len(_tB))
_rT = _mwu(_tB, _tA, alternative="two-sided")
_put("5.8", "trust", "", "U", _rT.statistic)
_put("5.8", "trust", "", "p", _rT.pvalue)
_put("5.8", "trust", "", "r", 2 * float(_rT.statistic) / (len(_tA) * len(_tB)) - 1)

# ---- 5.9 intervention quality ---------------------------------------------
_qA, _qB = A["IQ"].dropna(), B["IQ"].dropna()
_rQ = _mwu(_qB, _qA, alternative="two-sided")
_put("5.9", "intervention_quality", "", "U", _rQ.statistic)
_put("5.9", "intervention_quality", "", "p", _rQ.pvalue)

# ---- 5.1 covariate balance, and 4.12 ---------------------------------------
_W = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
      "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _yrs(v):
    t = str(v).strip()
    if _re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    m = _re.match(r"^(\d+(?:\.\d+)?)\s*years?\b", t, _re.I)
    if m:
        return float(m.group(1))
    m = _re.match(r"^([A-Za-z]+)\s*years?\b", t, _re.I)
    return float(_W[m.group(1).lower()]) if m and m.group(1).lower() in _W else None


_anl = d[d["Trust"].notna()]
_txt = pd.read_csv("data_local/responses.csv", keep_default_na=False, na_filter=False)
_txt.columns = [c.strip() for c in _txt.columns]
_yrs_raw = _txt["years"].reindex(_anl.index)
_ag = _anl["group"].str.upper()

_put("4.12", "prior_ai_use", "", "answered",
     float((_txt["ai_prior"].reindex(_anl.index).str.strip() != "").sum()))

_tab = pd.crosstab(_anl["role"].astype(str).str.strip(), _ag)
for _c in ("A", "B"):
    if _c not in _tab:
        _tab[_c] = 0
_put("5.1", "role", "", "chi2", _chi2c(_tab[["A", "B"]].values, correction=False)[0])

_y = _yrs_raw.map(_yrs)
_ok = _y.notna()
_yA, _yB = _y[_ok & (_ag == "A")], _y[_ok & (_ag == "B")]
_put("5.1", "experience", "", "usable", float(_ok.sum()))
_put("5.1", "experience", "A", "n", len(_yA))
_put("5.1", "experience", "B", "n", len(_yB))
_put("5.1", "experience", "A", "median", _yA.median())
_put("5.1", "experience", "B", "median", _yB.median())
_rY = _mwu(_yB, _yA, alternative="two-sided")
_put("5.1", "experience", "", "U", _rY.statistic)
_put("5.1", "experience", "", "r", 2 * float(_rY.statistic) / (len(_yA) * len(_yB)) - 1)
_put("5.1", "experience_vs_trust", "A", "rho",
     _spr(_y[_ok & (_ag == "A")], _anl.loc[_ok & (_ag == "A"), "Trust"])[0])
_put("5.1", "experience_vs_trust", "B", "rho",
     _spr(_y[_ok & (_ag == "B")], _anl.loc[_ok & (_ag == "B"), "Trust"])[0])

_y0 = _yrs_raw.map(lambda v: 0.0 if str(v).strip().lower() == "none" else _yrs(v))
_ok0 = _y0.notna()
_a0, _b0 = _y0[_ok0 & (_ag == "A")], _y0[_ok0 & (_ag == "B")]
_r0 = _mwu(_b0, _a0, alternative="two-sided")
_put("5.1", "experience_word_as_zero", "", "U", _r0.statistic)
_put("5.1", "experience_word_as_zero", "", "r",
     2 * float(_r0.statistic) / (len(_a0) * len(_b0)) - 1)
_put("5.1", "experience_vs_trust_word_as_zero", "B", "rho",
     _spr(_y0[_ok0 & (_ag == "B")], _anl.loc[_ok0 & (_ag == "B"), "Trust"])[0])

_agg = pd.DataFrame(_rows, columns=["section", "measure", "group", "statistic", "value"])
assert not _agg.empty, "no aggregates computed"
assert _agg.duplicated(["section", "measure", "group", "statistic"]).sum() == 0
_agg.to_csv("outputs/educator_aggregates.csv", index=False)
print(f"\nSTEP 3: wrote outputs/educator_aggregates.csv ({len(_agg)} rows, group level only)")
print(_agg.to_string(index=False))
