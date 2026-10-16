"""
Project Cinque — trust-study analysis (between-subjects Group A vs Group B).
Reads responses.csv, scores the SDT/TAM subscales, and runs the correct test
for two independent groups: the Mann-Whitney U test (NOT Wilcoxon signed-rank,
which is for paired data).

All outcomes are tested TWO-SIDED, matching Section 4.10 of the manuscript.
Both interpreting theories were selected after the analysis was complete, so no
directional prediction was available in advance.

Run:  python analyze_trust_study.py responses.csv
"""
import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

# subscale -> column prefixes (see column layout in the Analysis Guide)
SCALES = {
    "Trust":            [f"T{i}" for i in range(1, 9)],
    "Competence (IMI)": [f"C{i}" for i in range(1, 5)],
    "Autonomy (IMI)":   [f"A{i}" for i in range(1, 5)],
    "Usefulness (TAM)": [f"U{i}" for i in range(1, 5)],
    "Ease of use (TAM)":[f"E{i}" for i in range(1, 5)],
}
REVERSE = ["T1", "T2", "T3", "T4"]          # reverse-coded trust items (1..5)
PERCASE = ["appropriate", "trust", "confidence"]   # per-case 1..5 items, s1..s5
QUALITY = [f"s{i}_quality" for i in range(1, 6)]    # 0/1/2 intervention-quality (coder-entered)

# Participant codes excluded in the Section 5.10 robustness re-test: the two
# participants whose open-text responses were duplicated boilerplate.
# Leave empty to skip the re-test arm.
EXCLUDE_BOILERPLATE = []        # e.g. ["P07", "P23"]

def cronbach_alpha(df):
    df = df.dropna(); k = df.shape[1]
    if k < 2 or len(df) < 2: return np.nan
    var_sum = df.var(axis=0, ddof=1).sum(); tot_var = df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - var_sum / tot_var) if tot_var else np.nan

def participant_col(d):
    for c in d.columns:
        if c.strip().lower() in ("participant", "participant_code", "code", "id", "pid"):
            return c
    raise KeyError(f"no participant column found; columns are {list(d.columns)}")

def analyse(d, label):
    print(f"\n{'=' * 72}\nARM: {label}  (n = {len(d)})\n{'=' * 72}")
    print("\n=== Reliability (Cronbach's alpha) ===")
    for name, items in SCALES.items():
        items = [i for i in items if i in d]
        d[name] = d[items].mean(axis=1)
        print(f"  {name:20s} alpha = {cronbach_alpha(d[items]):.3f}  ({len(items)} items)")
    for k in PERCASE:
        cols = [f"s{i}_{k}" for i in range(1, 6) if f"s{i}_{k}" in d]
        if cols: d[f"case_{k}"] = d[cols].mean(axis=1)
    qcols = [c for c in QUALITY if c in d]
    if qcols: d["intervention_quality"] = d[qcols].mean(axis=1)

    outcomes = list(SCALES) + [c for c in ["case_appropriate","case_trust","case_confidence","intervention_quality"] if c in d]
    A = d[d["group"].str.upper() == "A"]; B = d[d["group"].str.upper() == "B"]
    print(f"\nN: Group A = {len(A)}, Group B = {len(B)}")
    print("\n=== Group B vs A — Mann-Whitney U (two-sided, all outcomes) ===")
    print(f"{'Outcome':22s} {'A mean':>8s} {'B mean':>8s} {'U':>8s} {'p':>9s} {'r':>7s}  Bonf<.05?")
    m = len(outcomes); alpha = 0.05 / m
    for o in outcomes:
        a, b = A[o].dropna(), B[o].dropna()
        if len(a) < 3 or len(b) < 3:
            print(f"{o:22s}  (insufficient n)"); continue
        res = mannwhitneyu(b, a, alternative="two-sided")    # no predicted direction
        U, p = float(res.statistic), float(res.pvalue)
        r = 2 * U / (len(a) * len(b)) - 1                    # rank-biserial effect size
        sig = "YES" if p < alpha else "no"
        print(f"{o:22s} {a.mean():8.2f} {b.mean():8.2f} {U:8.1f} {p:9.5f} {r:7.2f}   {sig}")
    print(f"\nBonferroni-corrected threshold = {alpha:.4f} ({m} outcomes).")
    print("Effect size r (rank-biserial): ~0.1 small, ~0.3 medium, ~0.5 large.")

def main(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    for c in REVERSE:
        if c in d: d[c] = 6 - d[c]                     # reverse on 1..5, applied once
    analyse(d.copy(), "full sample")
    if EXCLUDE_BOILERPLATE:
        col = participant_col(d)
        keep = ~d[col].astype(str).str.strip().isin(EXCLUDE_BOILERPLATE)
        dropped = (~keep).sum()
        if dropped != len(EXCLUDE_BOILERPLATE):
            print(f"\n[warn] matched {dropped} of {len(EXCLUDE_BOILERPLATE)} codes in column '{col}'")
        analyse(d[keep].copy(), "boilerplate excluded (Section 5.10 re-test)")
    else:
        print("\n[note] EXCLUDE_BOILERPLATE is empty; Section 5.10 re-test arm skipped.")
    print("\nReminder: open-ended responses are analysed separately via thematic analysis (Braun & Clarke).")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "responses.csv")
