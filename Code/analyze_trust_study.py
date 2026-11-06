"""
Project Cinque — trust-study analysis (between-subjects Group A vs Group B).
Reads responses.csv, scores the SDT/TAM subscales, and runs the correct test
for two independent groups: the Mann-Whitney U test (NOT Wilcoxon signed-rank,
which is for paired data).

All outcomes are tested TWO-SIDED, matching Section 4.10 of the manuscript.
The study materials stated directional hypotheses for trust, competence and
autonomy. The two-sided test is the conservative choice and does not rely on
them.

Run:  python analyze_trust_study.py responses.csv
"""
import sys, re, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr

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
EXCLUDE_BOILERPLATE = ["P07", "P09"]   # facilitator-issued codes, both Group A

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
        res = mannwhitneyu(b, a, alternative="two-sided")    # two-sided (Section 4.10)
        U, p = float(res.statistic), float(res.pvalue)
        r = 2 * U / (len(a) * len(b)) - 1                    # rank-biserial effect size
        sig = "YES" if p < alpha else "no"
        print(f"{o:22s} {a.mean():8.2f} {b.mean():8.2f} {U:8.1f} {p:9.5f} {r:7.2f}   {sig}")
    print(f"\nBonferroni-corrected threshold = {alpha:.4f} ({m} outcomes).")
    print("Effect size r (rank-biserial): ~0.1 small, ~0.3 medium, ~0.5 large.")

YEARS_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

def parse_years(v):
    """The years field is free text: '4', '4 years', 'Four years', blank, and in
    two rows an answer belonging to another column. Returns None if unusable."""
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*years?\b", s, re.I)
    if m:
        return float(m.group(1))
    m = re.match(r"^([A-Za-z]+)\s*years?\b", s, re.I)
    if m and m.group(1).lower() in YEARS_WORDS:
        return float(YEARS_WORDS[m.group(1).lower()])
    return None

def balance(d):
    """Covariate balance between the two allocation blocks (Section 5.1).

    Allocation followed the pre-assigned participant-code sequence in four
    consecutive blocks of ten by condition (Section 4.12), not at random, so the
    conditions are checked for balance on the covariates the booklet recorded.
    """
    print(f"\n{'=' * 72}\nCOVARIATE BALANCE (Section 5.1)\n{'=' * 72}")
    g = d["group"].astype(str).str.strip().str.upper()

    if "role" in d:
        tab = pd.crosstab(d["role"].astype(str).str.strip(), g)
        for c in ("A", "B"):
            if c not in tab: tab[c] = 0
        tab = tab[["A", "B"]]
        print("\n=== Role by condition ===")
        print(tab.to_string())
        chi2, pchi, dof, exp = chi2_contingency(tab.values, correction=False)
        print(f"  chi-square({dof}) = {chi2:.2f}   p = {pchi:.3f}   "
              f"smallest expected cell {exp.min():.1f}")

    if "years" not in d:
        print("\n[note] no years column; experience balance skipped.")
        return
    # pandas maps the literal string "None" to NaN on read; re-read the column
    # with NA conversion off so a participant's "None" survives as text.
    yr = d["years_raw"] if "years_raw" in d else d["years"]
    y = yr.map(parse_years)
    ok = y.notna()
    yA, yB = y[ok & (g == "A")], y[ok & (g == "B")]
    print("\n=== Teaching experience ===")
    print(f"  usable entries {int(ok.sum())} of {len(d)}   "
          f"(A {len(yA)}, B {len(yB)}; unusable A {int(((~ok) & (g == 'A')).sum())}, "
          f"B {int(((~ok) & (g == 'B')).sum())})")
    print(f"  median A {yA.median():.1f}   median B {yB.median():.1f}")
    rY = mannwhitneyu(yB, yA, alternative="two-sided")
    rr = 2 * float(rY.statistic) / (len(yA) * len(yB)) - 1
    print(f"  U = {float(rY.statistic):.1f}   p = {float(rY.pvalue):.5f}   r = {rr:+.3f}")

    tcols = [c for c in SCALES["Trust"] if c in d]
    trust = d[tcols].mean(axis=1)          # T1-T4 already reversed by main()
    print("\n=== Experience against trust: does the imbalance explain 5.8? ===")
    for lbl, m in (("pooled", ok), ("Group A", ok & (g == "A")), ("Group B", ok & (g == "B"))):
        rho, prho = spearmanr(y[m], trust[m])
        print(f"  {lbl:8s} n = {int(m.sum()):2d}   Spearman rho = {rho:+.3f}   p = {prho:.3f}")
    # Sensitivity (Section 5.1): one entry reads "None", which may mean zero
    # years rather than a non-answer. Both readings are reported so that the
    # conclusion does not rest on how that one cell is read.
    y0 = yr.map(lambda v: 0.0 if str(v).strip().lower() == "none"
                else parse_years(v))
    ok0 = y0.notna()
    a0, b0 = y0[ok0 & (g == "A")], y0[ok0 & (g == "B")]
    r0 = mannwhitneyu(b0, a0, alternative="two-sided")
    rr0 = 2 * float(r0.statistic) / (len(a0) * len(b0)) - 1
    rhoB0 = float(spearmanr(y0[ok0 & (g == "B")], trust[ok0 & (g == "B")])[0])
    print("\n=== Sensitivity: reading the word entry as zero years ===")
    print(f"  usable {int(ok0.sum())}   n A {len(a0)}, B {len(b0)}")
    print(f"  U = {float(r0.statistic):.1f}   p = {float(r0.pvalue):.5f}   "
          f"r = {rr0:+.3f}   rho in the explanation condition = {rhoB0:+.3f}")

    print("\n  Near-zero within-condition correlations mean the experience imbalance")
    print("  does not account for the Section 5.8 trust difference under either reading.")

def main(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    # keep the untouched text of the free-text columns: pandas turns a literal
    # "None" into NaN, and two participants answered exactly that.
    _raw = pd.read_csv(path, keep_default_na=False, na_filter=False)
    _raw.columns = [c.strip() for c in _raw.columns]
    d["years_raw"] = _raw["years"]
    for c in REVERSE:
        if c in d: d[c] = 6 - d[c]                     # reverse on 1..5, applied once
    balance(d.copy())
    analyse(d.copy(), "full sample")
    if EXCLUDE_BOILERPLATE:
        col = participant_col(d)
        # Case- and whitespace-insensitive, and tolerant of a stray prefix:
        # responses.csv stores these codes inconsistently ("p07", "Code p37"),
        # so a plain case-sensitive isin() matched 0 of 2 and the re-test arm
        # silently ran on the full sample.
        _norm = (d[col].astype(str).str.strip().str.upper()
                 .str.replace(r"^CODE\s+", "", regex=True))
        _want = {c.strip().upper() for c in EXCLUDE_BOILERPLATE}
        keep = ~_norm.isin(_want)
        dropped = int((~keep).sum())
        if dropped != len(EXCLUDE_BOILERPLATE):
            raise SystemExit(
                f"\n[stop] matched {dropped} of {len(EXCLUDE_BOILERPLATE)} codes "
                f"in column '{col}'. Section 5.10's re-test needs both. "
                f"Codes present: {sorted(_norm.unique())}")
        print(f"\n[ok] excluded {dropped} participants: {sorted(_want)}")
        analyse(d[keep].copy(), "boilerplate excluded (Section 5.10 re-test)")
    else:
        print("\n[note] EXCLUDE_BOILERPLATE is empty; Section 5.10 re-test arm skipped.")
    print("\nReminder: open-ended responses are analysed separately via thematic analysis (Braun & Clarke).")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "responses.csv")
