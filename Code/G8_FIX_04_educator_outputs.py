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
# These must stay identical to EXCLUDE_BOILERPLATE in analyze_trust_study.py line 31.
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
# analyze_trust_study.py line 31 already carries the same two codes.

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
