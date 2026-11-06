"""
Acceptance gate for the Group 8 release. Run this against the released outputs
BEFORE you tag the commit you cite in the manuscript.

Every figure below is quoted from GROUP8_MANUSCRIPT_D8.docx. If an assertion
fails, either the released run does not reproduce the paper or the paper is
wrong. Both are things to find now rather than in review.

Point the CONFIG block at your files. Nothing else needs editing.

NOTE ON SENSITIVE INPUTS. TRUST_CSV holds educator responses and must never be
committed. The gate reads it locally, before tagging. Keep it in .gitignore.
"""
import sys, json, re, pathlib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
PER_SEED    = "outputs/per_seed_metrics.csv"      # one row per (config, seed)
SEED42      = "outputs/seed42_predictions.csv"    # y_true, proba per config, fee status
BUDGETS     = "outputs/adaptation_budgets.csv"    # arm, budget, seed, auc
EXCLUSIONS  = "outputs/educator_exclusions.json"
TRUST_CSV   = "data_local/responses.csv"          # NOT COMMITTED: raw educator responses
OULAD_SEED42 = "outputs/oulad_seed42_predictions.csv"  # y_true, proba_full/pruned/cand, imd_band
OULAD_SEEDS  = "outputs/oulad_per_seed_metrics.csv"    # config, seed, auc
TRANSFER     = "outputs/transfer_matched_ceiling.csv"  # seed, zero_shot, matched_ceiling
CV55         = "outputs/repeated_cv55.csv"             # config, fold, auc

COL_CONFIG, COL_SEED, COL_AUC = "config", "seed", "auc"
SPLITS20 = [42, 0, 123, 2024, 7] + list(range(100, 115))   # Section 4.8
SEEDS5   = [42, 0, 123, 2024, 7]
TOL = 0.0006          # the paper reports three decimals
ALLOW_SKIP = False    # set True only to run a partial gate; a skip still blocks a tag
# -------------------------------------------------------------------------

fails, passes, skips = [], [], []


def check(label, got, want, tol=TOL):
    ok = got is not None and abs(got - want) <= tol
    (passes if ok else fails).append(
        f"{label:54s} got {got if got is None else round(got, 5)!s:>9s}  want {want}")


def check_eq(label, got, want):
    (passes if got == want else fails).append(f"{label:54s} got {got!r}  want {want!r}")


def block_skipped(label, exc):
    skips.append(f"{label:54s} {type(exc).__name__}: {exc}")


def mean_over(df, cfg, seeds):
    s = df[(df[COL_CONFIG] == cfg) & (df[COL_SEED].isin(seeds))]
    return float(s[COL_AUC].mean()) if len(s) == len(seeds) else None


def paired(df, a, b, seeds):
    x = df[(df[COL_CONFIG] == a) & (df[COL_SEED].isin(seeds))].sort_values(COL_SEED)
    y = df[(df[COL_CONFIG] == b) & (df[COL_SEED].isin(seeds))].sort_values(COL_SEED)
    if len(x) != len(seeds) or len(y) != len(seeds):
        return None
    return float((x[COL_AUC].values - y[COL_AUC].values).mean())


def paired_diffs(df, a, b, seeds):
    p = df[df[COL_SEED].isin(seeds)].pivot_table(
        index=COL_SEED, columns=COL_CONFIG, values=COL_AUC)
    return (p[a] - p[b]).values


def cronbach(df):
    df = df.dropna()
    k = df.shape[1]
    if k < 2 or len(df) < 2:
        return np.nan
    v = df.var(axis=0, ddof=1).sum()
    t = df.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - v / t)) if t else np.nan


def eo_gap(y_true, proba, groups, tau, min_n=30):
    """Equal-opportunity gap at the operating point tau. Groups under min_n
    in total are not audited. Mirrors the protocol in Section 4.9."""
    y_true, proba, groups = np.asarray(y_true), np.asarray(proba), np.asarray(groups)
    yhat = (proba >= tau).astype(int)
    rec = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < min_n:
            continue
        pos = m & (y_true == 1)
        if pos.sum() == 0:
            continue
        rec[g] = float(yhat[pos].mean())
    if len(rec) < 2:
        return float("nan"), rec
    return max(rec.values()) - min(rec.values()), rec


# ---- 1. Table 1 and 5.2, the 20-split discrimination block ---------------
try:
    ps = pd.read_csv(PER_SEED)
    check("Table 1  full, 20-split mean AUC",     mean_over(ps, "full", SPLITS20),     0.868)
    check("Table 1  pruned, 20-split mean AUC",   mean_over(ps, "pruned", SPLITS20),   0.871)
    check("Table 1  deployed, 20-split mean AUC", mean_over(ps, "deployed", SPLITS20), 0.861)
    check("5.2      paired diff, pruned - full",
          paired(ps, "pruned", "full", SPLITS20), 0.0034, 0.0004)
    check("4.11     paired diff, pruned - deployed",
          paired(ps, "pruned", "deployed", SPLITS20), 0.0106, 0.0004)
    check("4.11     deployed seed-42 single split",
          float(ps[(ps[COL_CONFIG] == "deployed") & (ps[COL_SEED] == 42)][COL_AUC].iloc[0]),
          0.852)
    check("5.13     transferable 9-col, 20-split", mean_over(ps, "transfer9", SPLITS20), 0.851)
    check("5.2      pruned, five-seed mean",       mean_over(ps, "pruned", SEEDS5),     0.880)

    # TOST against the pre-specified delta = 0.01 AUC bound (Sections 4.6, 4.11).
    # D8 reports t(19) = -5.84 and t(19) = 0.29. The fix guide's sample said 0.28
    # for the second; D8 says 0.29 in both places it appears, and the guide's own
    # rule is that the gate takes the number printed in D8.
    dd = paired_diffs(ps, "pruned", "full", SPLITS20)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    check("4.6      TOST t(19), upper bound", float((dd.mean() - 0.01) / se), -5.84, 0.02)

    dd2 = paired_diffs(ps, "pruned", "deployed", SPLITS20)
    se2 = dd2.std(ddof=1) / np.sqrt(len(dd2))
    check("4.11     TOST t(19), fairness screen", float((dd2.mean() - 0.01) / se2), 0.29, 0.02)
except Exception as e:
    block_skipped("per-seed metrics unreadable", e)

# ---- 2. 5.4, the fairness audit AT THE SCREENING THRESHOLD --------------
# This is the one the released script got wrong. It audited at 0.5.
try:
    s42 = pd.read_csv(SEED42)
    gp, rp = eo_gap(s42["y_true"], s42["proba_pruned"],   s42["fee_status"], 0.0755)
    gd, rd = eo_gap(s42["y_true"], s42["proba_deployed"], s42["fee_status"], 0.0833)
    check("5.4      EO gap, pruned at tau = 0.0755",   gp, 0.490, 0.0015)
    check("5.4      EO gap, deployed at tau = 0.0833", gd, 0.385, 0.0015)
    check("5.4      pruned recall, fee-paid",   rp.get("Paid"),    0.316, 0.0015)
    check("5.4      pruned recall, partial",    rp.get("Partial"), 0.481, 0.0015)
    check("5.4      pruned recall, unpaid",     rp.get("Unpaid"),  0.806, 0.0015)
    check("5.4      deployed recall, fee-paid", rd.get("Paid"),    0.421, 0.0015)
    check("5.4      deployed recall, partial",  rd.get("Partial"), 0.519, 0.0015)
    check("5.4      deployed recall, unpaid",   rd.get("Unpaid"),  0.806, 0.0015)

    # the gaps must NOT be the 0.5-threshold values the stale script produced
    g5p, _ = eo_gap(s42["y_true"], s42["proba_pruned"],   s42["fee_status"], 0.5)
    g5d, _ = eo_gap(s42["y_true"], s42["proba_deployed"], s42["fee_status"], 0.5)
    ok = abs(gp - g5p) > 0.01 and abs(gd - g5d) > 0.01
    (passes if ok else fails).append(
        f"{'5.4      audited at tau, not at 0.5':54s} "
        f"tau gaps {gp:.3f}/{gd:.3f} vs 0.5 gaps {g5p:.3f}/{g5d:.3f}")

    yhat = (s42["proba_deployed"].values >= 0.0833).astype(int)
    pos = s42["y_true"].values == 1
    check("4.4      deployed recall at tau = 0.0833", float(yhat[pos].mean()), 0.673, 0.002)
    check("4.4      true positives at tau = 0.0833", float((yhat[pos] == 1).sum()), 76.0, 0.5)
    check("5.5      precision at tau = 0.0833",
          float(s42["y_true"].values[yhat == 1].mean()), 0.484, 0.002)
except Exception as e:
    block_skipped("seed-42 predictions unreadable", e)

# ---- 3. 5.13, the adaptation budgets ------------------------------------
try:
    bd = pd.read_csv(BUDGETS)
    piv = bd.pivot_table(index=["budget", "seed"], columns="arm", values="auc")
    d = piv["finetuned"] - piv["scratch"]
    check("5.13     mean benefit across all cells", float(d.mean()), -0.0074, 0.0006)
    check("5.13     negative cells out of 30", float(int((d < 0).sum())), 29.0, 0.5)
    for b, want in [(100, -0.0117), (250, -0.0058), (500, -0.0060),
                    (1000, -0.0060), (2500, -0.0051), (5000, -0.0098)]:
        check(f"5.13     per-budget benefit at {b}",
              float(d.xs(b, level="budget").mean()), want, 0.0006)
    check("5.13     from-scratch at 100 records",
          float(piv.xs(100, level="budget")["scratch"].mean()), 0.862, 0.002)
except Exception as e:
    block_skipped("budget outputs unreadable", e)

# ---- 4. 5.10, the exclusion list must be recorded -----------------------
# 51 different pairs give U = 277.0, so the re-test is not reproducible
# without naming the two participants that were dropped.
try:
    ex = json.load(open(EXCLUSIONS))
    ids = ex.get("excluded_participant_ids", [])
    (passes if len(ids) == 2 else fails).append(
        f"{'5.10     two excluded participants named':54s} got {ids}")
    # Both are in the predictions-only arm: only nA=17/nB=20 yields r = +.63.
    check_eq("5.10     both from predictions-only (r = +.63)",
             ex.get("excluded_from_condition"), "predictions_only")
except Exception as e:
    block_skipped("5.10     exclusion record missing", e)

# ---- 5. Educator study, and the reverse-coding that twice went missing --
# Sections 4.9, 5.8, 5.9, 5.10. T1-T4 are negatively keyed and MUST be
# recoded 6 - x before the composite is formed. Without it the trust effect
# disappears: alpha reads .788, U reads 170.5, and the direction reverses.
try:
    tr = pd.read_csv(TRUST_CSV)
    tr.columns = [c.strip() for c in tr.columns]
    NEG = ["T1", "T2", "T3", "T4"]
    ALL8 = [f"T{i}" for i in range(1, 9)]

    raw_alpha = cronbach(tr[ALL8])
    rec = tr.copy()
    for c in NEG:
        rec[c] = 6 - rec[c]
    rec_alpha = cronbach(rec[ALL8])

    check("4.9      trust alpha, items as stored (raw)", raw_alpha, 0.788, 0.003)
    check("4.9      trust alpha, after 6 - x on T1-T4",  rec_alpha, 0.718, 0.003)
    (passes if rec_alpha < raw_alpha else fails).append(
        f"{'4.9      recoding lowers alpha (two-block scale)':54s} "
        f"{raw_alpha:.3f} -> {rec_alpha:.3f}")

    from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr
    rec["Trust"] = rec[ALL8].mean(axis=1)
    for name, pre, n in (("Competence (IMI)", "C", 4), ("Autonomy (IMI)", "A", 4),
                         ("Usefulness (TAM)", "U", 4), ("Ease of use (TAM)", "E", 4)):
        cols = [f"{pre}{i}" for i in range(1, n + 1) if f"{pre}{i}" in rec]
        rec[name] = rec[cols].mean(axis=1)
    qcols = [f"s{i}_quality" for i in range(1, 6) if f"s{i}_quality" in rec]
    if qcols:
        rec["intervention_quality"] = rec[qcols].mean(axis=1)

    g = rec["group"].str.upper()
    A = rec.loc[g == "A", "Trust"].dropna()
    B = rec.loc[g == "B", "Trust"].dropna()
    check("5.8      Group A trust mean (predictions only)", float(A.mean()), 2.82, 0.006)
    check("5.8      Group B trust mean (+ explanations)",   float(B.mean()), 3.55, 0.006)
    check("5.8      analysed n, Group A", float(len(A)), 19.0, 0.5)
    check("5.8      analysed n, Group B", float(len(B)), 20.0, 0.5)

    rT = mannwhitneyu(B, A, alternative="two-sided")
    check("5.8      trust U",             float(rT.statistic), 305.0, 0.5)
    check("5.8      trust p (two-sided)", float(rT.pvalue),    0.001, 0.0006)
    check("5.8      trust rank-biserial r",
          2 * float(rT.statistic) / (len(A) * len(B)) - 1, 0.61, 0.006)
    (passes if float(rT.pvalue) < 0.0056 else fails).append(
        f"{'5.8      trust clears Bonferroni .0056':54s} p = {float(rT.pvalue):.5f}")

    if "intervention_quality" in rec:
        qA = rec.loc[g == "A", "intervention_quality"].dropna()
        qB = rec.loc[g == "B", "intervention_quality"].dropna()
        rQ = mannwhitneyu(qB, qA, alternative="two-sided")
        check("5.9      intervention-quality U", float(rQ.statistic), 231.0, 0.5)
        check("5.9      intervention-quality p", float(rQ.pvalue),    0.249, 0.0015)

    for pre, want in (("C", 0.80), ("A", 0.82), ("U", 0.88), ("E", 0.90)):
        cols = [f"{pre}{i}" for i in range(1, 5) if f"{pre}{i}" in rec]
        check(f"4.9      alpha, {pre}-scale", cronbach(rec[cols]), want, 0.006)

    # ---- 5.1 covariate balance between the two allocation blocks ----------
    # Allocation ran off the pre-assigned code sequence in four blocks of ten
    # (Section 4.12), not at random, so Section 5.1 reports a balance check.
    # These re-derive it. p values are deliberately not gated: they depend on
    # scipy's tie handling. The statistics and effect sizes are exact.
    _W = {"one":1,"two":2,"three":3,"four":4,"five":5,
          "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    def _yrs(v):
        t = str(v).strip()
        if re.fullmatch(r"\d+(\.\d+)?", t): return float(t)
        m = re.match(r"^(\d+(?:\.\d+)?)\s*years?\b", t, re.I)
        if m: return float(m.group(1))
        m = re.match(r"^([A-Za-z]+)\s*years?\b", t, re.I)
        return float(_W[m.group(1).lower()]) if m and m.group(1).lower() in _W else None

    anl = rec[rec["Trust"].notna()]                 # the 39 analysed participants
    # pandas maps the literal string "None" to NaN on read, and two participants
    # answered "None" - one in years, one in prior AI use. Re-read those columns
    # with NA conversion off so the answers survive as text.
    _txt = pd.read_csv(TRUST_CSV, keep_default_na=False, na_filter=False)
    _txt.columns = [c.strip() for c in _txt.columns]
    _yrs_raw = _txt["years"].reindex(anl.index)
    check("4.12     prior-AI-use item answered by all 39",
          float((_txt["ai_prior"].reindex(anl.index).str.strip() != "").sum()), 39.0, 0.5)
    ag = anl["group"].str.upper()
    _tab = pd.crosstab(anl["role"].astype(str).str.strip(), ag)
    for _c in ("A", "B"):
        if _c not in _tab: _tab[_c] = 0
    _chi2, _pchi, _dof, _exp = chi2_contingency(_tab[["A", "B"]].values, correction=False)
    check("5.1      role chi-square across conditions", float(_chi2), 5.58, 0.02)

    _y = _yrs_raw.map(_yrs)
    _ok = _y.notna()
    _yA, _yB = _y[_ok & (ag == "A")], _y[_ok & (ag == "B")]
    check("5.1      experience, usable entries", float(_ok.sum()), 33.0, 0.5)
    check("5.1      experience n, Group A", float(len(_yA)), 16.0, 0.5)
    check("5.1      experience n, Group B", float(len(_yB)), 17.0, 0.5)
    check("5.1      experience median, Group A", float(_yA.median()), 2.0, 0.01)
    check("5.1      experience median, Group B", float(_yB.median()), 4.0, 0.01)
    _rY = mannwhitneyu(_yB, _yA, alternative="two-sided")
    check("5.1      experience U (B vs A)", float(_rY.statistic), 211.5, 0.5)
    check("5.1      experience rank-biserial r",
          2 * float(_rY.statistic) / (len(_yA) * len(_yB)) - 1, 0.555, 0.005)

    # The imbalance only matters if experience predicts trust. It does not,
    # within either condition; the pooled value is induced by the group gap.
    _rhoA = float(spearmanr(_y[_ok & (ag == "A")], anl.loc[_ok & (ag == "A"), "Trust"])[0])
    _rhoB = float(spearmanr(_y[_ok & (ag == "B")], anl.loc[_ok & (ag == "B"), "Trust"])[0])
    check("5.1      experience vs trust rho, Group A", _rhoA, -0.137, 0.01)
    check("5.1      experience vs trust rho, Group B", _rhoB, +0.024, 0.01)
    # 5.1 reports a sensitivity reading in which the one word-valued entry is
    # taken as zero years. Gate it too, so neither reading rests on assertion.
    _y0 = _yrs_raw.map(lambda v: 0.0 if str(v).strip().lower() == "none"
                       else _yrs(v))
    _ok0 = _y0.notna()
    _a0, _b0 = _y0[_ok0 & (ag == "A")], _y0[_ok0 & (ag == "B")]
    _r0 = mannwhitneyu(_b0, _a0, alternative="two-sided")
    check("5.1      experience U, word entry read as 0", float(_r0.statistic), 212.0, 0.5)
    check("5.1      experience r, word entry read as 0",
          2 * float(_r0.statistic) / (len(_a0) * len(_b0)) - 1, 0.472, 0.005)
    check("5.1      rho Group B, word entry read as 0",
          float(spearmanr(_y0[_ok0 & (ag == "B")], anl.loc[_ok0 & (ag == "B"), "Trust"])[0]),
          0.182, 0.01)

    (passes if abs(_rhoA) < 0.3 and abs(_rhoB) < 0.3 else fails).append(
        f"{'5.1      imbalance does not explain the 5.8 trust gap':54s} "
        f"rho {_rhoA:+.3f} / {_rhoB:+.3f}")
except Exception as e:
    block_skipped("5.8      educator responses unreadable", e)

# ---- 6. 5.12, the OULAD replication -------------------------------------
try:
    op = pd.read_csv(OULAD_SEEDS)
    check("5.12     OULAD full, five-seed mean AUC",   mean_over(op, "full", SEEDS5),   0.936)
    check("5.12     OULAD pruned, five-seed mean AUC", mean_over(op, "pruned", SEEDS5), 0.936)

    o42 = pd.read_csv(OULAD_SEED42)
    check("5.12     OULAD held-out partition size", float(len(o42)), 4955.0, 0.5)
    yh_f = (o42["proba_full"].values   >= 0.5).astype(int)
    yh_p = (o42["proba_pruned"].values >= 0.5).astype(int)
    yt = o42["y_true"].values
    b = int(((yh_f == yt) & (yh_p != yt)).sum())
    c = int(((yh_f != yt) & (yh_p == yt)).sum())
    check("5.12     McNemar b (full right, pruned wrong)", float(b), 102.0, 0.5)
    check("5.12     McNemar c (full wrong, pruned right)", float(c), 97.0, 0.5)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    check("5.12     McNemar chi-square", float(chi2), 0.08, 0.006)

    # M1: tau is selected on a 25% validation split of the training partition
    # (Section 4.4), not on the test set. The old 0.4416 / 0.4533 were test-set
    # values and audit at the wrong operating point.
    gp24, _ = eo_gap(o42["y_true"], o42["proba_pruned"], o42["imd_band"], 0.4719)
    gc23, _ = eo_gap(o42["y_true"], o42["proba_cand"],   o42["imd_band"], 0.4874)
    check("5.12     OULAD EO gap, 24-col pruned", gp24, 0.119, 0.0015)
    check("5.12     OULAD EO gap, 23-col candidate", gc23, 0.160, 0.0015)
    check("5.12     OULAD EO change (removal rejected)", gc23 - gp24, 0.041, 0.0015)
    (passes if gc23 > gp24 else fails).append(
        f"{'5.12     Algorithm 1 line 12 rejects removal':54s} "
        f"{gp24:.3f} -> {gc23:.3f}")
except Exception as e:
    block_skipped("5.12     OULAD replication outputs unreadable", e)

# ---- 7. 5.13, zero-shot transfer and the matched ceiling ----------------
try:
    tf = pd.read_csv(TRANSFER)
    zs, ceil = tf["zero_shot"].values, tf["matched_ceiling"].values
    check("5.13     zero-shot mean AUC",        float(zs.mean()),   0.642, 0.0015)
    check("5.13     zero-shot SD",              float(zs.std(ddof=1)), 0.047, 0.0015)
    check("5.13     matched nine-column ceiling", float(ceil.mean()), 0.926, 0.0015)
    check("5.13     matched ceiling SD",        float(ceil.std(ddof=1)), 0.001, 0.0015)
    check("5.13     matched transfer gap",      float(ceil.mean() - zs.mean()), 0.284, 0.0015)
    # the ceiling must NOT be the 0.937 full-schema figure the old notebooks used
    (passes if abs(float(ceil.mean()) - 0.937) > 0.005 else fails).append(
        f"{'5.13     ceiling is nine-column, not 0.937 full-schema':54s} "
        f"got {float(ceil.mean()):.4f}")
except Exception as e:
    block_skipped("5.13     transfer ceiling outputs unreadable", e)

# ---- 8. 4.11, the 5x5 repeated cross-validation variance check ----------
try:
    cv = pd.read_csv(CV55)
    for cfg, want in (("full", 0.865), ("pruned", 0.869), ("deployed", 0.858)):
        s = cv[cv[COL_CONFIG] == cfg][COL_AUC]
        check(f"4.11     5x5 CV mean AUC, {cfg}",
              float(s.mean()) if len(s) == 25 else None, want, 0.0015)
    check("4.11     5x5 CV fold count", float(len(cv)), 75.0, 0.5)
except Exception as e:
    block_skipped("4.11     repeated-CV outputs unreadable", e)

# ---- 9. no mock-data fallback, and no unseeded randomness ---------------
# Seeded RNG is legitimate: the bootstrap intervals in 5.4 and 5.12 use
# np.random.default_rng(42). What must not ship is fabricated data or a
# result that changes between runs.
MOCK = ("mock_oulad", "make_mock", "fake_data", "dummy_data", "simulate_responses")
UNSEEDED = re.compile(r"default_rng\(\s*\)|np\.random\.(?:rand|randn|randint|choice|seed)\(")
banned = []
SELF = pathlib.Path(__file__).resolve()
for p in list(pathlib.Path(".").rglob("*.py")) + list(pathlib.Path(".").rglob("*.ipynb")):
    if "test" in p.name or p.resolve() == SELF:
        continue
    t = p.read_text(errors="ignore")
    for marker in MOCK:
        if marker in t:
            banned.append(f"{p}: {marker}")
    for m in UNSEEDED.finditer(t):
        banned.append(f"{p}: unseeded RNG '{m.group(0)}'")
(passes if not banned else fails).append(
    f"{'no mock data and no unseeded randomness':54s} "
    f"{'clean' if not banned else banned}")

# ---- verdict ------------------------------------------------------------
print("=" * 82)
for line in passes:
    print("  pass  " + line)
for line in skips:
    print("  SKIP  " + line)
for line in fails:
    print("  FAIL  " + line)
print("=" * 82)
print(f"{len(passes)} passed, {len(fails)} failed, {len(skips)} skipped")

if fails:
    print("\nDo not tag this commit. Fix the failures or correct the manuscript.")
    sys.exit(1)
if skips and not ALLOW_SKIP:
    print("\nDo not tag this commit. A skipped block is an unverified claim.")
    print("Produce the missing outputs, or set ALLOW_SKIP = True and say so in the README.")
    sys.exit(2)
print("\nSafe to tag. Record this commit hash in the title page and Sections 4.15 and 7.")
