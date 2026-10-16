"""
Project Cinque — remaining analysis runs (1-11)
===============================================
Each RUN is a self-contained notebook cell. Paste them one at a time.
The header of each run names the notebook it belongs in and the variables it
expects to already exist in that session.

Source session  (rerun_pruning_v3_original_protocol.py / Model_Full_exp):
    X, y, df, params (= HPARAMS), FEE_COLS, pruned_cols, deployed_cols, SEEDS
OULAD session  (Oulad_exp__1_.ipynb):
    X, y, df, best, pruned_cols, cand_cols, SEEDS
TRANSFER session (RetTransUst.ipynb):
    K, yk, O, yo, BEST, SEEDS, O_tr, O_te, yo_tr, yo_te

Every run prints the numbers to paste back; several also write a CSV so the
figure and the table come from the same file.
"""

# ===========================================================================
# RUN 1 — Source guard at matched thresholds, with a bootstrap  [Source session]
# ===========================================================================
# Puts Source on the same protocol as OULAD: each configuration is scored at
# its OWN validation-selected threshold, and the change in the gap carries an
# interval. Expect the interval to include zero.

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

TARGET_RECALL = 0.75
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
ytr_v, yte_v = np.asarray(ytr), np.asarray(yte)
fee = df.loc[Xte.index, "fee_payment_status"].values
GROUPS = [g for g in pd.unique(fee) if (fee == g).sum() >= 30]     # rule on subgroup SIZE
print("subgroups kept:", {g: int((fee == g).sum()) for g in GROUPS})


def fit_k(Xa, ya, cols, seed=42):
    sc = MinMaxScaler().fit(Xa[cols])
    Xb, yb = SMOTE(random_state=seed).fit_resample(sc.transform(Xa[cols]), ya)
    return sc, XGBClassifier(**params).fit(Xb, yb)


def val_threshold_k(cols, target=TARGET_RECALL, seed=42):
    """Highest threshold reaching `target` recall on a validation split of the
    TRAINING partition. The test partition is never seen."""
    Xi, Xv, yi, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    sc, m = fit_k(Xi, yi, cols, seed)
    pv = m.predict_proba(sc.transform(Xv[cols]))[:, 1]
    yv_v = np.asarray(yv)
    ts = np.sort(np.unique(pv))[::-1]
    rec = np.array([(pv[yv_v == 1] >= t).mean() for t in ts])
    i = int(np.argmax(rec >= target))
    return float(ts[i]), float(rec[i])


def test_pred_k(cols, thr, seed=42):
    sc, m = fit_k(Xtr, ytr, cols, seed)
    return (m.predict_proba(sc.transform(Xte[cols]))[:, 1] >= thr).astype(int)


def gap_k(yh, idx):
    r = [yh[idx][(fee[idx] == g) & (yte_v[idx] == 1)].mean() for g in GROUPS]
    return float(np.nanmax(r) - np.nanmin(r))


def wilson(k, n, z=1.959964):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


preds, rows = {}, []
for name, cols in [("pruned_28", pruned_cols), ("deployed_26", deployed_cols)]:
    thr, vrec = val_threshold_k(cols)
    yh = test_pred_k(cols, thr)
    preds[name] = yh
    print(f"\n{name}: thr={thr:.4f} (val recall {vrec:.3f}) | "
          f"test recall {recall_score(yte_v, yh):.3f}, precision {precision_score(yte_v, yh):.3f}")
    for g in GROUPS:                                   # the subgroup table of record
        pos = (fee == g) & (yte_v == 1)
        tp, n = int(yh[pos].sum()), int(pos.sum())
        lo, hi = wilson(tp, n)
        rows.append(dict(config=name, threshold=round(thr, 4), subgroup=g, n_at_risk=n,
                         TP=tp, FN=n - tp, recall=round(tp / n, 3), ci_low=lo, ci_high=hi))
        print(f"   {g:<8} n={n:<3} TP={tp:<3} FN={n-tp:<3} recall={tp/n:.3f} [{lo}, {hi}]")
    print(f"   dEO = {gap_k(yh, np.arange(len(yte_v))):.3f}")

allidx = np.arange(len(yte_v))
g_pr, g_dep = gap_k(preds["pruned_28"], allidx), gap_k(preds["deployed_26"], allidx)
rng = np.random.default_rng(42)
diffs = [gap_k(preds["deployed_26"], i) - gap_k(preds["pruned_28"], i)
         for i in (rng.integers(0, len(yte_v), len(yte_v)) for _ in range(1000))]
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\ndEO pruned {g_pr:.3f} -> deployed {g_dep:.3f} | change {g_dep-g_pr:+.3f}, "
      f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
print(f"Algorithm 1 line 12: accept_removal = {g_dep < g_pr}")
pd.DataFrame(rows).to_csv("source_subgroup_table.csv", index=False)   # Figure 7b comes from THIS file


# ===========================================================================
# RUN 2 — SHAP on SCALED inputs (fixes Figure 5)              [Source session]
# ===========================================================================
# The bug: section 8 of the v3 script passed raw Xte42 to a model trained on
# MinMax-scaled inputs, so every attribution was for out-of-range inputs.

import xgboost as xgb, matplotlib.pyplot as plt

sc_dep, m_dep = fit_k(Xtr, ytr, deployed_cols)                    # same fit as the bundle
Xte_s = pd.DataFrame(sc_dep.transform(Xte[deployed_cols]), columns=deployed_cols)
contribs = m_dep.get_booster().predict(
    xgb.DMatrix(Xte_s, feature_names=deployed_cols), pred_contribs=True)[:, :-1]
sv = pd.Series(np.abs(contribs).mean(axis=0), index=deployed_cols).sort_values(ascending=False)
sv.to_csv("cinque_v3_shap_values_SCALED.csv")

MAPPABLE = ["n_module_views", "n_logins", "engagement_span_days", "std_module_score",
            "total_score", "score_improvement", "total_attempts", "modules_incomplete",
            "gender_Female"]                                       # the 9 transfer columns
ENGAGEMENT = [c for c in deployed_cols if c.startswith(("n_", "pct_", "total_quiz",
                                                        "avg_quiz", "engagement_"))]
print("mappable share :", round(sv[MAPPABLE].sum() / sv.sum(), 3), "  (old, buggy value 0.676)")
print("engagement share:", round(sv[ENGAGEMENT].sum() / sv.sum(), 3), ENGAGEMENT)
print("gender rank    :", list(sv.index).index("gender_Female") + 1, "of", len(sv))
print(sv.head(10).round(3))

top = sv.head(10)[::-1]                                            # regenerate Figure 5
plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.barh(top.index, top.values, color="#5b7d99")
for i, v in enumerate(top.values):
    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("cinque_v3_FIG5_shap_SCALED.png", bbox_inches="tight", dpi=300)


# ===========================================================================
# RUN 3 — Matched transfer ceiling                         [TRANSFER session]
# ===========================================================================
# The 0.937 currently used as the ceiling belongs to a 23/24-column OULAD model
# on a different partition. The matched ceiling is a NINE-column model trained
# from scratch on the full OULAD pool and evaluated on the SAME held-out 7,422.

from sklearn.metrics import roc_auc_score

COLS = list(K.columns)
zs, ceil = [], []
for s in SEEDS:
    # zero-shot arm, unchanged: Source-9 applied cold, same evaluation partition
    Xtr_k, _, ytr_k, _ = train_test_split(K[COLS], yk, test_size=0.20, stratify=yk, random_state=s)
    sc_k = MinMaxScaler().fit(Xtr_k)
    Xb, yb = SMOTE(random_state=s).fit_resample(sc_k.transform(Xtr_k), ytr_k)
    src = XGBClassifier(**BEST, eval_metric="auc", random_state=42).fit(Xb, yb)
    zs.append(roc_auc_score(yo_te, src.predict_proba(sc_k.transform(O_te[COLS]))[:, 1]))

    # matched ceiling: 9 columns, full OULAD pool, same O_te
    sc_o = MinMaxScaler().fit(O_tr[COLS])
    Xb, yb = SMOTE(random_state=s).fit_resample(sc_o.transform(O_tr[COLS]), yo_tr)
    m = XGBClassifier(**BEST, eval_metric="auc", random_state=42).fit(Xb, yb)
    ceil.append(roc_auc_score(yo_te, m.predict_proba(sc_o.transform(O_te[COLS]))[:, 1]))

zs, ceil = np.array(zs), np.array(ceil)
print(f"n_train (full pool) = {len(O_tr):,} | eval = {len(O_te):,}")
print(f"zero-shot Source-9 -> OULAD : {zs.mean():.4f} +/- {zs.std(ddof=1):.4f}")
print(f"MATCHED ceiling (9 cols, OULAD from scratch, same partition): "
      f"{ceil.mean():.4f} +/- {ceil.std(ddof=1):.4f}")
print(f"matched transfer gap       : {ceil.mean()-zs.mean():+.4f}   "
      f"(old, confounded gap vs 0.937: {0.937-zs.mean():+.4f})")
pd.DataFrame(dict(seed=SEEDS, zero_shot=zs, matched_ceiling=ceil)).to_csv(
    "transfer_matched_ceiling.csv", index=False)


# ===========================================================================
# RUN 4 — Precision at MATCHED recall, both datasets    [Source then OULAD]
# ===========================================================================
# Replaces "the precision collapse is Source-specific". The two datasets are
# currently compared at different recalls (0.673 vs 0.739), so the comparison
# only shows that OULAD separates the classes better.
# Run block A in the Source session, block B in the OULAD session.

from sklearn.metrics import precision_recall_curve, average_precision_score


def precision_at_recall(y_true, p, target=0.70):
    pr, rc, _ = precision_recall_curve(y_true, p)
    ok = rc >= target
    return float(pr[ok].max()) if ok.any() else float("nan")


# --- block A: Source deployed ---
sc_a, m_a = fit_k(Xtr, ytr, deployed_cols)
p_source = m_a.predict_proba(sc_a.transform(Xte[deployed_cols]))[:, 1]
print(f"Source deployed: AP={average_precision_score(yte_v, p_source):.3f} | "
      f"precision@recall0.70={precision_at_recall(yte_v, p_source):.3f} | "
      f"prevalence={yte_v.mean():.3f}")

# --- block B: OULAD configuration of record (paste in the OULAD session) ---
# sc_b, m_b = fit(Xtr, ytr, pruned_cols)          # `fit` from the guard cell
# p_oulad = m_b.predict_proba(sc_b.transform(Xte[pruned_cols]))[:, 1]
# print(f"OULAD pruned_24: AP={average_precision_score(yte, p_oulad):.3f} | "
#       f"precision@recall0.70={precision_at_recall(yte, p_oulad):.3f} | "
#       f"prevalence={np.mean(yte):.3f}")


# ===========================================================================
# RUN 5 — Operating-point metrics across five seeds           [Source session]
# ===========================================================================
# Table 5 currently reports tau = 0.5 only, and the operating point only on
# seed 42. The threshold is re-derived per seed: a threshold is part of the
# fitted model, so it cannot be carried across splits.

from sklearn.metrics import f1_score, roc_auc_score

rows = []
for s in SEEDS:
    Xtr_s_, Xte_s_, ytr_s_, yte_s_ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    yte_sv = np.asarray(yte_s_)
    Xi, Xv, yi, yv = train_test_split(Xtr_s_, ytr_s_, test_size=0.25, stratify=ytr_s_, random_state=s)
    sc_i, m_i = fit_k(Xi, yi, deployed_cols, seed=s)
    pv = m_i.predict_proba(sc_i.transform(Xv[deployed_cols]))[:, 1]
    yv_v = np.asarray(yv)
    ts = np.sort(np.unique(pv))[::-1]
    rec_v = np.array([(pv[yv_v == 1] >= t).mean() for t in ts])
    thr = float(ts[int(np.argmax(rec_v >= TARGET_RECALL))])

    sc_f, m_f = fit_k(Xtr_s_, ytr_s_, deployed_cols, seed=s)
    p = m_f.predict_proba(sc_f.transform(Xte_s_[deployed_cols]))[:, 1]
    yh = (p >= thr).astype(int)
    rows.append(dict(seed=s, threshold=round(thr, 4), auc=roc_auc_score(yte_sv, p),
                     recall=recall_score(yte_sv, yh), precision=precision_score(yte_sv, yh),
                     f1_at_risk=f1_score(yte_sv, yh)))
r5 = pd.DataFrame(rows)
print(r5.round(3).to_string(index=False))
print("\nmean +/- sd:\n", r5.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())
r5.to_csv("source_operating_point_per_seed.csv", index=False)


# ===========================================================================
# RUN 6 — Logistic regression at ITS OWN threshold            [Source session]
# ===========================================================================
# Scoring LR at XGBoost's 0.0833 flags nearly everyone and says nothing.

from sklearn.linear_model import LogisticRegression


def fit_lr(Xa, ya, cols, seed=42):
    sc = MinMaxScaler().fit(Xa[cols])
    Xb, yb = SMOTE(random_state=seed).fit_resample(sc.transform(Xa[cols]), ya)
    return sc, LogisticRegression(max_iter=1000, C=1.0).fit(Xb, yb)


Xi, Xv, yi, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)
sc_l, m_l = fit_lr(Xi, yi, deployed_cols)
pv = m_l.predict_proba(sc_l.transform(Xv[deployed_cols]))[:, 1]
yv_v = np.asarray(yv)
ts = np.sort(np.unique(pv))[::-1]
rec_v = np.array([(pv[yv_v == 1] >= t).mean() for t in ts])
thr_lr = float(ts[int(np.argmax(rec_v >= TARGET_RECALL))])

sc_l2, m_l2 = fit_lr(Xtr, ytr, deployed_cols)
p_lr = m_l2.predict_proba(sc_l2.transform(Xte[deployed_cols]))[:, 1]
yh_lr = (p_lr >= thr_lr).astype(int)
print(f"LR  own threshold {thr_lr:.4f}: recall {recall_score(yte_v, yh_lr):.3f}, "
      f"precision {precision_score(yte_v, yh_lr):.3f}, AP {average_precision_score(yte_v, p_lr):.3f}")
print(f"LR  precision @ XGB's recall (0.673): {precision_at_recall(yte_v, p_lr, 0.673):.3f}")
print(f"XGB precision @ its own recall 0.673: 0.484")


# ===========================================================================
# RUN 7 — Two small diagnostics                               [Source session]
# ===========================================================================
# (a) validation recall at the deployed tau = 0.0833 (the number missing from
#     the FATAL 2 paragraph); (b) which fee dummy was lost at 43 -> 28.

Xi, Xv, yi, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)
sc_d, m_d = fit_k(Xi, yi, deployed_cols)
pv = m_d.predict_proba(sc_d.transform(Xv[deployed_cols]))[:, 1]
yv_v = np.asarray(yv)
for t in (0.0833,):
    print(f"validation (n={len(yv_v)}, at-risk={int(yv_v.sum())}) recall at tau={t}: "
          f"{(pv[yv_v == 1] >= t).mean():.3f}, precision {precision_score(yv_v, (pv >= t).astype(int)):.3f}")
print("fee dummies:", FEE_COLS)
print("surviving the 43->28 prune:", [c for c in FEE_COLS if c in pruned_cols])
print("already pruned before the screen:", [c for c in FEE_COLS if c not in pruned_cols])


# ===========================================================================
# RUN 8 — Is total_score redundant?                           [Source session]
# ===========================================================================
# total_score is the sum of the six module scores, which are also in the model.

no_total = [c for c in deployed_cols if c != "total_score"]
out = []
for s in SEEDS:
    Xtr_s_, Xte_s_, ytr_s_, yte_s_ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    row = dict(seed=s)
    for nm, cols in [("deployed_26", deployed_cols), ("no_total_25", no_total)]:
        sc_, m_ = fit_k(Xtr_s_, ytr_s_, cols, seed=s)
        row[nm] = roc_auc_score(np.asarray(yte_s_), m_.predict_proba(sc_.transform(Xte_s_[cols]))[:, 1])
    out.append(row)
r8 = pd.DataFrame(out)
print(r8.round(4).to_string(index=False))
print("mean difference (26 - 25):", round((r8.deployed_26 - r8.no_total_25).mean(), 4))
# also confirm the arithmetic identity, on the raw frame:
mods = [c for c in df.columns if c.startswith("score_m")]
print("max |total_score - sum(module scores)|:", float((df["total_score"] - df[mods].sum(axis=1)).abs().max()))


# ===========================================================================
# RUN 9 — TOST equivalence on the paired per-seed AUCs        [Source session]
# ===========================================================================
# Non-significant McNemar is NOT equivalence. Set delta BEFORE looking.
# Feed this the per-seed AUCs from the leak-free tuning run.

from scipy import stats

DELTA = 0.01          # declare this in the manuscript as pre-specified


def tost_paired(a, b, delta=DELTA):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (d.mean() + delta) / se          # H0: diff <= -delta
    t_hi = (d.mean() - delta) / se          # H0: diff >= +delta
    p = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    return d.mean(), se, p


# replace these with the per-seed AUCs from the leak-free run
full = [0.8868, 0.0, 0.0, 0.0, 0.0]
pruned = [0.8796, 0.0, 0.0, 0.0, 0.0]
deployed = [0.8719, 0.0, 0.0, 0.0, 0.0]
for nm, arr in [("full vs pruned", pruned), ("full vs deployed", deployed)]:
    m_, se_, p_ = tost_paired(full, arr)
    print(f"{nm}: mean diff {m_:+.4f} (SE {se_:.4f}), TOST p = {p_:.3f} at delta = {DELTA} "
          f"-> {'equivalent' if p_ < .05 else 'NOT shown equivalent'}")


# ===========================================================================
# RUN 10 — PSI before the percentile transform               [OULAD session]
# ===========================================================================
# Percentile ranks are uniform by construction, so PSI on them is ~0 whatever
# the underlying distributions look like. Compute it on the raw variables.

def psi(a, b, bins=10):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    cuts = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return float("nan")
    pa = np.histogram(a, bins=cuts)[0] / len(a) + 1e-6
    pb = np.histogram(b, bins=cuts)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


source_raw = pd.read_excel(SOURCE_DATA)          # raw, before to_pct()
oulad_raw = pd.read_parquet(f"{OUT_DIR}/oulad_engagement.parquet")
for kcol, ocol in [("n_module_views", "n_clicks"), ("n_logins", "n_sessions"),
                   ("engagement_span_days", "engagement_span_days")]:
    print(f"PSI raw {kcol} vs {ocol}: {psi(source_raw[kcol], oulad_raw[ocol]):.2f}")
    print(f"   Source median {source_raw[kcol].median():.0f} | OULAD median {oulad_raw[ocol].median():.0f}")


# ===========================================================================
# RUN 11 — Archive every run next to the code that produced it
# ===========================================================================
# Reviewers reproduce from the repository, so each reported number needs a file.

import json, os, sys, sklearn, xgboost, imblearn

OUT = "cinque_final_outputs"
os.makedirs(OUT, exist_ok=True)
json.dump(dict(python=sys.version.split()[0], sklearn=sklearn.__version__,
               xgboost=xgboost.__version__, imblearn=imblearn.__version__,
               seeds=SEEDS, target_recall=TARGET_RECALL, tost_delta=DELTA),
          open(f"{OUT}/environment.json", "w"), indent=2)
for f in ["source_subgroup_table.csv", "cinque_v3_shap_values_SCALED.csv",
          "cinque_v3_FIG5_shap_SCALED.png", "transfer_matched_ceiling.csv",
          "source_operating_point_per_seed.csv"]:
    if os.path.exists(f):
        os.replace(f, f"{OUT}/{f}")
print(sorted(os.listdir(OUT)))
# Then, in Model_Full_exp.ipynb, delete or mark cells 25-38: they are the
# superseded v2 run (26 -> 23 features, AUC 0.860, tau = 0.0442).
