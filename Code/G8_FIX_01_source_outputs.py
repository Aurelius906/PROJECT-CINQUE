"""
GROUP 8 - FIX PACK 01 - SOURCE-INSTITUTION OUTPUTS            (exemplar code)
==============================================================================
WHAT THIS IS
  Look-and-do-same code. It regenerates, from YOUR data and YOUR pipeline,
  every source-institution number in Manuscript D that has no released code
  today, and writes each one to outputs/ so acceptance_gate.py can check it.

WHERE TO RUN
  The same Colab session as rerun_pruning_v3_original_protocol.py, AFTER its
  sections 0-2 have run. These names must already exist:
      df, X, y, params, FEE_COLS, pruned_cols, deployed_cols, CATS
  For block B also run CELL 4 of cinque_final_runs.py first (it builds K).

HOW TO USE
  1. Run block by block (each block starts with "# ===== BLOCK").
  2. Each block prints GOT vs PAPER. If they differ, the CODE wins:
     correct the manuscript to the GOT value and list it in your Return Note.
  3. Never edit a number to match the paper. Never commit model_ready_dataset_1.xlsx.
  4. When all blocks are done: run acceptance_gate.py (pointed at outputs/).

Time: about 2-3 hours of Colab CPU in total. Blocks F and H are the slowest.
"""
import os, json, time, tracemalloc
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import xgboost as xgb

os.makedirs("outputs", exist_ok=True)
SEEDS5 = [42, 0, 123, 2024, 7]
SPLITS20 = SEEDS5 + list(range(100, 115))           # Section 4.8
DELTA = 0.01                                         # Section 4.10, pre-specified
TARGET_RECALL = 0.75
LOG = {}                                             # everything printed also goes here


def say(label, got, paper=None, nd=4):
    """Print GOT vs PAPER and keep it for the run log."""
    g = None if got is None else round(float(got), nd)
    differs = False
    if paper is not None and g is not None:
        dp = len(str(paper).split(".")[1]) if "." in str(paper) else 0   # decimals the paper prints
        differs = round(float(got), dp) != round(float(paper), dp)
    flag = "   <-- DIFFERS: fix the paper" if differs else ""
    print(f"{label:58s} GOT {g!s:>9}   PAPER {paper!s:>9}{flag}")
    LOG[label] = dict(got=g, paper=paper, differs=differs)


def fit(Xa, ya, cols, seed, model_params=None):
    sc = MinMaxScaler().fit(Xa[cols])
    Xb, yb = SMOTE(random_state=seed).fit_resample(sc.transform(Xa[cols]), ya)
    m = XGBClassifier(**(model_params or params)).fit(Xb, yb)
    return sc, m


def proba(sc, m, Xb_, cols):
    return m.predict_proba(sc.transform(Xb_[cols]))[:, 1]


def val_threshold(Xtr_, ytr_, cols, seed, target=TARGET_RECALL):
    """Highest threshold reaching target recall on a 25% validation split of
    the TRAINING partition (Section 4.4). Test data never touched."""
    Xi, Xv, yi, yv = train_test_split(Xtr_, ytr_, test_size=0.25, stratify=ytr_, random_state=seed)
    sc, m = fit(Xi, yi, cols, seed)
    pv, yv = proba(sc, m, Xv, cols), np.asarray(yv)
    ts = np.sort(np.unique(pv))[::-1]
    rec = np.array([(pv[yv == 1] >= t).mean() for t in ts])
    i = int(np.argmax(rec >= target))
    return float(ts[i]), float(rec[i])


def tost_paired(a, b, delta=DELTA):
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    t_lo, t_hi = (d.mean() + delta) / se, (d.mean() - delta) / se
    p = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    tcrit = stats.t.ppf(0.975, n - 1)
    return dict(mean=d.mean(), sd=d.std(ddof=1), lo=d.mean() - tcrit * se,
                hi=d.mean() + tcrit * se, t_upper=t_hi, p=p, n=n)


CONFIGS = {"full": list(X.columns), "pruned": pruned_cols, "deployed": deployed_cols}

# ===== BLOCK A - Table 1, 5.2, 4.11: the 20-split comparison and TOST ========
# This is the headline equivalence result. No released code produces it today.
rows = []
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    for name, cols in CONFIGS.items():
        sc, m = fit(Xtr, ytr, cols, s)
        p = proba(sc, m, Xte, cols); yh = (p >= 0.5).astype(int)
        rows.append(dict(config=name, seed=s, auc=roc_auc_score(yte, p),
                         ap=average_precision_score(yte, p),
                         macro_f1=f1_score(yte, yh, average="macro"),
                         precision_05=precision_score(yte, yh, zero_division=0),
                         recall_05=recall_score(yte, yh)))
ps = pd.DataFrame(rows)

# transfer9 (Section 5.13, 0.851): needs K from cinque_final_runs.py CELL 4
if "K" in globals():
    for s in SPLITS20:
        Ktr, Kte, ytr, yte = train_test_split(K, y.values, test_size=0.20, stratify=y.values, random_state=s)
        sc = MinMaxScaler().fit(Ktr)
        Xb, yb = SMOTE(random_state=s).fit_resample(sc.transform(Ktr), ytr)
        m = XGBClassifier(**params).fit(Xb, yb)
        ps.loc[len(ps)] = dict(config="transfer9", seed=s,
                               auc=roc_auc_score(yte, m.predict_proba(sc.transform(Kte))[:, 1]))
ps.to_csv("outputs/per_seed_metrics.csv", index=False)

piv = ps.pivot(index="seed", columns="config", values="auc").loc[SPLITS20]
for cfg, want in [("full", 0.868), ("pruned", 0.871), ("deployed", 0.861), ("transfer9", 0.851)]:
    if cfg in piv:
        say(f"Table 1  {cfg} 20-split mean AUC", piv[cfg].mean(), want, 3)
        say(f"Table 1  {cfg} 20-split SD", piv[cfg].std(ddof=1),
        # Table 1 for the first three, Section 5.13 for transfer9 (0.851 +/- 0.011)
        {"full": 0.018, "pruned": 0.016,
         "deployed": 0.021, "transfer9": 0.011}.get(cfg), 3)
t1 = tost_paired(piv["pruned"], piv["full"])
say("4.6/5.2  paired diff pruned-full", t1["mean"], 0.0034)
say("4.6/5.2  95% CI low", t1["lo"], 0.0010); say("4.6/5.2  95% CI high", t1["hi"], 0.0058)
say("4.6/5.2  TOST t(19) upper", t1["t_upper"], -5.84, 2)
say("4.10     SD of 20 paired differences", t1["sd"], 0.0051)
t2 = tost_paired(piv["pruned"], piv["deployed"])
say("4.11     paired diff pruned-deployed", t2["mean"], 0.0106)
say("4.11     95% CI low", t2["lo"], 0.0062); say("4.11     95% CI high", t2["hi"], 0.0151)
say("4.11     TOST p", t2["p"], 0.614, 3)
say("4.11     deployed seed-42 single split", piv.loc[42, "deployed"], 0.852, 3)
say("5.2      pruned five-seed mean", piv.loc[SEEDS5, "pruned"].mean(), 0.880, 3)
for cfg, want in [("full", 0.788), ("pruned", 0.792), ("deployed", 0.786)]:
    say(f"Table 1  macro-F1 {cfg}", ps[ps.config == cfg].macro_f1.mean(), want, 3)

# ===== BLOCK B - seed-42 prediction file + operating point (4.4, 5.4, 5.6) =====
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
yte_v = np.asarray(yte)
s42 = pd.DataFrame({"y_true": yte_v,
                    "fee_status": df.loc[Xte.index, "fee_payment_status"].astype(str).values,
                    "gender": df.loc[Xte.index, "gender"].astype(str).values})
for name, cols in CONFIGS.items():
    sc, m = fit(Xtr, ytr, cols, 42)
    s42[f"proba_{name}"] = proba(sc, m, Xte, cols)
thr = {}
for name in ["pruned", "deployed"]:
    thr[name], vrec = val_threshold(Xtr, ytr, CONFIGS[name], 42)
    say(f"4.4      tau ({name})", thr[name], {"pruned": 0.0755, "deployed": 0.0833}[name])
s42.to_csv("outputs/seed42_predictions.csv", index=False)

yh = (s42.proba_deployed >= thr["deployed"]).astype(int)
tp = int(((yh == 1) & (yte_v == 1)).sum()); fp = int(((yh == 1) & (yte_v == 0)).sum())
say("5.6      deployed TP at tau", tp, 76, 1); say("5.6      deployed FP at tau", fp, 81, 1)
yb05 = (s42.proba_full >= 0.5).astype(int)
say("5.6      baseline TP at 0.5", int(((yb05 == 1) & (yte_v == 1)).sum()), 62, 1)
say("5.6      baseline FP at 0.5", int(((yb05 == 1) & (yte_v == 0)).sum()), 22, 1)
# The same deployed model at 0.5, so the reader can separate the effect of the
# model from the effect of the threshold. Section 5.6 reports both: "At the same
# 0.5 threshold, the deployed model recovered 60 of 113 at-risk students with 21
# false positives." They are published figures, so they are checked, not just
# printed.
yd05 = (s42.proba_deployed >= 0.5).astype(int)
say("5.6      deployed TP at 0.5", int(((yd05 == 1) & (yte_v == 1)).sum()), 60, 1)
say("5.6      deployed FP at 0.5", int(((yd05 == 1) & (yte_v == 0)).sum()), 21, 1)

# ===== BLOCK C - 5x5 repeated CV file (4.8, 5.2) ===============================
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
cv = []
for fold, (tr, te) in enumerate(rskf.split(X, y)):
    for name, cols in CONFIGS.items():
        sc, m = fit(X.iloc[tr], y.iloc[tr], cols, 42)
        cv.append(dict(config=name, fold=fold, auc=roc_auc_score(y.iloc[te], proba(sc, m, X.iloc[te], cols))))
cv = pd.DataFrame(cv); cv.to_csv("outputs/repeated_cv55.csv", index=False)
for cfg, want in [("full", 0.864), ("pruned", 0.868), ("deployed", 0.858)]:
    say(f"5.2      5x5 CV {cfg}", cv[cv.config == cfg].auc.mean(), want, 3)

# ===== BLOCK D - 4.2 label-rebuild checks and fee-engagement association ======
mods = [c for c in df.columns if c.startswith("score_m")]
# At-risk is the positive class and a HIGHER summed score means LOWER risk,
# so the score is negated. Without the minus sign the check prints 1 - 0.784.
say("4.2      AUC of summed module score alone", roc_auc_score(y, -df["total_score"]), 0.784, 3)
dt_auc = []
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(df[mods], y, test_size=0.20, stratify=y, random_state=s)
    dt = DecisionTreeClassifier(max_depth=2, random_state=s).fit(Xtr, ytr)
    dt_auc.append(roc_auc_score(yte, dt.predict_proba(Xte)[:, 1]))
say("4.2      depth-2 tree on score columns", np.mean(dt_auc), 0.746, 3)
score_cols = [c for c in deployed_cols if c.startswith("score_m") or c in ("total_score",)]
no_score = [c for c in deployed_cols if c not in score_cols]
a_ns = []
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    sc, m = fit(Xtr, ytr, no_score, s); a_ns.append(roc_auc_score(yte, proba(sc, m, Xte, no_score)))
say("4.2      deployed minus all score columns", np.mean(a_ns), 0.839, 3)
fee_ord = df["fee_payment_status"].map({"Unpaid": 0, "Partial": 1, "Paid": 2})
rho = stats.spearmanr(fee_ord, df["n_module_views"])
say("4.2      Spearman rho fee vs module views", rho.statistic, 0.284, 3)
# bootstrap CI for rho (seeded)
rng = np.random.default_rng(42); idx = np.arange(len(df)); boots = []
for _ in range(2000):
    i = rng.choice(idx, len(idx), replace=True)
    boots.append(stats.spearmanr(fee_ord.values[i], df["n_module_views"].values[i]).statistic)
say("4.2      rho 95% CI low", np.percentile(boots, 2.5), 0.251, 3)
say("4.2      rho 95% CI high", np.percentile(boots, 97.5), 0.317, 3)
H = stats.kruskal(*[df.loc[df.fee_payment_status == g, "n_module_views"] for g in ["Unpaid", "Partial", "Paid"]])
say("4.2      Kruskal-Wallis H", H.statistic, 315.4, 1)
for col, want in [("n_logins", 0.145), ("engagement_span_days", 0.093)]:
    say(f"4.2      Spearman rho fee vs {col}", stats.spearmanr(fee_ord, df[col]).statistic, want, 3)

# ===== BLOCK E - 4.6 retention-criterion comparison (gain, total_gain, weight) =
Xtr0, _, ytr0, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
sc0 = MinMaxScaler().fit(Xtr0)
Xb0, yb0 = SMOTE(random_state=42).fit_resample(pd.DataFrame(sc0.transform(Xtr0), columns=X.columns), ytr0)
base = XGBClassifier(**params).fit(Xb0, yb0)
crit_sets = {}
for itype in ["gain", "total_gain", "weight"]:
    imp = pd.Series(base.get_booster().get_score(importance_type=itype)).reindex(X.columns).fillna(0)
    imp = imp.sort_values(ascending=False); cum = imp.cumsum() / imp.sum()
    keep = list(cum[cum <= 0.95].index)
    if len(keep) < len(cum): keep.append(cum.index[len(keep)])
    crit_sets[itype] = keep
    inter = len(set(keep) & set(pruned_cols)); union = len(set(keep) | set(pruned_cols))
    say(f"4.6      {itype}: columns retained", len(keep), {"gain": 31, "total_gain": 24, "weight": 29}[itype], 1)
    say(f"4.6      {itype}: overlap with SHAP set", inter, {"gain": 24, "total_gain": 23, "weight": 26}[itype], 1)
    say(f"4.6      {itype}: Jaccard", inter / union, {"gain": 0.686, "total_gain": 0.793, "weight": 0.839}[itype], 3)
    print("         fee indicators kept:", [c for c in FEE_COLS if c in keep])
g_auc = []
for s in SEEDS5:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    sc, m = fit(Xtr, ytr, crit_sets["gain"], s); g_auc.append(roc_auc_score(yte, proba(sc, m, Xte, crit_sets["gain"])))
d = piv.loc[SEEDS5, "pruned"].values - np.array(g_auc)
tt = stats.ttest_1samp(d, 0)
say("4.6/5.2  gain-pruned five-seed mean", np.mean(g_auc), 0.8712)
say("4.6/5.2  SHAP minus gain, mean", d.mean(), 0.0090)
say("4.6/5.2  t(4)", tt.statistic, 2.20, 2); say("4.6/5.2  p", tt.pvalue, 0.093, 3)

# ===== BLOCK F - 4.7 SMOTE inside each CV fold (tuning sensitivity) ===========
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
# Table S4 row 2 records this 54-combination grid and its selected values
# (depth 4, 100 trees, lr 0.05, subsample 1.0). 54 combinations x 5 folds.
pipe = ImbPipeline([("sc", MinMaxScaler()), ("sm", SMOTE(random_state=42)),
                    ("xgb", XGBClassifier(eval_metric="auc", random_state=42, n_jobs=-1))])
grid = {"xgb__max_depth": [4, 6, 8], "xgb__n_estimators": [100, 200, 300],
        "xgb__learning_rate": [0.01, 0.05, 0.1], "xgb__subsample": [0.8, 1.0]}
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                  scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
p_sens = {k.replace("xgb__", ""): v for k, v in gs.best_params_.items()}
p_sens.update(eval_metric="auc", random_state=42, n_jobs=-1)
print("4.7 documented search winner:", p_sens, "CV AUC", round(gs.best_score_, 3))
print("TABLE S4 row 2 records: depth 4, 100 trees, lr 0.05, subsample 1.0")
sens = {k: [] for k in CONFIGS}
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    for name, cols in CONFIGS.items():
        sc, m = fit(Xtr, ytr, cols, s, p_sens); sens[name].append(roc_auc_score(yte, proba(sc, m, Xte, cols)))
for cfg, want in [("full", 0.882), ("pruned", 0.883), ("deployed", 0.875)]:
    say(f"4.7      sensitivity {cfg}", np.mean(sens[cfg]), want, 3)
r = tost_paired(sens["pruned"], sens["full"]); say("4.7      pruned-full diff", r["mean"], 0.0005)
r = tost_paired(sens["pruned"], sens["deployed"]); say("4.7      pruned-deployed diff", r["mean"], 0.0083)
r = tost_paired(sens["full"], sens["deployed"]); say("4.7      full-deployed diff (write as full minus deployed)", r["mean"], 0.0077)
pd.DataFrame(sens, index=SPLITS20).to_csv("outputs/sensitivity_smote_in_fold.csv")

# ===== BLOCK G - 5.3 perturbation of the six top SHAP features ===============
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
sc, m = fit(Xtr, ytr, deployed_cols, 42)
sv = pd.read_csv("cinque_v3_shap_values_SCALED.csv", index_col=0).iloc[:, 0]   # from RUN 2
top6 = list(sv.index[:6]); base_auc = roc_auc_score(yte, proba(sc, m, Xte, deployed_cols))
pert = []
for lvl in [0.05, 0.10, 0.20]:
    for draw in range(20):
        rng = np.random.default_rng(1000 * int(lvl * 100) + draw)
        Xp = Xte.copy()
        for c in top6:
            sd = Xtr[c].std()
            Xp[c] = np.clip(np.round(Xte[c] + rng.normal(0, lvl * sd, len(Xte))), Xtr[c].min(), Xtr[c].max())
        changed = float((Xp[top6].values != Xte[top6].values).mean())
        pp = proba(sc, m, Xp, deployed_cols); yhp = (pp >= thr["deployed"]).astype(int)
        pert.append(dict(level=lvl, draw=draw, cells_changed=changed,
                         d_auc=roc_auc_score(yte, pp) - base_auc,
                         recall=recall_score(yte, yhp), precision=precision_score(yte, yhp)))
pert = pd.DataFrame(pert); pert.to_csv("outputs/perturbation_5_3.csv", index=False)
print(pert.groupby("level").agg(cells=("cells_changed", "mean"), d_auc=("d_auc", "mean"),
                                sd=("d_auc", "std"), recall=("recall", "mean"),
                                precision=("precision", "mean")).round(4))
print("PAPER 5.3: cells 26/31/41%, dAUC +0.0013/+0.0003/-0.0022, SD 0.0025->0.0072")

# ===== BLOCK H - 5.5 gender gap on BOTH bases (fixes contradiction C8) ========
ge = s42["gender"].values
def eo(yh_, grp, yt):
    r = {g: yh_[(grp == g) & (yt == 1)].mean() for g in np.unique(grp) if (grp == g).sum() >= 30}
    return max(r.values()) - min(r.values()), r
g42, r42 = eo((s42.proba_deployed >= thr["deployed"]).values.astype(int), ge, yte_v)
say("5.4      gender EO gap, seed 42, deployed at tau", g42, 0.121, 3)
gaps, recs = [], []
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    t_s, _ = val_threshold(Xtr, ytr, deployed_cols, s)
    sc, m = fit(Xtr, ytr, deployed_cols, s)
    yh_ = (proba(sc, m, Xte, deployed_cols) >= t_s).astype(int)
    g_, r_ = eo(yh_, df.loc[Xte.index, "gender"].astype(str).values, np.asarray(yte))
    gaps.append(g_); recs.append(r_)
say("5.5      gender EO gap, 20-split mean, deployed at per-split tau", np.mean(gaps), 0.053, 3)
fg, _ = eo((s42.proba_pruned >= thr["pruned"]).values.astype(int), s42.fee_status.values, yte_v)
say("5.5      fee EO gap, seed 42, pruned at tau (same basis as 0.121)", fg, 0.490, 3)

# ===== BLOCK I - 5.7 complete-record analysis, AS THE PAPER DESCRIBES IT ======
# Paper 4.11: randomised search, 40 draws, 7 hyperparameters, run ONCE on the
# training partition of the FIRST split, then held fixed across 20 splits.
# The released RUN D does something else (nested, 25 draws, 10 seeds).
# Use THIS block if the paper text is what you did. If RUN D is what you did,
# keep RUN D, run it on all 20 splits, and rewrite 4.11 and 5.7 to match.
mods = [c for c in df.columns if c.startswith("score_m")]
complete = ((df[mods] == 0).sum(axis=1) == 0).values
Xc, yc = X[complete], y[complete]
say("5.7      complete-record n", len(Xc), 3089, 1)
# Grid and protocol exactly as Table S4 states: 2,916 combinations, 40 drawn
# at random, five-fold stratified CV, cost-sensitive class weighting (no SMOTE),
# run once on the training partition of the first split.
GRID = dict(max_depth=[2, 3, 4, 5], n_estimators=[300, 600, 1000],
            learning_rate=[0.01, 0.03, 0.05], subsample=[0.7, 0.9, 1.0],
            colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 5, 10], reg_lambda=[1, 5, 20])
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.20, stratify=yc, random_state=SPLITS20[0])
sc = MinMaxScaler().fit(Xtr[deployed_cols])
spw0 = (np.asarray(ytr) == 0).sum() / (np.asarray(ytr) == 1).sum()
srch = RandomizedSearchCV(XGBClassifier(eval_metric="auc", random_state=42, n_jobs=-1, scale_pos_weight=spw0),
                          GRID, n_iter=40, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          random_state=42).fit(sc.transform(Xtr[deployed_cols]), ytr)
p_ct = {**srch.best_params_, "eval_metric": "auc", "random_state": 42, "n_jobs": -1}
print("5.7 re-tuned configuration:", p_ct)
json.dump(dict(best_params=srch.best_params_, cv_auc=round(srch.best_score_, 3),
               n_iter=40, space_size=2916, random_state=42,
               tuned_on_seed=SPLITS20[0], cv_folds=5, class_weighting=True),
          open("outputs/complete_record_retune.json", "w"), indent=2)
print("TABLE S4 row 3 records: depth 2, 600 trees, lr 0.01, subsample 0.7,",
      "colsample 0.8, min child weight 10, lambda 20")
print("If the printed configuration differs, the CODE wins: Table S4 row 3 and 5.7 take it.")
arms = {k: [] for k in ["deployed", "retuned_smote", "retuned_weighted", "logreg"]}
for s in SPLITS20:
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.20, stratify=yc, random_state=s)
    sc = MinMaxScaler().fit(Xtr[deployed_cols]); A, B = sc.transform(Xtr[deployed_cols]), sc.transform(Xte[deployed_cols])
    Xb, yb = SMOTE(random_state=s).fit_resample(A, ytr)
    arms["deployed"].append(roc_auc_score(yte, XGBClassifier(**params).fit(Xb, yb).predict_proba(B)[:, 1]))
    arms["retuned_smote"].append(roc_auc_score(yte, XGBClassifier(**p_ct).fit(Xb, yb).predict_proba(B)[:, 1]))
    spw = (np.asarray(ytr) == 0).sum() / (np.asarray(ytr) == 1).sum()
    arms["retuned_weighted"].append(roc_auc_score(yte, XGBClassifier(**p_ct, scale_pos_weight=spw).fit(A, ytr).predict_proba(B)[:, 1]))
    arms["logreg"].append(roc_auc_score(yte, LogisticRegression(max_iter=2000, class_weight="balanced").fit(A, ytr).predict_proba(B)[:, 1]))
arms = pd.DataFrame(arms, index=SPLITS20); arms.to_csv("outputs/complete_records_5_7.csv")
for k, want in [("deployed", 0.757), ("retuned_smote", 0.791), ("retuned_weighted", 0.801), ("logreg", 0.800)]:
    say(f"5.7      {k}", arms[k].mean(), want, 3)
say("5.7      retuned_smote - deployed", tost_paired(arms.retuned_smote, arms.deployed)["mean"], 0.0341)
say("5.7      weighted - retuned_smote", tost_paired(arms.retuned_weighted, arms.retuned_smote)["mean"], 0.0102)

# ===== BLOCK J - 6.7 / 6.8 fee proxy recoverable from engagement ==============
eng3 = ["n_logins", "n_module_views", "engagement_span_days"]
two = df.fee_payment_status.isin(["Unpaid", "Paid"]).values
Xe, ye = df.loc[two, eng3], (df.loc[two, "fee_payment_status"] == "Unpaid").astype(int)
aucs, bals = [], []
from sklearn.metrics import balanced_accuracy_score
for s in SPLITS20:
    a, b_, ya, yb_ = train_test_split(Xe, ye, test_size=0.20, stratify=ye, random_state=s)
    aucs.append(roc_auc_score(yb_, DecisionTreeClassifier(max_depth=3, random_state=s).fit(a, ya).predict_proba(b_)[:, 1]))
    a, b_, ya, yb_ = train_test_split(df[eng3], df.fee_payment_status, test_size=0.20,
                                      stratify=df.fee_payment_status, random_state=s)
    bals.append(balanced_accuracy_score(yb_, DecisionTreeClassifier(max_depth=3, random_state=s).fit(a, ya).predict(b_)))
say("6.7      depth-3 tree, unpaid vs paid AUC", np.mean(aucs), 0.716, 3)
say("6.7      three-level balanced accuracy", np.mean(bals), 0.439, 3)

# ===== BLOCK K - 5.14 timings, measured as Section 5.14 states ===============
# 5.14 says: minimum of three repeats per seed, then the median across the five
# fixed seeds. The minimum is used because contention on a shared runtime can
# only add time. The committed run is the run of record: do not chase an
# earlier number, and copy these six values straight into 5.14.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
tim = {}
for name, cols in CONFIGS.items():
    per_seed = []
    for s in SEEDS5:
        reps = []
        for _ in range(3):
            t0 = time.perf_counter(); fit(Xtr, ytr, cols, s); reps.append(time.perf_counter() - t0)
        per_seed.append(min(reps))
    tim[f"fit_s_{name}"] = round(float(np.median(per_seed)), 2)
tracemalloc.start(); sc, m = fit(Xtr, ytr, deployed_cols, 42)
tim["peak_mb_deployed_fit"] = round(tracemalloc.get_traced_memory()[1] / 1e6, 1); tracemalloc.stop()
A = sc.transform(Xte[deployed_cols])
t0 = time.perf_counter(); m.predict_proba(A)
tim["inference_us_per_record"] = round((time.perf_counter() - t0) / len(A) * 1e6, 1)
t0 = time.perf_counter(); m.get_booster().predict(xgb.DMatrix(A), pred_contribs=True)
tim["shap_ms_per_record"] = round((time.perf_counter() - t0) / len(A) * 1e3, 2)
json.dump(tim, open("outputs/timings_5_14.json", "w"), indent=2)
LOG["5.14     timings of record"] = dict(got=tim, paper=None, differs=False)
print("5.14 timings of record:", tim)

# ===== BLOCK L - 6.8 total_score redundancy TOST ==============================
no_total = [c for c in deployed_cols if c != "total_score"]
a26, a25 = [], []
for s in SEEDS5:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=s)
    sc, m = fit(Xtr, ytr, deployed_cols, s); a26.append(roc_auc_score(yte, proba(sc, m, Xte, deployed_cols)))
    sc, m = fit(Xtr, ytr, no_total, s); a25.append(roc_auc_score(yte, proba(sc, m, Xte, no_total)))
r = tost_paired(a26, a25)
say("6.8      26 minus 25 (no total_score)", r["mean"], 0.0012); say("6.8      TOST p", r["p"], 0.005, 3)

# ===== SAVE THE RUN LOG =======================================================
import sklearn, imblearn, sys
LOG["_environment"] = dict(python=sys.version.split()[0], xgboost=xgb.__version__,
                           sklearn=sklearn.__version__, imblearn=imblearn.__version__,
                           params=params, pruned_cols=pruned_cols, deployed_cols=deployed_cols,
                           tau=thr)
json.dump(LOG, open("outputs/run_log_source.json", "w"), indent=2, default=str)
diffs = [k for k, v in LOG.items() if isinstance(v, dict) and v.get("differs")]
print("\nLines that differ from the paper (correct the PAPER, list them in the Return Note):")
for k in diffs: print("   ", k, LOG[k])
