"""
Project Cinque — SHAP-guided pruning: full reproducible rerun
=============================================================
Rebuilds the lost pruning section end-to-end and re-establishes the numbers of
record from ONE script, so tables, figures, and the deployed bundle all come
from the same runs.

Pipeline (as recorded):
  1. Baseline XGBoost on all 43 encoded features
  2. SHAP-guided pruning: retain features covering 95% cumulative mean |SHAP|
  3. Fairness screen: drop fee_payment_status encodings -> deployed feature set
  4. Evaluation: multi-seed 80/20 splits (+ optional repeated stratified CV),
     McNemar equivalence test, ROC curves, SHAP summary, threshold selection
  5. Saves: metrics CSV, ROC/SHAP figures + arrays, new locked bundle

Run in Colab or locally:  python rerun_pruning.py
Requires: xgboost, scikit-learn, pandas, numpy, matplotlib, joblib
          (shap NOT required — uses XGBoost's native pred_contribs)
"""

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from scipy.stats import chi2
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------------------------------
# 0. CONFIG — edit paths only
# ----------------------------------------------------------------------------
DATA_PATH   = "model_ready_dataset_1.xlsx"      # the REAL 3,325-row dataset
OLD_BUNDLE  = None                              # v3 uses HPARAMS (grid winner) for protocol fidelity
OUT_PREFIX  = "cinque_v3"

SEEDS          = [42, 0, 123, 2024, 7]          # ORIGINAL notebook seeds
RETAIN         = 0.95                            # cumulative |SHAP| retention for pruning
TARGET_RECALL  = 0.75                            # screening-threshold target (recorded protocol)
RUN_REPEATED_CV = True                           # also run 5x5 repeated stratified CV (CAN-DO Step 3)

HPARAMS = dict(                                  # grid-search winner under the ORIGINAL protocol
    n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1,
)                                                # (GridSearchCV over depth {4,6,8} x trees {100,200,300}
                                                 #  x lr {0.01,0.1}, 5-fold stratified on SMOTE-balanced
                                                 #  scaled seed-42 train; CV AUC 0.983 - inflated by SMOTE,
                                                 #  report held-out numbers only)

CATS = {
    "gender": ["Female", "Male"],
    "programme": ["Actuarial Science", "Biological Science", "Chemistry", "Computer Science", "Physics"],
    "level": ["1st Year", "2nd Year"],
    "fee_payment_status": ["Paid", "Partial", "Unpaid"],
}

# ----------------------------------------------------------------------------
# 1. LOAD + SANITY GATES (this is what prevents another 0.703 incident)
# ----------------------------------------------------------------------------
df = pd.read_excel(DATA_PATH)
assert df.shape[0] == 3325, f"WRONG DATASET: expected 3,325 rows, got {df.shape[0]}"
y = (df["at_risk"] == 0).astype(int)             # encoding is inverted: at_risk==0 IS the at-risk minority
assert y.sum() == 565, f"WRONG LABELS: expected 565 at-risk, got {y.sum()}"
print(f"[ok] dataset {df.shape}, at-risk = {y.sum()} ({y.mean():.1%})")

drop_cols = ["participant_id", "at_risk"]
raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
for c, cats in CATS.items():
    raw[c] = pd.Categorical(raw[c], categories=cats)
X = pd.get_dummies(raw, columns=list(CATS))
assert X.shape[1] == 43, f"expected 43 encoded features, got {X.shape[1]}"
print(f"[ok] encoded feature matrix: {X.shape}")

if OLD_BUNDLE:
    _old = joblib.load(OLD_BUNDLE)
    params = _old["model"].get_params()
    params.pop("missing", None)
    print("[ok] hyperparameters cloned from existing bundle")
else:
    params = HPARAMS

FEE_COLS = [c for c in X.columns if c.startswith("fee_payment_status_")]

# ----------------------------------------------------------------------------
# 2. SHAP-GUIDED PRUNING (defined on the seed-42 training fold, as recorded)
# ----------------------------------------------------------------------------
Xtr0, Xte0, ytr0, yte0 = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

_sc0 = MinMaxScaler().fit(Xtr0)
Xtr0_s = pd.DataFrame(_sc0.transform(Xtr0), columns=X.columns)
Xb0, yb0 = SMOTE(random_state=42).fit_resample(Xtr0_s, ytr0)
base = XGBClassifier(**params).fit(Xb0, yb0)
booster = base.get_booster()
dm = xgb.DMatrix(Xtr0_s, feature_names=list(X.columns))
contribs = booster.predict(dm, pred_contribs=True)[:, :-1]           # drop bias column
mean_abs = pd.Series(np.abs(contribs).mean(axis=0), index=X.columns).sort_values(ascending=False)

cum = mean_abs.cumsum() / mean_abs.sum()
pruned_cols = list(cum[cum <= RETAIN].index)
if len(pruned_cols) < len(cum):                                      # include the feature that crosses the line
    pruned_cols.append(cum.index[len(pruned_cols)])
print(f"[ok] pruning at {RETAIN:.0%} cumulative |SHAP|: 43 -> {len(pruned_cols)} features")

deployed_cols = [c for c in pruned_cols if c not in FEE_COLS]         # fairness screen
print(f"[ok] fairness screen (drop fee status): {len(pruned_cols)} -> {len(deployed_cols)} features")

CONFIGS = {"full_43": list(X.columns), "pruned": pruned_cols, "deployed": deployed_cols}

# ----------------------------------------------------------------------------
# 3. MULTI-SEED EVALUATION — the numbers of record
# ----------------------------------------------------------------------------
def evaluate(cols, Xtr, Xte, ytr, yte, seed=42):
    sc = MinMaxScaler().fit(Xtr[cols])
    Xtr_s = sc.transform(Xtr[cols]); Xte_s = sc.transform(Xte[cols])
    Xb, yb = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    m = XGBClassifier(**params).fit(Xb, yb)
    p = m.predict_proba(Xte_s)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return m, p, dict(
        auc=roc_auc_score(yte, p),
        macro_f1=f1_score(yte, yhat, average="macro"),
        precision=precision_score(yte, yhat, zero_division=0),
        recall=recall_score(yte, yhat),
    )

rows, seed42_probs, seed42_models = [], {}, {}
for seed in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    for name, cols in CONFIGS.items():
        m, p, met = evaluate(cols, Xtr, Xte, ytr, yte, seed=seed)
        rows.append(dict(seed=seed, config=name, **met))
        if seed == 42:
            seed42_probs[name], seed42_models[name] = p, m
    if seed == 42:
        Xte42, yte42, Xtr42, ytr42 = Xte, yte, Xtr, ytr

res = pd.DataFrame(rows)
summary = res.drop(columns=["seed"]).groupby("config").agg(["mean", "std"]).round(4)
print("\n===== MULTI-SEED SUMMARY (mean / std over", len(SEEDS), "seeds) =====")
print(summary.to_string())
res.to_csv(f"{OUT_PREFIX}_metrics_per_seed.csv", index=False)
summary.to_csv(f"{OUT_PREFIX}_metrics_summary.csv")

# ----------------------------------------------------------------------------
# 4. McNEMAR EQUIVALENCE (full vs pruned, seed-42 test set) — with chi^2 this time
# ----------------------------------------------------------------------------
yh_full   = (seed42_probs["full_43"] >= 0.5).astype(int)
yh_pruned = (seed42_probs["pruned"]  >= 0.5).astype(int)
b = int(((yh_full == yte42.values) & (yh_pruned != yte42.values)).sum())
c = int(((yh_full != yte42.values) & (yh_pruned == yte42.values)).sum())
chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0   # continuity-corrected
p_mcnemar = float(chi2.sf(chi2_stat, df=1)) if (b + c) > 0 else 1.0
print(f"\n[McNemar] discordant pairs b={b}, c={c}, chi2={chi2_stat:.3f}, p={p_mcnemar:.3f}")

# ----------------------------------------------------------------------------
# 5. FAIRNESS: equal-opportunity gap by fee status (deployed vs pruned)
# ----------------------------------------------------------------------------
fee_raw = df.loc[Xte42.index, "fee_payment_status"]
def eo_gap(probs, thresh=0.5):
    yhat = (probs >= thresh).astype(int)
    rec = {}
    for g in CATS["fee_payment_status"]:
        gmask = fee_raw.values == g
        if gmask.sum() < 30:                       # ORIGINAL subgroup filter
            continue
        mask = gmask & (yte42.values == 1)
        if mask.sum() > 0:
            rec[g] = yhat[mask].mean()
    return rec, (max(rec.values()) - min(rec.values()))
rec_pruned, gap_pruned = eo_gap(seed42_probs["pruned"])
rec_dep, gap_dep = eo_gap(seed42_probs["deployed"])
print(f"[fairness] EO recall by fee status — pruned: { {k: round(v,3) for k,v in rec_pruned.items()} } gap={gap_pruned:.2f}")
print(f"[fairness] EO recall by fee status — deployed: { {k: round(v,3) for k,v in rec_dep.items()} } gap={gap_dep:.2f}")

# ----------------------------------------------------------------------------
# 6. SCREENING THRESHOLD for the deployed model (target recall on validation)
# ----------------------------------------------------------------------------
Xtr_in, Xval, ytr_in, yval = train_test_split(Xtr42, ytr42, test_size=0.25, stratify=ytr42, random_state=42)
_sct = MinMaxScaler().fit(Xtr_in[deployed_cols])
_Xb, _yb = SMOTE(random_state=42).fit_resample(_sct.transform(Xtr_in[deployed_cols]), ytr_in)
m_thresh = XGBClassifier(**params).fit(_Xb, _yb)
pv = m_thresh.predict_proba(_sct.transform(Xval[deployed_cols]))[:, 1]
ths = np.unique(pv)
ok = [t for t in ths if recall_score(yval, (pv >= t).astype(int)) >= TARGET_RECALL]
threshold = float(max(ok)) if ok else 0.5
p_dep_te = seed42_probs["deployed"]
rec_at = recall_score(yte42, (p_dep_te >= threshold).astype(int))
prec_at = precision_score(yte42, (p_dep_te >= threshold).astype(int), zero_division=0)
print(f"[threshold] {threshold:.4f} -> test recall {rec_at:.3f}, precision {prec_at:.3f}")

# ----------------------------------------------------------------------------
# 7. OPTIONAL: repeated stratified CV (CAN-DO Step 3 — variance done properly)
# ----------------------------------------------------------------------------
if RUN_REPEATED_CV:
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    cvrows = []
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        for name, cols in CONFIGS.items():
            sc = MinMaxScaler().fit(X.iloc[tr][cols])
            Xb, yb = SMOTE(random_state=42).fit_resample(sc.transform(X.iloc[tr][cols]), y.iloc[tr])
            m = XGBClassifier(**params).fit(Xb, yb)
            p = m.predict_proba(sc.transform(X.iloc[te][cols]))[:, 1]
            cvrows.append(dict(fold=fold, config=name, auc=roc_auc_score(y.iloc[te], p)))
    cv = pd.DataFrame(cvrows)
    print("\n===== REPEATED 5x5 STRATIFIED CV (AUC mean / std over 25 folds) =====")
    print(cv.groupby("config")["auc"].agg(["mean", "std"]).round(4).to_string())
    cv.to_csv(f"{OUT_PREFIX}_repeated_cv.csv", index=False)

# ----------------------------------------------------------------------------
# 8. FIGURES — ROC (Figure 4) and SHAP summary (Figure 5), from THESE runs
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(5.4, 4.4))
styles = {"full_43": ("--", "#5b7d99", "Full, 43 features"),
          "pruned":  ("-.", "#3d7a4f", f"Pruned, {len(pruned_cols)} features"),
          "deployed": ("-", "#1c4a2c", f"Deployed, {len(deployed_cols)} features")}
for name, (ls, ccol, lab) in styles.items():
    fpr, tpr, _ = roc_curve(yte42, seed42_probs[name])
    auc = roc_auc_score(yte42, seed42_probs[name])
    ax.plot(fpr, tpr, ls, color=ccol, lw=1.8, label=f"{lab} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], ":", color="#999", lw=1, label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", fontsize=8, frameon=True)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{OUT_PREFIX}_FIG4_roc.png", bbox_inches="tight", dpi=300); plt.close()

dm_dep = xgb.DMatrix(Xte42[deployed_cols], feature_names=deployed_cols)
sc_dep = seed42_models["deployed"].get_booster().predict(dm_dep, pred_contribs=True)[:, :-1]
sv = pd.Series(np.abs(sc_dep).mean(axis=0), index=deployed_cols).sort_values(ascending=False)
top = sv.head(10)[::-1]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.barh(top.index, top.values, color="#5b7d99")
for i, v in enumerate(top.values):
    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(f"{OUT_PREFIX}_FIG5_shap.png", bbox_inches="tight", dpi=300); plt.close()
sv.to_csv(f"{OUT_PREFIX}_shap_values.csv")
print(f"\n[figures] {OUT_PREFIX}_FIG4_roc.png, {OUT_PREFIX}_FIG5_shap.png written")

# ----------------------------------------------------------------------------
# 9. NEW LOCKED BUNDLE (same schema as the previous one, fully documented)
# ----------------------------------------------------------------------------
scaler = MinMaxScaler().fit(Xtr42[deployed_cols])
_Xbf, _ybf = SMOTE(random_state=42).fit_resample(
    pd.DataFrame(scaler.transform(Xtr42[deployed_cols]), columns=deployed_cols), ytr42)
final_model = XGBClassifier(**params).fit(_Xbf, _ybf)
p_final = final_model.predict_proba(
    pd.DataFrame(scaler.transform(Xte42[deployed_cols]), columns=deployed_cols))[:, 1]
bundle = dict(
    model=final_model, scaler=scaler, feature_names=deployed_cols,
    categories=CATS, raw_feature_order=list(raw.columns), cat_cols=list(CATS),
    decision_threshold=threshold,
    meta=dict(version="3.0 (original protocol: per-seed MinMax + SMOTE, grid-search params, original seeds)",
              model=f"XGBoost + SHAP-pruned (retain={RETAIN}) minus fee_payment_status",
              n_features=len(deployed_cols), dropped_fee_for_fairness=True,
              test_AUC=round(float(roc_auc_score(yte42, p_final)), 3),
              test_recall_at_threshold=round(float(recall_score(yte42, (p_final >= threshold).astype(int))), 3),
              test_precision_at_threshold=round(float(precision_score(yte42, (p_final >= threshold).astype(int), zero_division=0)), 3),
              mcnemar_full_vs_pruned=dict(b=b, c=c, chi2=round(chi2_stat, 3), p=round(p_mcnemar, 3)),
              eo_gap_pruned=round(float(gap_pruned), 2), eo_gap_deployed=round(float(gap_dep), 2),
              positive_class="at_risk(target=1)", trained_on="80% stratified split, seed42",
              seeds=SEEDS, script="rerun_pruning.py"),
)
joblib.dump(bundle, f"{OUT_PREFIX}_model_bundle.joblib")
with open(f"{OUT_PREFIX}_run_summary.json", "w") as f:
    json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
               for k, v in bundle["meta"].items() if k != "model"}, f, indent=2, default=str)
print(f"[bundle] {OUT_PREFIX}_model_bundle.joblib written — meta: {bundle['meta']['test_AUC']} AUC, "
      f"threshold {threshold:.4f}")
print("\nDONE. Report the numbers in *_metrics_summary.csv (and *_repeated_cv.csv) as the record;")
print("Figure 4/5 and Table 3 now come from the same runs by construction.")
