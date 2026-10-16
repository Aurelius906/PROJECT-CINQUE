"""
Project Cinque — final runs, one sitting
========================================
Run these in ONE Colab runtime, in order. Cell 1 puts every input in Drive so a
disconnect costs a reload rather than a rebuild.

Files needed (put them in Drive once, in the folder set as BASE in cell 1):
    model_ready_dataset_1.xlsx     source-institution records
    studentInfo.csv                OULAD
    studentAssessment.csv          OULAD
    assessments.csv                OULAD
    studentVle.csv                 OULAD  (or oulad_engagement.parquet if you kept it)

Two questions are answered here:
    RUN C  does the transfer failure survive matching the populations?  (Objective 4b)
    RUN D  what is the complete-records AUC without a model-selection caveat?  (Section 5.13)
"""

# =============================================================================
# CELL 1 — mount, locate inputs, imports
# =============================================================================
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import glob, os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE = "/content/drive/MyDrive/cinque"          # adjust to your folder
os.makedirs(BASE, exist_ok=True)

def find(name):
    hits = (glob.glob(f"{BASE}/**/{name}", recursive=True)
            + glob.glob(f"/content/**/{name}", recursive=True)
            + glob.glob("/content/drive/MyDrive/**/" + name, recursive=True))
    assert hits, f"{name} not found — put it in {BASE}"
    return hits[0]

SEEDS   = [42, 0, 123, 2024, 7]
SEEDS20 = SEEDS + list(range(100, 115))
PARAMS_V3       = dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                       eval_metric="auc", random_state=42, n_jobs=-1)   # model of record
BEST_TRANSFER   = dict(max_depth=8, n_estimators=300, learning_rate=0.1, subsample=0.8)
print("inputs visible:", [os.path.basename(find(f)) for f in
      ["model_ready_dataset_1.xlsx", "studentInfo.csv", "studentAssessment.csv", "assessments.csv"]])


# =============================================================================
# CELL 2 — source-institution frame and design matrix
# =============================================================================
CATS = {"gender": ["Female", "Male"],
        "programme": ["Actuarial Science", "Biological Science", "Chemistry",
                      "Computer Science", "Physics"],
        "level": ["1st Year", "2nd Year"],
        "fee_payment_status": ["Paid", "Partial", "Unpaid"]}

df = pd.read_excel(find("model_ready_dataset_1.xlsx"))
y  = (df["at_risk"] == 0).astype(int)
assert len(df) == 3325 and y.sum() == 565, (len(df), int(y.sum()))

raw = df.drop(columns=[c for c in ["participant_id", "at_risk"] if c in df.columns])
for c, cats in CATS.items():
    raw[c] = pd.Categorical(raw[c], categories=cats)
X = pd.get_dummies(raw, columns=list(CATS))
assert X.shape[1] == 43, X.shape

FEE_COLS = [c for c in X.columns if c.startswith("fee_payment_status_")]
mods     = [c for c in df.columns if c.startswith("score_m")]
unsat    = (df[mods] == 0).sum(axis=1)                 # trailing zeros = modules not sat
print("source OK |", X.shape, "| at-risk", int(y.sum()),
      "| unsat>0:", int((unsat > 0).sum()), "| unsat>=3:", int((unsat >= 3).sum()))


# =============================================================================
# CELL 3 — pruning, to recover pruned_cols and deployed_cols
# =============================================================================
import xgboost as xgb
RETAIN = 0.95
Xtr0, _, ytr0, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
sc0 = MinMaxScaler().fit(Xtr0)
Xtr0_s = pd.DataFrame(sc0.transform(Xtr0), columns=X.columns)
Xb0, yb0 = SMOTE(random_state=42).fit_resample(Xtr0_s, ytr0)
base = XGBClassifier(**PARAMS_V3).fit(Xb0, yb0)
contrib = base.get_booster().predict(
    xgb.DMatrix(Xtr0_s, feature_names=list(X.columns)), pred_contribs=True)[:, :-1]
mean_abs = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
cum = mean_abs.cumsum() / mean_abs.sum()
pruned_cols = list(cum[cum <= RETAIN].index)
if len(pruned_cols) < len(cum):
    pruned_cols.append(cum.index[len(pruned_cols)])
deployed_cols = [c for c in pruned_cols if c not in FEE_COLS]
assert (len(pruned_cols), len(deployed_cols)) == (28, 26), (len(pruned_cols), len(deployed_cols))
print("pruning OK | 43 ->", len(pruned_cols), "->", len(deployed_cols))


# =============================================================================
# CELL 4 — transfer objects (K, O, fixed OULAD partition), saved to Drive
# =============================================================================
def pct(s):
    a = np.asarray(s, float); r = rankdata(a, method="average")
    return 100.0 * (r - 1) / (len(r) - 1)

K = pd.DataFrame({
    "eng_views_pct": pct(df["n_module_views"]), "eng_sess_pct": pct(df["n_logins"]),
    "eng_span_pct": pct(df["engagement_span_days"]), "score_sd": df["std_module_score"],
    "score_mean": df["total_score"] / 6.0, "score_improve": df["score_improvement"],
    "n_attempts": df["total_attempts"], "n_incomplete": df["modules_incomplete"],
    "is_female": (df["gender"] == "Female").astype(int)})
yk = y.values
assert len(K) == len(df)                              # row alignment for the unsat mask

KEY = ["id_student", "code_module", "code_presentation"]
info = pd.read_csv(find("studentInfo.csv"))
info["y"] = (info.final_result == "Withdrawn").astype(int)
sa = pd.read_csv(find("studentAssessment.csv")).merge(
    pd.read_csv(find("assessments.csv"))[["id_assessment", "code_module",
                                          "code_presentation", "assessment_type", "date"]],
    on="id_assessment", how="left")
sa = sa[sa.assessment_type == "TMA"].sort_values(KEY + ["date"])
sc_ = sa.groupby(KEY).agg(
    score_mean=("score", "mean"), score_sd=("score", "std"),
    score_improve=("score", lambda v: float(v.iloc[-1] - v.iloc[0]) if len(v) > 1 else 0.0),
    n_assess=("score", "count")).reset_index()
sc_["score_sd"] = sc_["score_sd"].fillna(0)

pq = glob.glob(f"{BASE}/**/oulad_engagement.parquet", recursive=True) \
     + glob.glob("/content/**/oulad_engagement.parquet", recursive=True)
if pq:
    eng = pd.read_parquet(pq[0])
else:
    vle = pd.read_csv(find("studentVle.csv"))         # a few minutes
    eng = (vle.groupby(KEY).agg(n_clicks=("sum_click", "sum"), n_sessions=("date", "nunique"),
           engagement_span_days=("date", lambda v: int(v.max() - v.min()))).reset_index())
    eng.to_parquet(f"{BASE}/oulad_engagement.parquet")

ou = info[KEY + ["gender", "num_of_prev_attempts", "y"]].merge(sc_, on=KEY).merge(eng, on=KEY)
O = pd.DataFrame({
    "eng_views_pct": pct(ou["n_clicks"]), "eng_sess_pct": pct(ou["n_sessions"]),
    "eng_span_pct": pct(ou["engagement_span_days"]), "score_sd": ou["score_sd"],
    "score_mean": ou["score_mean"] / 10.0, "score_improve": ou["score_improve"].fillna(0),
    "n_attempts": ou["num_of_prev_attempts"], "n_incomplete": ou["n_assess"].max() - ou["n_assess"],
    "is_female": (ou["gender"] == "F").astype(int)})
yo = ou["y"].values
ok = O["score_mean"].notna(); O = O[ok].reset_index(drop=True); yo = yo[ok.values]

O_tr, O_te, yo_tr, yo_te = train_test_split(O, yo, test_size=0.30, stratify=yo, random_state=1)
COLS = list(K.columns)
K.to_parquet(f"{BASE}/K.parquet"); O.to_parquet(f"{BASE}/O.parquet")
np.save(f"{BASE}/yk.npy", yk); np.save(f"{BASE}/yo.npy", yo)
print(f"K {K.shape} | O {O.shape} | pool {len(O_tr):,} | eval {len(O_te):,}")
# expect O (24740, 9), pool 17,318, eval 7,422 — the partition 0.6417 was measured on


# =============================================================================
# RUN C — does the transfer failure survive matching the populations?
# =============================================================================
def zero_shot(mask=None, cols=COLS):
    out = []
    Ksub = K if mask is None else K[mask]
    ysub = yk if mask is None else yk[mask]
    for s in SEEDS:
        Xtr_k, _, ytr_k, _ = train_test_split(Ksub[cols], ysub, test_size=0.20,
                                              stratify=ysub, random_state=s)
        sc_k = MinMaxScaler().fit(Xtr_k)
        Xb, yb = SMOTE(random_state=s).fit_resample(sc_k.transform(Xtr_k), ytr_k)
        m = XGBClassifier(**BEST_TRANSFER, eval_metric="auc", random_state=42).fit(Xb, yb)
        out.append(roc_auc_score(yo_te, m.predict_proba(sc_k.transform(O_te[cols]))[:, 1]))
    return np.array(out)

full_cohort = zero_shot()
matched     = zero_shot(mask=(unsat <= 2).values)
print(f"zero-shot, full source cohort      : {full_cohort.mean():.4f} ± {full_cohort.std(ddof=1):.4f}")
print(f"zero-shot, disengaged excluded (<=2): {matched.mean():.4f} ± {matched.std(ddof=1):.4f}")
print(f"difference: {matched.mean()-full_cohort.mean():+.4f}   (ceiling for reference: 0.9258)")
# near 0.642  -> failure survives population matching; the last confound is closed
# well above  -> part of the gap was population mismatch; Section 5.12 must say so


# =============================================================================
# RUN D — complete-records AUC with selection nested inside each split
# =============================================================================
GRID = dict(max_depth=[2, 3, 4, 5], n_estimators=[300, 600], learning_rate=[0.01, 0.03, 0.05],
            subsample=[0.7, 0.9, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
            min_child_weight=[1, 5, 10], reg_lambda=[1, 5, 20])
keep2 = (unsat <= 2).values
Xc, yc = X[keep2], y[keep2]
print("analysis sample:", Xc.shape, "| at-risk", int(np.asarray(yc).sum()))

rows = []
for s in SEEDS20[:10]:                                   # 10 seeds; raise if time allows
    Xtr_, Xte_, ytr_, yte_ = train_test_split(Xc, yc, test_size=0.20, stratify=yc, random_state=s)
    sc = MinMaxScaler().fit(Xtr_[deployed_cols])
    Xtr_s, Xte_s = sc.transform(Xtr_[deployed_cols]), sc.transform(Xte_[deployed_cols])
    yv = np.asarray(yte_); spw = (np.asarray(ytr_) == 0).sum() / (np.asarray(ytr_) == 1).sum()
    srch = RandomizedSearchCV(XGBClassifier(eval_metric="auc", scale_pos_weight=spw, n_jobs=-1),
                              GRID, n_iter=25, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=s),
                              random_state=s, n_jobs=1)
    srch.fit(Xtr_s, ytr_)                                # selection inside the training fold only
    p = srch.best_estimator_.predict_proba(Xte_s)[:, 1]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_s, ytr_)
    rows.append(dict(seed=s, nested=roc_auc_score(yv, p),
                     logreg=roc_auc_score(yv, lr.predict_proba(Xte_s)[:, 1]),
                     depth=srch.best_params_["max_depth"]))
    print("  seed", s, "done")
rd = pd.DataFrame(rows)
print(rd.round(4).to_string(index=False))
print("\nnested mean:", rd.nested.mean().round(4), "±", rd.nested.std(ddof=1).round(4))
print("logreg mean:", rd.logreg.mean().round(4), "±", rd.logreg.std(ddof=1).round(4))
print("modal depth:", rd.depth.mode().tolist())
# compare with 0.809 (selected on the subsample) and 0.757 (deployed configuration)


# =============================================================================
# CELL 7 — archive everything
# =============================================================================
OUT = f"{BASE}/final_runs"; os.makedirs(OUT, exist_ok=True)
pd.DataFrame(dict(seed=SEEDS, zero_shot_full=full_cohort,
                  zero_shot_matched=matched)).to_csv(f"{OUT}/runC_population_matched_transfer.csv", index=False)
rd.to_csv(f"{OUT}/runD_nested_complete_records.csv", index=False)
json.dump(dict(params_v3=PARAMS_V3, best_transfer=BEST_TRANSFER, seeds=SEEDS, seeds20=SEEDS20,
               pruned_cols=pruned_cols, deployed_cols=deployed_cols),
          open(f"{OUT}/config.json", "w"), indent=2)
print("archived to", OUT, "|", sorted(os.listdir(OUT)))
