"""
GROUP 8 - FIX PACK 03 - TRANSFER AND ADAPTATION OUTPUTS         (exemplar code)
==============================================================================
WHERE TO RUN
  The RetTransUst.ipynb session (or cinque_final_runs.py CELL 4), after Stage 1,
  Stage 2 and the scaler-control cell have run.
  Names that must exist: K, yk, O, yo, O_tr, O_te, yo_tr, yo_te, BEST, SEEDS
  and the Stage 2 results frame (called `df` in the notebook) and `ctl`.

WHAT IT FIXES
  - writes transfer_matched_ceiling.csv and adaptation_budgets.csv into outputs/
  - re-runs the six-well-aligned-column test of 5.16 so it has released code
  - redraws Figure 5 with the MATCHED 0.926 ceiling as the reference line (C11)
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)
stage2 = df.copy()                    # the Stage 2 frame: seed, budget, arm, auc

# ===== BLOCK 1 - zero-shot and matched ceiling, any column subset ===========
def transfer(cols):
    zs, ceil = [], []
    for s in SEEDS:
        Xtr_k, _, ytr_k, _ = train_test_split(K[cols], yk, test_size=0.20, stratify=yk, random_state=s)
        sc_k = MinMaxScaler().fit(Xtr_k)
        Xb, yb = SMOTE(random_state=s).fit_resample(sc_k.transform(Xtr_k), ytr_k)
        src = XGBClassifier(**BEST, eval_metric="auc", random_state=42).fit(Xb, yb)
        zs.append(roc_auc_score(yo_te, src.predict_proba(sc_k.transform(O_te[cols]))[:, 1]))
        sc_o = MinMaxScaler().fit(O_tr[cols])
        Xb, yb = SMOTE(random_state=s).fit_resample(sc_o.transform(O_tr[cols]), yo_tr)
        m = XGBClassifier(**BEST, eval_metric="auc", random_state=42).fit(Xb, yb)
        ceil.append(roc_auc_score(yo_te, m.predict_proba(sc_o.transform(O_te[cols]))[:, 1]))
    return np.array(zs), np.array(ceil)

ALL9 = list(K.columns)
zs, ceil = transfer(ALL9)
pd.DataFrame(dict(seed=SEEDS, zero_shot=zs, matched_ceiling=ceil)).to_csv(
    "outputs/transfer_matched_ceiling.csv", index=False)
print(f"9 columns: zero-shot {zs.mean():.3f} +/- {zs.std(ddof=1):.3f} | ceiling {ceil.mean():.3f} +/- {ceil.std(ddof=1):.3f}")
print("PAPER 5.13: 0.642 +/- 0.047 | 0.926 +/- 0.001 | gap 0.284")

WELL6 = [c for c in ALL9 if c not in ("score_improve", "n_attempts", "n_incomplete")]
zs6, ceil6 = transfer(WELL6)
pd.DataFrame(dict(seed=SEEDS, zero_shot=zs6, matched_ceiling=ceil6)).to_csv(
    "outputs/transfer_six_aligned_columns.csv", index=False)
print(f"6 columns: zero-shot {zs6.mean():.3f} +/- {zs6.std(ddof=1):.3f} | ceiling {ceil6.mean():.3f}")
print("PAPER 5.16: 0.570 +/- 0.031 | 0.919 | gap 0.349")

# ===== BLOCK 2 - adaptation budgets file ====================================
bud = stage2[stage2.arm.isin(["fine-tuned", "from-scratch", "scratch"])].copy()
bud["arm"] = bud["arm"].replace({"fine-tuned": "finetuned", "from-scratch": "scratch"})
bud[["arm", "budget", "seed", "auc"]].to_csv("outputs/adaptation_budgets.csv", index=False)
piv = bud.pivot_table(index=["budget", "seed"], columns="arm", values="auc")
d = piv["finetuned"] - piv["scratch"]
print("\nmean benefit", round(d.mean(), 4), "(PAPER -0.0074) | negative cells", int((d < 0).sum()), "of", len(d), "(PAPER 29 of 30)")
print(d.groupby(level="budget").mean().round(4), "\nPAPER: 100 -0.0117, 1000 -0.0060, 2500 -0.0051, 5000 -0.0098")
try:
    ctl.to_csv("outputs/adaptation_scaler_control.csv")
except NameError:
    print("ctl missing: re-run the scaler-control cell, then this block.")

# ===== BLOCK 3 - Figure 5 with the matched ceiling (fixes C11) ==============
summ = bud.groupby(["arm", "budget"]).auc.agg(["mean", "std"]).reset_index()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
for arm, lab, ls in [("finetuned", "Fine-tuned from source model", "-o"), ("scratch", "Trained from scratch", "-s")]:
    t = summ[summ.arm == arm]
    a1.errorbar(t.budget, t["mean"], yerr=t["std"], fmt=ls, ms=4, capsize=2, label=lab)
a1.axhline(ceil.mean(), ls="--", c="grey", lw=1, label=f"Matched nine-column ceiling ({ceil.mean():.3f})")
a1.axhline(zs.mean(), ls=":", c="grey", lw=1, label=f"Zero-shot ({zs.mean():.3f})")
a1.set_xscale("log"); a1.set_xlabel("OULAD training records"); a1.set_ylabel("Held-out AUC")
a1.legend(fontsize=7, loc="lower right"); a1.set_title("(a)", loc="left")
g = d.groupby(level="budget").agg(["mean", "std"])
a2.errorbar(g.index, g["mean"], yerr=g["std"], fmt="-o", ms=4, capsize=2)
a2.axhline(0, c="grey", lw=1); a2.set_xscale("log")
a2.set_xlabel("OULAD training records"); a2.set_ylabel("Fine-tuned minus scratch (AUC)"); a2.set_title("(b)", loc="left")
for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("outputs/Figure5_adaptation_matched_ceiling.png", dpi=300)
print("Figure 5 saved. New caption text is in the Fix Pack report, row C11.")
