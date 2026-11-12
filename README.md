# PROJECT-CINQUE

Code and outputs for: *Trust Without Detectable Decision Gains: Explained
At-Risk Predictions, Educator Decisions, and the Limits of Model Portability.*

Every figure reported in the manuscript and the supplement is produced by the
scripts below and checked by `Code/acceptance_gate.py`. Where a script's output
and the manuscript disagree, the script is correct and the manuscript is wrong.

## Run order (each script and the sections it produces)

| # | Script | Produces |
|---|---|---|
| 1 | `Code/rerun_pruning_v3_original_protocol.py` | 4.4, 4.6, 5.4, Figure S3, the deployed bundle |
| 2 | `Code/G8_FIX_01_source_outputs.py` | 4.2, 4.6, 4.7, 4.11, 4.13, 5.2–5.7, 5.14 |
| 3 | `Code/Oulad_exp.ipynb`, then `Code/G8_FIX_02_verify_oulad.py` | 4.16, 5.12, Table 4, Table S9 |
| 4 | `Code/RetTransUst.ipynb`, then `Code/G8_FIX_03_*.py` | 4.17, 5.13, Figure 5, Table S10 |
| 5 | `Code/analyze_trust_study.py`, then `Code/G8_FIX_04_*.py` | 4.9, 4.10, 4.12, 5.1, 5.8–5.10 |
| 6 | `Code/acceptance_gate.py` | checks every reported number against `outputs/` |
| 7 | `Code/G8_M3_release.py` | assembles `RELEASE/` locally (not committed) |

Scripts 1 and 2 must run before 6: the gate reads their outputs. Script 3 is
self-contained and can run in any order relative to the others.

## Environment

Python 3.13.18, XGBoost 3.4.1, scikit-learn 1.6.1, imbalanced-learn 0.14.2.
Also pandas 2.2.3, NumPy 2.1.3, SciPy 1.16.3, SHAP 0.54.1.

```
pip install -r Code/requirements.txt
```

`Code/requirements.lock.txt` records the full environment of the run of record,
which was a Google Colab session. It contains Colab-only entries and is a record,
not an installable file; `Code/requirements.txt` is the one to install from.

The lock file is generated with `pip freeze` at tag time and is the record of
what the reported numbers were produced with. The four versions named first are
the ones stated in the manuscript.

## Verify

```
python Code/acceptance_gate.py            # last line must read: Safe to tag
python Code/G8_FIX_02_verify_oulad.py     # exit status 0
```

The gate checks every reported number against the committed outputs. The second
script is independent of it: it re-derives every OULAD figure from the four
committed OULAD artefacts alone, without the notebook, the model, or the raw
data, and cross-checks those four files against one another.

## Data

Institutional records are not shared, under the terms of the custodian's
authorisation, so the scripts that read them cannot be run from this repository
alone. Every figure they produce is committed in `outputs/`, and
`Code/acceptance_gate.py` re-derives each reported number from those files, so
the results stay checkable without the records themselves.

OULAD is public: Kuzilek, Hlosta and Zdrahal (2017), CC BY 4.0. It is downloaded
at run time by `Code/Oulad_exp.ipynb` and is not redistributed here.

Raw educator responses are not shared, in line with the consent given. The
group-level results they support are committed as
`outputs/educator_aggregates.csv`, and the acceptance gate checks the reported
figures against them.

## What this repository does not hold

- the institutional extract the models were fitted on
- raw educator responses or their coding rounds
- any API token or credential
- trained model weights (held by the data custodian, available on reasonable
  request)
- `seed42_predictions.csv`, `complete_records_5_7.csv` and
  `perturbation_5_3.csv`, which carry one row per student with the outcome and
  fee-payment status. De-identified, but still individual-level institutional
  data. The acceptance gate reads them locally before tagging; the log it
  produces is committed as `outputs/acceptance_gate_log.txt`.

## outputs/

Written by the scripts above and committed as the record of the reported
figures. The four OULAD artefacts are:

| File | Holds |
|---|---|
| `oulad_per_seed_metrics.csv` | AUC per seed and configuration, 5 seeds × 3 configurations |
| `oulad_seed42_predictions.csv` | seed-42 probabilities, labels and deprivation band |
| `oulad_shap_values.csv` | mean abs SHAP over all 43 encoded features, the ranking behind the 43 → 24 prune |
| `run_log_oulad.json` | thresholds, gaps, bootstrap interval, McNemar, and the Algorithm 1 line-12 decision |

## Reading the stored figures

Two stored values are easy to misread, and both are named so that they cannot
be taken as reported:

`eo_gap_at_0_5_superseded` in the model bundle records the fee-status gap at the
**default 0.5 threshold**. The reported gaps in Section 5.4 are measured at each
configuration's own validation-selected screening threshold and are different
numbers. The gate checks the reported pair, not these.

`mcnemar_full_vs_pruned` is **correctness** discordance, following Dietterich
(1998): *b* counts cases the full model classifies correctly and the pruned
model does not, and *c* the reverse. It is not prediction discordance, which
counts mere disagreement and gives different totals.
