# =============================================================================
# M3 RELEASE STEP  -  run AFTER rerun_pruning_v3_original_protocol.py.
#
# This replaces the old release cell, which no longer works: it popped
# meta["eo_gap_pruned"] and meta["eo_gap_deployed"], which the M3 rename moved
# inside eo_gap_at_0_5_superseded, and it DELETED meta["mcnemar_full_vs_pruned"],
# which after the reversion is a published figure (Section 5.12, Table 4) and
# must stay. Both are now assertions instead: a stale bundle is caught, not
# silently shipped.
#
# WHAT IT DOES
#   1. finds cinque_v3_model_bundle.joblib and validates the meta shape
#   2. scans everything releasable for institution-identifying strings
#   3. writes RELEASE/ : the run summary always, the weights only if approved
#   4. prints the SHA-256 of each released file, for the F5 title-page hash
#   5. prints the availability statement to paste into the title page and 4.15
#
# TWO SWITCHES
#   RELEASE_WEIGHTS - N10. Leave False until the data custodian AND your
#                     supervisor have approved releasing trained weights.
#   DRY_RUN         - prints the plan and writes nothing. Read it, then flip.
# =============================================================================
RELEASE_WEIGHTS = False
DRY_RUN = True

import os as _zos, re as _zre, json as _zjson, glob as _zglob, hashlib as _zhash
import shutil as _zsh
import joblib as _zjl

_zROOTS = [".", "outputs", "/content", "/content/outputs",
           "/content/drive/MyDrive/PROJECT_CINQUE_OUTPUTS"]
_zREL = "RELEASE"
_zSENS = ["knust", "kumasi", "ghana", "hussrec", "dcs/30", "kaggle",
          "model_ready_dataset", "responses.csv", "coding_rounds"]
_zfail = []

print("DRY RUN - nothing will be written" if DRY_RUN else "APPLYING")
print(f"  release weights: {RELEASE_WEIGHTS}"
      f"{'' if RELEASE_WEIGHTS else '   (N10 not yet approved)'}\n")

# ---- 1. locate the bundle and the summary ----------------------------------
def _zfind(name):
    for r in _zROOTS:
        p = _zos.path.join(r, name)
        if _zos.path.isfile(p):
            return p
    hits = sorted(_zglob.glob(f"**/{name}", recursive=True))
    return hits[0] if hits else None

_zb = _zfind("cinque_v3_model_bundle.joblib")
_zs = _zfind("cinque_v3_run_summary.json")
print(f"  bundle   {_zb or 'NOT FOUND'}")
print(f"  summary  {_zs or 'NOT FOUND'}")
if not _zb:
    _zfail.append("cinque_v3_model_bundle.joblib not found - run the rerun script first")

# ---- 2. validate the meta against the CURRENT shape ------------------------
_zmeta = None
if _zb:
    _zmeta = _zjl.load(_zb)["meta"]
    print("\n  meta shape:")

    def _zck(label, ok, detail=""):
        print(f"    {label:<46} {'OK' if ok else 'FAIL'}   {detail}")
        if not ok:
            _zfail.append(f"{label}: {detail or 'failed'}")

    _zmc = _zmeta.get("mcnemar_full_vs_pruned")
    _zck("mcnemar_full_vs_pruned present", _zmc is not None,
         str(_zmc) if _zmc is not None else "MISSING - the old release cell deleted it")
    _zok = isinstance(_zmc, dict) and {"b", "c", "chi2", "p"} <= set(_zmc)
    _zck("  and KEPT, not deleted (5.12, Table 4)", _zok,
         "b, c, chi2, p all recorded" if _zok else "incomplete or deleted")
    _zsp = _zmeta.get("eo_gap_at_0_5_superseded")
    _zck("eo_gap_at_0_5_superseded present", isinstance(_zsp, dict),
         str(_zsp) if _zsp is not None else "MISSING - re-run the corrected script")
    for _zk in ("eo_gap_pruned", "eo_gap_deployed"):
        _zck(f"{_zk} absent from the top level", _zk not in _zmeta,
             "stale bundle: re-run the corrected script" if _zk in _zmeta else "renamed, as expected")
    _zck("written by the corrected script",
         _zmeta.get("script") == "rerun_pruning_v3_original_protocol.py",
         str(_zmeta.get("script")))

# ---- 3. scan everything releasable for identifying strings -----------------
if _zmeta:
    print("\n  anonymisation scan:")
    _zblob = _zjson.dumps(_zmeta, default=str).lower()
    _zhits = [t for t in _zSENS if t in _zblob]
    _zidx = _zre.findall(r"\b\d{7,8}\b", _zblob)
    print(f"    sensitive terms in the meta                  "
          f"{'FAIL  ' + str(_zhits) if _zhits else 'none'}")
    print(f"    7-8 digit numbers (index-number shaped)      "
          f"{'FAIL  ' + str(_zidx) if _zidx else 'none'}")
    if _zhits:
        _zfail.append(f"meta carries {_zhits}")
    if _zidx:
        _zfail.append(f"meta carries index-number-shaped values {_zidx}")

# ---- 4. assemble the release ------------------------------------------------
_zplan = []
if _zs:
    _zplan.append((_zs, f"{_zREL}/cinque_v3_run_summary.json", "always released"))
if _zb:
    _zplan.append((_zb, f"{_zREL}/cinque_v3_model_bundle.joblib",
                   "weights, released" if RELEASE_WEIGHTS else "weights, HELD BACK (N10)"))
print(f"\n  {_zREL}/ contents:")
for _zsrc, _zdst, _zwhy in _zplan:
    _zskip = ("HELD BACK" in _zwhy)
    print(f"    {'skip ' if _zskip else 'copy '} {_zos.path.basename(_zdst):<38} "
          f"{_zos.path.getsize(_zsrc):>10,} bytes   {_zwhy}")
print(f"    write NOTES.md{'':<30} {'':>10}         "
      f"explains the superseded key and the weights decision")
print(f"    write SHA256SUMS.txt{'':<24} {'':>10}         "
      f"digests for the F5 title page")

if _zfail:
    print(f"\n  STOPPING: {len(_zfail)} problem(s). Nothing written.")
    for _zf in _zfail:
        print(f"    - {_zf}")
elif DRY_RUN:
    print("\n  DRY_RUN is True. Set it to False and re-run to write RELEASE/.")
else:
    _zos.makedirs(_zREL, exist_ok=True)
    _zdone = []
    for _zsrc, _zdst, _zwhy in _zplan:
        if "HELD BACK" in _zwhy:
            continue
        _zsh.copy2(_zsrc, _zdst)
        _zdone.append(_zdst)
        print(f"    wrote {_zdst}")
    # ---- 4b. a notes file, so the deposit explains itself ------------------
    # The summary is copied byte for byte, so its digest is reproducible by
    # re-running the script. Explanations therefore go in a separate file
    # rather than being injected into the JSON.
    _zsp = (_zmeta or {}).get("eo_gap_at_0_5_superseded") or {}
    _zmc = (_zmeta or {}).get("mcnemar_full_vs_pruned") or {}
    _znotes = f"""# Project Cinque - deposited files

## What is here
- `cinque_v3_run_summary.json` - hyperparameters, operating threshold and every
  reported metric. Copied byte for byte from the script's output, so its
  SHA-256 can be reproduced by re-running the script.
{"- `cinque_v3_model_bundle.joblib` - the trained model, scaler and feature order."
 if RELEASE_WEIGHTS else "- The trained weights are NOT deposited. See below."}
- `SHA256SUMS.txt` - digests of the files above.

## Reading `eo_gap_at_0_5_superseded` - IMPORTANT
That key records {_zsp.get("pruned", "?")} (pruned) and {_zsp.get("deployed", "?")} (deployed).
**These are not the equal-opportunity gaps the paper reports.** They are measured
at the default 0.5 probability threshold and are retained for provenance only.

Section 5.4 reports the gaps at each configuration's own validation-selected
screening threshold, tau = 0.0755 (pruned) and tau = 0.0833 (deployed), which
give **0.490** and **0.385**. Those are the reported figures, and
`acceptance_gate.py` block 2 checks them. The key is named `_superseded` so that
no reader mistakes the 0.5-threshold values for the reported ones.

## Reading `mcnemar_full_vs_pruned`
b = {_zmc.get("b", "?")}, c = {_zmc.get("c", "?")}, chi-square = {_zmc.get("chi2", "?")}, p = {_zmc.get("p", "?")}.
This is **correctness** discordance, following Dietterich (1998): b counts cases
the full model classifies correctly and the pruned model does not, and c the
reverse. It is not prediction discordance, which counts cases where the two
models merely disagree and would give different totals.

## Model weights
{"The trained bundle is deposited with this record." if RELEASE_WEIGHTS else
"The weights are held by the data custodian and are not released: the model was"
"\nfitted on an institutional extract that cannot be redistributed. Requests for"
"\naccess go to the custodian."}

## Reproducing
Run `Code/rerun_pruning_v3_original_protocol.py`. Every reported figure comes
from that one script, so the tables and figures are consistent by construction.
The OULAD cross-context replication is separate: `Code/Oulad_exp.ipynb` produces
it and `Code/G8_FIX_02_verify_oulad.py` re-derives every OULAD figure from the
four committed artefacts alone.
"""
    with open(f"{_zREL}/NOTES.md", "w") as _zfh:
        _zfh.write(_znotes)
    _zdone.append(f"{_zREL}/NOTES.md")
    print(f"    wrote {_zREL}/NOTES.md  ({len(_znotes):,} chars)")

    # ---- 5. hashes for the F5 title page -----------------------------------
    print("\n  SHA-256, for the title page and the Zenodo record:")
    for _zd in _zdone:
        _zh = _zhash.sha256(open(_zd, "rb").read()).hexdigest()
        print(f"    {_zos.path.basename(_zd)}")
        print(f"      {_zh}")
    with open(f"{_zREL}/SHA256SUMS.txt", "w") as _zfh:
        for _zd in _zdone:
            _zfh.write(f"{_zhash.sha256(open(_zd,'rb').read()).hexdigest()}  "
                       f"{_zos.path.basename(_zd)}\n")
    print(f"    wrote {_zREL}/SHA256SUMS.txt")

# ---- 6. the wording that goes in the title page and 4.15 -------------------
print("\n  availability statement to paste (N10):")
if RELEASE_WEIGHTS:
    print("    The trained model bundle and its run summary are deposited with")
    print("    this record. The SHA-256 digests above identify the exact files.")
else:
    print("    The run summary, which records the hyperparameters, the operating")
    print("    threshold and every reported metric, is deposited with this record.")
    print("    The trained weights are held by the data custodian and are not")
    print("    released, because the model was fitted on an institutional extract")
    print("    that cannot be redistributed. Requests go to the custodian.")
    print("    [Confirm this wording with your supervisor before it ships.]")
