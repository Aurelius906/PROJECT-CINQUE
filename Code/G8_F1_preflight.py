# =============================================================================
# F1 PREFLIGHT  -  run from the REPOSITORY ROOT before you commit.
#
# Checks the things the guide requires that nothing else checks:
#   1. Code/ holds G8_FIX_01 to G8_FIX_04, plus the scripts the README names
#   2. outputs/per_seed_metrics.csv has full, pruned, deployed AND transfer9
#      across all 20 seeds (42, 0, 123, 2024, 7, 100-114) - F1's "done when"
#   3. Data/ holds the mock dataset the README promises a reviewer
#   4. nothing forbidden is staged, and .gitignore covers what it must
#   5. no file in the repo carries an identifying string
#
# Read-only. Prints a verdict; writes nothing, stages nothing.
# =============================================================================
import os as _p_os, re as _p_re, sys as _p_sys, glob as _p_glob
import subprocess as _p_sub

_p_fail, _p_warn = [], []
def _p_ck(label, ok, detail="", hard=True):
    print(f"  {label:<52} {'OK  ' if ok else 'FAIL'}   {detail}")
    if not ok:
        (_p_fail if hard else _p_warn).append(f"{label}: {detail}")


def _p_main():
    global _p_fail, _p_warn
    _p_fail, _p_warn = [], []
    print("=" * 78 + f"\nF1 PREFLIGHT   cwd {_p_os.getcwd()}\n" + "=" * 78)

    _p_isrepo = _p_os.path.isdir("Code") or _p_os.path.isdir(".git")
    if not _p_isrepo:
        print("\n  STOP. This is not the repository root: no Code/ and no .git here.")
        print("  Nothing below would mean anything. Change to the checkout first, e.g.")
        print("    %cd /content/<your-repo>        # or wherever you cloned it")
        print("    !git rev-parse --show-toplevel  # prints the root if you are inside it")
        print("  Then re-run. If the repository is not cloned in this session yet, clone")
        print("  it first - the four OULAD artefacts in /content/outputs are what you")
        print("  will copy into it.")
        print("\n" + "=" * 78 + "\nNOT RUN\n" + "=" * 78)
        return

    # ---- 1. the fix scripts the guide requires in Code/ ------------------------
    print("\n1. Code/ - the guide requires G8_FIX_01 to G8_FIX_04")
    for _p_n in ("01", "02", "03", "04"):
        _p_hit = _p_glob.glob(f"Code/G8_FIX_{_p_n}*")
        _p_ck(f"G8_FIX_{_p_n}*", bool(_p_hit),
              _p_os.path.basename(_p_hit[0]) if _p_hit else "MISSING from Code/")
    print("\n   the other scripts the README names")
    for _p_f in ("rerun_pruning_v3_original_protocol.py", "acceptance_gate.py",
                 "analyze_trust_study.py", "Oulad_exp.ipynb", "RetTransUst.ipynb"):
        _p_hit = _p_glob.glob(f"Code/{_p_f}") or _p_glob.glob(f"Code/*{_p_f}")
        _p_ck(_p_f, bool(_p_hit), "" if _p_hit else "not in Code/ - fix the README or add it")

    # ---- 2. F1's stated "done when" --------------------------------------------
    print("\n2. outputs/per_seed_metrics.csv - F1's done-when condition")
    _p_ps = "outputs/per_seed_metrics.csv"
    if not _p_os.path.exists(_p_ps):
        _p_ck("file exists", False, "MISSING - this is F1's done-when; it comes from G8_FIX_01")
    else:
        import pandas as _p_pd
        _p_d = _p_pd.read_csv(_p_ps)
        _p_cfg = set(_p_d.get("config", _p_pd.Series(dtype=str)).astype(str))
        _p_sd = set(int(s) for s in _p_d.get("seed", _p_pd.Series(dtype=int)))
        _p_want_c = {"full", "pruned", "deployed", "transfer9"}
        _p_want_s = {42, 0, 123, 2024, 7} | set(range(100, 115))
        _p_ck("configurations", _p_want_c <= _p_cfg,
              f"have {sorted(_p_cfg)}" + ("" if _p_want_c <= _p_cfg
              else f" | MISSING {sorted(_p_want_c - _p_cfg)}"))
        if "transfer9" not in _p_cfg:
            print("       transfer9 is the 9-feature transfer arm (4.17, 5.13). It is not")
            print("       produced by the 3-configuration run. Either add that arm to the")
            print("       20-split block, or concatenate the rows G8_FIX_03 writes into")
            print("       this file. F1's done-when names all four configurations.")
        _p_ck("all 20 seeds", _p_want_s <= _p_sd,
              f"{len(_p_sd)} seeds" + ("" if _p_want_s <= _p_sd
              else f" | MISSING {sorted(_p_want_s - _p_sd)}"))
        _p_ck("rows = configs x seeds", len(_p_d) >= 80, f"{len(_p_d)} rows", hard=False)

    # ---- 3. the mock dataset the README promises -------------------------------
    print("\n3. Data/ - the README tells a reviewer a mock dataset is here")
    _p_data = [f for f in _p_glob.glob("Data/*") if _p_os.path.isfile(f)]
    _p_ck("Data/ holds at least one file", bool(_p_data),
          f"{len(_p_data)} file(s): {[_p_os.path.basename(f) for f in _p_data][:4]}"
          if _p_data else "EMPTY or absent - cut that README line or add the mock data")

    # ---- 4. what must never be committed ---------------------------------------
    print("\n4. nothing forbidden is tracked or staged")
    try:
        _p_tracked = _p_sub.run(["git", "ls-files"], capture_output=True, text=True,
                                timeout=30).stdout.splitlines()
    except Exception as _p_e:
        _p_tracked = []
        print(f"     (git unavailable: {type(_p_e).__name__})")
    _p_never = ["model_ready_dataset", "kaggle.json", "data_local/responses.csv",
                "data_local/coding_rounds.csv", "copy_of_oulad_exp",
                "Group8_Project Cinque.docx", ".xlsx"]
    if not _p_tracked:
        # Without a tracked list every check below would pass because there is
        # nothing to find. That is a false pass, so report it as a failure.
        _p_ck("git has tracked files to check", False,
              "git listed nothing - not a repository, or nothing committed yet. "
              "The forbidden-file checks CANNOT run and prove nothing.")
    else:
        print(f"     {len(_p_tracked)} tracked files")
        for _p_pat in _p_never:
            _p_bad = [f for f in _p_tracked if _p_pat.lower() in f.lower()]
            _p_ck(f"not tracked: {_p_pat}", not _p_bad, str(_p_bad) if _p_bad else "")
    _p_gi = open(".gitignore").read() if _p_os.path.exists(".gitignore") else ""
    for _p_need in ("data_local/", "*.xlsx"):
        _p_ck(f".gitignore covers {_p_need}", _p_need in _p_gi,
              "" if _p_need in _p_gi else "add it before staging")

    # ---- 5. identifying strings anywhere in the repo ---------------------------
    print("\n5. anonymisation sweep over tracked text files")
    _p_ids = ["knust", "kumasi", "hussrec", "dcs/30", "aurelius906",
              "4dd985d", "22665808", "1See85_w3L", "1Xi6kHj8", "1jgaI4M9"]
    _p_found = {}
    for _p_f in _p_tracked:
        if _p_os.path.splitext(_p_f)[1].lower() not in (
                ".py", ".md", ".txt", ".json", ".csv", ".ipynb", ".yml", ".yaml", ".cfg"):
            continue
        try:
            _p_t = open(_p_f, encoding="utf8", errors="ignore").read().lower()
        except Exception:
            continue
        for _p_s in _p_ids:
            if _p_s.lower() in _p_t:
                _p_found.setdefault(_p_s, []).append(_p_f)
    if not _p_tracked:
        _p_ck("sweep could run", False, "no tracked files, so nothing was swept")
    else:
        _p_ck("no identifying strings in tracked text", not _p_found,
              "" if not _p_found else "; ".join(f"{k} in {v[:2]}" for k, v in _p_found.items()))
    print("     (index numbers: check Group8_Project Cinque.docx separately - F4)")

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 78)
    if _p_fail:
        print(f"NOT READY for F1: {len(_p_fail)} blocker(s)")
        for _p_f in _p_fail:
            print("   -", _p_f)
    else:
        print("READY for F1. Stage Code/ outputs/ Data/ .gitignore README.md, then run the gate.")
    if _p_warn:
        print(f"\n{len(_p_warn)} soft warning(s):")
        for _p_w in _p_warn:
            print("   -", _p_w)


_p_main()
