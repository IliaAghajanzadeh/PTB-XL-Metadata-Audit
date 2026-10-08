# =====================================================================================
# Project: PTB-XL Metadata Audit  -  ph1 POST-HOC metadata-only evaluation on the test fold (fold 10)
# Code version: ph1 | Code revision: r2
# Date: 2026-10-05
# Status: POST-HOC (ph1): not pre-registered; reported separately from pre-registered results
#         (final_run_protocol.json post_fold10_rules 3). This cell never modifies, re-saves or overwrites an
#         existing file: every output goes to the write-once folder outputs\ph1.
#
# Purpose: does the metadata-only result of v0.2.1 / v0.3 (H1, H2; fold 9; confirmatory-after-the-fact) hold on
#   the test fold? The v0.2.1 pipeline (decision table: blocks B0-B5, LogisticRegression and HistGradientBoosting,
#   seeds 42 / 123 / 2024) and the v0.3 H2 paired test (delta = macro-AUROC(B3) - macro-AUROC(B1);
#   LogisticRegression primary, HistGradientBoosting seed 42 robustness) are re-run with EXACTLY ONE CHANGE:
#       training rows    folds 1-8 -> folds 1-9 (19,601 records)
#       evaluation rows  fold 9    -> fold 10   (2,198 records, 1,904 patients)
#   Encoder and scalers are fit on the training rows only. Bootstrap: patient level, the locked fold-10 BOOT_IDX
#   (N_BOOT 2000, BOOT_SEED 7) shared with v0.9. The analysis plan is the constant PH1_PLAN (verbatim).
#
# Revisions:
#   r1  stopped at the Part A gate (A2) on the real data: only the LogisticRegression x B2_acquisition rows (all three
#       seeds; MI, CD, HYP, MACRO; decision-table row LR x B2) differed from v0.2.1 (max |diff| AUROC 2.9e-6, AUPRC
#       5.7e-5; attributed to the scipy 1.11.4 -> 1.15.3 change of L-BFGS-B); every other A2 cell, A1 and A3 reproduced
#       exactly; nothing was computed on fold 10. The r1 output folder was renamed to outputs\ph1_failed_1.
#   r2  the core chat's change request after the r1 run - these are the ONLY changes:
#       (1) A2 tolerance per row and column (A2_TOLERANCE): LogisticRegression x B2_acquisition rows only - AUROC, CI
#           bounds, macro_auroc_mean, delta_vs_B1_physio 1e-5, AUPRC 1e-4; every other A2 cell 1e-12 (all
#           HistGradientBoosting rows, all B0 / B1 / B3 rows, LR B4 / B5, macro_auroc_std_across_seeds, *_chance);
#           identifier columns exact. A3 keeps 1e-12.
#       (2) new check A2b: every LogisticRegression row that is not exact keeps its 4-decimal AUROC; reported, it never
#           stops the run.
#       (3) ph1_analysis_plan.json gets the field amendment_r2 (AMENDMENT_R2, verbatim); PH1_PLAN and its sha256 are
#           unchanged. AMENDMENT_R2, the hashes of the r1 ph1_partA_failure.json and the r1 attempt record also go into
#           ph1_results.json; STEP 0 verifies that file (self_sha256 == the copy the core chat read, else STOP).
#       (4) CODE_REVISION = "r2".
#       Part B, the outputs and every other check are unchanged.
#
# Phases (each starts only after the previous one passed; every failed check STOPS the run):
#   STEP 0  1 versions: python 3.10.20, numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2 (the
#             environment of v0.4-v0.9; v0.2.1 and v0.3 ran with scipy 1.11.4 - Part A is the test that their
#             results reproduce under scipy 1.15.3); os.cpu_count() and threadpoolctl.threadpool_info() are
#             recorded; no thread setting is changed
#           2 file sha256 (+ self_sha256) of every input; the reference implementation REF is read from
#             final_run_protocol.json, sha256-asserted and exec'd (as v0.9)
#           3 snapshot (sha256 of every file) of preregistration\, outputs\v0.2.1, outputs\v0.3, outputs\v0.9
#           4 outputs\ph1 must not exist (write-once) -> created; earlier attempts outputs\ph1_failed_* recorded;
#             r2: outputs\ph1_failed_1\ph1_partA_failure.json (the r1 attempt) verified - self_sha256 == expected
#             (STOP otherwise), file sha256 == expected (WARN otherwise)
#           5 ph1_analysis_plan.json (PH1_PLAN + r2 field amendment_r2, exclusive create, self_sha256) BEFORE any
#             model is fit
#   PART A  validation gate on known data - train folds 1-8, evaluate fold 9 - with the SAME two functions as B:
#           A1 labels (inside fit_predict, before any fit): fold-9 positives NORM 955, MI 540, STTC 528, CD 495, HYP 268
#           A2 decision table (12 rows x 5 numeric columns) == v0.2.1_decision_table.csv and all 216 metric rows ==
#              v0.2.1_metrics_full.csv (identifier columns exact; numeric |diff| <= the r2 tolerance of each cell,
#              A2_TOLERANCE: 1e-12 everywhere except the LogisticRegression x B2_acquisition rows - AUROC, CI bounds,
#              macro_auroc_mean, delta_vs_B1_physio 1e-5, AUPRC 1e-4)
#           A2b (r2; reported, never stops the run) every LogisticRegression row that is not exact keeps its 4-decimal
#              AUROC (metric rows: auroc; decision table: macro_auroc_mean)
#           A3 H2 on the A2 seed-42 fits with REF (build_boot_idx(1942, 2000, 7)) == every field of
#              pre_registration.json H2.results_per_learner (numbers 1e-12, verdict strings exact) and the per-draw
#              deltas == v0.3_bootstrap_deltas.csv (1e-12)
#           any A1 / A2 / A3 failure -> ph1_partA_failure.json (diagnostics) and STOP; nothing is computed on fold 10
#   PART B  B0 (before any fold-10 model is fit): fold-10 ids / patients / BOOT_IDX == FRP locked hashes; labels
#              (v0.2.1 code) == v0.9_fold10_labels.csv and counts 963 / 550 / 521 / 496 / 262; encoder fit on folds
#              1-9 == FRP locked categories; no fold-10 record on __UNKNOWN__ or on an all-zero training column;
#              folds 1-9 level strings unchanged by the fold-10 values
#           B1 fit on folds 1-9, predict fold 10 (36 configurations) -> ph1_fold10_probs.csv and
#              ph1_frozen_outputs.json BEFORE any fold-10 metric; the file is read back (round trip, bitwise ==)
#           B2 decision table, B3 H2-style paired test (REF + BOOT_IDX), B4 metric rows: computed ONLY from the
#              probability file read back from disk
#   OUTPUT  snapshots re-checked -> ph1_results.json (written last of the results) -> ph1_manifest.json ->
#           snapshots re-checked again -> printed summary
#
# ONE CODE PATH FOR PARTS A AND B - both parts call the same two function objects:
#   (a) fit_predict()        v0.2.1 label code, Section 1 preprocessing, Section 2 feature blocks and the Section 3
#                            fits; every input is an argument; returns the long-format probabilities, the
#                            evaluation labels and every fitted object; a part-specific gate (A1 / B0) runs inside
#                            it after preprocessing and before the first fit.
#   (b) metrics_from_probs() v0.2.1 metric rows, marginal bootstrap CI, decision table and H1 verdict; v0.3 H2
#                            statistics, MDE, verdict strings and disagreement list on the REF paired bootstrap.
#   Neither function reads module-level state other than constants, helper functions and REF (asserted on their
#   bytecode before Part A); Part B asserts that no object fitted in Part A is reused.
#   A3 reuses the A2 seed-42 fits instead of v0.3's refit (v0.3 cell lines 368-415) and v0.3's BOOT_ROWS rebuild
#   (lines 443-456); the equivalence is TESTED against pre_registration.json and v0.3_bootstrap_deltas.csv.
#
# HARD RULES: no ECG signal (no wfdb, no torch, no records100); nothing is written outside outputs\ph1; every
#   write is '<name>.partial' (exclusive create) + exclusive rename; nothing else is computed: no Section 0a
#   prevalence table, no chance-baseline CSV, no Section 6 coefficients, no other block / learner / threshold /
#   tuning / selection. Learner disagreement is reported, never resolved by selection.
#
# CRASH PROTOCOL: if a run stops or crashes, RENAME outputs\ph1 to outputs\ph1_failed_<n> (never delete it) and
#   run the cell again. Every earlier attempt is recorded in ph1_results.json; a ph1_fold10_probs.csv left by an
#   earlier attempt is compared byte for byte with the new one (WARN if different).
#
# CODE PROVENANCE MARKERS (sources: the main code cell of the v0.2.1 and v0.3 notebooks; v0.9_cell.py, sha256
#   9af1773dba86039cfa2b191f2aaf0e54c02f9ccb90880b60471074e2c6fe4c7b):
#   '# >>> verbatim from vX (label)' ... '# <<< verbatim from vX': byte-identical to a contiguous excerpt of vX.
#   '# >>> adapted from vX (label) | ...' ... '# <<< adapted from vX': identical to a contiguous excerpt of vX
#     after undoing exactly what the marker declares, in this order: delete the lines tagged '# [ph1 added]';
#     remove 'indent: N' leading spaces; undo each 'literal: old => new' (separated by ' ;; '); undo each
#     'renames: old->new' (whole identifiers only). Comments, docstrings and string literals inside copied blocks
#     are not edited; where one no longer fits, a '# NOTE (ph1)' line stands above the block.
#   Everything without a marker is new ph1 code.
#
# Outputs (OUT_DIR = <PROJECT_ROOT>\outputs\ph1), all write-once:
#   ph1_analysis_plan.json   ph1_fold10_probs.csv   ph1_frozen_outputs.json   ph1_decision_table.csv
#   ph1_bootstrap_deltas.csv   ph1_metric_rows.csv   ph1_results.json   ph1_manifest.json
#   (a Part A failure leaves only ph1_analysis_plan.json and ph1_partA_failure.json)
# =====================================================================================

# >>> verbatim from v0.2.1 (imports, display options)
import os, sys, json, ast, hashlib, platform, time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 220)
# <<< verbatim from v0.2.1
import re, math, ntpath, dis, types, builtins
import scipy                                         # version assert only
import sklearn                                       # version assert only
from threadpoolctl import threadpool_info            # recorded only (no thread setting is changed)

# ------------------------------- configuration ---------------------------------------
PROJECT_ROOT = r"D:\ptbxl-metadata-audit"
DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
PREREG_DIR  = os.path.join(PROJECT_ROOT, "preregistration")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_DIR     = os.path.join(OUTPUTS_DIR, "ph1")
V021_DIR    = os.path.join(OUTPUTS_DIR, "v0.2.1")
V03_DIR     = os.path.join(OUTPUTS_DIR, "v0.3")
V09_DIR     = os.path.join(OUTPUTS_DIR, "v0.9")
SNAPSHOT_DIRS = {"preregistration": PREREG_DIR, "outputs\\v0.2.1": V021_DIR, "outputs\\v0.3": V03_DIR,
                 "outputs\\v0.9": V09_DIR}

DB_CSV  = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_CSV = os.path.join(DATA_ROOT, "scp_statements.csv")
SEAL_PATH           = os.path.join(PREREG_DIR, "pre_registration.json")
FINAL_PROTOCOL_PATH = os.path.join(PREREG_DIR, "final_run_protocol.json")
V021_DECISION_TABLE_CSV = os.path.join(V021_DIR, "v0.2.1_decision_table.csv")
V021_METRICS_FULL_CSV   = os.path.join(V021_DIR, "v0.2.1_metrics_full.csv")
V021_MANIFEST           = os.path.join(V021_DIR, "v0.2.1_manifest.json")
V03_BOOT_DELTAS_CSV     = os.path.join(V03_DIR, "v0.3_bootstrap_deltas.csv")
V03_MANIFEST            = os.path.join(V03_DIR, "v0.3_manifest.json")
V09_LABELS_CSV          = os.path.join(V09_DIR, "v0.9_fold10_labels.csv")

CODE_VERSION = "ph1"
CODE_REVISION = "r2"
PREFIX = CODE_VERSION + "_"
POSTHOC_LABEL = "POST-HOC (ph1): not pre-registered; reported separately from pre-registered results"

PLAN_PATH          = os.path.join(OUT_DIR, PREFIX + "analysis_plan.json")
PARTA_FAILURE_PATH = os.path.join(OUT_DIR, PREFIX + "partA_failure.json")
PROBS_PATH         = os.path.join(OUT_DIR, PREFIX + "fold10_probs.csv")
FROZEN_PATH        = os.path.join(OUT_DIR, PREFIX + "frozen_outputs.json")
DECISION_TABLE_PATH = os.path.join(OUT_DIR, PREFIX + "decision_table.csv")
BOOT_DELTAS_PATH   = os.path.join(OUT_DIR, PREFIX + "bootstrap_deltas.csv")
METRIC_ROWS_PATH   = os.path.join(OUT_DIR, PREFIX + "metric_rows.csv")
RESULTS_PATH       = os.path.join(OUT_DIR, PREFIX + "results.json")
MANIFEST_PATH      = os.path.join(OUT_DIR, PREFIX + "manifest.json")
FAILED_ATTEMPT_PREFIX = "ph1_failed_"
R1_ATTEMPT_FOLDER = FAILED_ATTEMPT_PREFIX + "1"                  # r2: the r1 attempt named in AMENDMENT_R2
R1_FAILURE_PATH = os.path.join(OUTPUTS_DIR, R1_ATTEMPT_FOLDER, PREFIX + "partA_failure.json")

# >>> verbatim from v0.2.1 (superclasses)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# <<< verbatim from v0.2.1
ACQ_COLS = ["device", "site", "nurse"]
# >>> verbatim from v0.2.1 (seeds, bootstrap)
SEEDS = [42, 123, 2024]
N_BOOT = 2000
BOOT_SEED = 7
# <<< verbatim from v0.2.1
# >>> verbatim from v0.2.1 (H1 threshold, learners)
VERDICT_THRESHOLD = 0.60
LEARNERS = ["LogisticRegression", "HistGradientBoosting"]
# <<< verbatim from v0.2.1
# >>> verbatim from v0.3 (H2 model seed)
SEED = 42
# <<< verbatim from v0.3
# >>> verbatim from v0.3 (learner roles)
PRIMARY_LEARNER = "LogisticRegression"
ROBUSTNESS_LEARNER = "HistGradientBoosting"
# <<< verbatim from v0.3
# >>> verbatim from v0.3 (compared blocks)
COMPARED_BLOCKS = ["B1_physio", "B3_physio_plus_acquisition"]
# <<< verbatim from v0.3
# v0.3 constants: the values used by ph1 are READ from pre_registration.json and asserted equal to these
# >>> verbatim from v0.3 (alpha, power, z values, SESOI)
ALPHA = 0.05
POWER = 0.80
Z_ALPHA_2 = 1.959964
Z_POWER = 0.841621
SESOI_H2 = 0.02
SESOI_H3 = 0.01
# <<< verbatim from v0.3

BLOCK_NAMES = ["B0_intercept", "B1_physio", "B2_acquisition", "B3_physio_plus_acquisition", "B4_temporal", "B5_all"]
CONFIGS = [(b, l, s) for b in BLOCK_NAMES for l in LEARNERS for s in SEEDS]      # 36; the probability-file order
PROB_COLUMNS = ["p_" + c for c in SUPERCLASSES]
PROBS_CSV_COLUMNS = ["ecg_id", "patient_id", "block", "learner", "seed"] + PROB_COLUMNS
DT_ID_COLUMNS = ["block", "learner"]
DT_NUMERIC_COLUMNS = ["macro_auroc_mean", "macro_auroc_std_across_seeds", "ci_low_mean", "ci_high_mean",
                      "delta_vs_B1_physio"]
MF_ID_COLUMNS = ["block", "learner", "seed", "class"]
MF_NUMERIC_COLUMNS = ["auroc", "auroc_chance", "auprc", "auprc_chance", "ci_low", "ci_high"]
PART_A_TOL = 1e-12                       # A3, and every A2 cell outside the r2 relaxed cells (only the core chat
                                         # can change a tolerance, in a new revision, before Part B runs)
# r2 (amendment_r2): A2 tolerance per row and column, set by the core chat after the r1 run, before anything was computed
# on fold 10. Only the LogisticRegression x B2_acquisition rows get a wider tolerance, and only in these columns; every
# other A2 cell keeps PART_A_TOL (all HistGradientBoosting rows, all B0 / B1 / B3 rows, LR B4 / B5 rows,
# macro_auroc_std_across_seeds, auroc_chance, auprc_chance); identifier columns are compared exactly.
A2_TOLERANCE = {
    "default": PART_A_TOL,
    "relaxed_rows": {"block": "B2_acquisition", "learner": "LogisticRegression"},
    "relaxed_columns": {"auroc": 1e-5, "ci_low": 1e-5, "ci_high": 1e-5, "auprc": 1e-4,                  # metric rows
                        "macro_auroc_mean": 1e-5, "ci_low_mean": 1e-5, "ci_high_mean": 1e-5,          # decision table
                        "delta_vs_B1_physio": 1e-5},
}
EXPECTED_N_FITTED = 4 + len(BLOCK_NAMES) * len(LEARNERS) * len(SEEDS) * len(SUPERCLASSES)   # 4 LR scalers + 180
AGE_SCALER_TOL = {"mean": 1e-12, "scale": 1e-12, "var": 1e-9}                   # as v0.8 (folds 1-9 age scaler)

EXPECTED_VERSIONS = {"python": "3.10.20", "numpy": "1.26.4", "pandas": "2.3.3", "scipy": "1.15.3",
                     "sklearn": "1.7.2"}
VERSIONS_NOTE = ("Assert python 3.10.20, numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2. This is the "
                 "environment of v0.4-v0.9. v0.2.1 and v0.3 ran with scipy 1.11.4; Part A is the test that their "
                 "results reproduce under scipy 1.15.3.")

# >>> verbatim from v0.3 (input hashes)
# v0.2.1 input hashes (must match exactly)
EXPECTED_INPUT_HASHES = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}
# <<< verbatim from v0.3
EXPECTED_SELF_SHA256 = {
    "pre_registration":   "5183f54569f3358a5339b16a2c3c3be49f5c7798eabe6b0bb25946119c2ba469",
    "final_run_protocol": "834acddec288862d0418e9ee70204898c9d371788eed7eb362f4c0d63d305c6b",
}
EXPECTED_FILE_SHA256 = {
    "pre_registration":   "11dd446272654fc80d65b2fdd40259fa674bbb419a44fde84e589b6e297cde32",
    "final_run_protocol": "e1a5260e72d2e8b0d27dc709639468b7f9a48a7c8e2f8af2d663e105dbd329d4",
}
EXPECTED_REFERENCE_CODE_SHA256 = "48ac837c07880f0bd88a7bd01475e7da3edbeab46019f8288976884ad7662856"
EXPECTED_V021_DECISION_TABLE_SHA256 = "20bffb2c6df30e16a070184f91f7211bddc14ede9bc2de39273b5be1cc32359b"  # core chat copy
EXPECTED_V021_MANIFEST_SHA256 = "37ca9b785ac2c1d78dd3e28f729dc64d289d42ae80803adacbe8a406122b23f6"  # v0.2.1 printout; WARN only
EXPECTED_V03_MANIFEST_SHA256 = "c7e1bec283d11a3d4cd94a3f2509640651e8069d26424cfd14b5d6b975053e35"
EXPECTED_V09_LABELS_SHA256 = "29051bd87676a59b70b3a79a91075f70a7cddd028b2c297461c7bbae3f15b3b0"
# r2: outputs\ph1_failed_1\ph1_partA_failure.json of the r1 attempt, as the core chat read it
EXPECTED_R1_FAILURE_SELF_SHA256 = "d48b8c77ccbdc957d54f0907503e7463197fb2570e6ec26c7bb67e2c40c77807"
EXPECTED_R1_FAILURE_FILE_SHA256 = "d7b7d5e1bbdb07f02ca58664a116b83ac1c7109c023f5d6b11f28e916d6e9415"  # WARN only

# >>> verbatim from v0.9 (fold-9 positives, folds 1-9 records)
EXPECTED_FOLD9_POSITIVES = {"NORM": 955, "MI": 540, "STTC": 528, "CD": 495, "HYP": 268}
EXPECTED_N_RECORDS_FOLDS1TO9 = 19601
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (fold-10 records)
EXPECTED_N_FOLD10_RECORDS = 2198
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (fold-10 patients)
EXPECTED_N_FOLD10_PATIENTS = 1904
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (fold-10 label counts disclosed in FRP)
# fold-10 label counts disclosed in FRP disclosures_fold10_exposure (computed in v0.1.1 / v0.1.2 with the same
# label rule as v0.2.1); asserted to appear verbatim in FRP, then checked in Part C before any metric
EXPECTED_FOLD10_POSITIVES = {"NORM": 963, "MI": 550, "STTC": 521, "CD": 496, "HYP": 262}
# <<< verbatim from v0.9
# split sizes printed by v0.2.1 / v0.3 (Part A) and locked in FRP (Part B); patient counts add up (disjoint folds)
EXPECTED_SPLITS = {
    "A": {"n_train": 17418, "n_eval": 2183, "n_train_patients": 15023, "n_eval_patients": 1942},
    "B": {"n_train": 19601, "n_eval": 2198, "n_train_patients": 15023 + 1942, "n_eval_patients": 1904},
}

# the ph1 analysis plan, VERBATIM (prompt 1, section 3); its UTF-8 sha256 is asserted below
PH1_PLAN = (
    "ph1 — POST-HOC analysis (not pre-registered). Question: does the metadata-only result of "
    "v0.2.1/v0.3 (H1, H2; fold 9; confirmatory-after-the-fact) hold on the test fold? Design: identical "
    "to v0.2.1 (decision table: blocks B0–B5, LogisticRegression and HistGradientBoosting, seeds as in "
    "v0.2.1) and v0.3 (H2 paired test: delta = macro-AUROC(B3) − macro-AUROC(B1); LogisticRegression "
    "primary, HistGradientBoosting seed 42 robustness), with exactly one change: training rows = folds "
    "1–9 (instead of 1–8), evaluation rows = fold 10 (instead of 9). Encoder and scalers are fit on the "
    "training rows. Bootstrap: patient level, the locked fold-10 BOOT_IDX (N_BOOT 2000, BOOT_SEED 7) "
    "shared with the v0.9 analysis. Rules applied post hoc, unchanged: difference test (95% percentile "
    "CI of the paired delta excludes 0), TOST with SESOI_H2 = 0.02 (90% percentile CI within ±0.02), "
    "achieved equivalence bound; learner disagreement is reported, never resolved by selection. Nothing "
    "else is computed. All results are reported regardless of outcome, labelled post hoc, separately "
    "from the pre-registered results. Validation gate: before any fold-10 model is fit, the same code "
    "with training folds 1–8 and evaluation fold 9 must reproduce the v0.2.1 decision table and the v0.3 "
    "H2 results to within 1e-12; otherwise the run stops."
)
EXPECTED_PH1_PLAN_SHA256 = "bb9669de4e54b0530db3eef0b42fe4fd8378da4a084d142df6750bc61b182c17"
assert hashlib.sha256(PH1_PLAN.encode("utf-8")).hexdigest() == EXPECTED_PH1_PLAN_SHA256, \
    "PH1_PLAN text differs from the locked plan (sha256) - aborting."
# r2: the plan amendment of the core chat (change request after the r1 run), VERBATIM; written as the field
# 'amendment_r2' of ph1_analysis_plan.json (PH1_PLAN above is unchanged) and copied into ph1_results.json
AMENDMENT_R2 = (
    "r2 (2026-10-05): the r1 run stopped at the Part A gate because the LogisticRegression "
    "B2_acquisition rows (all three seeds; classes MI, CD, HYP) differed from v0.2.1 by at most 2.9e-6 "
    "(AUROC) and 5.7e-5 (AUPRC), attributed to the scipy 1.11.4 -> 1.15.3 change of the L-BFGS-B "
    "implementation; every other metric row and decision-table row (all HistGradientBoosting rows, "
    "LogisticRegression B0/B1/B3/B4/B5) and all H2 fields and per-draw deltas reproduced to 1e-12. The "
    "gate tolerance for the LogisticRegression B2_acquisition rows only was set to 1e-5 (AUROC, CIs) and "
    "1e-4 (AUPRC) before anything was computed on fold 10. r1 attempt: outputs\\ph1_failed_1."
)
EXPECTED_AMENDMENT_R2_SHA256 = "e0255ebbd7588f84523577bd5bb3d480fe52d8d0bbada4668e2e59d57f651094"
assert hashlib.sha256(AMENDMENT_R2.encode("utf-8")).hexdigest() == EXPECTED_AMENDMENT_R2_SHA256, \
    "AMENDMENT_R2 text differs from the core chat's amendment (sha256) - aborting."

# >>> verbatim from v0.9 (output paths outside the dataset folder)
_root_abs = os.path.abspath(DATA_ROOT)
for _p in (OUT_DIR, PREREG_DIR):
    assert not os.path.abspath(_p).lower().startswith(_root_abs.lower()), \
        "output path is inside the dataset folder - aborting: " + _p
# <<< verbatim from v0.9

# >>> verbatim from v0.9 (run start, warnings)
RUN_STARTED = datetime.now().astimezone().isoformat()
WARNINGS = []
# <<< verbatim from v0.9
CHECKS = []                              # every Part A / Part B check record (stored in ph1_results.json)
PART_A_DIAG = {}                         # Part A diagnostics (stored in ph1_partA_failure.json on a failure)


# >>> verbatim from v0.2.1 (sha256_of)
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
# <<< verbatim from v0.2.1


# >>> verbatim from v0.2.1 (hdr, warn, note, _json_default, fast_auc, to_level_string, encode_categorical)
def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def warn(msg):
    WARNINGS.append(msg)
    print("[WARN] " + msg)


def note(msg):
    print("[NOTE] " + msg)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fast_auc(y, s):
    """Binary AUROC via the Mann-Whitney U statistic (average-rank tie handling).
    Used instead of sklearn.metrics.roc_auc_score for speed inside the bootstrap
    loop; both compute the identical quantity."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(np.asarray(s))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def to_level_string(series, missing="__MISSING__"):
    """Human-readable, zero-padded categorical level for numeric columns (e.g.
    '007' for site) so sorting and printed feature names stay legible."""
    if pd.api.types.is_numeric_dtype(series):
        nn = series.dropna()
        integral = bool(len(nn) == 0 or np.all(np.isclose(nn.values, np.round(nn.values))))
        if integral:
            width = max(3, len(str(int(nn.max()))) if len(nn) else 3)
            return series.apply(lambda v: missing if pd.isna(v) else str(int(v)).zfill(width))
    return series.apply(lambda v: missing if pd.isna(v) else str(v).strip())


def encode_categorical(level_series, train_mask, col_name):
    """One-hot encode a pre-stringified categorical column. The category list is
    fixed from the training rows only, always includes an explicit '__MISSING__'
    level, and always includes a catch-all '__UNKNOWN__' level for anything seen
    only outside training (e.g. only in fold 9). Both splits therefore produce the
    exact same set of columns."""
    train_vals = level_series[train_mask]
    train_categories = sorted(set(train_vals.unique().tolist()) | {"__MISSING__"})
    known = set(train_categories)
    remapped = level_series.where(level_series.isin(known), other="__UNKNOWN__")
    categories = train_categories + ["__UNKNOWN__"]
    cat = pd.Categorical(remapped, categories=categories)
    dummies = pd.get_dummies(cat)
    dummies.columns = ["{}={}".format(col_name, c) for c in dummies.columns]
    dummies.index = level_series.index
    return dummies, categories
# <<< verbatim from v0.2.1


# >>> verbatim from v0.3 (canonical JSON, self_sha256 procedure)
def canonical_json_bytes(d):
    """Canonical serialization used for self_sha256 (see SELF_SHA256_PROCEDURE)."""
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_json_default).encode("utf-8")


SELF_SHA256_PROCEDURE = (
    "Load the JSON file; remove the key 'self_sha256' (the key must be absent, not empty); "
    "serialize the remaining dict with json.dumps(d, sort_keys=True, ensure_ascii=False, "
    "separators=(',', ':')); compute SHA-256 of the UTF-8 bytes; the hex digest must equal "
    "the stored 'self_sha256'. (At write time numpy scalars were converted to plain JSON "
    "types first, so the loaded file hashes identically.)"
)
# <<< verbatim from v0.3


# >>> verbatim from v0.9 (verify_self_sha256)
def verify_self_sha256(path):
    """Return (stored, recomputed, loaded_dict) for a self-hashed JSON document."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    stored = d.pop("self_sha256", None)
    recomputed = hashlib.sha256(canonical_json_bytes(d)).hexdigest()
    d["self_sha256"] = stored
    return stored, recomputed, d
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (plain, manifest_entry)
def plain(o):
    return json.loads(json.dumps(o, ensure_ascii=False, default=_json_default))


def manifest_entry(manifest, basename):
    """Return the unique 'outputs' entry of a project manifest whose file basename matches
    (works for full Windows paths and bare names)."""
    hits = [o for o in manifest["outputs"] if ntpath.basename(o["file"]) == basename]
    assert len(hits) == 1, "expected exactly one manifest entry for {}, found {}".format(basename, len(hits))
    return hits[0]
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (write_once_json)
def write_once_json(path, doc):
    """Normalize to plain JSON types, add self_sha256, exclusive-create, verify from disk."""
    doc = json.loads(json.dumps(doc, ensure_ascii=False, default=_json_default))
    assert "self_sha256" not in doc
    sha = hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
    doc["self_sha256"] = sha
    with open(path, "x", encoding="utf-8") as f:          # refuses to overwrite
        json.dump(doc, f, ensure_ascii=False, indent=2)
    s_, r_, _ = verify_self_sha256(path)
    assert s_ == r_ == sha, "self_sha256 verification FAILED for " + path
    return sha
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (sha256_bytes)
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (_find_nonfinite)
def _find_nonfinite(o, path="$"):
    if isinstance(o, float) and not math.isfinite(o):
        return [path]
    if isinstance(o, dict):
        return [p for k, v in o.items() for p in _find_nonfinite(v, path + "." + str(k))]
    if isinstance(o, list):
        return [p for i, v in enumerate(o) for p in _find_nonfinite(v, "{}[{}]".format(path, i))]
    return []
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (_partial)
def _partial(p):
    return p + ".partial"
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (csv_bytes)
def csv_bytes(df):
    """Deterministic CSV bytes: UTF-8 with BOM, '\\n' line ends, pandas default float repr, NaN as empty."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
# <<< verbatim from v0.9


# >>> verbatim from v0.9 (exclusive_rename, write_once_bytes, write_once_json_atomic)
def exclusive_rename(src, dst):
    """Move a completed '.partial' file to its final name; never replaces an existing file."""
    assert not os.path.exists(dst), "{} already exists - a write-once file is never overwritten - aborting.".format(dst)
    os.rename(src, dst)                # on Windows os.rename itself also refuses an existing target


def write_once_bytes(path, data):
    """Write-once bytes: exclusive create of '<path>.partial', fsync, exclusive rename, verify."""
    assert not os.path.exists(path), "{} already exists - aborting.".format(path)
    with open(_partial(path), "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    exclusive_rename(_partial(path), path)
    got = sha256_of(path)
    assert got == sha256_bytes(data), "write-once verification FAILED for " + path
    return got


def write_once_json_atomic(path, doc):
    """write_once_json (v0.7, exclusive create + self_sha256) on '<path>.partial', then exclusive rename."""
    assert not os.path.exists(path), "{} already exists - aborting.".format(path)
    sha = write_once_json(_partial(path), doc)
    exclusive_rename(_partial(path), path)
    s_, r_, _ = verify_self_sha256(path)
    assert s_ == r_ == sha, "self_sha256 verification FAILED for " + path
    return sha
# <<< verbatim from v0.9


def tree_snapshot(snap_dir):
    """Sorted (relative path, sha256) of every file in snap_dir and its subfolders (v0.9 dir_snapshot, recursive)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(snap_dir):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            out.append((os.path.relpath(p, snap_dir).replace(os.sep, "/"), sha256_of(p)))
    return sorted(out)


def snapshot_all():
    """Snapshots of the four read-only folders plus the two input CSVs (re-checked at the end)."""
    snap = {label: tree_snapshot(d) for label, d in SNAPSHOT_DIRS.items()}
    snap["data"] = [(os.path.basename(p), sha256_of(p)) for p in (DB_CSV, SCP_CSV)]
    return snap


def snapshot_diff(a, b):
    """Folders whose (path, sha256) lists differ between two snapshot_all() results, with the differing paths."""
    out = {}
    for k in sorted(set(a) | set(b)):
        da, db = dict(a.get(k, [])), dict(b.get(k, []))
        changed = sorted(p for p in set(da) | set(db) if da.get(p) != db.get(p))
        if changed:
            out[k] = changed
    return out


def json_safe(o):
    """Non-finite floats -> strings (diagnostics may hold inf / nan); everything else as plain JSON."""
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def check(section, cid, description, ok, stop=True, gating=True, **fields):
    """Record one check (id, section, description, kind, expected, observed, max_abs_diff, tolerance, status ...);
    a failed check STOPS the run unless stop=False (Part A collects every comparison first). r2: a non-gating check
    (gating=False; A2b) is PASS or WARN, a WARN is added to WARNINGS, and it never stops the run."""
    rec = {"id": cid, "section": section, "description": description,
           "status": "PASS" if ok else ("FAIL" if gating else "WARN")}
    if not gating:
        rec["gating"] = False
    rec.update(fields)
    rec = json_safe(plain(rec))
    CHECKS.append(rec)
    extra = ""
    if rec.get("max_abs_diff") is not None:
        extra = " | max |diff| = {} (tolerance {})".format(rec["max_abs_diff"], rec.get("tolerance"))
    print("  [{}] {:4s} {} - {}{}".format(rec["status"], section, cid, description, extra))
    if not gating:
        if not ok:
            warn("non-gating check {} is WARN - {} | observed {}".format(cid, description, str(rec.get("observed"))[:400]))
    elif stop:
        assert ok, "STOP: check {} failed - {} | expected {} | observed {}".format(
            cid, description, str(rec.get("expected"))[:400], str(rec.get("observed"))[:400])
    return rec


def abs_diff_matrix(got, exp, num_cols):
    """|got - exp| per row and numeric column (frames with the same rows); 0 where both values are NaN, inf where only
    one is NaN."""
    D = np.zeros((len(exp), len(num_cols)), dtype=np.float64)
    for j, c in enumerate(num_cols):
        a = got[c].to_numpy(dtype=np.float64)
        b = exp[c].to_numpy(dtype=np.float64)
        na, nb = np.isnan(a), np.isnan(b)
        d = np.abs(np.where(na | nb, 0.0, a - b))
        d[na != nb] = np.inf
        D[:, j] = d
    return D


def cell_tolerance(tol, ids, column):
    """r2: tolerance of one cell. tol is one number (every cell) or an A2 tolerance spec (A2_TOLERANCE): the relaxed
    value for a relaxed column of a row whose identifiers match relaxed_rows, the default for every other cell."""
    if not isinstance(tol, dict):
        return float(tol)
    if all(ids.get(k) == v for k, v in tol["relaxed_rows"].items()):
        return float(tol["relaxed_columns"].get(column, tol["default"]))
    return float(tol["default"])


def compare_frames(section, cid, description, got, exp, id_cols, num_cols, tol):
    """Identifier columns exactly (same rows, same order), numeric columns |diff| <= the tolerance of each cell (NaN
    only where the expected value is NaN). tol: one number, or (r2, A2) a tolerance spec (cell_tolerance). Returns
    (check record, diagnostics with the max |diff| per column, per row and per tolerance, and every cell over its
    tolerance)."""
    diag = {"columns_expected": list(exp.columns), "columns_observed": list(got.columns),
            "n_rows_expected": int(len(exp)), "n_rows_observed": int(len(got))}
    shape_ok = list(got.columns) == list(exp.columns) and len(got) == len(exp)
    ids_ok = shape_ok and all(np.array_equal(got[c].to_numpy(), exp[c].to_numpy()) for c in id_cols)
    per_col, max_all, by_tol, over = {}, float("inf"), {}, None
    if shape_ok:
        D = abs_diff_matrix(got, exp, num_cols)
        row_ids = [{c: exp[c].iloc[i] for c in id_cols} for i in range(len(exp))]
        T = np.array([[cell_tolerance(tol, ids, c) for c in num_cols] for ids in row_ids],
                     dtype=np.float64).reshape(len(exp), len(num_cols))
        per_col_by_tol = {}
        for j, c in enumerate(num_cols):
            per_col[c] = float(D[:, j].max()) if len(D) else 0.0
            per_col_by_tol[c] = {"{:g}".format(t): float(D[T[:, j] == t, j].max()) for t in sorted(set(T[:, j].tolist()))}
        row_max = D.max(axis=1) if len(num_cols) else np.zeros(len(exp))
        max_all = max(per_col.values()) if per_col else 0.0
        by_tol = {"{:g}".format(t): float(D[T == t].max()) for t in sorted(set(T.ravel().tolist()))}
        over = [dict(row_ids[i], column=num_cols[j], abs_diff=float(D[i, j]), tolerance=float(T[i, j]))
                for i, j in zip(*np.nonzero(~(D <= T)))]
        diag["max_abs_diff_per_column"] = per_col
        diag["max_abs_diff_per_column_by_tolerance"] = per_col_by_tol
        diag["max_abs_diff_by_tolerance"] = by_tol
        diag["cells_over_tolerance"] = over
        diag["max_abs_diff_per_row"] = [dict(row_ids[i], max_abs_diff=float(row_max[i])) for i in range(len(exp))]
        if "auroc" in num_cols:
            changed = D[:, num_cols.index("auroc")] > 0
            diag["any_auroc_changed"] = bool(changed.any())
            diag["n_auroc_changed"] = int(changed.sum())
    ok = bool(shape_ok and ids_ok and over == [])
    rec = check(section, cid, description, ok, stop=False, kind="numeric", max_abs_diff=max_all, tolerance=tol,
                shape_equal=shape_ok, identifiers_equal=ids_ok, max_abs_diff_per_column=per_col,
                max_abs_diff_by_tolerance=by_tol, n_cells_over_tolerance=None if over is None else len(over))
    return rec, diag


def lr_rows_auroc_4dp(got, exp, id_cols, num_cols, auroc_col):
    """r2, A2b: every LogisticRegression row whose numeric values are not all exactly equal to v0.2.1, with its AUROC
    (column auroc_col) here and in v0.2.1 and both rounded to 4 decimals ('{:.4f}'); None if the two frames do not hold
    the same rows (A2 fails then)."""
    if list(got.columns) != list(exp.columns) or len(got) != len(exp) or \
            not all(np.array_equal(got[c].to_numpy(), exp[c].to_numpy()) for c in id_cols):
        return None
    D = abs_diff_matrix(got, exp, num_cols)
    rows = []
    for i in range(len(exp)):
        if exp["learner"].iloc[i] == "LogisticRegression" and D[i].max() > 0:
            e, g = float(exp[auroc_col].iloc[i]), float(got[auroc_col].iloc[i])
            rows.append(dict({c: exp[c].iloc[i] for c in id_cols}, column=auroc_col, v021=e, ph1=g,
                             v021_4dp="{:.4f}".format(e), ph1_4dp="{:.4f}".format(g),
                             unchanged_4dp="{:.4f}".format(e) == "{:.4f}".format(g)))
    return rows


def compare_value(got, exp, tol):
    """(ok, |diff|): numbers and lists of numbers within tol; strings / booleans / None exactly (|diff| None);
    a missing or non-numeric observed value is a FAIL with |diff| = inf."""
    if exp is None or isinstance(exp, (str, bool)):
        return got == exp, None
    try:
        if isinstance(exp, list):
            if not isinstance(got, list) or len(got) != len(exp):
                return False, float("inf")
            d = max([abs(float(g) - float(e)) for g, e in zip(got, exp)] or [0.0])
        else:
            d = abs(float(got) - float(exp))
    except (TypeError, ValueError):
        return False, float("inf")
    if not math.isfinite(d):
        return False, float("inf")
    return d <= tol, d


def bitwise_frame_equal(a, b):
    """(equal, first differing column): same columns, dtypes and values; float columns compared bit for bit."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, "columns / rows"
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype != y.dtype:
            return False, "dtype of " + c
        if x.dtype.kind == "f":
            same = np.array_equal(np.ascontiguousarray(x).view(np.int64), np.ascontiguousarray(y).view(np.int64))
        else:
            same = np.array_equal(x, y)
        if not same:
            return False, c
    return True, None


def global_names(fn):
    """Module-level names a function (and every nested code object) loads, builtins excluded."""
    names, stack = set(), [fn.__code__]
    while stack:
        co = stack.pop()
        names |= {ins.argval for ins in dis.get_instructions(co) if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
        stack.extend(c for c in co.co_consts if isinstance(c, types.CodeType))
    return names - set(dir(builtins))


class PartAFailure(RuntimeError):
    """The Part A validation gate did not pass."""


# =====================================================================================
# (a) fit_predict - labels, preprocessing, feature blocks, fits and predictions (Parts A and B)
# =====================================================================================
# NOTE (ph1): the v0.2.1 sections LABEL DERIVATION, 1 (preprocessing), 2 (feature blocks) and 3 (fits) as ONE
# function of its inputs. The v0.2.1 working set df9 is the argument 'frame' (Part A: folds 1-9; Part B: folds 1-9
# followed by fold 10); the masks use max_train_fold / eval_fold (Part A: 8 / 9 = v0.2.1; Part B: 9 / 10);
# scp_statements.csv is read once in STEP 0 and passed as 'scp'. The metric statements of the v0.2.1 fit loop move
# to metrics_from_probs (no metric here); the Section 6 coefficient capture is not copied. The words 'folds 1-8'
# and 'fold 9' in copied comments and docstrings describe v0.2.1.
def fit_predict(frame, scp, max_train_fold, eval_fold, gate):
    """(a) Returns {'probs': long-format probabilities of the evaluation rows (CONFIGS order; evaluation rows in
    frame order), 'labels_eval': v0.2.1 labels of the evaluation rows, 'info': preprocessing facts, 'fitted': every
    fitted scaler / model, 'scalers': LR scaler per block}. gate(info) runs after preprocessing, before any fit."""
# >>> adapted from v0.2.1 (train / eval masks) | indent: 4 | literal: <= 8 => <= max_train_fold ;; == 9 => == eval_fold | renames: df9->frame
    mask_train = frame["strat_fold"] <= max_train_fold
    mask_eval = frame["strat_fold"] == eval_fold
# <<< adapted from v0.2.1
    assert set(frame["strat_fold"].unique()) == set(range(1, eval_fold + 1)), \
        "working set is not exactly folds 1-{} - STOP.".format(eval_fold)
    assert frame.index.is_unique and bool((mask_train ^ mask_eval).all()), "row masks do not partition the frame"
    train_patients = set(frame.loc[mask_train, "patient_id"])
    eval_patients = set(frame.loc[mask_eval, "patient_id"])
    assert train_patients.isdisjoint(eval_patients), "patient-level leakage between train and eval rows - STOP."
    print("train folds 1-{}: {} records / {} patients | eval fold {}: {} records / {} patients | disjoint: PASS".format(
        max_train_fold, int(mask_train.sum()), len(train_patients), eval_fold, int(mask_eval.sum()),
        len(eval_patients)))

# >>> adapted from v0.2.1 (label derivation; scp is an argument) | indent: 4 | renames: df9->frame
    diag = scp[scp["diagnostic"].fillna(0) == 1]
    code2class = {k: v for k, v in diag["diagnostic_class"].items() if isinstance(v, str)}
    parsed = frame["scp_codes"].apply(ast.literal_eval)
    rec_classes = parsed.apply(lambda d: {code2class[k] for k in d if k in code2class})
    lab = pd.DataFrame({c: rec_classes.apply(lambda s, c=c: int(c in s)) for c in SUPERCLASSES},
                       index=frame.index)
    for c in SUPERCLASSES:
        n_pos_train = int(lab.loc[mask_train, c].sum())
        n_pos_eval = int(lab.loc[mask_eval, c].sum())
        if n_pos_train == 0 or n_pos_eval == 0:
            warn("{}: zero positive examples in {} set - this class cannot be modeled.".format(
                c, "train" if n_pos_train == 0 else "eval"))
# <<< adapted from v0.2.1
    labels_eval = lab.loc[mask_eval]

# >>> adapted from v0.2.1 (Section 1: age, sex, recording_year, level strings, encoder fit on the training rows) | indent: 4 | renames: df9->frame
    age_raw = frame["age"]
    age_sentinel_mask = (age_raw == 300)
    n_age_sentinel = int(age_sentinel_mask.sum())
    age_clean = age_raw.where(~age_sentinel_mask, np.nan)
    age_censored = age_sentinel_mask.astype(float)
    age_imputed = age_clean.fillna(90.0)
    print("age: sentinel value 300 (ages above 89, per the v1.0.2 changelog) affects",
          n_age_sentinel, "records ({:.2f}%); replaced with NaN, flagged via age_censored, "
          "imputed with 90.".format(n_age_sentinel / len(frame) * 100))

    sex = frame["sex"].astype(float)
    print("sex: used as-is (0/1), no transformation.")

    rd = pd.to_datetime(frame["recording_date"], errors="coerce")
    recording_year = rd.dt.year.astype(float)
    n_year_missing = int(recording_year.isna().sum())
    if n_year_missing > 0:
        train_median_year = float(recording_year[mask_train].median())
        recording_year = recording_year.fillna(train_median_year)
        warn("{} records had an unparsable recording_date; recording_year imputed with the "
             "training-set median ({}). This decision was not in the original spec and is "
             "logged here for transparency.".format(n_year_missing, train_median_year))
    else:
        print("recording_year: derived from recording_date, no missing values.")

    print("height / weight: excluded entirely (68%/57% missing; physiologically impossible "
          "values such as a minimum height of 6cm and a minimum weight of 5kg).")

    device_level = to_level_string(frame["device"])
    site_level = to_level_string(frame["site"])
    nurse_level = to_level_string(frame["nurse"])

    device_dummies, device_categories = encode_categorical(device_level, mask_train, "device")
    site_dummies, site_categories = encode_categorical(site_level, mask_train, "site")
    nurse_dummies, nurse_categories = encode_categorical(nurse_level, mask_train, "nurse")
# <<< adapted from v0.2.1
    for name, dummies, cats in [("device", device_dummies, device_categories), ("site", site_dummies, site_categories),
                                ("nurse", nurse_dummies, nurse_categories)]:
        print("{}: {} training levels (incl. __MISSING__) + 1 __UNKNOWN__ catch-all -> {} one-hot columns | "
              "__UNKNOWN__ active on {} eval records".format(name, len(cats) - 1, dummies.shape[1],
                                                            int(dummies.loc[mask_eval, "{}=__UNKNOWN__".format(name)].sum())))

# >>> adapted from v0.2.1 (Section 1 feature frame, Section 2 feature blocks, design matrix shapes) | indent: 4
    FEATURES = pd.concat([
        age_imputed.rename("age_imputed"),
        age_censored.rename("age_censored"),
        sex.rename("sex"),
        device_dummies, site_dummies, nurse_dummies,
        recording_year.rename("recording_year"),
    ], axis=1)

    B1_cols = ["age_imputed", "age_censored", "sex"]
    B2_cols = list(device_dummies.columns) + list(site_dummies.columns) + list(nurse_dummies.columns)
    B3_cols = B1_cols + B2_cols
    B4_cols = ["recording_year"]
    B5_cols = B1_cols + B2_cols + B4_cols

    BLOCKS = {
        "B0_intercept": [],
        "B1_physio": B1_cols,
        "B2_acquisition": B2_cols,
        "B3_physio_plus_acquisition": B3_cols,
        "B4_temporal": B4_cols,
        "B5_all": B5_cols,
    }

    print("\n--- feature block design matrix shapes ---")
    for name, cols in BLOCKS.items():
        if name == "B0_intercept":
            print("  {:28s} train=({}, 1) [constant zero column]  eval=({}, 1)".format(
                name, int(mask_train.sum()), int(mask_eval.sum())))
        else:
            print("  {:28s} train=({}, {})  eval=({}, {})".format(
                name, int(mask_train.sum()), len(cols), int(mask_eval.sum()), len(cols)))
# <<< adapted from v0.2.1
    assert list(BLOCKS) == BLOCK_NAMES
    info = {"frame": frame, "lab": lab, "labels_eval": labels_eval, "mask_train": mask_train, "mask_eval": mask_eval,
            "levels": {"device": device_level, "site": site_level, "nurse": nurse_level},
            "categories": {"device": device_categories, "site": site_categories, "nurse": nurse_categories},
            "dummies": {"device": device_dummies, "site": site_dummies, "nurse": nurse_dummies},
            "feature_columns": list(FEATURES.columns), "block_widths": {k: len(v) for k, v in BLOCKS.items()},
            "n_age_sentinel": n_age_sentinel, "n_year_missing": n_year_missing,
            "n_train": int(mask_train.sum()), "n_eval": int(mask_eval.sum()),
            "n_train_patients": len(train_patients), "n_eval_patients": len(eval_patients)}
    gate(info)                                         # A1 (Part A) / B0 (Part B) - before any model is fit

# >>> adapted from v0.2.1 (training labels) | indent: 4
    y_train_full = lab.loc[mask_train]
# <<< adapted from v0.2.1
    FITTED, SCALERS, PROBS = [], {}, {}
    t_fit = time.time()
# >>> adapted from v0.2.1 (Section 3 fit loop up to the training labels) | indent: 4
    for block_name, cols in BLOCKS.items():
        if block_name == "B0_intercept":
            Xtr_raw = np.zeros((int(mask_train.sum()), 1))
            Xev_raw = np.zeros((int(mask_eval.sum()), 1))
        else:
            Xtr_raw = FEATURES.loc[mask_train, cols].to_numpy(dtype=float)
            Xev_raw = FEATURES.loc[mask_eval, cols].to_numpy(dtype=float)

        # Only continuous columns are standardized for LogisticRegression. Binary /
        # near-binary columns (sex, age_censored) and every one-hot dummy column
        # (device=*, site=*, nurse=*) pass through unscaled on their raw 0/1 scale,
        # so a rare category's coefficient is not inflated or deflated by its own
        # tiny within-column standard deviation.
        block_cols = cols if block_name != "B0_intercept" else []
        continuous_cols = [c for c in block_cols if c in ("age_imputed", "recording_year")]
        if continuous_cols:
            cont_idx = [block_cols.index(c) for c in continuous_cols]
            rest_idx = [i for i in range(len(block_cols)) if i not in cont_idx]

            scaler = StandardScaler().fit(Xtr_raw[:, cont_idx])
            FITTED.append(scaler)  # [ph1 added]
            SCALERS[block_name] = scaler  # [ph1 added]
            Xtr_cont = scaler.transform(Xtr_raw[:, cont_idx])
            Xev_cont = scaler.transform(Xev_raw[:, cont_idx])

            Xtr_lr = np.empty_like(Xtr_raw)
            Xev_lr = np.empty_like(Xev_raw)
            Xtr_lr[:, cont_idx] = Xtr_cont
            Xev_lr[:, cont_idx] = Xev_cont
            if rest_idx:
                Xtr_lr[:, rest_idx] = Xtr_raw[:, rest_idx]
                Xev_lr[:, rest_idx] = Xev_raw[:, rest_idx]
        else:
            # No continuous columns in this block (B0, B2) -> nothing to scale.
            Xtr_lr = Xtr_raw
            Xev_lr = Xev_raw

        for learner in LEARNERS:
            for seed in SEEDS:
                scores_eval = {}
                models_this = {}
                per_class_auroc, per_class_auprc = [], []

                for sc in SUPERCLASSES:
                    ytr = y_train_full[sc].to_numpy()
# <<< adapted from v0.2.1
# NOTE (ph1): v0.2.1 line 'yev = y_eval_full[sc].to_numpy()' is not copied - no evaluation label is used here.
# >>> adapted from v0.2.1 (Section 3 fit + predict_proba per class) | indent: 4

                    if learner == "LogisticRegression":
                        model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
                        model.fit(Xtr_lr, ytr)
                        pos_col = list(model.classes_).index(1)
                        proba = model.predict_proba(Xev_lr)[:, pos_col]
                    else:
                        model = HistGradientBoostingClassifier(random_state=seed)
                        model.fit(Xtr_raw, ytr)
                        pos_col = list(model.classes_).index(1)
                        proba = model.predict_proba(Xev_raw)[:, pos_col]

                    scores_eval[sc] = proba
                    models_this[sc] = model
# <<< adapted from v0.2.1
                PROBS[(block_name, learner, seed)] = scores_eval
                FITTED.extend(models_this[sc] for sc in SUPERCLASSES)
        print("  fitted {:28s} | {} learners x {} seeds x {} classes | elapsed {:.1f}s".format(
            block_name, len(LEARNERS), len(SEEDS), len(SUPERCLASSES), time.time() - t_fit))

    ev_ecg = frame.index[mask_eval].to_numpy().astype(np.int64)
    ev_pid_raw = frame.loc[mask_eval, "patient_id"].to_numpy()
    assert np.all(ev_pid_raw == np.round(ev_pid_raw)), "non-integral patient_id in the evaluation rows"
    ev_pid = ev_pid_raw.astype(np.int64)
    chunks = []
    for block_name, learner, seed in CONFIGS:
        chunk = pd.DataFrame({"ecg_id": ev_ecg, "patient_id": ev_pid})
        chunk["block"] = block_name
        chunk["learner"] = learner
        chunk["seed"] = np.int64(seed)
        for sc in SUPERCLASSES:
            chunk["p_" + sc] = np.asarray(PROBS[(block_name, learner, seed)][sc], dtype=np.float64)
        chunks.append(chunk)
    probs_long = pd.concat(chunks, ignore_index=True)
    assert list(probs_long.columns) == PROBS_CSV_COLUMNS and len(probs_long) == len(CONFIGS) * len(ev_ecg)
    return {"probs": probs_long, "labels_eval": labels_eval, "info": info, "fitted": FITTED, "scalers": SCALERS}


# =====================================================================================
# (b) metrics_from_probs - v0.2.1 metric rows / decision table / H1 and v0.3 H2 from a probability frame
# =====================================================================================
# NOTE (ph1): every number is computed from the long-format probability frame (Part B: the file read back from disk)
# and the evaluation labels. v0.2.1 (Sections 3+4 metrics, 5): fast_auc / AUPRC per class, macro, the v0.2.1
# patient-level marginal bootstrap (its index matrix is returned and compared with BOOT_IDX), the decision table and
# the B3 >= 0.60 verdict. v0.3 (Parts A-C) on the seed-42 configurations of the SAME frame: the paired per-draw
# deltas come from REF.paired_bootstrap on boot_idx (strict NaN rule; v0.3 used a per-class nan-skip - identical
# when no draw is NaN); point values from REF.macro_auroc_strict (asserted == the v0.2.1 macro values); mean, SE,
# percentile CIs, MDE, verdict strings, achieved bound and disagreement list from the v0.3 code; the v0.3 CIs are
# asserted == REF.summarize_deltas. SESOI_H2 and the MDE z values are arguments (read from pre_registration.json).
def metrics_from_probs(probs_long, labels_eval, boot_idx, sesoi_h2, z_alpha_2, z_power):
    """(b) Returns the v0.2.1 metric rows, decision table and H1 verdict, the v0.3 H2 statistics per learner, the
    per-draw deltas and the v0.2.1 bootstrap index matrix of this evaluation fold."""
    n_eval = len(labels_eval)
    assert list(probs_long.columns) == PROBS_CSV_COLUMNS, "probability frame columns {}".format(list(probs_long.columns))
    assert len(probs_long) == len(CONFIGS) * n_eval, "probability frame has {} rows, expected {}".format(
        len(probs_long), len(CONFIGS) * n_eval)
    assert list(labels_eval.columns) == SUPERCLASSES
    eval_ecg_ids = labels_eval.index.to_numpy()
    P, eval_patient_ids = {}, None
    for k, (block_name, learner, seed) in enumerate(CONFIGS):
        chunk = probs_long.iloc[k * n_eval:(k + 1) * n_eval]
        assert (chunk["block"] == block_name).all() and (chunk["learner"] == learner).all() and \
            (chunk["seed"] == seed).all(), "probability frame order differs from CONFIGS at {}".format(k)
        assert np.array_equal(chunk["ecg_id"].to_numpy(), eval_ecg_ids), "ecg_id order differs at {}".format(k)
        if eval_patient_ids is None:
            eval_patient_ids = chunk["patient_id"].to_numpy().astype(np.int64)
        assert np.array_equal(chunk["patient_id"].to_numpy(), eval_patient_ids), "patient_id differs at {}".format(k)
        P[(block_name, learner, seed)] = {sc: chunk["p_" + sc].to_numpy(dtype=np.float64) for sc in SUPERCLASSES}

# >>> adapted from v0.2.1 (chance AUPRC baseline = evaluation-fold prevalence) | indent: 4 | literal: lab.loc[mask_eval] => labels_eval | renames: prev_fold9->prev_eval
    prev_eval = labels_eval.mean()
# <<< adapted from v0.2.1
# >>> adapted from v0.2.1 (evaluation labels) | indent: 4 | literal: lab.loc[mask_eval] => labels_eval
    y_eval_full = labels_eval
# <<< adapted from v0.2.1
# >>> adapted from v0.2.1 (patient-level bootstrap resamples, v0.2.1 construction) | indent: 4 | literal: df9.loc[mask_eval, "patient_id"].to_numpy() => eval_patient_ids
    # --- precompute patient-level bootstrap resamples once; reused for every config ---
    eval_patient_arr = eval_patient_ids
    unique_eval_patients = np.unique(eval_patient_arr)
    n_eval_patients = len(unique_eval_patients)
    patient_row_list = [np.where(eval_patient_arr == p)[0] for p in unique_eval_patients]

    t0 = time.time()
    rng_boot = np.random.default_rng(BOOT_SEED)
    BOOT_ROWS = []
    BOOT_SAMP = []  # [ph1 added]
    for _ in range(N_BOOT):
        samp = rng_boot.integers(0, n_eval_patients, size=n_eval_patients)
        BOOT_SAMP.append(samp)  # [ph1 added]
        BOOT_ROWS.append(np.concatenate([patient_row_list[i] for i in samp]))
# <<< adapted from v0.2.1
    print("v0.2.1 marginal bootstrap: {} patient-level resamples of the evaluation fold ({} patients, {} records) "
          "in {:.2f}s".format(N_BOOT, n_eval_patients, n_eval, time.time() - t0))

# >>> adapted from v0.2.1 (metric rows list) | indent: 4
    METRIC_ROWS = []
# <<< adapted from v0.2.1
    MACRO_AUROC = {}
    t_all = time.time()
    for block_name in BLOCK_NAMES:
        for learner in LEARNERS:
            for seed in SEEDS:
                scores_eval = P[(block_name, learner, seed)]
                per_class_auroc, per_class_auprc = [], []
                for sc in SUPERCLASSES:
                    yev = y_eval_full[sc].to_numpy()
                    proba = scores_eval[sc]
# >>> adapted from v0.2.1 (per-class AUROC / AUPRC rows) | indent: 4 | renames: prev_fold9->prev_eval
                    auroc = fast_auc(yev, proba)
                    auprc = float(average_precision_score(yev, proba)) if (0 < yev.sum() < len(yev)) else float("nan")
                    per_class_auroc.append(auroc)
                    per_class_auprc.append(auprc)

                    METRIC_ROWS.append({
                        "block": block_name, "learner": learner, "seed": seed, "class": sc,
                        "auroc": auroc, "auroc_chance": 0.5,
                        "auprc": auprc, "auprc_chance": float(prev_eval[sc]),
                        "ci_low": float("nan"), "ci_high": float("nan"),
                    })
# <<< adapted from v0.2.1
# >>> adapted from v0.2.1 (macro AUROC / AUPRC + patient-level bootstrap 95% CI) | indent: 4 | renames: prev_fold9->prev_eval
                macro_auroc = float(np.nanmean(per_class_auroc))
                macro_auprc = float(np.nanmean(per_class_auprc))
                macro_auprc_chance = float(np.mean([prev_eval[c] for c in SUPERCLASSES]))

                boot_macros = np.empty(N_BOOT)
                for bi, rows in enumerate(BOOT_ROWS):
                    accs = []
                    for sc in SUPERCLASSES:
                        yb = y_eval_full[sc].to_numpy()[rows]
                        sb = scores_eval[sc][rows]
                        a = fast_auc(yb, sb)
                        if not np.isnan(a):
                            accs.append(a)
                    boot_macros[bi] = np.mean(accs) if accs else np.nan
                n_boot_valid = int(np.sum(~np.isnan(boot_macros)))
                ci_low, ci_high = np.nanpercentile(boot_macros, [2.5, 97.5])

                METRIC_ROWS.append({
                    "block": block_name, "learner": learner, "seed": seed, "class": "MACRO",
                    "auroc": macro_auroc, "auroc_chance": 0.5,
                    "auprc": macro_auprc, "auprc_chance": macro_auprc_chance,
                    "ci_low": float(ci_low), "ci_high": float(ci_high),
                })
                print("  {:28s} | {:22s} | seed={:5d} | macro-AUROC={:.4f} (chance 0.50) "
                      "95%CI[{:.4f},{:.4f}] n_boot_valid={} | macro-AUPRC={:.4f} (chance {:.4f})".format(
                          block_name, learner, seed, macro_auroc, ci_low, ci_high, n_boot_valid,
                          macro_auprc, macro_auprc_chance))
# <<< adapted from v0.2.1
                MACRO_AUROC[(block_name, learner, seed)] = macro_auroc
    print("metric rows + marginal bootstrap: {:.1f}s".format(time.time() - t_all))
# >>> adapted from v0.2.1 (metric rows frame) | indent: 4
    metrics_df = pd.DataFrame(METRIC_ROWS)
# <<< adapted from v0.2.1

# >>> adapted from v0.2.1 (Section 5 decision table) | indent: 4 | renames: BLOCKS->BLOCK_NAMES
    macro_df = metrics_df[metrics_df["class"] == "MACRO"]
    decision_rows = []
    for block_name in BLOCK_NAMES:
        for learner in LEARNERS:
            sub = macro_df[(macro_df["block"] == block_name) & (macro_df["learner"] == learner)]
            decision_rows.append({
                "block": block_name, "learner": learner,
                "macro_auroc_mean": float(sub["auroc"].mean()),
                "macro_auroc_std_across_seeds": float(sub["auroc"].std(ddof=0)),
                "ci_low_mean": float(sub["ci_low"].mean()),
                "ci_high_mean": float(sub["ci_high"].mean()),
            })
    decision_df = pd.DataFrame(decision_rows).set_index(["block", "learner"])

    b1_ref = {learner: decision_df.loc[("B1_physio", learner), "macro_auroc_mean"] for learner in LEARNERS}
    decision_df["delta_vs_B1_physio"] = [
        row["macro_auroc_mean"] - b1_ref[idx[1]] for idx, row in decision_df.iterrows()
    ]

    print(decision_df.round(5).to_string())
    print("\nNote: macro_auroc_mean / ci_*_mean average the 3 per-seed point estimates and CI "
          "bounds. LogisticRegression uses lbfgs (deterministic given fixed data), so its "
          "std across seeds should be ~0; any residual is float-precision noise, not real "
          "seed sensitivity.")
# <<< adapted from v0.2.1
    print("\n--- H1 rule: does B3_physio_plus_acquisition reach macro-AUROC >= {:.2f} on the evaluation fold? ---".format(
        VERDICT_THRESHOLD))
# >>> adapted from v0.2.1 (B3 >= 0.60 verdict) | indent: 4
    VERDICT = {}
    for learner in LEARNERS:
        val = decision_df.loc[("B3_physio_plus_acquisition", learner), "macro_auroc_mean"]
        passed = bool(val >= VERDICT_THRESHOLD)
        VERDICT[learner] = {"macro_auroc_mean": float(val), "passes_threshold": passed}
        print("  [{}] macro-AUROC = {:.4f} -> {}".format(learner, val, "PASS" if passed else "FAIL"))
    overall_pass = any(v["passes_threshold"] for v in VERDICT.values())
    print("overall: {} (at least one learner reaches the threshold)".format(
        "PASS" if overall_pass else "FAIL"))
    VERDICT["overall_pass"] = overall_pass
# <<< adapted from v0.2.1

    # --- H2 (v0.3) on the seed-42 configurations of this frame; per-draw deltas from REF on boot_idx ---
    Y_EVAL = labels_eval[SUPERCLASSES].to_numpy(dtype=np.int64)
    ref_patients, ref_rows = REF["patient_rows_in_file_order"](eval_patient_ids)
    assert np.array_equal(ref_patients, unique_eval_patients) and len(ref_rows) == len(patient_row_list) and \
        all(np.array_equal(a, b) for a, b in zip(ref_rows, patient_row_list)), \
        "REF patient grouping differs from the v0.2.1 grouping"
    assert boot_idx.shape == (N_BOOT, len(ref_patients)), "boot_idx shape {}".format(boot_idx.shape)
# >>> adapted from v0.3 (point macro-AUROC per learner and block) | indent: 4
    POINT_MACRO = {learner: {} for learner in LEARNERS}
# <<< adapted from v0.3
    DELTAS = {}
    t0 = time.time()
    for learner in LEARNERS:
        P_ARM = {}
        for block_name in COMPARED_BLOCKS:
            P_ARM[block_name] = np.column_stack([P[(block_name, learner, SEED)][sc] for sc in SUPERCLASSES])
            POINT_MACRO[learner][block_name] = REF["macro_auroc_strict"](Y_EVAL, P_ARM[block_name])
            assert POINT_MACRO[learner][block_name] == MACRO_AUROC[(block_name, learner, SEED)], \
                "REF macro_auroc_strict != v0.2.1 macro-AUROC for {} {} seed {}".format(block_name, learner, SEED)
        DELTAS[learner], _ = REF["paired_bootstrap"](Y_EVAL, P_ARM["B1_physio"], P_ARM["B3_physio_plus_acquisition"],
                                                     ref_rows, boot_idx)
    print("H2 paired bootstrap (REF, delta = B3 - B1, seed {}): {:.1f}s".format(SEED, time.time() - t0))
# >>> adapted from v0.3 (H2 statistics per learner) | indent: 4
    H2_STATS = {}
    for learner in LEARNERS:
        d = DELTAS[learner]
        n_invalid = int(np.sum(np.isnan(d)))
        point_delta = POINT_MACRO[learner]["B3_physio_plus_acquisition"] - POINT_MACRO[learner]["B1_physio"]
        mean_delta = float(np.nanmean(d))
        se_delta = float(np.nanstd(d, ddof=1))
        ci95 = [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]
        ci90 = [float(x) for x in np.nanpercentile(d, [5, 95])]
        H2_STATS[learner] = {
            "point_B1": POINT_MACRO[learner]["B1_physio"],
            "point_B3": POINT_MACRO[learner]["B3_physio_plus_acquisition"],
            "point_delta": float(point_delta), "mean_delta": mean_delta, "SE_delta": se_delta,
            "ci95": ci95, "ci90": ci90, "n_boot_valid": N_BOOT - n_invalid, "n_boot_invalid": n_invalid,
        }
        if n_invalid > 0:
            warn("{}: {} invalid (NaN) bootstrap replicates.".format(learner, n_invalid))
        print("  [{:20s}] point delta={:+.4f} | mean delta={:+.4f} | SE={:.4f} | "
              "95%CI[{:+.4f},{:+.4f}] | 90%CI[{:+.4f},{:+.4f}] | invalid replicates={}".format(
                  learner, point_delta, mean_delta, se_delta, ci95[0], ci95[1], ci90[0], ci90[1], n_invalid))
# <<< adapted from v0.3
    REF_SUMMARY = {}
    for learner in LEARNERS:
        S = REF["summarize_deltas"](H2_STATS[learner]["point_delta"], DELTAS[learner])
        assert [S["lo95"], S["hi95"]] == H2_STATS[learner]["ci95"] and [S["lo90"], S["hi90"]] == H2_STATS[learner]["ci90"], \
            "v0.3 nanpercentile CIs != REF summarize_deltas for " + learner
        assert S["point_delta"] == H2_STATS[learner]["point_delta"] and S["n_valid"] == H2_STATS[learner]["n_boot_valid"] \
            and S["n_nan"] == H2_STATS[learner]["n_boot_invalid"], "REF summary counts differ for " + learner
        REF_SUMMARY[learner] = S
# >>> adapted from v0.3 (MDE) | indent: 4 | renames: Z_ALPHA_2->z_alpha_2, Z_POWER->z_power
    for learner in LEARNERS:
        mde = round((z_alpha_2 + z_power) * H2_STATS[learner]["SE_delta"], 4)
        H2_STATS[learner]["MDE"] = mde
        print("  [{:20s}] MDE = ({} + {}) * {:.6f} = {:.4f}".format(
            learner, z_alpha_2, z_power, H2_STATS[learner]["SE_delta"], mde))
# <<< adapted from v0.3
# >>> adapted from v0.3 (difference test, TOST, achieved bound, learner disagreement) | indent: 4 | renames: SESOI_H2->sesoi_h2
    for learner in LEARNERS:
        s = H2_STATS[learner]
        lo95, hi95 = s["ci95"]
        lo90, hi90 = s["ci90"]
        zero_in_ci = (lo95 <= 0.0 <= hi95)
        if zero_in_ci:
            diff_verdict = "no statistically significant difference between B1 and B3"
        else:
            diff_verdict = "significant difference: B3 {} than B1 by {:+.4f} (point delta)".format(
                "higher" if s["point_delta"] > 0 else "lower", s["point_delta"])
        tost_pass = (lo90 >= -sesoi_h2) and (hi90 <= sesoi_h2)
        tost_verdict = "practically negligible" if tost_pass else "inconclusive / not shown negligible"
        s.update({
            "difference_significant": (not zero_in_ci), "difference_verdict": diff_verdict,
            "tost_pass": tost_pass, "tost_verdict": tost_verdict,
            "achieved_equivalence_bound": float(max(abs(lo90), abs(hi90))),
        })
        print("  [{:20s}] difference test: 95%CI[{:+.4f},{:+.4f}] -> {}".format(learner, lo95, hi95, diff_verdict))
        print("  [{:20s}] TOST (+/-{:.2f}): 90%CI[{:+.4f},{:+.4f}] -> {}".format(
            learner, sesoi_h2, lo90, hi90, tost_verdict))
        print("  [{:20s}] achieved_equivalence_bound = {:.4f}".format(learner, s["achieved_equivalence_bound"]))

    DISAGREEMENTS = []
    if H2_STATS[PRIMARY_LEARNER]["difference_significant"] != H2_STATS[ROBUSTNESS_LEARNER]["difference_significant"]:
        DISAGREEMENTS.append("difference test: {} -> '{}'; {} -> '{}'".format(
            PRIMARY_LEARNER, H2_STATS[PRIMARY_LEARNER]["difference_verdict"],
            ROBUSTNESS_LEARNER, H2_STATS[ROBUSTNESS_LEARNER]["difference_verdict"]))
    if H2_STATS[PRIMARY_LEARNER]["tost_pass"] != H2_STATS[ROBUSTNESS_LEARNER]["tost_pass"]:
        DISAGREEMENTS.append("TOST: {} -> '{}'; {} -> '{}'".format(
            PRIMARY_LEARNER, H2_STATS[PRIMARY_LEARNER]["tost_verdict"],
            ROBUSTNESS_LEARNER, H2_STATS[ROBUSTNESS_LEARNER]["tost_verdict"]))
    print()
    if DISAGREEMENTS:
        print("LEARNER DISAGREEMENT (reported, NOT resolved by choosing one learner):")
        for dmsg in DISAGREEMENTS:
            print("  - " + dmsg)
    else:
        print("Learners agree on both tests.")
# <<< adapted from v0.3

# >>> adapted from v0.3 (results block per learner, pre_registration.json field names) | indent: 4
    def _learner_block(learner):
        s = H2_STATS[learner]
        return {
            "point_macro_auroc_B1": s["point_B1"], "point_macro_auroc_B3": s["point_B3"],
            "point_delta": s["point_delta"], "mean_delta": s["mean_delta"], "SE_delta": s["SE_delta"],
            "ci95_percentile": s["ci95"], "ci90_percentile": s["ci90"],
            "n_boot_valid": s["n_boot_valid"], "n_boot_invalid": s["n_boot_invalid"],
            "MDE": s["MDE"],
            "difference_test_verdict": s["difference_verdict"],
            "tost_verdict": s["tost_verdict"],
            "achieved_equivalence_bound": s["achieved_equivalence_bound"],
        }
# <<< adapted from v0.3
# >>> adapted from v0.3 (per-draw deltas frame) | indent: 4
    boot_df = pd.DataFrame({
        "replicate": np.arange(N_BOOT),
        "delta_LR": DELTAS["LogisticRegression"],
        "delta_HGB": DELTAS["HistGradientBoosting"],
    })
# <<< adapted from v0.3
    results_per_learner = {}
    for learner in LEARNERS:
        results_per_learner[learner] = _learner_block(learner)
        results_per_learner[learner]["n_patients"] = int(len(ref_patients))
    return {"metric_rows": metrics_df, "decision_table": decision_df, "h1_verdict": VERDICT, "macro_auroc": MACRO_AUROC,
            "boot_samp": np.array(BOOT_SAMP, dtype=np.int64), "v021_patient_row_list": patient_row_list,
            "eval_patient_ids": eval_patient_ids, "h2_stats": H2_STATS, "results_per_learner": results_per_learner,
            "learner_disagreement": DISAGREEMENTS, "ref_summaries": REF_SUMMARY, "deltas": DELTAS, "boot_df": boot_df}


ALLOWED_FUNCTION_GLOBALS = {
    "np", "pd", "ast", "time", "StandardScaler", "LogisticRegression", "HistGradientBoostingClassifier",
    "average_precision_score", "fast_auc", "to_level_string", "encode_categorical", "warn", "REF",
    "SUPERCLASSES", "SEEDS", "LEARNERS", "N_BOOT", "BOOT_SEED", "VERDICT_THRESHOLD", "SEED", "COMPARED_BLOCKS",
    "PRIMARY_LEARNER", "ROBUSTNESS_LEARNER", "BLOCK_NAMES", "CONFIGS", "PROBS_CSV_COLUMNS"}


# =====================================================================================
# Part-specific gates (called inside fit_predict after preprocessing, before the first fit)
# =====================================================================================
def gate_part_a(info):
    """A1: v0.2.1 label counts on fold 9, split sizes, folds 1-8 encoder (FRP development widths)."""
    pos9 = {c: int(info["labels_eval"][c].sum()) for c in SUPERCLASSES}
    check("A1", "A1_fold9_positives", "fold-9 positives (v0.2.1 label code) == v0.2.1 counts",
          pos9 == EXPECTED_FOLD9_POSITIVES, kind="exact", expected=EXPECTED_FOLD9_POSITIVES, observed=pos9)
    split = {k: info[k] for k in ("n_train", "n_eval", "n_train_patients", "n_eval_patients")}
    check("A1", "A1_split_sizes", "train folds 1-8 / eval fold 9 records and patients == v0.2.1 / v0.3",
          split == EXPECTED_SPLITS["A"], kind="exact", expected=EXPECTED_SPLITS["A"], observed=split)
    exp18 = {c: [v for v in LQ_ENC["categories"][c] if v not in LQ_ENC["new_levels_vs_folds1to8"][c]] for c in ACQ_COLS}
    widths = {c: len(info["categories"][c]) for c in ACQ_COLS}
    check("A1", "A1_encoder_folds1to8", "folds 1-8 categories == FRP folds 1-9 categories minus new_levels_vs_folds1to8"
          " (widths == FRP widths_folds1to8_development)",
          info["categories"] == exp18 and widths == LQ_ENC["widths_folds1to8_development"], kind="exact",
          expected=LQ_ENC["widths_folds1to8_development"], observed=widths)


def gate_part_b(info):
    """B0 checks that need the preprocessing of the folds 1-9 + fold 10 frame (before any fold-10 model is fit)."""
    m_tr, m_ev, frame = info["mask_train"], info["mask_eval"], info["frame"]
    ev = np.stack([frame.index[m_ev].to_numpy().astype(np.int64),
                   frame.loc[m_ev, "patient_id"].to_numpy().astype(np.int64)], axis=1)
    check("B0", "B0_eval_rows_file_order", "evaluation rows == fold-10 (ecg_id, patient_id) in ptbxl_database.csv "
          "file order (F10_IDS)", np.array_equal(ev, F10_IDS), kind="exact")
    lab10 = info["labels_eval"]
    cols_ok = list(V09_LABELS.columns) == ["ecg_id", "patient_id"] + SUPERCLASSES
    same_ids = cols_ok and len(V09_LABELS) == len(lab10) and \
        np.array_equal(V09_LABELS["ecg_id"].to_numpy(), lab10.index.to_numpy()) and \
        np.array_equal(V09_LABELS["patient_id"].to_numpy(), ev[:, 1])
    n_diff = int((V09_LABELS[SUPERCLASSES].to_numpy(dtype=np.int64) != lab10[SUPERCLASSES].to_numpy(dtype=np.int64)).sum()) \
        if same_ids else None
    check("B0", "B0_labels_eq_v09", "fold-10 labels (v0.2.1 code) == v0.9_fold10_labels.csv (same rows, order, values)",
          bool(same_ids) and n_diff == 0, kind="exact",
          observed={"columns_equal": cols_ok, "same_rows_and_order": bool(same_ids), "n_differing_label_values": n_diff})
    pos10 = {c: int(lab10[c].sum()) for c in SUPERCLASSES}
    check("B0", "B0_fold10_positives", "fold-10 positives == the counts disclosed in FRP",
          pos10 == EXPECTED_FOLD10_POSITIVES, kind="exact", expected=EXPECTED_FOLD10_POSITIVES, observed=pos10)
    check("B0", "B0_encoder_eq_FRP", "device / site / nurse categories fit on folds 1-9 == FRP "
          "locked_quantities.categorical_encoder.categories", info["categories"] == LQ_ENC["categories"], kind="exact",
          expected={c: len(LQ_ENC["categories"][c]) for c in ACQ_COLS},
          observed={c: len(info["categories"][c]) for c in ACQ_COLS})
    n_unknown = {c: int(info["dummies"][c].loc[m_ev, "{}=__UNKNOWN__".format(c)].sum()) for c in ACQ_COLS}
    check("B0", "B0_fold10_unknown_zero", "fold-10 records mapped to __UNKNOWN__ == 0",
          sum(n_unknown.values()) == 0, kind="exact", expected=0, observed=n_unknown)
    allzero = sorted(col for c in ACQ_COLS for col in info["dummies"][c].columns
                     if int(info["dummies"][c].loc[m_tr, col].sum()) == 0)
    on_allzero = {c: int(info["dummies"][c].loc[m_ev, [col for col in allzero if col.startswith(c + "=")]].to_numpy().sum())
                  for c in ACQ_COLS}
    check("B0", "B0_untrained_columns", "all-zero training columns == FRP allzero_training_columns_folds1to9 and 0 "
          "fold-10 records on them", allzero == sorted(LQ_ENC["allzero_training_columns_folds1to9"]) and
          sum(on_allzero.values()) == 0, kind="exact", expected=sorted(LQ_ENC["allzero_training_columns_folds1to9"]),
          observed={"allzero": allzero, "fold10_records_on_them": on_allzero})
    same_levels = {c: bool(info["levels"][c].loc[df9.index].equals(to_level_string(df9[c]))) for c in ACQ_COLS}
    check("B0", "B0_level_strings_unchanged", "level strings of the folds 1-9 rows are unchanged by the fold-10 "
          "values (FRP fold10_level_mapping)", all(same_levels.values()), kind="exact", observed=same_levels)
    split = {k: info[k] for k in ("n_train", "n_eval", "n_train_patients", "n_eval_patients")}
    check("B0", "B0_split_sizes", "train folds 1-9 / eval fold 10 records and patients (patient-disjoint)",
          split == EXPECTED_SPLITS["B"], kind="exact", expected=EXPECTED_SPLITS["B"], observed=split)


# =====================================================================================
# STEP 0 - ENVIRONMENT, INPUTS, GUARDS (nothing is written before step 4)
# =====================================================================================
hdr("STEP 0 - ENVIRONMENT, INPUTS, GUARDS")
print("run started  :", RUN_STARTED)
print("code version :", CODE_VERSION, "| revision:", CODE_REVISION)
print("status       :", POSTHOC_LABEL)

# --- 1. versions ---
VERSIONS = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "os": platform.platform()}
print("versions     :", VERSIONS)
print(VERSIONS_NOTE)
_ver_diff = {k: (VERSIONS[k], v) for k, v in EXPECTED_VERSIONS.items() if VERSIONS[k] != v}
assert not _ver_diff, "STOP: environment differs from the required versions (now, required): {}".format(_ver_diff)
CPU_COUNT = os.cpu_count()
THREADPOOL_INFO = json_safe(plain(threadpool_info()))
print("os.cpu_count() = {} | thread pools: {}".format(CPU_COUNT, [(d.get("internal_api"), d.get("num_threads"))
                                                                 for d in THREADPOOL_INFO]))

# --- 2. hashes of every input; REF ---
for p in (DB_CSV, SCP_CSV):
    assert os.path.isfile(p), "file not found: " + p
INPUT_HASHES = {os.path.basename(p): sha256_of(p) for p in (DB_CSV, SCP_CSV)}
assert INPUT_HASHES == EXPECTED_INPUT_HASHES, "STOP: input CSV hashes {} != expected - aborting.".format(INPUT_HASHES)
for k, v in INPUT_HASHES.items():
    print("{:28s} {} -> PASS".format(k, v))
DOCS, DOC_SELF, DOC_FILE = {}, {}, {}
for key, path in (("pre_registration", SEAL_PATH), ("final_run_protocol", FINAL_PROTOCOL_PATH)):
    assert os.path.isfile(path), key + " not found: " + path + " - aborting."
    _s, _r, _d = verify_self_sha256(path)
    assert _s == _r == EXPECTED_SELF_SHA256[key], \
        "STOP: {}: self_sha256 stored {} / recomputed {} / expected {}.".format(key, _s, _r, EXPECTED_SELF_SHA256[key])
    _f = sha256_of(path)
    assert _f == EXPECTED_FILE_SHA256[key], "STOP: {}: file sha256 {} != {}.".format(key, _f, EXPECTED_FILE_SHA256[key])
    DOCS[key], DOC_SELF[key], DOC_FILE[key] = _d, _s, _f
    print("{:28s} self {} | file {} -> PASS".format(os.path.basename(path), _s, _f))
SEAL, FRP = DOCS["pre_registration"], DOCS["final_run_protocol"]
assert FRP["input_files_sha256"] == INPUT_HASHES, "final_run_protocol input_files_sha256 != the input CSVs - aborting."
assert FRP["parent_hashes"]["pre_registration"] == {"self_sha256": DOC_SELF["pre_registration"],
                                                    "file_sha256": DOC_FILE["pre_registration"]}
assert FRP["earlier_manifests_sha256"]["v0.3_manifest.json"] == EXPECTED_V03_MANIFEST_SHA256
# >>> verbatim from v0.9 (reference implementation exec)
REFERENCE_CODE = FRP["reference_implementation"]["code"]
REFERENCE_CODE_SHA256 = hashlib.sha256(REFERENCE_CODE.encode("utf-8")).hexdigest()
assert REFERENCE_CODE_SHA256 == FRP["reference_implementation"]["sha256_of_utf8_code"] == EXPECTED_REFERENCE_CODE_SHA256, \
    "reference implementation sha256 mismatch - aborting."
REF = {}
exec(REFERENCE_CODE, REF)
# <<< verbatim from v0.9
print("reference implementation: sha256 {} -> PASS (exec'd into REF)".format(REFERENCE_CODE_SHA256))

V021_MAN_SHA = sha256_of(V021_MANIFEST)
if V021_MAN_SHA != EXPECTED_V021_MANIFEST_SHA256:
    warn("v0.2.1_manifest.json sha256 {} != {} (the hash printed by v0.2.1); recorded, not a STOP".format(
        V021_MAN_SHA, EXPECTED_V021_MANIFEST_SHA256))
with open(V021_MANIFEST, "r", encoding="utf-8") as f:
    V021_MAN = json.load(f)
V021_DT_SHA = sha256_of(V021_DECISION_TABLE_CSV)
print("v0.2.1_decision_table.csv  sha256 {} (core chat copy {})".format(V021_DT_SHA, EXPECTED_V021_DECISION_TABLE_SHA256))
assert V021_DT_SHA == EXPECTED_V021_DECISION_TABLE_SHA256 == \
    manifest_entry(V021_MAN, os.path.basename(V021_DECISION_TABLE_CSV))["sha256"], \
    "STOP: v0.2.1_decision_table.csv differs from the core chat's copy / the v0.2.1 manifest - aborting."
V021_MF_SHA = sha256_of(V021_METRICS_FULL_CSV)
assert V021_MF_SHA == manifest_entry(V021_MAN, os.path.basename(V021_METRICS_FULL_CSV))["sha256"], \
    "STOP: v0.2.1_metrics_full.csv differs from the v0.2.1 manifest - aborting."
print("v0.2.1_metrics_full.csv    sha256 {} == v0.2.1 manifest -> PASS".format(V021_MF_SHA))
V03_MAN_SHA = sha256_of(V03_MANIFEST)
assert V03_MAN_SHA == EXPECTED_V03_MANIFEST_SHA256, \
    "STOP: v0.3_manifest.json sha256 {} != {} - aborting.".format(V03_MAN_SHA, EXPECTED_V03_MANIFEST_SHA256)
with open(V03_MANIFEST, "r", encoding="utf-8") as f:
    V03_MAN = json.load(f)
V03_DEL_SHA = sha256_of(V03_BOOT_DELTAS_CSV)
assert V03_DEL_SHA == manifest_entry(V03_MAN, os.path.basename(V03_BOOT_DELTAS_CSV))["sha256"], \
    "STOP: v0.3_bootstrap_deltas.csv differs from the v0.3 manifest - aborting."
assert manifest_entry(V03_MAN, os.path.basename(SEAL_PATH))["sha256"] == DOC_FILE["pre_registration"]
print("v0.3_manifest.json         sha256 {} -> PASS | v0.3_bootstrap_deltas.csv {} == manifest -> PASS".format(
    V03_MAN_SHA, V03_DEL_SHA))
V09_LABELS_SHA = sha256_of(V09_LABELS_CSV)
assert V09_LABELS_SHA == EXPECTED_V09_LABELS_SHA256, \
    "STOP: v0.9_fold10_labels.csv sha256 {} != {} - aborting.".format(V09_LABELS_SHA, EXPECTED_V09_LABELS_SHA256)
print("v0.9_fold10_labels.csv     sha256 {} -> PASS (not parsed before Part A has passed)".format(V09_LABELS_SHA))

# locked values read from the documents (asserted equal to the v0.2.1 / v0.3 constants)
PREREG_SESOI_H2 = float(SEAL["SESOI_H2"])
PREREG_Z_ALPHA_2 = float(SEAL["MDE"]["z_alpha_2"])
PREREG_Z_POWER = float(SEAL["MDE"]["z_power"])
assert (PREREG_SESOI_H2, PREREG_Z_ALPHA_2, PREREG_Z_POWER) == (SESOI_H2, Z_ALPHA_2, Z_POWER), \
    "pre_registration.json SESOI_H2 / z values differ from v0.3 - aborting."
assert (SEAL["MDE"]["alpha"], SEAL["MDE"]["power"], SEAL["MDE"]["N_BOOT"], SEAL["MDE"]["BOOT_SEED"],
        SEAL["MDE"]["model_seed"]) == (ALPHA, POWER, N_BOOT, BOOT_SEED, SEED)
assert SEAL["primary_learner"] == PRIMARY_LEARNER and SEAL["robustness_learner"] == ROBUSTNESS_LEARNER
BOOT_SPEC = FRP["bootstrap"]
assert (int(BOOT_SPEC["N_BOOT"]), int(BOOT_SPEC["BOOT_SEED"]), int(BOOT_SPEC["n_patients"])) == \
    (N_BOOT, BOOT_SEED, EXPECTED_N_FOLD10_PATIENTS), "FRP bootstrap spec differs - aborting."
LOCKED_HASHES_FRP = BOOT_SPEC["locked_hashes"]
LQ = FRP["locked_quantities"]
LQ_ENC = LQ["categorical_encoder"]
assert LQ["fit_rows"].startswith("folds 1-9") and LQ["fold10"]["n_records"] == EXPECTED_N_FOLD10_RECORDS
_disclosed = "(" + ", ".join("{} {}".format(c, EXPECTED_FOLD10_POSITIVES[c]) for c in SUPERCLASSES) + \
             " of {})".format(EXPECTED_N_FOLD10_RECORDS)
assert _disclosed in FRP["disclosures_fold10_exposure"]["fold10_labels"][0], \
    "the expected fold-10 label counts {} are not the ones disclosed in FRP - aborting.".format(_disclosed)
POST_FOLD10_RULE_3 = FRP["post_fold10_rules"][2]
assert POST_FOLD10_RULE_3.startswith("3. ") and "POST-HOC" in POST_FOLD10_RULE_3
print("pre_registration.json: SESOI_H2 {} | MDE z {} + {} | N_BOOT {} BOOT_SEED {} | final_run_protocol: bootstrap "
      "N_BOOT {} BOOT_SEED {} n_patients {} -> PASS".format(PREREG_SESOI_H2, PREREG_Z_ALPHA_2, PREREG_Z_POWER,
                                                            SEAL["MDE"]["N_BOOT"], SEAL["MDE"]["BOOT_SEED"], N_BOOT,
                                                            BOOT_SEED, BOOT_SPEC["n_patients"]))
print("post_fold10_rules 3: " + POST_FOLD10_RULE_3)
INPUT_FILES_SHA256 = {
    "ptbxl_database.csv": INPUT_HASHES["ptbxl_database.csv"], "scp_statements.csv": INPUT_HASHES["scp_statements.csv"],
    "pre_registration.json": {"file_sha256": DOC_FILE["pre_registration"], "self_sha256": DOC_SELF["pre_registration"]},
    "final_run_protocol.json": {"file_sha256": DOC_FILE["final_run_protocol"],
                                "self_sha256": DOC_SELF["final_run_protocol"]},
    "reference_implementation_code_utf8": REFERENCE_CODE_SHA256,
    "v0.2.1_decision_table.csv": V021_DT_SHA, "v0.2.1_metrics_full.csv": V021_MF_SHA,
    "v0.2.1_manifest.json": V021_MAN_SHA, "v0.3_bootstrap_deltas.csv": V03_DEL_SHA, "v0.3_manifest.json": V03_MAN_SHA,
    "v0.9_fold10_labels.csv": V09_LABELS_SHA,
}

# --- 3. snapshots (re-checked before the results and after the manifest) ---
for _label, _d in SNAPSHOT_DIRS.items():
    assert os.path.isdir(_d), "folder not found: " + _d
SNAPSHOT_START = snapshot_all()
for _label, _snap in SNAPSHOT_START.items():
    print("snapshot {:18s}: {} files".format(_label, len(_snap)))

# --- 4. write-once output folder; earlier attempts ---
assert not os.path.exists(OUT_DIR), (
    "STOP: {} already exists. ph1 is write-once and never overwrites. If an earlier run stopped or crashed, RENAME "
    "that folder to outputs\\ph1_failed_<n> (never delete it) and run this cell again.".format(OUT_DIR))
PREVIOUS_ATTEMPTS = []
for _name in sorted(os.listdir(OUTPUTS_DIR)):
    _p = os.path.join(OUTPUTS_DIR, _name)
    if _name.startswith(FAILED_ATTEMPT_PREFIX) and os.path.isdir(_p):
        PREVIOUS_ATTEMPTS.append({"folder": _name, "files": [{"file": n, "sha256": h} for n, h in tree_snapshot(_p)]})
print("earlier attempts (outputs\\{}*): {}".format(FAILED_ATTEMPT_PREFIX, [a["folder"] for a in PREVIOUS_ATTEMPTS] or "none"))
# r2: the r1 attempt named in AMENDMENT_R2 - its ph1_partA_failure.json must be the file the core chat read (self_sha256;
# STOP otherwise). A different file sha256 with the same self_sha256 (same content, e.g. other line endings) is a WARN.
assert os.path.isfile(R1_FAILURE_PATH), (
    "STOP: {} not found. AMENDMENT_R2 names outputs\\{} as the r1 attempt: rename the r1 output folder (it holds "
    "ph1_analysis_plan.json and ph1_partA_failure.json) to that name and run this cell again.".format(
        R1_FAILURE_PATH, R1_ATTEMPT_FOLDER))
_s, _r, R1_FAILURE_DOC = verify_self_sha256(R1_FAILURE_PATH)
assert _s == _r == EXPECTED_R1_FAILURE_SELF_SHA256, \
    "STOP: r1 attempt {}: self_sha256 stored {} / recomputed {} / expected {} - aborting.".format(
        R1_FAILURE_PATH, _s, _r, EXPECTED_R1_FAILURE_SELF_SHA256)
R1_FAILURE_FILE_SHA = sha256_of(R1_FAILURE_PATH)
if R1_FAILURE_FILE_SHA != EXPECTED_R1_FAILURE_FILE_SHA256:
    warn("outputs\\{}\\{} file sha256 {} != {} (the copy the core chat read); its self_sha256 matches, so the content is "
         "the same; recorded, not a STOP".format(R1_ATTEMPT_FOLDER, os.path.basename(R1_FAILURE_PATH), R1_FAILURE_FILE_SHA,
                                                 EXPECTED_R1_FAILURE_FILE_SHA256))
R1_ATTEMPT = {
    "folder": "outputs\\" + R1_ATTEMPT_FOLDER, "code_revision": R1_FAILURE_DOC["code_revision"],
    "timestamp": R1_FAILURE_DOC["timestamp"], "outcome": R1_FAILURE_DOC["exception"],
    "failed_checks": R1_FAILURE_DOC["failed_checks"], "plan_self_sha256": R1_FAILURE_DOC["plan_self_sha256"],
    "partA_failure_file": os.path.basename(R1_FAILURE_PATH),
    "partA_failure_self_sha256": _s, "partA_failure_file_sha256": R1_FAILURE_FILE_SHA,
    "expected_self_sha256": EXPECTED_R1_FAILURE_SELF_SHA256, "expected_file_sha256": EXPECTED_R1_FAILURE_FILE_SHA256,
    "file_sha256_equals_expected": R1_FAILURE_FILE_SHA == EXPECTED_R1_FAILURE_FILE_SHA256,
    "files": [a["files"] for a in PREVIOUS_ATTEMPTS if a["folder"] == R1_ATTEMPT_FOLDER][0],
}
print("r1 attempt outputs\\{}: {} self_sha256 {} -> PASS | file sha256 {} ({}) | revision {} | failed checks {}".format(
    R1_ATTEMPT_FOLDER, R1_ATTEMPT["partA_failure_file"], _s, R1_FAILURE_FILE_SHA,
    "== expected" if R1_ATTEMPT["file_sha256_equals_expected"] else "!= expected: WARN", R1_ATTEMPT["code_revision"],
    R1_ATTEMPT["failed_checks"]))
os.makedirs(OUT_DIR)
print("created {}".format(OUT_DIR))

# --- 5. analysis plan (exclusive create, self_sha256) - before any model is fit ---
PLAN_DOC = {
    "document": "ph1 analysis plan - written before any model is fit",
    "label": POSTHOC_LABEL,
    "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
    "timestamp": datetime.now().astimezone().isoformat(),
    "versions": VERSIONS,
    "plan": PH1_PLAN,
    "plan_utf8_sha256": hashlib.sha256(PH1_PLAN.encode("utf-8")).hexdigest(),
    "amendment_r2": AMENDMENT_R2,
    "input_files_sha256": INPUT_FILES_SHA256,
    "self_sha256_procedure": SELF_SHA256_PROCEDURE,
}
PLAN_SELF_SHA = write_once_json_atomic(PLAN_PATH, PLAN_DOC)
PLAN_FILE_SHA = sha256_of(PLAN_PATH)
_s, _r, _plan_disk = verify_self_sha256(PLAN_PATH)
assert _s == _r == PLAN_SELF_SHA and _plan_disk["plan"] == PH1_PLAN and _plan_disk["amendment_r2"] == AMENDMENT_R2
print("[saved] {} (self_sha256 {})".format(PLAN_PATH, PLAN_SELF_SHA))

# --- the two shared functions read no module-level state other than constants, helpers and REF ---
for _fn in (fit_predict, metrics_from_probs):
    _extra = sorted(global_names(_fn) - ALLOWED_FUNCTION_GLOBALS)
    assert not _extra, "{} reads module-level names outside the allowed set: {}".format(_fn.__name__, _extra)
print("fit_predict / metrics_from_probs read only constants, helper functions and REF (bytecode check): PASS")

# --- data: one full read (as v0.2.1); fold-10 rows kept apart and untouched until Part A has passed ---
scp = pd.read_csv(SCP_CSV, index_col=0)
df_all = pd.read_csv(DB_CSV, index_col="ecg_id")
print("full dataset shape:", df_all.shape)
assert set(df_all["strat_fold"].unique()) == set(range(1, 11)) and df_all.index.is_unique
# >>> verbatim from v0.2.1 (working set folds 1-9)
df9 = df_all[df_all["strat_fold"] <= 9].copy()
# <<< verbatim from v0.2.1
df10 = df_all[df_all["strat_fold"] == 10].copy()
assert len(df9) == EXPECTED_N_RECORDS_FOLDS1TO9 and len(df10) == EXPECTED_N_FOLD10_RECORDS and \
    len(df9) + len(df10) == len(df_all), "fold sizes differ - aborting."
del df_all
print("folds 1-9: {} records (Part A) | fold 10: {} records in a separate frame, not used before Part A has passed".format(
    len(df9), len(df10)))


# =====================================================================================
# PART A - VALIDATION GATE (train folds 1-8, evaluate fold 9; the same two functions as Part B)
# =====================================================================================
hdr("PART A - VALIDATION GATE: folds 1-8 -> fold 9 must reproduce v0.2.1 and v0.3 (A2: per-cell tolerance of r2; A3: {:g})"
    .format(PART_A_TOL))
try:
    BOOT_IDX9 = REF["build_boot_idx"](EXPECTED_SPLITS["A"]["n_eval_patients"], N_BOOT, BOOT_SEED)
    A_FIT = fit_predict(df9, scp, 8, 9, gate_part_a)                       # A1 runs inside, before any fit
    A_MET = metrics_from_probs(A_FIT["probs"], A_FIT["labels_eval"], BOOT_IDX9, PREREG_SESOI_H2, PREREG_Z_ALPHA_2,
                               PREREG_Z_POWER)

    hdr("PART A - A2: decision table and metric rows vs the v0.2.1 files")
    print("A2 tolerance (r2, amendment_r2): rows {} -> {}; every other cell {:g}; identifier columns exact".format(
        A2_TOLERANCE["relaxed_rows"], A2_TOLERANCE["relaxed_columns"], A2_TOLERANCE["default"]))
    V021_DT = pd.read_csv(V021_DECISION_TABLE_CSV, encoding="utf-8-sig", float_precision="round_trip")
    V021_MF = pd.read_csv(V021_METRICS_FULL_CSV, encoding="utf-8-sig", float_precision="round_trip")
    A_DT = A_MET["decision_table"].reset_index()
    for _key, _cid, _desc, _got, _exp, _ids, _nums in (
            ("decision_table", "A2_decision_table", "12 rows x 5 numeric columns == v0.2.1_decision_table.csv",
             A_DT, V021_DT, DT_ID_COLUMNS, DT_NUMERIC_COLUMNS),
            ("metrics_full", "A2_metric_rows", "all 216 metric rows == v0.2.1_metrics_full.csv",
             A_MET["metric_rows"], V021_MF, MF_ID_COLUMNS, MF_NUMERIC_COLUMNS)):
        _rec, PART_A_DIAG[_key] = compare_frames("A2", _cid, _desc, _got, _exp, _ids, _nums, A2_TOLERANCE)
        for _c, _v in _rec["max_abs_diff_per_column"].items():
            print("      max |diff| {:30s} {} | by tolerance {}".format(
                _c, _v, PART_A_DIAG[_key]["max_abs_diff_per_column_by_tolerance"][_c]))
        for _cell in (PART_A_DIAG[_key].get("cells_over_tolerance") or [])[:24]:
            print("      over tolerance: {}".format(_cell))
    print("      any AUROC changed vs v0.2.1: {} ({} rows)".format(PART_A_DIAG["metrics_full"].get("any_auroc_changed"),
                                                               PART_A_DIAG["metrics_full"].get("n_auroc_changed")))

    hdr("PART A - A2b (r2; reported, never stops the run): 4-decimal AUROC of the LogisticRegression rows that are not exact")
    PART_A_DIAG["a2b"] = {
        "decision_table": lr_rows_auroc_4dp(A_DT, V021_DT, DT_ID_COLUMNS, DT_NUMERIC_COLUMNS, "macro_auroc_mean"),
        "metric_rows": lr_rows_auroc_4dp(A_MET["metric_rows"], V021_MF, MF_ID_COLUMNS, MF_NUMERIC_COLUMNS, "auroc")}
    _a2b_obs = {k: ("not evaluated: the rows differ from v0.2.1 (A2 fails)" if v is None else
                    {"n_LR_rows_not_exact": len(v), "rows_4dp_changed": [r for r in v if not r["unchanged_4dp"]]})
                for k, v in PART_A_DIAG["a2b"].items()}
    for _k, _v in PART_A_DIAG["a2b"].items():
        for _row in _v or []:
            print("      {:14s} {} -> 4-decimal AUROC v0.2.1 {} | ph1 {} | {}".format(
                _k, {c: _row[c] for c in (DT_ID_COLUMNS if _k == "decision_table" else MF_ID_COLUMNS)}, _row["v021_4dp"],
                _row["ph1_4dp"], "unchanged" if _row["unchanged_4dp"] else "CHANGED"))
    A2B_REC = check("A2b", "A2b_LR_auroc_4dp_unchanged", "every LogisticRegression row that is not exact keeps its "
                    "4-decimal AUROC (metric rows: auroc; decision table: macro_auroc_mean); reported, does not stop the run",
                    all(v is not None and all(r["unchanged_4dp"] for r in v) for v in PART_A_DIAG["a2b"].values()),
                    gating=False, kind="exact", observed=_a2b_obs)

    hdr("PART A - A3: H2 vs pre_registration.json and v0.3_bootstrap_deltas.csv")
    # A3 uses the seed-42 configurations of the A2 fits (metrics_from_probs) instead of v0.3's refit of B1 / B3 (v0.3
    # cell lines 368-415) and v0.3's BOOT_ROWS rebuild (v0.3 cell lines 443-456). That equivalence is TESTED here: every
    # field of pre_registration.json H2.results_per_learner and every per-draw delta of v0.3_bootstrap_deltas.csv.
    _exp_rpl = SEAL["H2"]["results_per_learner"]
    PART_A_DIAG["h2_fields"] = []
    for learner in LEARNERS:
        _got = A_MET["results_per_learner"][learner]
        check("A3", "A3_fields_{}".format(learner), "{}: field set == pre_registration.json fields + n_patients".format(
            learner), sorted(_got) == sorted(list(_exp_rpl[learner]) + ["n_patients"]), stop=False, kind="exact",
            expected=sorted(_exp_rpl[learner]), observed=sorted(_got))
        for _key, _exp_v in _exp_rpl[learner].items():
            _ok, _d = compare_value(_got.get(_key), _exp_v, PART_A_TOL)
            PART_A_DIAG["h2_fields"].append({"learner": learner, "field": _key, "expected": _exp_v,
                                             "observed": _got.get(_key), "abs_diff": _d, "pass": bool(_ok)})
            check("A3", "A3_{}_{}".format(learner, _key), "{} {} == pre_registration.json".format(learner, _key), _ok,
                  stop=False, kind="exact" if _d is None else "numeric", expected=_exp_v, observed=_got.get(_key),
                  max_abs_diff=_d, tolerance=None if _d is None else PART_A_TOL)
        check("A3", "A3_n_patients_{}".format(learner), "{} n_patients == 1942".format(learner),
              _got.get("n_patients") == EXPECTED_SPLITS["A"]["n_eval_patients"], stop=False, kind="exact",
              expected=EXPECTED_SPLITS["A"]["n_eval_patients"], observed=_got.get("n_patients"))
    check("A3", "A3_learner_disagreement", "learner disagreement list == pre_registration.json (exact strings)",
          A_MET["learner_disagreement"] == SEAL["H2"]["learner_disagreement"], stop=False, kind="exact",
          expected=SEAL["H2"]["learner_disagreement"], observed=A_MET["learner_disagreement"])
    V03_DEL = pd.read_csv(V03_BOOT_DELTAS_CSV, encoding="utf-8-sig", float_precision="round_trip")
    _rec, PART_A_DIAG["bootstrap_deltas"] = compare_frames(
        "A3", "A3_per_draw_deltas", "2000 per-draw deltas (LR, HGB) == v0.3_bootstrap_deltas.csv",
        A_MET["boot_df"], V03_DEL, ["replicate"], ["delta_LR", "delta_HGB"], PART_A_TOL)
    PART_A_DIAG["bootstrap_deltas"].pop("max_abs_diff_per_row", None)
    check("A3", "A3_v021_index_matrix_eq_REF", "v0.2.1 marginal-bootstrap index matrix on fold 9 == "
          "REF.build_boot_idx(1942, 2000, 7)", np.array_equal(A_MET["boot_samp"], BOOT_IDX9), stop=False, kind="exact")

    A_FAILED = [r["id"] for r in CHECKS if r["section"] in ("A1", "A2", "A3") and r["status"] != "PASS"]
    if A_FAILED:
        raise PartAFailure("Part A validation gate FAILED ({} check(s): {}) - nothing is computed on fold 10.".format(
            len(A_FAILED), A_FAILED))
except Exception as _e:
    PART_A_FAILURE_DOC = {
        "document": "ph1 Part A failure - the validation gate did not pass; no fold-10 model was fit and no fold-10 "
                    "metric was computed",
        "label": POSTHOC_LABEL, "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
        "timestamp": datetime.now().astimezone().isoformat(), "plan_self_sha256": PLAN_SELF_SHA,
        "tolerance": {"A2": A2_TOLERANCE, "A3": PART_A_TOL}, "exception": "{}: {}".format(type(_e).__name__, _e),
        "failed_checks": [r["id"] for r in CHECKS if r["status"] == "FAIL"],
        "checks": CHECKS, "diagnostics": PART_A_DIAG, "versions": VERSIONS, "cpu_count": CPU_COUNT,
        "threadpool_info": THREADPOOL_INFO, "warnings": WARNINGS,
    }
    _sha = write_once_json_atomic(PARTA_FAILURE_PATH, json_safe(plain(PART_A_FAILURE_DOC)))
    print("\n[saved] {} (self_sha256 {})".format(PARTA_FAILURE_PATH, _sha))
    print("PART A FAILED -> STOP. Report ph1_partA_failure.json to the core chat; for any re-run RENAME outputs\\ph1 to "
          "outputs\\ph1_failed_<n> first.")
    raise

PART_A_SUMMARY = {
    "tolerance": {"A2": A2_TOLERANCE, "A3": PART_A_TOL}, "all_pass": True,
    "decision_table_max_abs_diff_per_column": PART_A_DIAG["decision_table"]["max_abs_diff_per_column"],
    "decision_table_max_abs_diff_by_tolerance": PART_A_DIAG["decision_table"]["max_abs_diff_by_tolerance"],
    "metric_rows_max_abs_diff_per_column": PART_A_DIAG["metrics_full"]["max_abs_diff_per_column"],
    "metric_rows_max_abs_diff_by_tolerance": PART_A_DIAG["metrics_full"]["max_abs_diff_by_tolerance"],
    "metric_rows_any_auroc_changed": PART_A_DIAG["metrics_full"]["any_auroc_changed"],
    "metric_rows_n_auroc_changed": PART_A_DIAG["metrics_full"]["n_auroc_changed"],
    "h2_fields_max_abs_diff": max([r["abs_diff"] for r in PART_A_DIAG["h2_fields"] if r["abs_diff"] is not None] or [0.0]),
    "per_draw_deltas_max_abs_diff_per_column": PART_A_DIAG["bootstrap_deltas"]["max_abs_diff_per_column"],
    "A2b": {"status": A2B_REC["status"], "gating": False, "rows": PART_A_DIAG["a2b"]},
}
print("\nPART A PASSED: v0.2.1 decision table + 216 metric rows (A2, r2 per-cell tolerance) and v0.3 H2 (fields + per-draw "
      "deltas, A3, {:g}) reproduced; A2b {} -> fold 10 may be opened.".format(PART_A_TOL, A2B_REC["status"]))


# =====================================================================================
# PART B - POST-HOC EVALUATION ON FOLD 10 (train folds 1-9)
# =====================================================================================
hdr("PART B - B0 CHECKS (before any fold-10 model is fit)")
assert sha256_of(V09_LABELS_CSV) == V09_LABELS_SHA, "v0.9_fold10_labels.csv changed since STEP 0 - aborting."
V09_LABELS = pd.read_csv(V09_LABELS_CSV, encoding="utf-8-sig")
# >>> adapted from v0.9 (fold-10 ids, patients) | renames: fold10_meta->df10
fold10_pid = df10["patient_id"].to_numpy()
assert np.all(fold10_pid == np.round(fold10_pid)), "non-integral fold-10 patient_id"
fold10_pid = fold10_pid.astype(np.int64)
fold10_ecg = df10.index.to_numpy().astype(np.int64)
N_F10 = len(df10)
N_F10_PATIENTS = int(len(np.unique(fold10_pid)))
F10_DISJOINT = set(fold10_pid.tolist()).isdisjoint(set(df9["patient_id"].astype(np.int64).tolist()))
assert N_F10 == EXPECTED_N_FOLD10_RECORDS, "fold-10 records {} != 2198 - aborting.".format(N_F10)
assert N_F10_PATIENTS == EXPECTED_N_FOLD10_PATIENTS, "fold-10 patients {} != 1904 - aborting.".format(N_F10_PATIENTS)
assert F10_DISJOINT, "a fold-10 patient appears in folds 1-9 - aborting."
# <<< adapted from v0.9
# >>> verbatim from v0.9 (fold-10 patient grouping and BOOT_IDX with REF)
F10_PATIENTS, F10_ROWS = REF["patient_rows_in_file_order"](fold10_pid)
assert len(F10_PATIENTS) == N_F10_PATIENTS and sum(len(r) for r in F10_ROWS) == N_F10
BOOT_IDX = REF["build_boot_idx"](N_F10_PATIENTS, N_BOOT, BOOT_SEED)
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (fold-10 id array)
F10_IDS = np.ascontiguousarray(np.stack([fold10_ecg, fold10_pid], axis=1).astype("<i8"))
# <<< verbatim from v0.9
LOCK_HASHES = {
    "fold10_ids_file_order_sha256": sha256_bytes(F10_IDS.tobytes(order="C")),
    "fold10_patients_sorted_sha256": sha256_bytes(np.ascontiguousarray(F10_PATIENTS.astype("<i8")).tobytes()),
    "boot_idx_sha256": sha256_bytes(np.ascontiguousarray(BOOT_IDX.astype("<i8")).tobytes()),
}
for _k, _v in LOCK_HASHES.items():
    check("B0", "B0_" + _k, "{} == FRP bootstrap.locked_hashes (FRP serialization)".format(_k),
          _v == LOCKED_HASHES_FRP[_k], kind="exact", expected=LOCKED_HASHES_FRP[_k], observed=_v)
check("B0", "B0_boot_idx_shape", "BOOT_IDX = REF.build_boot_idx(1904, 2000, 7)",
      BOOT_IDX.shape == (N_BOOT, EXPECTED_N_FOLD10_PATIENTS), kind="exact", observed=list(BOOT_IDX.shape))
check("B0", "B0_fold10_patients_disjoint", "fold-10 patients disjoint from folds 1-9", bool(F10_DISJOINT), kind="exact")

hdr("PART B - B1: FIT ON FOLDS 1-9, PREDICT FOLD 10 (B0 label / encoder checks run inside, before the first fit)")
A_FITTED_IDS = {id(o) for o in A_FIT["fitted"]}          # A_FIT stays referenced, so these ids stay unique
df_work_B = pd.concat([df9, df10])                        # folds 1-9 (file order) followed by fold 10 (file order)
B_FIT = fit_predict(df_work_B, scp, 9, 10, gate_part_b)
check("B1", "B1_no_partA_object_reused", "no scaler / model fitted in Part A is reused in Part B",
      not any(id(o) in A_FITTED_IDS for o in B_FIT["fitted"]) and len(B_FIT["fitted"]) == len(A_FIT["fitted"])
      == EXPECTED_N_FITTED, kind="exact", expected=EXPECTED_N_FITTED, observed=len(B_FIT["fitted"]))
_sc_rows = {b: int(s.n_samples_seen_) for b, s in B_FIT["scalers"].items()}
check("B1", "B1_scalers_fit_on_training_rows", "every LR scaler was fit on the 19,601 training rows",
      sorted(_sc_rows) == ["B1_physio", "B3_physio_plus_acquisition", "B4_temporal", "B5_all"] and
      set(_sc_rows.values()) == {EXPECTED_SPLITS["B"]["n_train"]}, kind="exact", observed=_sc_rows)
_age = B_FIT["scalers"]["B1_physio"]
_age_diff = {"mean": abs(float(_age.mean_[0]) - LQ["age_scaler"]["mean"]),
             "scale": abs(float(_age.scale_[0]) - LQ["age_scaler"]["scale"]),
             "var": abs(float(_age.var_[0]) - LQ["age_scaler"]["var"])}
check("B1", "B1_age_scaler_eq_FRP", "B1 age_imputed scaler (folds 1-9) == FRP locked age_scaler (tol {})".format(
      AGE_SCALER_TOL), all(_age_diff[k] <= AGE_SCALER_TOL[k] for k in AGE_SCALER_TOL) and
      int(_age.n_samples_seen_) == LQ["age_scaler"]["n_samples_seen"], kind="numeric", observed=_age_diff,
      max_abs_diff=max(_age_diff.values()), tolerance=AGE_SCALER_TOL)
del A_FIT, A_FITTED_IDS

PROBS_BYTES = csv_bytes(B_FIT["probs"])
PROBS_SHA = write_once_bytes(PROBS_PATH, PROBS_BYTES)
print("[saved] {} ({} rows, {} bytes, sha256 {})".format(PROBS_PATH, len(B_FIT["probs"]), len(PROBS_BYTES), PROBS_SHA))
FROZEN_DOC = {
    "document": "ph1 frozen fold-10 probabilities - written BEFORE any fold-10 metric was computed",
    "label": POSTHOC_LABEL, "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
    "timestamp": datetime.now().astimezone().isoformat(), "plan_self_sha256": PLAN_SELF_SHA,
    "statement": "no fold-10 metric has been computed when this file is written",
    "outputs": {os.path.basename(PROBS_PATH): {"sha256": PROBS_SHA, "bytes": len(PROBS_BYTES),
                                                "n_rows": int(len(B_FIT["probs"])), "columns": PROBS_CSV_COLUMNS}},
    "row_order": "block (B0_intercept .. B5_all), learner (LogisticRegression, HistGradientBoosting), seed (42, 123, "
                 "2024), fold-10 records in ptbxl_database.csv file order",
    "n_configurations": len(CONFIGS), "training_rows": "folds 1-9 ({} records)".format(EXPECTED_SPLITS["B"]["n_train"]),
    "fold10_lock_hashes": LOCK_HASHES,
}
FROZEN_SELF_SHA = write_once_json_atomic(FROZEN_PATH, FROZEN_DOC)
FROZEN_FILE_SHA = sha256_of(FROZEN_PATH)
print("[saved] {} (self_sha256 {})".format(FROZEN_PATH, FROZEN_SELF_SHA))

# from here on every number comes from the probability file read back from disk
assert sha256_of(PROBS_PATH) == PROBS_SHA, "ph1_fold10_probs.csv changed after it was written - aborting."
PROBS_DISK = pd.read_csv(PROBS_PATH, encoding="utf-8-sig", float_precision="round_trip")
_eq, _where = bitwise_frame_equal(PROBS_DISK, B_FIT["probs"])
check("B1", "B1_probs_roundtrip", "ph1_fold10_probs.csv read back (float_precision='round_trip') == memory, bit for "
      "bit (columns, dtypes, values)", _eq, kind="exact", observed=_where)
B_LABELS_EVAL = B_FIT["labels_eval"]
del B_FIT

PREVIOUS_PROBS_COMPARISON = []
for _a in PREVIOUS_ATTEMPTS:
    _pp = os.path.join(OUTPUTS_DIR, _a["folder"], os.path.basename(PROBS_PATH))
    if os.path.isfile(_pp):
        with open(_pp, "rb") as f:
            _same = f.read() == PROBS_BYTES
        PREVIOUS_PROBS_COMPARISON.append({"folder": _a["folder"], "identical_bytes": bool(_same)})
        if not _same:
            warn("{}\\{} differs from this run's probability file (byte comparison)".format(
                _a["folder"], os.path.basename(PROBS_PATH)))
    else:
        PREVIOUS_PROBS_COMPARISON.append({"folder": _a["folder"], "identical_bytes": None,
                                          "note": "no probability file in that attempt"})
print("earlier attempts' probability files: {}".format(PREVIOUS_PROBS_COMPARISON or "none"))

hdr("PART B - B2 / B3 / B4: METRICS FROM THE PROBABILITY FILE (POST HOC)")
B_MET = metrics_from_probs(PROBS_DISK, B_LABELS_EVAL, BOOT_IDX, PREREG_SESOI_H2, PREREG_Z_ALPHA_2, PREREG_Z_POWER)
check("B2", "B2_v021_index_matrix_eq_BOOT_IDX", "v0.2.1 marginal-bootstrap index matrix on fold 10 == BOOT_IDX",
      np.array_equal(B_MET["boot_samp"], BOOT_IDX), kind="exact")
check("B2", "B2_patient_rows_eq_REF", "v0.2.1 patient grouping on fold 10 == REF.patient_rows_in_file_order",
      np.array_equal(B_MET["eval_patient_ids"], F10_IDS[:, 1]) and len(B_MET["v021_patient_row_list"]) == len(F10_ROWS)
      and all(np.array_equal(a, b) for a, b in zip(B_MET["v021_patient_row_list"], F10_ROWS)), kind="exact")
check("B3", "B3_n_patients", "H2-style test on 1904 fold-10 patients", all(
      B_MET["results_per_learner"][l]["n_patients"] == EXPECTED_N_FOLD10_PATIENTS for l in LEARNERS), kind="exact")

B_DT = B_MET["decision_table"].reset_index()
assert list(B_DT.columns) == DT_ID_COLUMNS + DT_NUMERIC_COLUMNS and len(B_DT) == 12
DT_SHA = write_once_bytes(DECISION_TABLE_PATH, csv_bytes(B_DT))
print("[saved] {} (sha256 {})".format(DECISION_TABLE_PATH, DT_SHA))
assert list(B_MET["boot_df"].columns) == ["replicate", "delta_LR", "delta_HGB"] and len(B_MET["boot_df"]) == N_BOOT
DELTAS_SHA = write_once_bytes(BOOT_DELTAS_PATH, csv_bytes(B_MET["boot_df"]))
print("[saved] {} (sha256 {})".format(BOOT_DELTAS_PATH, DELTAS_SHA))
assert list(B_MET["metric_rows"].columns) == MF_ID_COLUMNS + MF_NUMERIC_COLUMNS and len(B_MET["metric_rows"]) == 216
METRIC_ROWS_SHA = write_once_bytes(METRIC_ROWS_PATH, csv_bytes(B_MET["metric_rows"]))
print("[saved] {} (sha256 {})".format(METRIC_ROWS_PATH, METRIC_ROWS_SHA))


# =====================================================================================
# OUTPUT - results (written last of the results), manifest, final checks
# =====================================================================================
hdr("OUTPUT - RESULTS AND MANIFEST")
_diff = snapshot_diff(SNAPSHOT_START, snapshot_all())
check("OUT", "OUT_snapshots_before_results", "preregistration\\, outputs\\v0.2.1, v0.3, v0.9 and the input CSVs are "
      "unchanged since STEP 0", not _diff, kind="exact", observed=_diff)
OUTPUT_FILES_BEFORE_RESULTS = {os.path.basename(p): sha256_of(p) for p in (
    PLAN_PATH, PROBS_PATH, FROZEN_PATH, DECISION_TABLE_PATH, BOOT_DELTAS_PATH, METRIC_ROWS_PATH)}
assert OUTPUT_FILES_BEFORE_RESULTS[os.path.basename(PROBS_PATH)] == PROBS_SHA
B_RPL = B_MET["results_per_learner"]
RESULTS_DOC = {
    "document": "ph1 results - post-hoc metadata-only evaluation on fold 10 (train folds 1-9)",
    "label": POSTHOC_LABEL,
    "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
    "run_started": RUN_STARTED, "timestamp": datetime.now().astimezone().isoformat(),
    "versions": VERSIONS, "versions_note": VERSIONS_NOTE, "cpu_count": CPU_COUNT, "threadpool_info": THREADPOOL_INFO,
    "plan": {"file": os.path.basename(PLAN_PATH), "self_sha256": PLAN_SELF_SHA, "file_sha256": PLAN_FILE_SHA,
             "plan_utf8_sha256": EXPECTED_PH1_PLAN_SHA256, "amendment_r2_utf8_sha256": EXPECTED_AMENDMENT_R2_SHA256},
    "amendment_r2": AMENDMENT_R2,
    "r1_attempt": R1_ATTEMPT,
    "frozen_outputs": {"file": os.path.basename(FROZEN_PATH), "self_sha256": FROZEN_SELF_SHA,
                       "file_sha256": FROZEN_FILE_SHA},
    "input_files_sha256": INPUT_FILES_SHA256,
    "post_fold10_rules_3": POST_FOLD10_RULE_3,
    "previous_attempts": PREVIOUS_ATTEMPTS, "previous_attempts_probability_files": PREVIOUS_PROBS_COMPARISON,
    "part_A": PART_A_SUMMARY,
    "checks": CHECKS,
    "split": {"train": "folds 1-9", "eval": "fold 10", **EXPECTED_SPLITS["B"]},
    "fold10_positives": EXPECTED_FOLD10_POSITIVES,
    "fold10_lock_hashes": LOCK_HASHES,
    "decision_table": B_DT.to_dict(orient="records"),
    "H1_posthoc": {"label": POSTHOC_LABEL, "rule": "B3_physio_plus_acquisition macro_auroc_mean >= {:.2f} (v0.2.1)".format(
                       VERDICT_THRESHOLD), "verdict": B_MET["h1_verdict"]},
    "H2_posthoc": {
        "label": POSTHOC_LABEL,
        "comparison": "B3_physio_plus_acquisition vs B1_physio, delta = B3 - B1, seed 42 (as v0.3); train folds 1-9, "
                      "evaluate fold 10",
        "tests": SEAL["H2"]["tests"], "SESOI_H2": PREREG_SESOI_H2,
        "MDE_constants": {"z_alpha_2": PREREG_Z_ALPHA_2, "z_power": PREREG_Z_POWER, "formula": SEAL["MDE"]["formula"]},
        "bootstrap": "REF.paired_bootstrap on the locked fold-10 BOOT_IDX (N_BOOT {}, BOOT_SEED {}, {} patients)".format(
            N_BOOT, BOOT_SEED, EXPECTED_N_FOLD10_PATIENTS),
        "primary_learner": PRIMARY_LEARNER, "robustness_learner": ROBUSTNESS_LEARNER,
        "results_per_learner": B_RPL, "learner_disagreement": B_MET["learner_disagreement"],
        "learner_rule": "disagreement is reported, never resolved by selection",
        "ref_summaries": B_MET["ref_summaries"],
    },
    "v021_index_matrix_equals_BOOT_IDX": True,
    "outputs_before_results": OUTPUT_FILES_BEFORE_RESULTS,
    "warnings": WARNINGS,
    "self_sha256_procedure": SELF_SHA256_PROCEDURE,
}
RESULTS_DOC = plain(RESULTS_DOC)
assert not _find_nonfinite(RESULTS_DOC), "non-finite value in the results: {}".format(_find_nonfinite(RESULTS_DOC))
RESULTS_SELF_SHA = write_once_json_atomic(RESULTS_PATH, RESULTS_DOC)
print("[saved] {} (self_sha256 {})".format(RESULTS_PATH, RESULTS_SELF_SHA))

MANIFEST_DOC = {
    "document": "ph1 manifest", "label": POSTHOC_LABEL, "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
    "timestamp": datetime.now().astimezone().isoformat(), "output_dir": OUT_DIR,
    "outputs": [{"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256_of(p)} for p in (
        PLAN_PATH, PROBS_PATH, FROZEN_PATH, DECISION_TABLE_PATH, BOOT_DELTAS_PATH, METRIC_ROWS_PATH, RESULTS_PATH)],
    "plan_self_sha256": PLAN_SELF_SHA, "frozen_outputs_self_sha256": FROZEN_SELF_SHA,
    "results_self_sha256": RESULTS_SELF_SHA, "self_sha256_procedure": SELF_SHA256_PROCEDURE,
}
MANIFEST_SELF_SHA = write_once_json_atomic(MANIFEST_PATH, MANIFEST_DOC)
print("[saved] {} (self_sha256 {})".format(MANIFEST_PATH, MANIFEST_SELF_SHA))
_files = sorted(os.listdir(OUT_DIR))
_expected_files = sorted(os.path.basename(p) for p in (PLAN_PATH, PROBS_PATH, FROZEN_PATH, DECISION_TABLE_PATH,
                                                         BOOT_DELTAS_PATH, METRIC_ROWS_PATH, RESULTS_PATH, MANIFEST_PATH))
assert _files == _expected_files, "outputs\\ph1 holds {} - expected {}".format(_files, _expected_files)
_diff = snapshot_diff(SNAPSHOT_START, snapshot_all())
assert not _diff, "FAIL: read-only folders / input CSVs changed during the run: {}".format(_diff)
print("snapshots after the manifest: preregistration\\, outputs\\v0.2.1, v0.3, v0.9 and the input CSVs unchanged: PASS")


# =====================================================================================
# SUMMARY
# =====================================================================================
hdr("SUMMARY - ph1 (" + POSTHOC_LABEL + ")")
print("Part A gate (folds 1-8 -> fold 9) : PASS | A2 max |diff| by tolerance: decision table {} | metric rows {} | A3 "
      "max |diff|: H2 fields {} | per-draw deltas {} (tolerance {:g}) | A2b {}".format(
          PART_A_SUMMARY["decision_table_max_abs_diff_by_tolerance"], PART_A_SUMMARY["metric_rows_max_abs_diff_by_tolerance"],
          PART_A_SUMMARY["h2_fields_max_abs_diff"], max(PART_A_SUMMARY["per_draw_deltas_max_abs_diff_per_column"].values()),
          PART_A_TOL, PART_A_SUMMARY["A2b"]["status"]))
print("r1 attempt                        : {} ({}; {}) | ph1_partA_failure.json self_sha256 {} | amendment_r2 in the plan "
      "and the results".format(R1_ATTEMPT["folder"], R1_ATTEMPT["code_revision"], R1_ATTEMPT["failed_checks"],
                               R1_ATTEMPT["partA_failure_self_sha256"]))
print("fold 10 (eval)                    : {} records / {} patients | train folds 1-9: {} records / {} patients".format(
    EXPECTED_SPLITS["B"]["n_eval"], EXPECTED_SPLITS["B"]["n_eval_patients"], EXPECTED_SPLITS["B"]["n_train"],
    EXPECTED_SPLITS["B"]["n_train_patients"]))
print("\nfold-10 decision table (POST HOC; macro-AUROC mean across seeds):")
print(B_MET["decision_table"][["macro_auroc_mean", "ci_low_mean", "ci_high_mean", "delta_vs_B1_physio"]].round(4).to_string())
print("\nH1 rule post hoc (B3 >= {:.2f}):".format(VERDICT_THRESHOLD),
      {k: v["passes_threshold"] for k, v in B_MET["h1_verdict"].items() if k != "overall_pass"},
      "-> overall:", "PASS" if B_MET["h1_verdict"]["overall_pass"] else "FAIL")
print("\nH2-style paired test post hoc (delta = B3 - B1, seed 42, REF + locked BOOT_IDX, {} patients):".format(
    EXPECTED_N_FOLD10_PATIENTS))
for learner in LEARNERS:
    r = B_RPL[learner]
    print("  [{:20s}] delta={:+.4f} 95%CI[{:+.4f},{:+.4f}] 90%CI[{:+.4f},{:+.4f}] SE={:.4f} MDE={:.4f} | {} | TOST: {} | "
          "achieved bound={:.4f}".format(learner, r["point_delta"], r["ci95_percentile"][0], r["ci95_percentile"][1],
                                         r["ci90_percentile"][0], r["ci90_percentile"][1], r["SE_delta"], r["MDE"],
                                         r["difference_test_verdict"], r["tost_verdict"], r["achieved_equivalence_bound"]))
print("learner disagreement              :", "; ".join(B_MET["learner_disagreement"]) if B_MET["learner_disagreement"]
      else "none")
print("\nfiles (outputs\\ph1):")
for _o in MANIFEST_DOC["outputs"]:
    print("  {}  {:>10d} B  {}".format(_o["sha256"], _o["bytes"], _o["file"]))
print("  manifest self_sha256 {}  {}".format(MANIFEST_SELF_SHA, os.path.basename(MANIFEST_PATH)))
print("\nwarnings ({}):".format(len(WARNINGS)))
for w in WARNINGS:
    print("  - " + w)
print("\n" + CODE_VERSION + " " + CODE_REVISION + " finished successfully. " + POSTHOC_LABEL)
