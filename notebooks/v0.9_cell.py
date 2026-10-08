# =====================================================================================
# Project: PTB-XL Metadata Audit  -  v0.9 THE SINGLE OPENING OF FOLD 10 (H3 evaluation)
# Code version: v0.9 | Code revision: r1
#   (a fix made BEFORE fold 10 is opened increments CODE_REVISION and is re-reviewed and re-timestamped;
#    after fold 10 is opened a fix runs as v0.9.1 in its own folder - see POST_OPENING_POLICY)
# Date: 2026-10-04
# Locked by: final_run_protocol.json (FRP): stage_plan["v0.9"], v0.8_overrides["v0.9_evaluation"],
#            crash_and_rerun_rules["v0.9"], post_fold10_rules, bootstrap, H3_decision, reporting_order,
#            exploratory_fold10_comparisons, reference_implementation. FRP has priority over any prompt.
#
# Purpose - the phases run strictly in this order; each starts only after the previous one passed:
#   STEP 0 (part 1)  integrity: snapshots of PREREG_DIR (10 files) and outputs\v0.8; the five locked documents
#           (self + file sha256), parent chain, .ots == v0.7 manifest; manifests v0.4-v0.8 (v0.8: file sha256,
#           flags, exactly one 'completed' and no 'completed_recovered' per run, .ots present); state of
#           outputs\v0.9 and the pre-freeze STOP rule; environment == v0.8 manifest (except os), cuDNN 90100,
#           same GPU, wfdb 4.3.1; recipe and TRAINING_DETAILS == v0.5_training_details.json; FRP checks; input
#           CSV / v0.4 cache hashes; reference implementation sha256-asserted and exec'd (REF); the 39 v0.8
#           output files hash-verified against the v0.8 manifest
#   STEP 0 (part 2)  ptbxl_database.csv is read ONCE with a column whitelist (no label column); the folds 1-9
#           locked quantities are refit and asserted == FRP (encoder, age scaler, column order, lead
#           normalization; the FRP values are used); folds 1-8 development objects for Part A (suffix 18);
#           fold-10 METADATA only: lock hashes (fold-10 ids, patients, BOOT_IDX, PERM_IDX) == FRP, fold-10
#           level mapping and META10 - no fold-10 signal, no label
#   PART A  pre-opening validation - no fold-10 signal, no label, no metric:
#           A1 the v0.4 wfdb reader re-reads all 19,601 folds 1-9 records (chunks of 1000): bitwise == v0.4 cache
#           A2 predict_records reproduces the saved fold-9 probabilities of the 18 development cells (tol 1e-5)
#           A3 encode_with_categories reproduces the folds 1-8 dummies (79 columns) and META18 of the fold-9 rows
#           A4 the 18 v0.8 state_dicts load with strict=True (head widths 256/259/339; no forward pass)
#           A5 h3_verdict truth table: the nine v0.6 cases (synthetic CI bounds, no data; runs in every state)
#           afterwards every folds 1-8 object (*18) is deleted
#   PART B  identity guard -> 'started' -> the 2198 fold-10 records are read with the A1-validated reader ->
#           inference of the 18 v0.8 models + 120 acquisition-permutation passes -> identity guard -> FREEZE
#           (v0.9_frozen_outputs.json: sha256 of the 20 output files, written BEFORE any label is read and
#           BEFORE any metric is computed)
#   PART C  labels, only after the freeze file verifies: v0.2.1 label code on the fold-9 and fold-10 rows;
#           fold-9 counts == v0.2.1, fold-10 counts == the counts disclosed in FRP; v0.9_fold10_labels.csv
#   PART D  analysis with REF only (CPU, deterministic), FRP closed list: D0 fast_auc vs sklearn; D1 PRIMARY
#           (resnet1d_wang, seed ensemble) and D2 ROBUSTNESS (inception1d) H3 verdicts; D3 per seed
#           (secondary); D4 disagreements (reported, never resolved); D5 EXPLORATORY E1-E3; D6 descriptive
#           components; D7 sanity (WARN-only); v0.9_bootstrap_deltas.csv; v0.9_results.json (written LAST)
#   MANIFEST  v0.9_manifest.json (write-once, first complete execution) or v0.9_manifest_reexec_<n>.json
#
# BINDING FRP CLAUSES (meaning; the verbatim texts are read from FRP and stored in the outputs):
#   stage_plan["v0.9"]: the single opening of fold 10, the first and only stage that reads fold-10 signals or
#     labels; STEP 0 regenerates BOOT_IDX / PERM_IDX / the fold-10 id list with reference_implementation and
#     asserts the locked hashes; evaluation runs FROM THE SAVED, HASH-VERIFIED v0.8 state_dicts (no model is
#     re-trained or re-created by training); 7 windows, mean of sigmoid probabilities per record (as v0.5), with
#     the locked folds 1-9 lead normalization, age scaler and encoder applied to fold 10; the fold-10
#     probabilities of all 18 models are written to files and hashed BEFORE any metric is computed; fold-10
#     labels come from the v0.2.1 label code (verbatim).
#   v0.8_overrides["v0.9_evaluation"]: the 'evaluation' and 'metric' fields of v0.5_training_details.json apply
#     to fold-10 records, with 'folds 1-8 statistics' replaced by the locked folds 1-9 lead normalization and age
#     scaler.
#   crash_and_rerun_rules["v0.9"] -> post_fold10_rules 2: a re-execution uses only the saved, sha256-verified
#     v0.8 state_dicts and the saved, sha256-verified fold-10 probability files (see INTERPRETATION below).
#   post_fold10_rules 1 and 4: no model is trained after fold 10 is opened; the acquisition permutation is done
#     at inference time on the saved v0.8 state_dicts.
#   bootstrap, H3_decision, reporting_order, exploratory_fold10_comparisons, reference_implementation: executed
#     exactly, with REF (the reference implementation stored in FRP) for every fold-10 number.
#
# INTERPRETATION FIXED BEFORE FOLD 10 WAS OPENED (supervisor answer 1, 2026-10-04; constant
# INTERPRETATION_POST_FOLD10_RULE_2, stored in the freeze file, the results and the manifest):
#   post_fold10_rules 2 constrains where MODEL OUTPUTS come from: the saved, sha256-verified v0.8 state_dicts
#   and, after FREEZE, the frozen fold-10 probability files. It does not forbid reading ptbxl_database.csv to
#   derive labels. Before FREEZE (no label read, no metric computed) a crash may be followed by re-reading the
#   fold-10 signals to regenerate the probabilities from the same state_dicts (at most 2 pre-freeze attempts).
#
# HARD RULES (each one visible in the code):
#   1. No training: no optimizer, no backward pass, no model.train(). Models come only from the 18 v0.8
#      state_dicts: sha256-verified, torch.load(path, map_location=DEVICE, weights_only=True),
#      load_state_dict(..., strict=True); predict_records runs model.eval() under torch.no_grad().
#   2. No label of any fold is read before FREEZE: the STEP 0 read uses usecols (no scp_codes, report or any
#      other diagnostic column); scp_statements.csv is first read in Part C; derive_labels() verifies the freeze
#      file before it reads anything.
#   3. Every fold-10 AUROC, bootstrap index, CI and verdict comes from REF. The only other AUROC call is the D0
#      check that the locked 'metric' text requires (sklearn.metrics is imported inside D0 only). Nothing is
#      rounded before a comparison.
#   4. Closed output list (Part D): nothing else is computed on fold 10 - no AUPRC, no calibration, no
#      thresholds, no subgroups, no other arm pair, no signal_plus_all - signal_only.
#   5. PREREG_DIR and outputs\v0.8 are never written (name + sha256 snapshots asserted unchanged at the end);
#      outputs\v0.4 - outputs\v0.7 are only read.
#   6. Write-once files (freeze file, labels CSV, bootstrap-deltas CSV, results JSON, first manifest) are written
#      to '<name>.partial' and moved with an existence check + os.rename; on re-execution they are recomputed and
#      compared, never overwritten.
#   The words optimizer / AdamW / OneCycleLR / fast_auc / roc_auc_score / model.eval() also occur inside the
#   locked TRAINING_DETAILS text, which exists only for its equality assert with v0.5_training_details.json.
#
# DOCUMENTED CHANGES vs v0.8 (all deliberate):
#   (1)  CODE_VERSION v0.9, CODE_REVISION r1, OUT_DIR = outputs\v0.9; outputs\v0.8 becomes read-only (snapshot).
#   (2)  No training: the v0.8 PREFLIGHT, RandomCropDataset, make_loader, train_run, the training loop, the
#        init-hash / schedule helpers and the training-only constants are removed; the v0.8 stage_plan["v0.8"],
#        final_training_steps and shrink-factor checks are dropped (training texts).
#   (3)  STEP 0 adds: outputs\v0.8 snapshot; v0.8 manifest checks; the outputs\v0.9 state machine and the
#        pre-freeze STOP rule; environment compared with the v0.8 manifest (was: v0.7); wfdb imported (first
#        time since v0.4) and asserted 4.3.1; the FRP v0.9 checks; REF is exec'd (v0.8: hash only); the 39 v0.8
#        outputs are hash-verified.
#   (4)  STEP 0 (part 2): read_csv with usecols (no label column) instead of a full read; the fold-10 rows of the
#        same read become fold10_meta (column whitelist); the label derivation and the fold-9 label-count assert
#        move to Part C (after FREEZE). The rest of the v0.8 folds 1-9 refit is unchanged and asserted == FRP.
#   (5)  New: folds 1-8 development objects (v0.7 / v0.6 code, suffix 18, deleted after Part A); fold-10
#        metadata mapping and lock hashes (v0.6 code); META10; the identity guard.
#   (6)  New phases: Part A (A1-A5), Part B (inference, permutation passes, FREEZE), Part C (labels), Part D
#        (analysis) and the results JSON; the v0.8 training summary CSV has no counterpart.
#   (7)  Attempt log (v0.7 functions, append-only) events: started, failed, frozen, frozen_recovered,
#        deleted_partial_files, post_freeze_started, results_written, results_verified.
#   (8)  Manifest: write-once v0.9_manifest.json at the end of the first complete execution; every later
#        execution writes v0.9_manifest_reexec_<n>.json (v0.8: regenerated on every execution).
#
# CODE PROVENANCE MARKERS:
#   '# >>> verbatim from v0.x (label)' ... '# <<< verbatim from v0.x': byte-identical to a contiguous excerpt of
#     the v0.x cell.
#   '# >>> adapted from v0.x (label) | ...' ... '# <<< adapted from v0.x': identical to a contiguous excerpt of
#     the v0.x cell after undoing exactly what the marker declares, in this order: delete the lines tagged
#     '# [v0.9 added]'; remove 'indent: N' leading spaces; undo each 'literal: old => new' (separated by ' ;; ');
#     undo each 'renames: old->new' (whole identifiers only). Comments, docstrings and string literals inside
#     copied blocks are never edited; where one no longer fits, a '# NOTE (v0.9)' line stands above the block.
#   Everything without a marker is new v0.9 code.
#
# RE-EXECUTION (crash_and_rerun_rules["v0.9"] -> post_fold10_rules 2):
#   state "fresh"               outputs\v0.9 absent.
#   state "pre_freeze_attempt"  outputs\v0.9 present, no freeze file (a crash before FREEZE: no label read, no
#       metric computed). STEP 0 and Part A run again; Part B deletes the leftover probability / permutation
#       files (logged) and runs again. One attempt = one 'started' entry, logged immediately before the first
#       fold-10 file is opened (a failure in STEP 0 or Part A uses no attempt). With >= 2 'started' entries and
#       no freeze file the cell STOPS in STEP 0, before Part A.
#   state "post_freeze"         the freeze file exists: it must verify (self_sha256 and every listed file); A1-A4
#       and Part B are skipped (no record is read with wfdb, no model is loaded; A5 is synthetic and runs in every
#       state); Parts C and D are recomputed from the frozen files; every write-once file that exists must equal the recomputation (CSVs byte for byte;
#       results JSON field by field except 'timestamp', 'self_sha256' and versions.os); nothing is overwritten.
#       After FREEZE no probability is ever recomputed.
#   The append-only attempt log and new v0.9_manifest_reexec_<n>.json files are the only additions after FREEZE.
#
# Outputs (OUT_DIR = <PROJECT_ROOT>\outputs\v0.9):
#   v0.9_{arch}_{arm}_seed{seed}_lr0.01_fold10_probs.csv (18)   v0.9_{arch}_perm_probs.npy (2)
#   v0.9_frozen_outputs.json   v0.9_fold10_labels.csv   v0.9_bootstrap_deltas.csv   v0.9_results.json
#   v0.9_attempt_log.jsonl     v0.9_manifest.json  /  v0.9_manifest_reexec_<n>.json
# =====================================================================================

import os, sys, json, ast, hashlib, platform, time, random, math, re, gc, ntpath, copy
from datetime import datetime
import numpy as np
import pandas as pd
import scipy                                         # version assert only
import sklearn                                       # version assert only
from sklearn.preprocessing import StandardScaler     # age scalers (folds 1-9 locked, folds 1-8 for Part A)

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 220)

# ------------------------------- configuration ---------------------------------------
PROJECT_ROOT = r"D:\ptbxl-metadata-audit"
DATA_ROOT  = os.path.join(PROJECT_ROOT, "data", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
OUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.9")
V04_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.4")
V05_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.5")
V06_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.6")
V07_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.7")
V08_DIR    = os.path.join(PROJECT_ROOT, "outputs", "v0.8")
PREREG_DIR = os.path.join(PROJECT_ROOT, "preregistration")

# >>> verbatim from v0.8 (document paths)
SEAL_PATH             = os.path.join(PREREG_DIR, "pre_registration.json")
DEV_PROTOCOL_PATH     = os.path.join(PREREG_DIR, "dev_protocol.json")
TRAINING_DETAILS_PATH = os.path.join(PREREG_DIR, "v0.5_training_details.json")
LR_SELECTION_PATH     = os.path.join(PREREG_DIR, "lr_selection.json")
FINAL_PROTOCOL_PATH   = os.path.join(PREREG_DIR, "final_run_protocol.json")
DOC_PATHS = {"pre_registration": SEAL_PATH, "dev_protocol": DEV_PROTOCOL_PATH,
             "training_details": TRAINING_DETAILS_PATH, "lr_selection": LR_SELECTION_PATH,
             "final_run_protocol": FINAL_PROTOCOL_PATH}
# <<< verbatim from v0.8

V04_MANIFEST = os.path.join(V04_DIR, "v0.4_manifest.json")
V05_MANIFEST = os.path.join(V05_DIR, "v0.5_manifest.json")
V06_MANIFEST = os.path.join(V06_DIR, "v0.6_manifest.json")
V07_MANIFEST = os.path.join(V07_DIR, "v0.7_manifest.json")
V08_MANIFEST = os.path.join(V08_DIR, "v0.8_manifest.json")
V08_MANIFEST_OTS = V08_MANIFEST + ".ots"
# >>> verbatim from v0.8 (v0.4 cache paths)
SIGNALS_NPY = os.path.join(V04_DIR, "v0.4_signals_folds1to9_100hz.npy")
SIGNALS_INDEX = os.path.join(V04_DIR, "v0.4_signals_index.csv")
V04_LEAD_NORM_CSV = os.path.join(V04_DIR, "v0.4_lead_norm_params.csv")
# <<< verbatim from v0.8

CODE_VERSION = "v0.9"
CODE_REVISION = "r1"
PREFIX = CODE_VERSION + "_"
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ACQ_COLS = ["device", "site", "nurse"]

# --- the five locked documents: self_sha256 (canonical JSON) and file sha256 (bytes on disk) ---
# >>> verbatim from v0.8 (document hash constants)
EXPECTED_SELF_SHA256 = {
    "pre_registration":   "5183f54569f3358a5339b16a2c3c3be49f5c7798eabe6b0bb25946119c2ba469",
    "dev_protocol":       "c967ac38c8cc9382bf41fdd3785c8bbd730db196e349846b8052d856bbac5fc6",
    "training_details":   "c12c33a2f649984fb1b70a46445823c46b57d1b024d87655e16661789f04b46f",
    "lr_selection":       "c70cbfa65e0ab75321dfef9c64714a64705c60a967685d79a878a1d40450749d",
    "final_run_protocol": "834acddec288862d0418e9ee70204898c9d371788eed7eb362f4c0d63d305c6b",
}
EXPECTED_FILE_SHA256 = {
    "pre_registration":   "11dd446272654fc80d65b2fdd40259fa674bbb419a44fde84e589b6e297cde32",
    "dev_protocol":       "babf101c1d69b322daef6907175fdc4c17b5a53755a68f13a998eefa0148dc2d",
    "training_details":   "6480c191aeaed9ff1e6b1da5caa0c1e46908d453d30106a32beb7cf83678be11",
    "lr_selection":       "358d783b37bba6fcf61237a43817c1fd8594fe192f8ecdfca2c25b871a21415b",
    "final_run_protocol": "e1a5260e72d2e8b0d27dc709639468b7f9a48a7c8e2f8af2d663e105dbd329d4",
}
OTS_NOTE = "Bitcoin attestation of all five .ots files confirmed by the user on 2026-10-02 (not re-verified from Python)"

EXPECTED_V04_MANIFEST_SHA256 = "3781f90c4cf5a85ea612efabb1fd81cc61b77cbeb38a54cc644c10c66afe1d9b"
EXPECTED_V05_MANIFEST_SHA256 = "59968bfb50d8c919de0e8004ef878a44a115560de438b7e74582877495e95f87"
EXPECTED_V06_MANIFEST_SHA256 = "2cb7a1b023e03aad85bb11150ad38e203624300170bc0fa197014c727758c61a"
EXPECTED_REFERENCE_CODE_SHA256 = "48ac837c07880f0bd88a7bd01475e7da3edbeab46019f8288976884ad7662856"

EXPECTED_INPUT_HASHES = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}
EXPECTED_SIGNALS_NPY_SHA256 = "17558f9d09b39b990d7f3a5ca2ea10e1527980288a49a6828c89b7c3df61745a"
EXPECTED_SIGNALS_INDEX_SHA256 = "06b49ecdafc6a74c9bdbcfa0177bb357eb845ece187b40e3a777496a8cf72203"
EXPECTED_LEAD_NORM_CSV_SHA256 = "07fb6d6dc8f0f1e2379fdd45b6462f1102518621eb242ba585fb8600acff35aa"

EXPECTED_FOLD9_POSITIVES = {"NORM": 955, "MI": 540, "STTC": 528, "CD": 495, "HYP": 268}
EXPECTED_N_RECORDS_FOLDS1TO9 = 19601
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (v0.7 manifest, PREREG_DIR, fold-10 count, age sentinel, cuDNN)
EXPECTED_V07_MANIFEST_SHA256 = "b443264cb1096ebd35dec5340f27e31d2a4f09795165f1aaca27070c673c2dc6"
EXPECTED_V07_FLAGS = {"fold10_opened": False, "prereg_dir_unchanged": True, "h3_verdict_called": False}
EXPECTED_PREREG_N_FILES = 10
EXPECTED_N_FOLD10_RECORDS = 2198
EXPECTED_N_AGE_SENTINEL_FOLDS1TO9 = 259
EXPECTED_CUDNN_VERSION = 90100
# <<< verbatim from v0.8

# >>> verbatim from v0.8 (constants that dev_protocol.json defines; the FILE values are used)
EXPECTED_EPOCHS = 50
EXPECTED_LR_GRID = [1e-3, 3e-3, 1e-2]
EXPECTED_CROP = 250
EXPECTED_EVAL_STRIDE = 125
EXPECTED_WINDOW_STARTS = [0, 125, 250, 375, 500, 625, 750]
EXPECTED_WEIGHT_DECAY = 1e-2
EXPECTED_BATCH_SIZE = {"resnet1d_wang": 128, "inception1d": 128}
EXPECTED_SELECTED_LR = {"resnet1d_wang": 0.01, "inception1d": 0.01}

SEED = 42
FS = 100
N_SAMPLES = 1000
N_LEADS = 12
NUM_WORKERS = 0                   # Windows + Jupyter
# <<< verbatim from v0.8
# >>> verbatim from v0.7 (evaluation batch size, lr tie text constant, sklearn check tolerance)
EVAL_BATCH_SIZE = 512             # inference only; does not affect results (model.eval())
LR_TIE_TOL = 1e-4                 # appears in TRAINING_DETAILS (verbatim v0.5); not used for selection here
SKLEARN_CHECK_TOL = 1e-9
# <<< verbatim from v0.7

# >>> verbatim from v0.8 (archs, arms, seeds, attempts, embedding, chunk)
ARCHS = ["resnet1d_wang", "inception1d"]
ARMS = ["signal_only", "signal_plus_physio", "signal_plus_all"]
SEEDS = [42, 123, 2024]
MAX_ATTEMPTS = 2
EMBEDDING_DIM = 256
CHUNK = 1000
# <<< verbatim from v0.8

# final-run (folds 1-9) locked quantities; every one is also asserted equal to FRP below
# >>> verbatim from v0.8 (final-run widths)
EXPECTED_ONEHOT_WIDTHS_FINAL = {"device": 13, "site": 53, "nurse": 14}
EXPECTED_NEW_LEVELS = {"device": [], "site": ["049"], "nurse": []}
EXPECTED_META_DIM = {"signal_only": 0, "signal_plus_physio": 3, "signal_plus_all": 83}
EXPECTED_HEAD_IN = {"signal_only": 256, "signal_plus_physio": 259, "signal_plus_all": 339}
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (locked-quantity tolerances)
AGE_SCALER_TOL = {"mean": 1e-12, "scale": 1e-12, "var": 1e-9}
LEAD_NORM_ATOL = 1e-12
# <<< verbatim from v0.8

# folds 1-8 DEVELOPMENT constants (Part A only; suffix 18 -> deleted with every folds 1-8 object after Part A)
# >>> adapted from v0.7 (folds 1-8 widths, v0.5 age scaler) | renames: EXPECTED_ONEHOT_COUNTS->EXPECTED_ONEHOT_COUNTS18, AGE_SCALER_TOL->AGE_SCALER_TOL18
EXPECTED_ONEHOT_COUNTS18 = {"device": 13, "site": 52, "nurse": 14}
V05_AGE_SCALER_FOLDS1TO8 = {"mean": 59.7333, "scale": 16.8750}      # printed with 4 decimals in v0.5
AGE_SCALER_TOL18 = 5e-5
# <<< adapted from v0.7
# >>> adapted from v0.7 (folds 1-8 meta dims and head widths) | renames: EXPECTED_META_DIM->EXPECTED_META_DIM18, EXPECTED_HEAD_IN->EXPECTED_HEAD_IN18
EXPECTED_META_DIM18 = {"signal_only": 0, "signal_plus_physio": 3, "signal_plus_all": 82}
EXPECTED_HEAD_IN18 = {"signal_only": 256, "signal_plus_physio": 259, "signal_plus_all": 338}
# <<< adapted from v0.7
# >>> verbatim from v0.7 (the three development cells reused from v0.5)
REUSED_CELLS = [("resnet1d_wang", "signal_plus_physio", 42), ("inception1d", "signal_plus_physio", 42),
                ("resnet1d_wang", "signal_only", 42)]
# <<< verbatim from v0.7
# >>> verbatim from v0.8 (run order, as v0.7)
RUN_ORDER = [(arch, arm, seed) for arch in ARCHS for seed in SEEDS for arm in ARMS]      # as v0.7
assert len(RUN_ORDER) == len(ARCHS) * len(ARMS) * len(SEEDS) == 18
# <<< verbatim from v0.8

# v0.9 constants
EXPECTED_V08_MANIFEST_SHA256 = "9725a0ee502784ffc267b1021ec3cb427966708205db9a09ce7e8c1fc8f30415"
EXPECTED_V08_FLAGS = {"evaluation_performed": False, "metric_computed": False, "fold10_opened": False,
                      "wfdb_imported": False, "prereg_dir_unchanged": True,
                      "files_read_from_outputs_v0.7": ["v0.7_manifest.json"], "v0.9_started": False}
EXPECTED_V08_N_OUTPUTS = 39            # 36 run files + preflight CSV + attempt log + training summary CSV
EXPECTED_WFDB_VERSION = "4.3.1"
EXPECTED_N_FOLD10_PATIENTS = 1904
EXPECTED_FOLD10_UNTRAINED_COUNT = 0
EXPECTED_N_BOOT = 2000
EXPECTED_BOOT_SEED = 7
EXPECTED_N_PERM = 20
EXPECTED_PERM_SEED = 42
EXPECTED_SESOI_H3 = 0.01
NAN_FLAG_FRACTION = 0.01               # FRP bootstrap.nan_rule: "if > 1% of draws are NaN this is flagged"
# fold-10 label counts disclosed in FRP disclosures_fold10_exposure (computed in v0.1.1 / v0.1.2 with the same
# label rule as v0.2.1); asserted to appear verbatim in FRP, then checked in Part C before any metric
EXPECTED_FOLD10_POSITIVES = {"NORM": 963, "MI": 550, "STTC": 521, "CD": 496, "HYP": 262}
STAGE_PLAN_V09_PHRASES = ["single opening of fold 10", "FROM THE SAVED, HASH-VERIFIED v0.8 state_dicts",
                          "BEFORE any metric is computed", "v0.2.1 label code"]
V09_EVALUATION_PHRASE = "apply to fold-10 records in v0.9"
EXPECTED_EXPLORATORY_KEYS = {"rule", "physio_minus_signal_only", "per_class_deltas", "acquisition_permutation"}
A1_CHUNK = 1000
A2_TOL = 1e-5
STD_RATIO_WARN_RANGE = (0.5, 2.0)      # WARN-only signal sanity of fold 10 (per-lead std / FRP folds 1-9 std)
SANITY_GATE_MIN_AUROC = 0.90           # D7 WARN-only (dev_protocol pipeline_sanity_gate threshold)
SECONDARY_VERDICT_LABEL = "secondary (per-seed); not the H3 verdict"
RUN_LABEL = "v0.9_fold10"
DB_USECOLS = ["ecg_id", "patient_id", "age", "sex", "device", "site", "nurse", "filename_lr", "strat_fold"]
FOLD10_META_COLUMNS = ["patient_id", "age", "sex", "device", "site", "nurse", "filename_lr"]
LABEL_COLUMNS_NEVER_READ_BEFORE_FREEZE = ["scp_codes", "report"]

INTERPRETATION_POST_FOLD10_RULE_2 = (
    "post_fold10_rules 2 constrains where MODEL OUTPUTS come from: the saved, sha256-verified v0.8 state_dicts and, "
    "after FREEZE, the frozen fold-10 probability files. It does not forbid reading ptbxl_database.csv to derive "
    "labels. Before FREEZE (no label read, no metric computed) a crash may be followed by re-reading the fold-10 "
    "signals to regenerate the probabilities from the same state_dicts (at most 2 pre-freeze attempts). Fixed "
    "before fold 10 was opened (supervisor answer 1, 2026-10-04).")
POST_OPENING_POLICY = (
    "After fold 10 has been opened:\n"
    "1. The v0.9 outputs are never modified or overwritten.\n"
    "2. A bug is only a demonstrable deviation of the v0.9 code from the locked documents or from the behaviour "
    "validated in Part A, shown by a test that uses no fold-10 label and no fold-10 result. A different method, "
    "setting or threshold is never a bug fix: it is a post-hoc analysis (post_fold10_rules 3) and is reported as "
    "such. No locked setting (SESOI, CI levels, bootstrap, verdict table, closed list) is ever changed.\n"
    "3. A bug fix runs under a new version id (v0.9.1) in its own folder, and the fix is written down before the "
    "corrected code is run. Probabilities are regenerated from the same hash-verified v0.8 state_dicts only if the "
    "bug is in the inference path; otherwise the frozen v0.9 probability files are reused.\n"
    "4. Both the original and the corrected results are reported, together with the deviation and whether any "
    "metric had been computed before the bug was found.\n"
    "5. No model is ever retrained.")
FREEZE_STATEMENT = "no label of any fold has been read; no metric has been computed"

ATTEMPT_LOG_PATH = os.path.join(OUT_DIR, PREFIX + "attempt_log.jsonl")
FREEZE_PATH = os.path.join(OUT_DIR, PREFIX + "frozen_outputs.json")
LABELS_PATH = os.path.join(OUT_DIR, PREFIX + "fold10_labels.csv")
BOOT_CSV_PATH = os.path.join(OUT_DIR, PREFIX + "bootstrap_deltas.csv")
RESULTS_PATH = os.path.join(OUT_DIR, PREFIX + "results.json")
MANIFEST_PATH = os.path.join(OUT_DIR, PREFIX + "manifest.json")
REEXEC_MANIFEST_RE = re.compile(r"^v0\.9_manifest_reexec_(\d+)\.json$")

# >>> verbatim from v0.8 (output paths outside the dataset folder)
_root_abs = os.path.abspath(DATA_ROOT)
for _p in (OUT_DIR, PREREG_DIR):
    assert not os.path.abspath(_p).lower().startswith(_root_abs.lower()), \
        "output path is inside the dataset folder - aborting: " + _p
# <<< verbatim from v0.8

# >>> verbatim from v0.8 (run start, warnings)
RUN_STARTED = datetime.now().astimezone().isoformat()
WARNINGS = []
# <<< verbatim from v0.8
PART_B_WARNINGS = []                   # stored in the freeze file (Part B does not run after FREEZE)
ANALYSIS_WARNINGS = []                 # Parts C / D only; stored in the results JSON


# >>> verbatim from v0.8 (helpers 1/3)
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (helpers 2/3)
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


def verify_self_sha256(path):
    """Return (stored, recomputed, loaded_dict) for a self-hashed JSON document."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    stored = d.pop("self_sha256", None)
    recomputed = hashlib.sha256(canonical_json_bytes(d)).hexdigest()
    d["self_sha256"] = stored
    return stored, recomputed, d
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (helpers 3/3)
def plain(o):
    return json.loads(json.dumps(o, ensure_ascii=False, default=_json_default))


def manifest_entry(manifest, basename):
    """Return the unique 'outputs' entry of a project manifest whose file basename matches
    (works for full Windows paths and bare names)."""
    hits = [o for o in manifest["outputs"] if ntpath.basename(o["file"]) == basename]
    assert len(hits) == 1, "expected exactly one manifest entry for {}, found {}".format(basename, len(hits))
    return hits[0]


def load_verified_manifest(path, expected_sha, label):
    assert os.path.isfile(path), label + " not found: " + path + " - aborting (fix the path constant)."
    got = sha256_of(path)
    assert got == expected_sha, "{} file sha256 {} != expected {} - aborting.".format(label, got, expected_sha)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), got


def prereg_snapshot():
    """Sorted (name, sha256) of every file directly in PREREG_DIR (incl. the .ots files)."""
    out = []
    for name in sorted(os.listdir(PREREG_DIR)):
        p = os.path.join(PREREG_DIR, name)
        if os.path.isfile(p):
            out.append((name, sha256_of(p)))
    return out
# <<< verbatim from v0.8


# >>> verbatim from v0.7 (write_once_json)
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
# <<< verbatim from v0.7


# >>> verbatim from v0.6 (sha256_bytes)
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()
# <<< verbatim from v0.6


# >>> verbatim from v0.6 (remap_with_categories)
def remap_with_categories(level_series, categories):
    """Same remapping rule as encode_categorical, with a FIXED category list (whose last
    element is '__UNKNOWN__'): any level not among the training categories -> '__UNKNOWN__'."""
    assert categories[-1] == "__UNKNOWN__"
    known = set(categories[:-1])
    return level_series.where(level_series.isin(known), other="__UNKNOWN__")
# <<< verbatim from v0.6


# NOTE (v0.9): one function for A3 (fold-9 rows, folds 1-8 categories) and fold 10 (folds 1-9 categories): the v0.6
# remap rule, then the one-hot construction of encode_categorical (its last four statements, unchanged).
# >>> adapted from v0.8 (one-hot lines of encode_categorical)
def encode_with_categories(level_series, categories, col_name):  # [v0.9 added]
    remapped = remap_with_categories(level_series, categories)  # [v0.9 added]
    cat = pd.Categorical(remapped, categories=categories)
    dummies = pd.get_dummies(cat)
    dummies.columns = ["{}={}".format(col_name, c) for c in dummies.columns]
    dummies.index = level_series.index
    return dummies  # [v0.9 added]
# <<< adapted from v0.8


# >>> verbatim from v0.6 (_find_nonfinite)
def _find_nonfinite(o, path="$"):
    if isinstance(o, float) and not math.isfinite(o):
        return [path]
    if isinstance(o, dict):
        return [p for k, v in o.items() for p in _find_nonfinite(v, path + "." + str(k))]
    if isinstance(o, list):
        return [p for i, v in enumerate(o) for p in _find_nonfinite(v, "{}[{}]".format(path, i))]
    return []
# <<< verbatim from v0.6


# >>> verbatim from v0.8 (partial + append-only attempt log)
def _partial(p):
    return p + ".partial"
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (attempt log)
def read_attempt_log():
    entries = []
    if not os.path.exists(ATTEMPT_LOG_PATH):
        return entries
    with open(ATTEMPT_LOG_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    for i, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            warn("attempt log line {} is not valid JSON (partial write?) - ignored: {!r}".format(i + 1, line[:120]))
    if raw and not raw.endswith("\n"):
        with open(ATTEMPT_LOG_PATH, "a", encoding="utf-8") as f:   # terminate a partial last line (append-only)
            f.write("\n")
    return entries


def log_event(run, attempt, event, **fields):
    entry = {"run": run, "attempt": attempt, "event": event, "time": datetime.now().astimezone().isoformat(), **fields}
    with open(ATTEMPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(plain(entry), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    ATTEMPT_LOG.append(plain(entry))
    return entry
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (run naming)
def run_name(arch, arm, seed, lr):
    return "{}{}_{}_seed{}_lr{:g}".format(PREFIX, arch, arm, seed, lr)
# <<< verbatim from v0.8


def prob_path(arch, arm, seed, lr):
    return os.path.join(OUT_DIR, run_name(arch, arm, seed, lr) + "_fold10_probs.csv")


def perm_path(arch):
    return os.path.join(OUT_DIR, "{}{}_perm_probs.npy".format(PREFIX, arch))


# >>> verbatim from v0.7 (v0.5 run paths)
def v05_run_paths(arch, arm, seed, lr):
    base = os.path.join(V05_DIR, "v0.5_{}_{}_seed{}_lr{:g}".format(arch, arm, seed, lr))
    return {"probs": base + "_fold9_probs.csv", "loss": base + "_train_loss.csv", "state": base + "_state_dict.pt"}
# <<< verbatim from v0.7


# >>> adapted from v0.7 (v0.5 run paths -> v0.7 run paths) | literal: def v05_run_paths( => def v07_run_paths( ;; V05_DIR, "v0.5_ => V07_DIR, "v0.7_
def v07_run_paths(arch, arm, seed, lr):
    base = os.path.join(V07_DIR, "v0.7_{}_{}_seed{}_lr{:g}".format(arch, arm, seed, lr))
    return {"probs": base + "_fold9_probs.csv", "loss": base + "_train_loss.csv", "state": base + "_state_dict.pt"}
# <<< adapted from v0.7


# >>> adapted from v0.7 (v0.5 run paths -> v0.8 run paths) | literal: def v05_run_paths( => def v08_run_paths( ;; V05_DIR, "v0.5_ => V08_DIR, "v0.8_ ;; {"probs": base + "_fold9_probs.csv", "loss" => {"loss"
def v08_run_paths(arch, arm, seed, lr):
    base = os.path.join(V08_DIR, "v0.8_{}_{}_seed{}_lr{:g}".format(arch, arm, seed, lr))
    return {"loss": base + "_train_loss.csv", "state": base + "_state_dict.pt"}
# <<< adapted from v0.7


def dir_snapshot(snap_dir):
    """Sorted (name, sha256) of every file directly in snap_dir (prereg_snapshot for any folder)."""
    out = []
    for name in sorted(os.listdir(snap_dir)):
        p = os.path.join(snap_dir, name)
        if os.path.isfile(p):
            out.append((name, sha256_of(p)))
    return out


def peek_attempt_log():
    """READ-ONLY parse of the v0.9 attempt log (STEP 0 writes nothing; read_attempt_log may append a newline)."""
    if not os.path.isfile(ATTEMPT_LOG_PATH):
        return []
    entries = []
    with open(ATTEMPT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def sha256_f32(a):
    """sha256 of an array as little-endian float32, C order (X10, META10)."""
    return sha256_bytes(np.ascontiguousarray(np.asarray(a).astype("<f4")).tobytes())


def csv_bytes(df):
    """Deterministic CSV bytes: UTF-8 with BOM, '\\n' line ends, pandas default float repr, NaN as empty."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def write_csv_replace(df, path):
    """Pre-freeze outputs (probability CSVs): '<path>.partial' then os.replace; returns sha256."""
    with open(_partial(path), "wb") as f:
        f.write(csv_bytes(df))
        f.flush()
        os.fsync(f.fileno())
    os.replace(_partial(path), path)
    return sha256_of(path)


def write_npy_replace(arr, path):
    """Pre-freeze outputs (permutation arrays): np.save through a file handle (a path would get '.npy' appended)."""
    with open(_partial(path), "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(_partial(path), path)
    return sha256_of(path)


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


def compare_or_write_csv(path, df, label):
    """Write-once CSV: written if absent; if present it must equal the recomputation byte for byte."""
    data = csv_bytes(df)
    if os.path.exists(path):
        with open(path, "rb") as f:
            disk = f.read()
        assert disk == data, "{} on disk differs from the recomputation - aborting (never overwritten).".format(label)
        print("[exists] {} == recomputation (byte for byte): PASS (not overwritten)".format(path))
        return sha256_of(path), False
    sha = write_once_bytes(path, data)
    print("[saved] {} (write-once)".format(path))
    return sha, True


def _comparable(doc):
    d = copy.deepcopy(doc)
    for k in ("timestamp", "self_sha256"):
        d.pop(k, None)
    if isinstance(d.get("versions"), dict):
        d["versions"].pop("os", None)
    return d


def compare_or_write_json(path, doc, label):
    """Write-once JSON: written if absent; if present it must equal the recomputation field by field, except
    'timestamp', 'self_sha256' and versions.os."""
    doc = plain(doc)
    assert not _find_nonfinite(doc), "non-finite value in {}: {}".format(label, _find_nonfinite(doc))
    if os.path.exists(path):
        s_, r_, disk = verify_self_sha256(path)
        assert s_ == r_, "{}: stored self_sha256 != recomputed - the file was modified - aborting.".format(label)
        mine, theirs = _comparable(doc), _comparable(disk)
        differing = sorted(set(mine) ^ set(theirs)) + sorted(k for k in set(mine) & set(theirs) if mine[k] != theirs[k])
        assert not differing, "{} differs from the recomputation in field(s) {} - aborting (never overwritten).".format(
            label, differing)
        print("[exists] {} == recomputation (field by field, except timestamp / self_sha256 / versions.os): PASS".format(path))
        return s_, False
    sha = write_once_json_atomic(path, doc)
    print("[saved] {} (write-once, self_sha256 {})".format(path, sha))
    return sha, True


def warn_part_b(msg):
    warn(msg)
    PART_B_WARNINGS.append(msg)


def warn_analysis(msg):
    warn(msg)
    ANALYSIS_WARNINGS.append(msg)


def frozen_output_names():
    names = [os.path.basename(prob_path(a, m, s, EXPECTED_SELECTED_LR[a])) for a, m, s in RUN_ORDER]
    return names + [os.path.basename(perm_path(a)) for a in ARCHS]


def verify_frozen_outputs():
    """The freeze file exists, its self_sha256 verifies, it was written by this code version / revision, and every
    listed output file exists with the recorded sha256. Returns the loaded document."""
    assert os.path.isfile(FREEZE_PATH), \
        "the freeze file {} does not exist - no label may be read before FREEZE - aborting.".format(FREEZE_PATH)
    s_, r_, d = verify_self_sha256(FREEZE_PATH)
    assert s_ == r_, "freeze file: stored self_sha256 {} != recomputed {} - aborting.".format(s_, r_)
    assert d["code_version"] == CODE_VERSION and d["code_revision"] == CODE_REVISION, \
        "freeze file written by {} {} - this cell is {} {} - aborting.".format(
            d["code_version"], d["code_revision"], CODE_VERSION, CODE_REVISION)
    assert sorted(d["outputs"]) == sorted(frozen_output_names()), "freeze file output list differs - aborting."
    for name, rec in d["outputs"].items():
        p = os.path.join(OUT_DIR, name)
        assert os.path.isfile(p), "frozen output missing: " + p + " - aborting."
        got = sha256_of(p)
        assert got == rec["sha256"], "frozen output {} changed: sha256 {} != frozen {} - aborting.".format(
            name, got, rec["sha256"])
    return d


# NOTE (v0.9): the v0.2.1 label code (via v0.7) as a function of the frame; it first verifies the freeze file.
# >>> adapted from v0.7 (label derivation, v0.2.1) | indent: 4 | renames: df9->frame
def derive_labels(frame):  # [v0.9 added]
    _frozen = verify_frozen_outputs()  # [v0.9 added]
    scp = pd.read_csv(SCP_CSV, index_col=0)
    diag = scp[scp["diagnostic"].fillna(0) == 1]
    code2class = {k: v for k, v in diag["diagnostic_class"].items() if isinstance(v, str)}
    parsed = frame["scp_codes"].apply(ast.literal_eval)
    rec_classes = parsed.apply(lambda d: {code2class[k] for k in d if k in code2class})
    lab = pd.DataFrame({c: rec_classes.apply(lambda s, c=c: int(c in s)) for c in SUPERCLASSES},
                       index=frame.index)
    return lab  # [v0.9 added]
# <<< adapted from v0.7


# =====================================================================================
# STEP 0 (part 1) - INTEGRITY (nothing is written before STEP 0 and Part A pass)
# =====================================================================================
hdr("STEP 0 (part 1) - INTEGRITY")

# --- 1. snapshots: PREREG_DIR and outputs\v0.8 (both asserted unchanged at the end of the cell) ---
# >>> verbatim from v0.8 (PREREG_DIR snapshot)
assert os.path.isdir(PREREG_DIR), "PREREG_DIR not found: " + PREREG_DIR
PREREG_SNAPSHOT_START = prereg_snapshot()
print("PREREG_DIR snapshot: {} files".format(len(PREREG_SNAPSHOT_START)))
for name, h in PREREG_SNAPSHOT_START:
    print("  {}  {}".format(h, name))
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (PREREG_DIR has 10 files)
assert len(PREREG_SNAPSHOT_START) == EXPECTED_PREREG_N_FILES, \
    "PREREG_DIR has {} files, expected {} - aborting.".format(len(PREREG_SNAPSHOT_START), EXPECTED_PREREG_N_FILES)
# <<< verbatim from v0.8
assert os.path.isdir(V08_DIR), "outputs\\v0.8 not found: " + V08_DIR
V08_SNAPSHOT_START = dir_snapshot(V08_DIR)
print("outputs\\v0.8 snapshot: {} files (read-only for v0.9)".format(len(V08_SNAPSHOT_START)))

# --- 2. the five locked documents ---
# >>> verbatim from v0.8 (five documents + parent chain)
DOCS, DOC_SELF, DOC_FILE = {}, {}, {}
for key, path in DOC_PATHS.items():
    assert os.path.isfile(path), key + " not found: " + path + " - aborting."
    _s, _r, _d = verify_self_sha256(path)
    assert _s == _r == EXPECTED_SELF_SHA256[key], \
        "{}: self_sha256 stored {} / recomputed {} / expected {} - aborting.".format(key, _s, _r, EXPECTED_SELF_SHA256[key])
    _f = sha256_of(path)
    assert _f == EXPECTED_FILE_SHA256[key], "{}: file sha256 {} != {} - aborting.".format(key, _f, EXPECTED_FILE_SHA256[key])
    DOCS[key], DOC_SELF[key], DOC_FILE[key] = _d, _s, _f
    print("{:18s} self {} | file {} -> PASS".format(key, _s, _f))
SEAL, DEV, TD_DOC, LR_SEL, FRP = (DOCS["pre_registration"], DOCS["dev_protocol"], DOCS["training_details"],
                                  DOCS["lr_selection"], DOCS["final_run_protocol"])
assert SEAL.get("sealed") is True
assert DEV["parent_seal_self_sha256"] == DOC_SELF["pre_registration"]
assert TD_DOC["parent_seal_self_sha256"] == DOC_SELF["pre_registration"]
assert TD_DOC["parent_dev_protocol_self_sha256"] == DOC_SELF["dev_protocol"]
assert LR_SEL["parent_hashes"] == {"pre_registration": DOC_SELF["pre_registration"],
                                   "dev_protocol": DOC_SELF["dev_protocol"],
                                   "training_details": DOC_SELF["training_details"]}
assert FRP["parent_hashes"] == {k: {"self_sha256": DOC_SELF[k], "file_sha256": DOC_FILE[k]}
                                for k in ("pre_registration", "dev_protocol", "training_details", "lr_selection")}, \
    "final_run_protocol parent_hashes differ from the four documents on disk - aborting."
assert FRP["code_version"] == "v0.6" and FRP["write_once"] is True
print("parent chain: dev_protocol -> seal; training_details -> seal + dev_protocol; lr_selection -> 3; "
      "final_run_protocol.parent_hashes == the four documents (self + file): PASS")
# <<< verbatim from v0.8

# --- OpenTimestamps proofs: existence, sha256, size (== v0.7 manifest, step 3) ---
# >>> verbatim from v0.8 (.ots files)
OTS_FILES = {}
for key, path in DOC_PATHS.items():
    ots = path + ".ots"
    assert os.path.isfile(ots), "OpenTimestamps proof missing: " + ots + " - aborting."
    OTS_FILES[os.path.basename(ots)] = {"sha256": sha256_of(ots), "bytes": os.path.getsize(ots)}
    print("  {}  {:6d} B  {}".format(OTS_FILES[os.path.basename(ots)]["sha256"], OTS_FILES[os.path.basename(ots)]["bytes"],
                                     os.path.basename(ots)))
print(OTS_NOTE)
# <<< verbatim from v0.8

# --- 3. earlier manifests (file sha256) ---
# >>> verbatim from v0.8 (v0.6 manifest)
V06_MAN, V06_MAN_SHA = load_verified_manifest(V06_MANIFEST, EXPECTED_V06_MANIFEST_SHA256, "v0.6_manifest.json")
assert V06_MAN["final_run_protocol"]["file_sha256"] == EXPECTED_FILE_SHA256["final_run_protocol"]
assert V06_MAN["final_run_protocol"]["self_sha256"] == EXPECTED_SELF_SHA256["final_run_protocol"]
print("v0.6_manifest.json    : {} | records final_run_protocol file sha256: PASS".format(V06_MAN_SHA))
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (v0.7 / v0.4 / v0.5 manifests)
V07_MAN, V07_MAN_SHA = load_verified_manifest(V07_MANIFEST, EXPECTED_V07_MANIFEST_SHA256, "v0.7_manifest.json")
assert V07_MAN["flags"] == EXPECTED_V07_FLAGS, "v0.7 manifest flags {} != {} - aborting.".format(V07_MAN["flags"], EXPECTED_V07_FLAGS)
assert V07_MAN["parent_documents"] == {k: {"self_sha256": DOC_SELF[k], "file_sha256": DOC_FILE[k]} for k in DOC_PATHS}, \
    "v0.7 manifest parent_documents differ from the five documents - aborting."
assert OTS_FILES == V07_MAN["ots_files"], "the .ots files (sha256 / size) differ from the v0.7 manifest - aborting."
print("v0.7_manifest.json    : {} | flags {} | parent_documents == the five documents | "
      ".ots sha256 + size == v0.7 manifest: PASS".format(V07_MAN_SHA, V07_MAN["flags"]))
V04_MAN, V04_MAN_SHA = load_verified_manifest(V04_MANIFEST, EXPECTED_V04_MANIFEST_SHA256, "v0.4_manifest.json")
V05_MAN, V05_MAN_SHA = load_verified_manifest(V05_MANIFEST, EXPECTED_V05_MANIFEST_SHA256, "v0.5_manifest.json")
assert FRP["earlier_manifests_sha256"]["v0.4_manifest.json"] == V04_MAN_SHA
assert FRP["earlier_manifests_sha256"]["v0.5_manifest.json"] == V05_MAN_SHA
V03_MAN_SHA_FROM_FRP = FRP["earlier_manifests_sha256"]["v0.3_manifest.json"]     # copied only; not opened
print("v0.4_manifest.json    : {} | v0.5_manifest.json: {} | both == final_run_protocol earlier_manifests: PASS".format(
    V04_MAN_SHA, V05_MAN_SHA))
# <<< verbatim from v0.8
V08_MAN, V08_MAN_SHA = load_verified_manifest(V08_MANIFEST, EXPECTED_V08_MANIFEST_SHA256, "v0.8_manifest.json")
assert V08_MAN["flags"] == EXPECTED_V08_FLAGS, "v0.8 manifest flags {} != {} - aborting.".format(V08_MAN["flags"], EXPECTED_V08_FLAGS)
assert V08_MAN["parent_documents"] == {k: {"self_sha256": DOC_SELF[k], "file_sha256": DOC_FILE[k]} for k in DOC_PATHS}, \
    "v0.8 manifest parent_documents differ from the five documents - aborting."
_v08_events = {}
for _e in V08_MAN["attempt_log_entries"]:
    _v08_events.setdefault(_e["run"], []).append(_e["event"])
for arch, arm, seed in RUN_ORDER:
    _nm = "v0.8_{}_{}_seed{}_lr{:g}".format(arch, arm, seed, EXPECTED_SELECTED_LR[arch])
    _ev = _v08_events.get(_nm, [])
    assert _ev.count("completed") == 1 and _ev.count("completed_recovered") == 0, \
        "v0.8 run {}: attempt-log events {} (need exactly one 'completed', no 'completed_recovered') - aborting.".format(_nm, _ev)
assert len(_v08_events) == len(RUN_ORDER), "v0.8 attempt log has runs other than the 18 - aborting."
assert os.path.isfile(V08_MANIFEST_OTS), "OpenTimestamps proof of the v0.8 manifest missing: " + V08_MANIFEST_OTS
V08_MANIFEST_OTS_INFO = {"sha256": sha256_of(V08_MANIFEST_OTS), "bytes": os.path.getsize(V08_MANIFEST_OTS),
                         "note": "recorded only; not compared with a constant ('ots upgrade' changes the file)"}
print("v0.8_manifest.json    : {} | flags == expected | parent_documents == the five documents | exactly one "
      "'completed' per run, no 'completed_recovered': PASS | .ots present ({} B, sha256 {})".format(
          V08_MAN_SHA, V08_MANIFEST_OTS_INFO["bytes"], V08_MANIFEST_OTS_INFO["sha256"]))

# --- 4. state of outputs\v0.9: fresh / pre-freeze attempt / post-freeze re-execution ---
PROB_NAMES = [os.path.basename(prob_path(a, m, s, EXPECTED_SELECTED_LR[a])) for a, m, s in RUN_ORDER]
PERM_NAMES = [os.path.basename(perm_path(a)) for a in ARCHS]
WRITE_ONCE_NAMES = [os.path.basename(p) for p in (FREEZE_PATH, LABELS_PATH, BOOT_CSV_PATH, RESULTS_PATH, MANIFEST_PATH)]
FROZEN = None
if not os.path.isdir(OUT_DIR):
    STATE = "fresh"
    print("outputs\\v0.9: absent -> state 'fresh' (fold 10 has never been opened)")
elif not os.path.isfile(FREEZE_PATH):
    STATE = "pre_freeze_attempt"
    _files = sorted(os.listdir(OUT_DIR))
    _allowed = set([os.path.basename(ATTEMPT_LOG_PATH), _partial(os.path.basename(FREEZE_PATH))]
                   + PROB_NAMES + PERM_NAMES + [_partial(n) for n in PROB_NAMES + PERM_NAMES])
    _unexpected = [f for f in _files if f not in _allowed]
    assert not _unexpected, ("outputs\\v0.9 has no freeze file but contains {} - inconsistent state, nothing is "
                             "deleted - STOP and report.".format(_unexpected))
    _log_peek = peek_attempt_log()
    _post_freeze_events = [e.get("event") for e in _log_peek if e.get("event") in
                           ("frozen", "frozen_recovered", "post_freeze_started", "results_written", "results_verified")]
    assert not _post_freeze_events, ("the attempt log records {} but the freeze file is missing - fold 10 was frozen "
                                     "before; no probability may be recomputed - STOP and report.".format(_post_freeze_events))
    _n_started = sum(e.get("event") == "started" for e in _log_peek)
    print("outputs\\v0.9: present, no freeze file -> state 'pre_freeze_attempt' ({} earlier 'started' entries; "
          "files: {})".format(_n_started, _files))
    if _n_started >= MAX_ATTEMPTS:
        raise RuntimeError("pre-freeze STOP: {} attempts already started and no freeze file exists -> STOP and report "
                           "(at most {} pre-freeze attempts; no setting may be changed).".format(_n_started, MAX_ATTEMPTS))
else:
    STATE = "post_freeze"
    FROZEN = verify_frozen_outputs()
    _files = sorted(os.listdir(OUT_DIR))
    _allowed = set([os.path.basename(ATTEMPT_LOG_PATH)] + PROB_NAMES + PERM_NAMES + WRITE_ONCE_NAMES
                   + [_partial(n) for n in WRITE_ONCE_NAMES])
    _unexpected = [f for f in _files if f not in _allowed and not REEXEC_MANIFEST_RE.match(f)
                   and not (f.endswith(".partial") and REEXEC_MANIFEST_RE.match(f[:-len(".partial")]))
                   and not f.endswith(".ots") and not f.endswith(".ots.bak")]
    assert not _unexpected, "outputs\\v0.9 contains unexpected files {} - aborting.".format(_unexpected)
    print("outputs\\v0.9: freeze file present and verified (self_sha256 {}; all {} frozen files unchanged) -> state "
          "'post_freeze': A1-A4 and Part B are skipped; Parts C and D are recomputed from the frozen files".format(
              FROZEN["self_sha256"], len(FROZEN["outputs"])))
RUN_PARTS_AB = STATE != "post_freeze"

# --- 5. environment (== v0.8 manifest) ---
import torch
import torch.nn as nn
# >>> verbatim from v0.8 (VERSIONS)
VERSIONS = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "os": platform.platform()}
# <<< verbatim from v0.8
print("versions:", VERSIONS)
_ver_diff = {k: (VERSIONS.get(k), V08_MAN["versions"].get(k)) for k in set(VERSIONS) | set(V08_MAN["versions"])
             if k != "os" and VERSIONS.get(k) != V08_MAN["versions"].get(k)}
assert not _ver_diff, "environment differs from v0.8 (now, v0.8): {} - aborting.".format(_ver_diff)
if VERSIONS["os"] != V08_MAN["versions"].get("os"):
    warn("os string differs from v0.8: now {!r}, v0.8 {!r}".format(VERSIONS["os"], V08_MAN["versions"].get("os")))
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False - no usable CUDA GPU in this kernel. Aborting.")
CUDNN_VERSION = torch.backends.cudnn.version()
assert CUDNN_VERSION == EXPECTED_CUDNN_VERSION == V08_MAN["cudnn"], \
    "cuDNN {} != {} / v0.8 {} - aborting.".format(CUDNN_VERSION, EXPECTED_CUDNN_VERSION, V08_MAN["cudnn"])
GPU_NAME = torch.cuda.get_device_name(0)
assert GPU_NAME == V08_MAN["gpu"], "GPU {!r} != v0.8 GPU {!r} - aborting.".format(GPU_NAME, V08_MAN["gpu"])
import wfdb                                          # first import since v0.4
WFDB_VERSION = getattr(wfdb, "__version__", "unknown")
assert WFDB_VERSION == EXPECTED_WFDB_VERSION, "wfdb {} != {} - aborting.".format(WFDB_VERSION, EXPECTED_WFDB_VERSION)
print("environment: versions == v0.8 manifest (except os) | cuDNN {} | GPU {} == v0.8 | wfdb {}: PASS".format(
    CUDNN_VERSION, GPU_NAME, WFDB_VERSION))

# --- 6. recipe values loaded FROM dev_protocol.json, then asserted equal to the constants ---
# >>> verbatim from v0.8 (recipe parsing)
_rec = DEV["training_recipe"]
EPOCHS = int(_rec["epochs"])
LR_GRID = [float(v) for v in DEV["tuning_grid"]["learning_rate"]]
WEIGHT_DECAY = float(_rec["weight_decay"])
BATCH_SIZE = {k: int(v) for k, v in _rec["batch_size"].items()}
_m = re.search(r"random (\d+)-sample crops", _rec["train_input"])
assert _m, "cannot parse the crop length from dev_protocol train_input"
CROP = int(_m.group(1))
_m = re.search(r"sliding windows of (\d+) samples, stride (\d+) \(window starts \[([0-9, ]+)\]\)", _rec["evaluation"])
assert _m, "cannot parse the evaluation windows from dev_protocol evaluation"
EVAL_WINDOW = int(_m.group(1))
EVAL_STRIDE = int(_m.group(2))
WINDOW_STARTS = [int(v) for v in _m.group(3).split(",")]
AGGREGATION = "mean" if "probability = mean over windows" in _rec["evaluation"] else None
assert "MEAN over sliding windows" in DEV["evaluation_aggregation_note"]
assert _rec["optimizer"] == "AdamW" and _rec["lr_schedule"] == "one-cycle"

assert EPOCHS == EXPECTED_EPOCHS, (EPOCHS, EXPECTED_EPOCHS)
assert LR_GRID == EXPECTED_LR_GRID, (LR_GRID, EXPECTED_LR_GRID)
assert WEIGHT_DECAY == EXPECTED_WEIGHT_DECAY, (WEIGHT_DECAY, EXPECTED_WEIGHT_DECAY)
assert BATCH_SIZE == EXPECTED_BATCH_SIZE, (BATCH_SIZE, EXPECTED_BATCH_SIZE)
assert CROP == EVAL_WINDOW == EXPECTED_CROP, (CROP, EVAL_WINDOW)
assert EVAL_STRIDE == EXPECTED_EVAL_STRIDE and WINDOW_STARTS == EXPECTED_WINDOW_STARTS
assert WINDOW_STARTS == list(range(0, N_SAMPLES - CROP + 1, EVAL_STRIDE))
assert AGGREGATION == "mean"
# <<< verbatim from v0.8
print("recipe from dev_protocol.json: epochs={} | weight_decay={} | batch sizes={} | crop={} | eval windows {} "
      "stride {} starts {} | aggregation={} -> all equal the constants: PASS".format(
          EPOCHS, WEIGHT_DECAY, BATCH_SIZE, CROP, EVAL_WINDOW, EVAL_STRIDE, WINDOW_STARTS, AGGREGATION))
# >>> verbatim from v0.8 (selected lr)
SELECTED_LR = {k: float(v) for k, v in LR_SEL["selected_lr"].items()}
assert SELECTED_LR == EXPECTED_SELECTED_LR, "selected_lr {} != {} - aborting.".format(SELECTED_LR, EXPECTED_SELECTED_LR)
# <<< verbatim from v0.8
print("selected lr (lr_selection.json): {} -> PASS".format(SELECTED_LR))

# --- TRAINING_DETAILS rebuilt VERBATIM (v0.5 / v0.7 / v0.8) and asserted equal to the locked file ---
# NOTE (v0.9): only for the equality assert with v0.5_training_details.json. FRP v0.8_overrides["v0.9_evaluation"]
# makes its 'evaluation' and 'metric' fields apply to fold-10 records, with the locked folds 1-9 lead normalization
# and age scaler instead of folds 1-8 statistics. Nothing is trained in v0.9.
# >>> verbatim from v0.8 (TRAINING_DETAILS)
ADAMW_BETAS = (0.9, 0.99)
ADAMW_EPS = 1e-8
ONECYCLE = {"pct_start": 0.3, "div_factor": 25.0, "final_div_factor": 1e4, "anneal_strategy": "cos",
            "cycle_momentum": True, "base_momentum": 0.85, "max_momentum": 0.95, "three_phase": False}

TRAINING_DETAILS = {
    "optimizer": {"name": "torch.optim.AdamW", "betas": list(ADAMW_BETAS), "eps": ADAMW_EPS,
                  "weight_decay": WEIGHT_DECAY, "weight_decay_source": "dev_protocol.json",
                  "applied_to": "all parameters (incl. BatchNorm and biases)"},
    "lr_scheduler": {"name": "torch.optim.lr_scheduler.OneCycleLR", "max_lr": "the run's lr",
                     "total_steps": "epochs * len(train_loader)", "stepped": "every batch (after optimizer.step())",
                     **ONECYCLE,
                     "momentum_note": ("cycle_momentum/base_momentum/max_momentum are torch defaults: AdamW "
                                       "beta1 is cycled 0.95 -> 0.85 -> 0.95 in the opposite phase of the lr")},
    "epochs": EPOCHS, "epochs_source": "dev_protocol.json",
    "batch_size": BATCH_SIZE, "batch_size_source": "dev_protocol.json",
    "train_data": ("folds 1-8; RandomCropDataset (v0.4): random {}-sample crop per record per epoch, start "
                   "uniform over [0, {}], per-lead normalized with folds 1-8 statistics".format(CROP, N_SAMPLES - CROP)),
    "data_order": ("DataLoader(shuffle=True, generator=torch.Generator().manual_seed(run_seed), num_workers=0, "
                   "pin_memory=True, drop_last=False); RandomCropDataset rng = np.random.default_rng(run_seed)"),
    "seeding": ("set_all_seeds(run_seed) (python random, numpy, torch, torch.cuda) immediately before model "
                "construction; cudnn.deterministic=True, cudnn.benchmark=False"),
    "training_loss": "BCEWithLogitsLoss (mean over batch and the 5 classes)",
    "training_loss_logging": ("per-epoch MEAN TRAINING loss = sample-weighted mean of batch losses over the "
                              "epoch (training data only), saved per run"),
    "validation": ("NO validation metric is computed at any epoch other than the end of epoch {}; no "
                   "checkpoint selection; the model after the last optimizer step is evaluated".format(EPOCHS)),
    "evaluation": ("model.eval(); torch.no_grad(); {} windows (starts {}) of {} samples per fold-9 record, "
                   "per-lead normalized with folds 1-8 statistics; per-window sigmoid PROBABILITIES averaged "
                   "per record (mean of probabilities, not of logits); metadata identical for all windows of a "
                   "record; inference batch size {} (no effect on results)".format(
                       len(WINDOW_STARTS), WINDOW_STARTS, CROP, EVAL_BATCH_SIZE)),
    "metric": ("macro-AUROC = mean over NORM, MI, STTC, CD, HYP of fast_auc (v0.2.1 function); verified once "
               "against sklearn.metrics.roc_auc_score (abs tol {})".format(SKLEARN_CHECK_TOL)),
    "lr_tie_break": ("highest fold-9 macro-AUROC; if two lrs are within {} of each other, choose the SMALLER lr "
                     "(implemented as: among lrs with macro-AUROC >= best - {}, take the smallest)".format(
                         LR_TIE_TOL, LR_TIE_TOL)),
    "per_run_outputs": ("v0.5_{arch}_{arm}_seed{seed}_lr{lr}_fold9_probs.csv (ecg_id, patient_id, 5 "
                        "probabilities), _train_loss.csv (per-epoch mean training loss), _state_dict.pt "
                        "(final model); all hashed in the manifest"),
    "determinism_note": ("torch.use_deterministic_algorithms is NOT enabled; some CUDA kernels may not be "
                         "bitwise reproducible across re-runs"),
}
assert plain(TRAINING_DETAILS) == TD_DOC["details"], \
    "v0.5_training_details.json on disk differs from this code's training definition - aborting."
TRAINING_DETAILS_SHA = DOC_SELF["training_details"]
# <<< verbatim from v0.8
V09_EVALUATION_TEXT = FRP["v0.8_overrides"]["v0.9_evaluation"]
assert V09_EVALUATION_PHRASE in V09_EVALUATION_TEXT, "FRP v0.8_overrides v0.9_evaluation text changed - aborting."
print("TRAINING_DETAILS (this code) == v0.5_training_details.json['details']: PASS | FRP v0.9_evaluation: " + V09_EVALUATION_TEXT)

# --- 7. FRP checks ---
LQ = FRP["locked_quantities"]
STAGE_PLAN_V09 = FRP["stage_plan"]["v0.9"]
for _ph in STAGE_PLAN_V09_PHRASES:
    assert _ph in STAGE_PLAN_V09, "final_run_protocol stage_plan v0.9 does not contain {!r} - aborting.".format(_ph)
CRASH_RULE_V09 = FRP["crash_and_rerun_rules"]["v0.9"]
POST_FOLD10_RULES = FRP["post_fold10_rules"]
BOOT_SPEC = FRP["bootstrap"]
N_BOOT = int(BOOT_SPEC["N_BOOT"])
BOOT_SEED = int(BOOT_SPEC["BOOT_SEED"])
assert N_BOOT == EXPECTED_N_BOOT and BOOT_SEED == EXPECTED_BOOT_SEED and \
    int(BOOT_SPEC["n_patients"]) == EXPECTED_N_FOLD10_PATIENTS, \
    "FRP bootstrap N_BOOT {} / BOOT_SEED {} / n_patients {} != {} / {} / {} - aborting.".format(
        N_BOOT, BOOT_SEED, BOOT_SPEC["n_patients"], EXPECTED_N_BOOT, EXPECTED_BOOT_SEED, EXPECTED_N_FOLD10_PATIENTS)
assert "if > 1% of draws are NaN this is flagged" in BOOT_SPEC["nan_rule"] and NAN_FLAG_FRACTION == 0.01
SESOI_H3 = float(FRP["H3_decision"]["SESOI_H3"])
assert SESOI_H3 == float(FRP["sealed_H3_protocol_verbatim"]["SESOI_H3"]) == EXPECTED_SESOI_H3, \
    "SESOI_H3 differs between H3_decision / sealed protocol / 0.01 - aborting."
SEALED_DECISION = FRP["sealed_H3_protocol_verbatim"]["decision"]
assert sorted(SEALED_DECISION) == ["a", "b", "c"] and FRP["sealed_H3_protocol_verbatim"]["seeds"] == SEEDS
assert FRP["sealed_H3_protocol_verbatim"]["primary_architecture"].startswith(ARCHS[0]) and \
    FRP["sealed_H3_protocol_verbatim"]["robustness_architecture"] == ARCHS[1]
H3_DECISION = FRP["H3_decision"]
REPORTING_ORDER = FRP["reporting_order"]
assert len(REPORTING_ORDER) == 5
EXPLORATORY = FRP["exploratory_fold10_comparisons"]
assert set(EXPLORATORY) == EXPECTED_EXPLORATORY_KEYS, "exploratory_fold10_comparisons keys {} - aborting.".format(sorted(EXPLORATORY))
_m = re.search(r"N_PERM = (\d+) sequential rng\.permutation\((\d+)\) calls on one default_rng\(PERM_SEED = (\d+)\)",
               EXPLORATORY["acquisition_permutation"]["repetitions"])
assert _m, "cannot parse N_PERM / permutation length / PERM_SEED from FRP acquisition_permutation - aborting."
N_PERM, PERM_LEN, PERM_SEED = int(_m.group(1)), int(_m.group(2)), int(_m.group(3))
assert (N_PERM, PERM_LEN, PERM_SEED) == (EXPECTED_N_PERM, EXPECTED_N_FOLD10_RECORDS, EXPECTED_PERM_SEED), \
    "FRP permutation spec {} != {} - aborting.".format((N_PERM, PERM_LEN, PERM_SEED),
                                                      (EXPECTED_N_PERM, EXPECTED_N_FOLD10_RECORDS, EXPECTED_PERM_SEED))
LOCKED_HASHES_FRP = BOOT_SPEC["locked_hashes"]
_disclosed = "(" + ", ".join("{} {}".format(c, EXPECTED_FOLD10_POSITIVES[c]) for c in SUPERCLASSES) + \
             " of {})".format(EXPECTED_N_FOLD10_RECORDS)
assert _disclosed in FRP["disclosures_fold10_exposure"]["fold10_labels"][0], \
    "the expected fold-10 label counts {} are not the ones disclosed in FRP - aborting.".format(_disclosed)
assert "macro-AUROC >= 0.90" in DEV["pipeline_sanity_gate"] and SANITY_GATE_MIN_AUROC == 0.90
print("final_run_protocol: stage_plan v0.9 phrases present | bootstrap N_BOOT {} BOOT_SEED {} n_patients {} | "
      "SESOI_H3 {} | exploratory keys {} | permutation N_PERM {} x {} records, PERM_SEED {} | disclosed fold-10 "
      "counts {} -> PASS".format(N_BOOT, BOOT_SEED, BOOT_SPEC["n_patients"], SESOI_H3, sorted(EXPLORATORY), N_PERM,
                                 PERM_LEN, PERM_SEED, _disclosed))
print("crash rule v0.9 (locked): " + CRASH_RULE_V09)
print("interpretation fixed before opening: " + INTERPRETATION_POST_FOLD10_RULE_2)

# --- 8. input CSVs, v0.4 signal cache, lead-norm csv ---
# >>> verbatim from v0.8 (input CSVs + v0.4 cache)
DB_CSV = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_CSV = os.path.join(DATA_ROOT, "scp_statements.csv")
for p in (DB_CSV, SCP_CSV):
    assert os.path.isfile(p), "file not found: " + p
INPUT_HASHES = {os.path.basename(p): sha256_of(p) for p in (DB_CSV, SCP_CSV)}
assert INPUT_HASHES == EXPECTED_INPUT_HASHES == FRP["input_files_sha256"], "input CSV hashes differ - aborting."
print("input CSV hashes      : equal to v0.4-v0.6 and to final_run_protocol -> PASS")
V04_MAN, V04_MAN_SHA = load_verified_manifest(V04_MANIFEST, EXPECTED_V04_MANIFEST_SHA256, "v0.4_manifest.json")
CACHE_HASHES = {}
for path, expected in [(SIGNALS_NPY, EXPECTED_SIGNALS_NPY_SHA256), (SIGNALS_INDEX, EXPECTED_SIGNALS_INDEX_SHA256),
                       (V04_LEAD_NORM_CSV, EXPECTED_LEAD_NORM_CSV_SHA256)]:
    assert os.path.isfile(path), "v0.4 file missing: " + path
    got = sha256_of(path)
    assert got == expected == manifest_entry(V04_MAN, os.path.basename(path))["sha256"] \
        == FRP["v0.4_signal_cache_sha256"][os.path.basename(path)], "v0.4 file hash mismatch for " + path + " - aborting."
    CACHE_HASHES[os.path.basename(path)] = got
    print("  {}  {}  -> PASS (== v0.4 manifest == final_run_protocol)".format(got, os.path.basename(path)))
# <<< verbatim from v0.8

# --- 9. reference implementation, read FROM the protocol file (never retyped), exec'd into REF ---
# >>> verbatim from v0.7 (reference implementation exec)
REFERENCE_CODE = FRP["reference_implementation"]["code"]
REFERENCE_CODE_SHA256 = hashlib.sha256(REFERENCE_CODE.encode("utf-8")).hexdigest()
assert REFERENCE_CODE_SHA256 == FRP["reference_implementation"]["sha256_of_utf8_code"] == EXPECTED_REFERENCE_CODE_SHA256, \
    "reference implementation sha256 mismatch - aborting."
REF = {}
exec(REFERENCE_CODE, REF)
# <<< verbatim from v0.7
print("reference implementation: sha256 {} -> PASS (exec'd into REF; every fold-10 AUROC, bootstrap index, CI and "
      "verdict uses REF)".format(REFERENCE_CODE_SHA256))

# --- 10. the v0.8 outputs (39 files incl. the 36 run files) == v0.8 manifest ---
_v08_snap = dict(V08_SNAPSHOT_START)
V08_OUTPUT_SHA = {}
for _o in V08_MAN["outputs"]:
    _name = ntpath.basename(_o["file"])
    assert _name in _v08_snap, "v0.8 output missing: " + _name + " - aborting."
    assert _v08_snap[_name] == _o["sha256"], "v0.8 output {} differs from the v0.8 manifest - aborting.".format(_name)
    V08_OUTPUT_SHA[_name] = _o["sha256"]
assert len(V08_OUTPUT_SHA) == len(V08_MAN["outputs"]) == EXPECTED_V08_N_OUTPUTS
V08_STATE_SHA = {}
for arch, arm, seed in RUN_ORDER:
    _paths = v08_run_paths(arch, arm, seed, SELECTED_LR[arch])
    _run = [r for r in V08_MAN["runs"] if (r["arch"], r["arm"], int(r["seed"])) == (arch, arm, seed)]
    assert len(_run) == 1, "v0.8 manifest runs: {} entries for {} {} {} - aborting.".format(len(_run), arch, arm, seed)
    for _kind in ("state", "loss"):
        assert os.path.basename(_paths[_kind]) in V08_OUTPUT_SHA, "v0.8 run file not in the manifest outputs: " + _paths[_kind]
    assert _run[0]["state_dict_sha256"] == V08_OUTPUT_SHA[os.path.basename(_paths["state"])] and \
        _run[0]["train_loss_sha256"] == V08_OUTPUT_SHA[os.path.basename(_paths["loss"])], \
        "v0.8 runs table hash != outputs hash for {} {} {} - aborting.".format(arch, arm, seed)
    V08_STATE_SHA[(arch, arm, seed)] = V08_OUTPUT_SHA[os.path.basename(_paths["state"])]
print("v0.8 outputs: {} files == v0.8 manifest; 18 state_dicts == runs[*].state_dict_sha256: PASS".format(len(V08_OUTPUT_SHA)))

# --- 11. the three development cells reused from v0.5 (read in A2 only): disk == FRP == v0.5 manifest ---
# >>> verbatim from v0.7 (reused v0.5 files: disk == FRP == v0.5 manifest)
REUSED_FILES = {}
for arch, arm, seed in REUSED_CELLS:
    for kind, p in v05_run_paths(arch, arm, seed, SELECTED_LR[arch]).items():
        name = os.path.basename(p)
        assert name in FRP["v0.7_reused_v0.5_files"], name + " is not listed in final_run_protocol - aborting."
        assert os.path.isfile(p), "reused v0.5 file missing: " + p
        got = sha256_of(p)
        assert got == FRP["v0.7_reused_v0.5_files"][name] == manifest_entry(V05_MAN, name)["sha256"], \
            "reused file hash mismatch (disk / protocol / v0.5 manifest) for " + name + " - aborting."
        REUSED_FILES[name] = {"path": p, "sha256": got}
assert set(REUSED_FILES) == set(FRP["v0.7_reused_v0.5_files"]), "reused-file set differs from the protocol - aborting."
print("v0.5_manifest.json    : {} | {} reused files: disk == final_run_protocol == v0.5 manifest -> PASS".format(
    V05_MAN_SHA, len(REUSED_FILES)))
# <<< verbatim from v0.7


# =====================================================================================
# STEP 0 (part 2) - DATA: folds 1-9 locked quantities (== FRP), folds 1-8 development objects,
#                   fold-10 METADATA (no signal, no label)
# =====================================================================================
hdr("STEP 0 (part 2) - DATA, LOCKED QUANTITIES, FOLD-10 METADATA (no fold-10 signal, no label)")

df_all = pd.read_csv(DB_CSV, index_col="ecg_id", usecols=DB_USECOLS)       # the ONLY read before FREEZE: no label column
assert set(df_all.columns) == set(DB_USECOLS) - {"ecg_id"} and df_all.index.name == "ecg_id"
assert not set(LABEL_COLUMNS_NEVER_READ_BEFORE_FREEZE) & set(df_all.columns)
N_SOURCE = int(len(df_all))
n_fold10_in_source = int((df_all["strat_fold"] == 10).sum())
assert n_fold10_in_source == EXPECTED_N_FOLD10_RECORDS, \
    "fold-10 rows in source CSV {} != {} - aborting.".format(n_fold10_in_source, EXPECTED_N_FOLD10_RECORDS)
fold10_meta = df_all.loc[df_all["strat_fold"] == 10, FOLD10_META_COLUMNS].copy()     # file order, whitelist only
df9 = df_all[df_all["strat_fold"] <= 9].copy()
del df_all
gc.collect()
FOLD10_ACCESS_LINE = (
    "fold-10 access: before FREEZE, pd.read_csv parsed ptbxl_database.csv once with usecols={} (the CSV tokenizer "
    "scans every byte of a row, but only these columns are converted and stored; no scp_codes, report or other "
    "diagnostic column enters memory); of the {} fold-10 rows, ecg_id (index), {} were copied into fold10_meta "
    "(strat_fold used only as the row filter) and the unrestricted frame was deleted. Fold-10 signals are read in "
    "Part B; fold-10 labels are first read in Part C, after the freeze file exists and verifies.".format(
        DB_USECOLS, n_fold10_in_source, ", ".join(FOLD10_META_COLUMNS)))
print(FOLD10_ACCESS_LINE)
# NOTE (v0.9): mask_train18 / mask_fold9 also select the folds 1-8 / fold-9 rows of the Part A development checks;
# the fold-9 label count assert moved to Part C (after FREEZE).
# >>> verbatim from v0.8 (folds 1-9 working set and masks)
assert set(df9["strat_fold"].unique()) == set(range(1, 10)), "working set is not exactly folds 1-9 - aborting."
assert (df9["strat_fold"] == 10).sum() == 0, "fold 10 leaked into the working set - aborting."
mask_train18 = df9["strat_fold"] <= 8          # folds 1-8: ONLY for the new_levels_vs_folds1to8 assert
mask_fold9 = df9["strat_fold"] == 9            # fold 9: ONLY for the fold-9 label COUNT assert (not a metric)
mask_final = df9["strat_fold"] <= 9            # folds 1-9: all True (final-run fit and training rows)
assert bool(mask_final.all())
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (sidecar index frame)
current_index = pd.DataFrame({
    "ecg_id": df9.index.to_numpy(),
    "strat_fold": df9["strat_fold"].to_numpy(),
    "patient_id": df9["patient_id"].to_numpy(),
})
N_REC = len(current_index)
assert N_REC == EXPECTED_N_RECORDS_FOLDS1TO9
# <<< verbatim from v0.8
# NOTE (v0.9): train_rows = all folds 1-9 rows = the rows the locked folds 1-9 lead normalization is fit on.
# >>> verbatim from v0.8 (folds 1-9 fit rows)
train_rows = np.arange(N_REC)                  # v0.8: ALL folds 1-9 rows are training rows
assert len(train_rows) == LQ["final_training_steps"]["n_records"]
# <<< verbatim from v0.8
print("records: folds 1-9 = {} | fold-10 metadata rows kept = {} (columns: {})".format(
    N_REC, len(fold10_meta), FOLD10_META_COLUMNS))

# --- physio columns (folds 1-9) ---
# >>> verbatim from v0.8 (physio columns)
age_raw = df9["age"]
age_sentinel_mask = (age_raw == 300)
n_age_sentinel = int(age_sentinel_mask.sum())
age_clean = age_raw.where(~age_sentinel_mask, np.nan)
age_censored = age_sentinel_mask.astype(float)
age_imputed = age_clean.fillna(90.0)
sex = df9["sex"].astype(float)
print("age: sentinel 300 affects", n_age_sentinel, "records; flagged via age_censored, imputed with 90.")
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (age sentinel count, physio order)
assert n_age_sentinel == EXPECTED_N_AGE_SENTINEL_FOLDS1TO9, \
    "age sentinel count {} != {} - aborting.".format(n_age_sentinel, EXPECTED_N_AGE_SENTINEL_FOLDS1TO9)
B1_cols = ["age_imputed", "age_censored", "sex"]
assert B1_cols == LQ["physio_columns"]["order"]
# <<< verbatim from v0.8

# --- categorical encoder refit on folds 1-9 (v0.6 ENC19 logic), asserted against FRP ---
# >>> verbatim from v0.8 (locked encoder spec)
LQ_ENC = LQ["categorical_encoder"]
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (level strings)
LEVELS = {col: to_level_string(df9[col]) for col in ACQ_COLS}
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (folds 1-9 encoder == FRP)
ENC19, CATS19, CATS18 = {}, {}, {}
for col in ACQ_COLS:
    ENC19[col], CATS19[col] = encode_categorical(LEVELS[col], mask_final, col)      # train_mask = all folds 1-9 rows
    _, CATS18[col] = encode_categorical(LEVELS[col], mask_train18, col)            # categories only (new-levels assert)
WIDTHS19 = {col: ENC19[col].shape[1] for col in ACQ_COLS}
NEW_LEVELS = {col: sorted(set(CATS19[col]) - set(CATS18[col])) for col in ACQ_COLS}
DROPPED_LEVELS = {col: sorted(set(CATS18[col]) - set(CATS19[col])) for col in ACQ_COLS}
del CATS18
for col in ACQ_COLS:
    assert CATS19[col] == LQ_ENC["categories"][col], \
        "{} categories differ from final_run_protocol (element by element) - aborting.".format(col)
assert WIDTHS19 == LQ_ENC["widths"] == EXPECTED_ONEHOT_WIDTHS_FINAL, "one-hot widths {} - aborting.".format(WIDTHS19)
assert NEW_LEVELS == LQ_ENC["new_levels_vs_folds1to8"] == EXPECTED_NEW_LEVELS, "new levels {} - aborting.".format(NEW_LEVELS)
assert all(len(v) == 0 for v in DROPPED_LEVELS.values()), "a folds 1-8 level vanished in folds 1-9 - aborting."
B3_COLS_FINAL = B1_cols + [c for col in ACQ_COLS for c in ENC19[col].columns]
assert B3_COLS_FINAL == LQ_ENC["signal_plus_all_column_order"], "signal_plus_all column order differs from FRP - aborting."
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (all-zero training columns, folds 1-9 fit)
ALLZERO_TRAIN_COLS_19 = sorted(c for col in ACQ_COLS for c in ENC19[col].columns
                               if int(ENC19[col][c].sum()) == 0)
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (all-zero columns == FRP)
assert ALLZERO_TRAIN_COLS_19 == LQ_ENC["allzero_training_columns_folds1to9"], \
    "all-zero training columns {} != FRP - aborting.".format(ALLZERO_TRAIN_COLS_19)
print("encoder (folds 1-9): widths {} | new levels vs folds 1-8 {} | all-zero training columns {} | "
      "categories + column order == final_run_protocol: PASS".format(WIDTHS19, NEW_LEVELS, ALLZERO_TRAIN_COLS_19))
# <<< verbatim from v0.8

# --- age_imputed scaler refit on folds 1-9, asserted against FRP (values read FROM FRP) ---
# >>> verbatim from v0.8 (folds 1-9 age scaler == FRP)
LQ_AGE = LQ["age_scaler"]
age_scaler = StandardScaler().fit(age_imputed[mask_final].to_numpy().reshape(-1, 1))
AGE_SCALER_DIFFS = {"mean": abs(float(age_scaler.mean_[0]) - float(LQ_AGE["mean"])),
                    "scale": abs(float(age_scaler.scale_[0]) - float(LQ_AGE["scale"])),
                    "var": abs(float(age_scaler.var_[0]) - float(LQ_AGE["var"]))}
for _k, _tol in AGE_SCALER_TOL.items():
    assert AGE_SCALER_DIFFS[_k] <= _tol, "age scaler {}: |diff| {} > {} - aborting.".format(_k, AGE_SCALER_DIFFS[_k], _tol)
assert int(age_scaler.n_samples_seen_) == int(LQ_AGE["n_samples_seen"]) == N_REC
age_imputed_std = pd.Series(age_scaler.transform(age_imputed.to_numpy().reshape(-1, 1)).ravel(),
                            index=df9.index, name="age_imputed")
print("age_imputed scaler (folds 1-9): mean={!r} scale={!r} | max abs diff vs FRP {} | n_samples_seen {}: PASS".format(
    float(age_scaler.mean_[0]), float(age_scaler.scale_[0]), AGE_SCALER_DIFFS, int(age_scaler.n_samples_seen_)))
# <<< verbatim from v0.8

# --- META matrices of folds 1-9 (v0.8 construction; kept for the locked-quantity checks only) ---
# >>> verbatim from v0.8 (folds 1-9 META)
META_FEATURES = pd.concat([age_imputed_std, age_censored.rename("age_censored"), sex.rename("sex"),
                           ENC19["device"], ENC19["site"], ENC19["nurse"]], axis=1)
ARM_COLUMNS = {"signal_only": [], "signal_plus_physio": B1_cols, "signal_plus_all": B3_COLS_FINAL}
assert EXPECTED_META_DIM == LQ["meta_dim"] and EXPECTED_HEAD_IN == LQ["head_first_linear_in_features"]
META = {}
for arm in ARMS:
    cols = ARM_COLUMNS[arm]
    if cols:
        META[arm] = META_FEATURES[cols].to_numpy(dtype=np.float32)
    else:
        META[arm] = np.zeros((N_REC, 0), dtype=np.float32)
    assert META[arm].dtype == np.float32 and META[arm].shape == (N_REC, EXPECTED_META_DIM[arm]), \
        "{}: metadata shape {} != {}".format(arm, META[arm].shape, (N_REC, EXPECTED_META_DIM[arm]))
    assert np.isfinite(META[arm]).all()
    assert EMBEDDING_DIM + EXPECTED_META_DIM[arm] == EXPECTED_HEAD_IN[arm]
    print("  {:20s} metadata dim = {:3d} | head first-Linear in_features {}".format(arm, META[arm].shape[1], EXPECTED_HEAD_IN[arm]))
assert list(META_FEATURES[ARM_COLUMNS["signal_plus_all"]].columns) == LQ_ENC["signal_plus_all_column_order"]
# indices (in hidden[0].weight columns) of the all-zero training columns, BY NAME from the locked column order
ZERO_COL_NAMES = list(LQ_ENC["allzero_training_columns_folds1to9"])
ZERO_COL_IDX = [EMBEDDING_DIM + LQ_ENC["signal_plus_all_column_order"].index(c) for c in ZERO_COL_NAMES]
assert all(float(np.abs(META["signal_plus_all"][:, i - EMBEDDING_DIM]).max()) == 0.0 for i in ZERO_COL_IDX)
print("all-zero columns {} -> hidden[0].weight column indices {} (by name, offset {})".format(
    ZERO_COL_NAMES, ZERO_COL_IDX, EMBEDDING_DIM))
# <<< verbatim from v0.8

# --- signal cache (v0.4), folds 1-9 only ---
# >>> verbatim from v0.8 (signal cache + sidecar index)
for path, expected in [(SIGNALS_NPY, EXPECTED_SIGNALS_NPY_SHA256), (SIGNALS_INDEX, EXPECTED_SIGNALS_INDEX_SHA256)]:
    assert os.path.isfile(path), "v0.4 cache file missing: " + path
    got = sha256_of(path)
    print("  {}  {}  -> {}".format(got, os.path.basename(path), "PASS" if got == expected else "FAIL"))
    assert got == expected, "v0.4 cache hash mismatch for " + path + " - aborting."
_old_idx = pd.read_csv(SIGNALS_INDEX, encoding="utf-8-sig")
assert list(_old_idx.columns) == list(current_index.columns) and _old_idx.astype(str).reset_index(drop=True).equals(
    current_index.astype(str).reset_index(drop=True)), "v0.4 sidecar index does not match folds 1-9 - aborting."
X = np.load(SIGNALS_NPY)
assert X.shape == (N_REC, N_LEADS, N_SAMPLES) and X.dtype == np.float32
print("signals loaded from the v0.4 cache: {} {} (folds 1-9 only)".format(X.shape, X.dtype))
# <<< verbatim from v0.8


# --- per-lead normalization refit on folds 1-9 (v0.6 code), asserted against FRP; FRP values are USED ---
# >>> verbatim from v0.8 (lead_norm_two_pass)
def lead_norm_two_pass(X, rows, chunk=CHUNK):
    """Per-lead mean / population std over the given rows; float64 two-pass in chunks of
    `chunk` records in row order (v0.4 Part B code, unchanged)."""
    _sum = np.zeros(N_LEADS, dtype=np.float64)
    for s in range(0, len(rows), chunk):
        _sum += X[rows[s:s + chunk]].astype(np.float64).sum(axis=(0, 2))
    _n_vals = len(rows) * N_SAMPLES
    mean = _sum / _n_vals
    _sq = np.zeros(N_LEADS, dtype=np.float64)
    for s in range(0, len(rows), chunk):
        xc = X[rows[s:s + chunk]].astype(np.float64) - mean[None, :, None]
        _sq += (xc ** 2).sum(axis=(0, 2))
    std = np.sqrt(_sq / _n_vals)
    return mean, std
# <<< verbatim from v0.8


# NOTE (v0.9): the two FRP float32 globals below are the ones passed to predict_records for fold 10 (there is no
# RandomCropDataset in v0.9); the identity guard asserts them bitwise against FRP before opening and before FREEZE.
# >>> verbatim from v0.8 (folds 1-9 lead normalization == FRP; FRP values used)
LQ_LN = LQ["lead_normalization"]
t0 = time.time()
LEAD_MEAN_19, LEAD_STD_19 = lead_norm_two_pass(X, train_rows)
FRP_LEAD_MEAN = np.asarray(LQ_LN["mean"], dtype=np.float64)
FRP_LEAD_STD = np.asarray(LQ_LN["std"], dtype=np.float64)
LEAD_NORM_MAX_ABS_DIFF = {"mean": float(np.max(np.abs(LEAD_MEAN_19 - FRP_LEAD_MEAN))),
                          "std": float(np.max(np.abs(LEAD_STD_19 - FRP_LEAD_STD)))}
assert np.allclose(LEAD_MEAN_19, FRP_LEAD_MEAN, rtol=0, atol=LEAD_NORM_ATOL) and \
    np.allclose(LEAD_STD_19, FRP_LEAD_STD, rtol=0, atol=LEAD_NORM_ATOL), \
    "folds 1-9 lead normalization differs from final_run_protocol (atol {}): {} - aborting.".format(
        LEAD_NORM_ATOL, LEAD_NORM_MAX_ABS_DIFF)
_v04_lead_names = pd.read_csv(V04_LEAD_NORM_CSV, encoding="utf-8-sig")["lead"].tolist()   # lead order only
assert LQ_LN["lead_order"] == _v04_lead_names and len(_v04_lead_names) == N_LEADS, "lead order differs - aborting."
assert np.all(FRP_LEAD_STD > 0) and np.all(np.isfinite(FRP_LEAD_MEAN))
# the model uses the FRP LOCKED values (RandomCropDataset reads these two globals)
LEAD_MEAN_F32 = np.asarray(LQ_LN["mean"], dtype=np.float64).astype(np.float32)[:, None]
LEAD_STD_F32 = np.asarray(LQ_LN["std"], dtype=np.float64).astype(np.float32)[:, None]
print("per-lead normalization (folds 1-9, {} records, {:.1f}s) == final_run_protocol (atol {}; max abs diff {}) | "
      "lead order == FRP == v0.4 csv: PASS | the model uses the FRP locked values".format(
          N_REC, time.time() - t0, LEAD_NORM_ATOL, LEAD_NORM_MAX_ABS_DIFF))
# <<< verbatim from v0.8

LOCKED_CHECKS = {
    "categorical_encoder": {"categories_equal_frp": True, "widths": WIDTHS19, "new_levels_vs_folds1to8": NEW_LEVELS,
                            "signal_plus_all_column_order_equal_frp": True,
                            "allzero_training_columns_folds1to9": ALLZERO_TRAIN_COLS_19},
    "meta_dim": {arm: int(META[arm].shape[1]) for arm in ARMS},
    "head_first_linear_in_features": EXPECTED_HEAD_IN,
    "age_scaler": {"mean": float(age_scaler.mean_[0]), "scale": float(age_scaler.scale_[0]),
                   "var": float(age_scaler.var_[0]), "n_samples_seen": int(age_scaler.n_samples_seen_),
                   "max_abs_diff_vs_frp": AGE_SCALER_DIFFS, "tolerances": AGE_SCALER_TOL},
    "lead_normalization": {"max_abs_diff_vs_frp": LEAD_NORM_MAX_ABS_DIFF, "atol": LEAD_NORM_ATOL,
                           "values_used": "FRP locked_quantities.lead_normalization (float64 -> float32)"},
    "age_sentinel_count_folds1to9": n_age_sentinel,
    "fold10_rows_in_source_csv": n_fold10_in_source,
}

# --- folds 1-8 DEVELOPMENT objects for Part A only (suffix 18; every one is deleted at the end of Part A) ---
# >>> adapted from v0.7 (folds 1-8 rows, fold-9 rows) | renames: train_rows->train_rows18, mask_train->mask_train18, mask_eval->mask_fold9
train_rows18 = np.where(mask_train18.to_numpy())[0]
eval_rows = np.where(mask_fold9.to_numpy())[0]
# <<< adapted from v0.7
# >>> adapted from v0.7 (folds 1-8 encoder, age scaler, META) | renames: device_level->device_level18, site_level->site_level18, nurse_level->nurse_level18, device_dummies->device_dummies18, site_dummies->site_dummies18, nurse_dummies->nurse_dummies18, device_categories->device_categories18, site_categories->site_categories18, nurse_categories->nurse_categories18, mask_train->mask_train18, EXPECTED_ONEHOT_COUNTS->EXPECTED_ONEHOT_COUNTS18, age_scaler->age_scaler18, age_imputed_std->age_imputed_std18, B2_cols->B2_cols18, B3_cols->B3_cols18, META_FEATURES->META_FEATURES18, ARM_COLUMNS->ARM_COLUMNS18, META->META18, EXPECTED_META_DIM->EXPECTED_META_DIM18, AGE_SCALER_TOL->AGE_SCALER_TOL18
device_level18 = to_level_string(df9["device"])
site_level18 = to_level_string(df9["site"])
nurse_level18 = to_level_string(df9["nurse"])
device_dummies18, device_categories18 = encode_categorical(device_level18, mask_train18, "device")
site_dummies18, site_categories18 = encode_categorical(site_level18, mask_train18, "site")
nurse_dummies18, nurse_categories18 = encode_categorical(nurse_level18, mask_train18, "nurse")
for name, dmy in [("device", device_dummies18), ("site", site_dummies18), ("nurse", nurse_dummies18)]:
    assert dmy.shape[1] == EXPECTED_ONEHOT_COUNTS18[name], \
        "{} one-hot width {} != v0.2.1's {}".format(name, dmy.shape[1], EXPECTED_ONEHOT_COUNTS18[name])

age_scaler18 = StandardScaler().fit(age_imputed[mask_train18].to_numpy().reshape(-1, 1))
age_imputed_std18 = pd.Series(age_scaler18.transform(age_imputed.to_numpy().reshape(-1, 1)).ravel(),
                            index=df9.index, name="age_imputed")

B1_cols = ["age_imputed", "age_censored", "sex"]
B2_cols18 = list(device_dummies18.columns) + list(site_dummies18.columns) + list(nurse_dummies18.columns)
B3_cols18 = B1_cols + B2_cols18

META_FEATURES18 = pd.concat([age_imputed_std18, age_censored.rename("age_censored"), sex.rename("sex"),
                           device_dummies18, site_dummies18, nurse_dummies18], axis=1)
ARM_COLUMNS18 = {"signal_only": [], "signal_plus_physio": B1_cols, "signal_plus_all": B3_cols18}
META18 = {}
for arm in ARMS:
    cols = ARM_COLUMNS18[arm]
    if cols:
        META18[arm] = META_FEATURES18[cols].to_numpy(dtype=np.float32)
    else:
        META18[arm] = np.zeros((N_REC, 0), dtype=np.float32)
    assert META18[arm].shape == (N_REC, EXPECTED_META_DIM18[arm]), \
        "{}: metadata dim {} != {}".format(arm, META18[arm].shape[1], EXPECTED_META_DIM18[arm])
    assert np.isfinite(META18[arm]).all()
    print("  {:20s} metadata dim = {:3d}{}".format(arm, META18[arm].shape[1],
          "" if not cols else "  ({} ... {})".format(", ".join(cols[:3]), cols[-1])))
assert ARM_COLUMNS18["signal_plus_physio"] == B1_cols and ARM_COLUMNS18["signal_plus_all"] == B1_cols + B2_cols18
print("column order: physio == v0.2.1 B1_physio, all == v0.2.1 B3_physio_plus_acquisition: PASS")
print("age_imputed scaler (folds 1-8): mean={:.4f}, scale={:.4f}".format(
    float(age_scaler18.mean_[0]), float(age_scaler18.scale_[0])))
assert abs(float(age_scaler18.mean_[0]) - V05_AGE_SCALER_FOLDS1TO8["mean"]) <= AGE_SCALER_TOL18 and \
    abs(float(age_scaler18.scale_[0]) - V05_AGE_SCALER_FOLDS1TO8["scale"]) <= AGE_SCALER_TOL18, \
    "folds 1-8 age scaler differs from v0.5 - aborting."
assert FRP["locked_quantities"]["meta_dim_development_folds1to8"] == EXPECTED_META_DIM18, \
    "final_run_protocol meta_dim_development_folds1to8 != {} - aborting.".format(EXPECTED_META_DIM18)
print("age scaler == v0.5 (tol {}) and meta dims == final_run_protocol meta_dim_development_folds1to8: PASS".format(
    AGE_SCALER_TOL18))
# <<< adapted from v0.7
CATS18 = {"device": device_categories18, "site": site_categories18, "nurse": nurse_categories18}
DUMMIES18 = {"device": device_dummies18, "site": site_dummies18, "nurse": nurse_dummies18}
assert all(sorted(set(CATS19[c]) - set(CATS18[c])) == NEW_LEVELS[c] for c in ACQ_COLS)
assert {arm: EMBEDDING_DIM + d for arm, d in EXPECTED_META_DIM18.items()} == EXPECTED_HEAD_IN18
# >>> adapted from v0.6 (folds 1-8 lead normalization == v0.4_lead_norm_params.csv) | renames: train_rows->train_rows18
LEAD_MEAN_18, LEAD_STD_18 = lead_norm_two_pass(X, train_rows18)
_v04_norm = pd.read_csv(V04_LEAD_NORM_CSV, encoding="utf-8-sig")
assert np.allclose(_v04_norm["mean"].to_numpy(), LEAD_MEAN_18, rtol=0, atol=1e-12) and \
    np.allclose(_v04_norm["std"].to_numpy(), LEAD_STD_18, rtol=0, atol=1e-12), \
    "folds 1-8 lead normalization differs from v0.4_lead_norm_params.csv - aborting."
# <<< adapted from v0.6
# >>> adapted from v0.7 (float32 copies) | renames: LEAD_MEAN_F32->LEAD_MEAN_F32_18, LEAD_STD_F32->LEAD_STD_F32_18, LEAD_MEAN->LEAD_MEAN_18, LEAD_STD->LEAD_STD_18
LEAD_MEAN_F32_18 = LEAD_MEAN_18.astype(np.float32)[:, None]
LEAD_STD_F32_18 = LEAD_STD_18.astype(np.float32)[:, None]
# <<< adapted from v0.7
print("folds 1-8 development objects (Part A only): one-hot widths {} | META18 dims {} | lead normalization == "
      "v0.4_lead_norm_params.csv (atol 1e-12): PASS".format(
          {c: len(CATS18[c]) for c in ACQ_COLS}, {arm: META18[arm].shape[1] for arm in ARMS}))

# --- fold-10 METADATA (v0.6 code): ids, patients, disjointness, level mapping counts ---
# >>> verbatim from v0.6 (fold-10 ids, patients, level mapping counts)
fold10_pid = fold10_meta["patient_id"].to_numpy()
assert np.all(fold10_pid == np.round(fold10_pid)), "non-integral fold-10 patient_id"
fold10_pid = fold10_pid.astype(np.int64)
fold10_ecg = fold10_meta.index.to_numpy().astype(np.int64)
N_F10 = len(fold10_meta)
N_F10_PATIENTS = int(len(np.unique(fold10_pid)))
F10_DISJOINT = set(fold10_pid.tolist()).isdisjoint(set(df9["patient_id"].astype(np.int64).tolist()))
assert N_F10 == EXPECTED_N_FOLD10_RECORDS, "fold-10 records {} != 2198 - aborting.".format(N_F10)
assert N_F10_PATIENTS == EXPECTED_N_FOLD10_PATIENTS, "fold-10 patients {} != 1904 - aborting.".format(N_F10_PATIENTS)
assert F10_DISJOINT, "a fold-10 patient appears in folds 1-9 - aborting."

F10_COUNTS = {"n_unknown": {}, "n_on_untrained_column": {}}
_any_unknown = np.zeros(N_F10, dtype=bool)
_any_untrained = np.zeros(N_F10, dtype=bool)
for col in ACQ_COLS:
    joint = to_level_string(pd.concat([df9[col], fold10_meta[col]]))
    assert (joint.iloc[:N_REC].to_numpy() == LEVELS[col].to_numpy()).all(), \
        "{}: level strings of folds 1-9 change when fold-10 values are present (zero-pad width) - aborting.".format(col)
    lvl10 = joint.iloc[N_REC:]
    mapped10 = remap_with_categories(lvl10, CATS19[col])
    is_unknown = (mapped10 == "__UNKNOWN__").to_numpy()
    is_untrained = ("{}=".format(col) + mapped10).isin(ALLZERO_TRAIN_COLS_19).to_numpy()
    F10_COUNTS["n_unknown"][col] = int(is_unknown.sum())
    F10_COUNTS["n_on_untrained_column"][col] = int(is_untrained.sum())
    _any_unknown |= is_unknown
    _any_untrained |= is_untrained
F10_COUNTS["n_unknown"]["any"] = int(_any_unknown.sum())
F10_COUNTS["n_on_untrained_column"]["any"] = int(_any_untrained.sum())
del _any_unknown, _any_untrained, lvl10, mapped10, is_unknown, is_untrained, joint
print("fold 10: n records {} | n patients {} | disjoint from folds 1-9 patients: {}".format(
    N_F10, N_F10_PATIENTS, "PASS" if F10_DISJOINT else "FAIL"))
print("fold 10 under the folds 1-9 encoder: __UNKNOWN__ counts {} | records on any all-zero training column {}".format(
    F10_COUNTS["n_unknown"], F10_COUNTS["n_on_untrained_column"]))
assert F10_COUNTS["n_on_untrained_column"]["any"] == EXPECTED_FOLD10_UNTRAINED_COUNT and \
    F10_COUNTS["n_unknown"]["any"] == 0, \
    "CROSS-CHECK FAILED: fold-10 records map to __UNKNOWN__ / untrained columns - STOP and report."
# <<< verbatim from v0.6
assert N_F10 == LQ["fold10"]["n_records"] and N_F10_PATIENTS == LQ["fold10"]["n_patients"] and \
    F10_DISJOINT == LQ["fold10"]["patient_disjoint_from_folds1to9"], "fold-10 counts differ from FRP - aborting."
assert F10_COUNTS["n_unknown"] == LQ["fold10"]["n_unknown_under_folds1to9_encoder"] and \
    F10_COUNTS["n_on_untrained_column"] == LQ["fold10"]["n_on_untrained_column_under_folds1to9_encoder"], \
    "fold-10 __UNKNOWN__ / untrained-column counts differ from FRP locked_quantities.fold10 - aborting."

# >>> verbatim from v0.6 (fold-10 index matrices and lock hashes)
# --- fold-10 bootstrap and permutation index matrices (patient_id / record count only) ---
t0 = time.time()
F10_PATIENTS, F10_ROWS = REF["patient_rows_in_file_order"](fold10_pid)
assert len(F10_PATIENTS) == N_F10_PATIENTS and sum(len(r) for r in F10_ROWS) == N_F10
BOOT_IDX = REF["build_boot_idx"](N_F10_PATIENTS, N_BOOT, BOOT_SEED)
PERM_IDX = REF["build_perm_idx"](N_F10, N_PERM, PERM_SEED)
assert BOOT_IDX.shape == (N_BOOT, N_F10_PATIENTS) and PERM_IDX.shape == (N_PERM, N_F10)
F10_IDS = np.ascontiguousarray(np.stack([fold10_ecg, fold10_pid], axis=1).astype("<i8"))
LOCK_HASHES = {
    "fold10_ids_file_order_sha256": sha256_bytes(F10_IDS.tobytes(order="C")),
    "fold10_patients_sorted_sha256": sha256_bytes(np.ascontiguousarray(F10_PATIENTS.astype("<i8")).tobytes()),
    "boot_idx_sha256": sha256_bytes(np.ascontiguousarray(BOOT_IDX.astype("<i8")).tobytes()),
    "perm_idx_sha256": sha256_bytes(np.ascontiguousarray(PERM_IDX.astype("<i8")).tobytes()),
}
# <<< verbatim from v0.6
for _k, _v in LOCK_HASHES.items():
    assert _v == LOCKED_HASHES_FRP[_k], "LOCK {}: {} != FRP {} - aborting.".format(_k, _v, LOCKED_HASHES_FRP[_k])
print("fold-10 locks == FRP bootstrap.locked_hashes: {} | BOOT_IDX {} | PERM_IDX {}: PASS".format(
    sorted(LOCK_HASHES), BOOT_IDX.shape, PERM_IDX.shape))

# --- fold-10 level mapping (FRP fold10_level_mapping) and META10 ---
ENC10 = {}
for col in ACQ_COLS:
    _joint10 = to_level_string(pd.concat([df9[col], fold10_meta[col]]))
    assert (_joint10.iloc[:N_REC].to_numpy() == LEVELS[col].to_numpy()).all(), "{}: folds 1-9 level strings changed".format(col)
    ENC10[col] = encode_with_categories(_joint10.iloc[N_REC:], CATS19[col], col)
    assert list(ENC10[col].columns) == ["{}={}".format(col, c) for c in CATS19[col]]
    assert ENC10[col].index.equals(fold10_meta.index) and (ENC10[col].to_numpy().sum(axis=1) == 1).all()
assert [c for col in ACQ_COLS for c in ENC10[col].columns] == LQ_ENC["signal_plus_all_column_order"][len(B1_cols):], \
    "fold-10 one-hot columns != FRP signal_plus_all_column_order block - aborting."
assert int(sum(ENC10[col]["{}=__UNKNOWN__".format(col)].sum() for col in ACQ_COLS)) == 0
print("fold 10:")
# >>> adapted from v0.8 (physio columns) | renames: age_raw->age_raw10, age_sentinel_mask->age_sentinel_mask10, n_age_sentinel->n_age_sentinel10, age_clean->age_clean10, age_censored->age_censored10, age_imputed->age_imputed10, sex->sex10, df9->fold10_meta
age_raw10 = fold10_meta["age"]
age_sentinel_mask10 = (age_raw10 == 300)
n_age_sentinel10 = int(age_sentinel_mask10.sum())
age_clean10 = age_raw10.where(~age_sentinel_mask10, np.nan)
age_censored10 = age_sentinel_mask10.astype(float)
age_imputed10 = age_clean10.fillna(90.0)
sex10 = fold10_meta["sex"].astype(float)
print("age: sentinel 300 affects", n_age_sentinel10, "records; flagged via age_censored, imputed with 90.")
# <<< adapted from v0.8
# NOTE (v0.9): age_scaler is the folds 1-9 scaler asserted == FRP above (identity guard re-checks it).
# >>> adapted from v0.8 (age_imputed standardized with the locked scaler) | renames: age_imputed_std->age_imputed_std10, age_imputed->age_imputed10, df9->fold10_meta
age_imputed_std10 = pd.Series(age_scaler.transform(age_imputed10.to_numpy().reshape(-1, 1)).ravel(),
                            index=fold10_meta.index, name="age_imputed")
# <<< adapted from v0.8
# >>> adapted from v0.8 (META feature frame) | renames: META_FEATURES->META_FEATURES10, age_imputed_std->age_imputed_std10, age_censored->age_censored10, sex->sex10, ENC19->ENC10
META_FEATURES10 = pd.concat([age_imputed_std10, age_censored10.rename("age_censored"), sex10.rename("sex"),
                           ENC10["device"], ENC10["site"], ENC10["nurse"]], axis=1)
# <<< adapted from v0.8
# >>> adapted from v0.8 (META matrices) | renames: META->META10, META_FEATURES->META_FEATURES10, N_REC->N_F10
META10 = {}
for arm in ARMS:
    cols = ARM_COLUMNS[arm]
    if cols:
        META10[arm] = META_FEATURES10[cols].to_numpy(dtype=np.float32)
    else:
        META10[arm] = np.zeros((N_F10, 0), dtype=np.float32)
    assert META10[arm].dtype == np.float32 and META10[arm].shape == (N_F10, EXPECTED_META_DIM[arm]), \
        "{}: metadata shape {} != {}".format(arm, META10[arm].shape, (N_F10, EXPECTED_META_DIM[arm]))
    assert np.isfinite(META10[arm]).all()
    assert EMBEDDING_DIM + EXPECTED_META_DIM[arm] == EXPECTED_HEAD_IN[arm]
    print("  {:20s} metadata dim = {:3d} | head first-Linear in_features {}".format(arm, META10[arm].shape[1], EXPECTED_HEAD_IN[arm]))
# <<< adapted from v0.8
META10_COLUMNS = list(META_FEATURES10[ARM_COLUMNS["signal_plus_all"]].columns)
assert META10_COLUMNS == LQ_ENC["signal_plus_all_column_order"], "META10 column order != FRP - aborting."
assert list(META_FEATURES10[ARM_COLUMNS["signal_plus_physio"]].columns) == B1_cols == LQ["physio_columns"]["order"]
META10_SHA256 = {arm: sha256_f32(META10[arm]) for arm in ARMS}
PHYSIO_IDX = [META10_COLUMNS.index(c) for c in B1_cols]
BLOCK_IDX = [i for i, c in enumerate(META10_COLUMNS) if "=" in c and c.split("=", 1)[0] in ACQ_COLS]
assert PHYSIO_IDX == [0, 1, 2] and len(BLOCK_IDX) == sum(WIDTHS19.values()) and \
    sorted(PHYSIO_IDX + BLOCK_IDX) == list(range(len(META10_COLUMNS)))
FILES9 = df9["filename_lr"].tolist()
FILES10 = fold10_meta["filename_lr"].tolist()
assert len(FILES9) == N_REC and len(FILES10) == N_F10 == len(set(FILES10))
assert set(FILES9).isdisjoint(FILES10), "a fold-10 record file is also a folds 1-9 record file - aborting."
ROWS10 = np.arange(N_F10)
print("META10: shapes {} | column order == FRP | permutation block = {} columns (device + site + nurse, by name), "
      "physio columns {} stay fixed".format({arm: META10[arm].shape for arm in ARMS}, len(BLOCK_IDX), B1_cols))
if STATE == "post_freeze":
    assert FROZEN["lock_hashes"] == LOCK_HASHES, "lock hashes differ from the freeze file - aborting."
    assert FROZEN["meta10_sha256"] == META10_SHA256 and FROZEN["meta10_columns"] == META10_COLUMNS, \
        "META10 differs from the freeze file - aborting."
    print("post-freeze: lock hashes and META10 recomputed == freeze file: PASS")


def identity_guard(where):
    """The objects used for fold 10 are the FRP-locked folds 1-9 objects (supervisor answer 3)."""
    _exp_mean = np.asarray(LQ_LN["mean"], dtype=np.float64).astype(np.float32)[:, None]
    _exp_std = np.asarray(LQ_LN["std"], dtype=np.float64).astype(np.float32)[:, None]
    _age_frp = ((age_imputed10.to_numpy(dtype=np.float64) - float(LQ_AGE["mean"])) / float(LQ_AGE["scale"])).astype(np.float32)
    checks = {
        "LEAD_MEAN_F32 bitwise == float32(FRP mean)": bool(
            LEAD_MEAN_F32.dtype == np.float32 and LEAD_MEAN_F32.shape == _exp_mean.shape
            and LEAD_MEAN_F32.tobytes() == _exp_mean.tobytes()),
        "LEAD_STD_F32 bitwise == float32(FRP std)": bool(
            LEAD_STD_F32.dtype == np.float32 and LEAD_STD_F32.shape == _exp_std.shape
            and LEAD_STD_F32.tobytes() == _exp_std.tobytes()),
        "age_scaler mean_ / scale_ / n_samples_seen_ == FRP": bool(
            float(age_scaler.mean_[0]) == float(LQ_AGE["mean"]) and float(age_scaler.scale_[0]) == float(LQ_AGE["scale"])
            and int(age_scaler.n_samples_seen_) == int(LQ_AGE["n_samples_seen"])),
        "CATS19 == FRP categories": bool(all(CATS19[c] == LQ_ENC["categories"][c] for c in ACQ_COLS)),
        "META10 column order == FRP": bool(META10_COLUMNS == LQ_ENC["signal_plus_all_column_order"]),
        "META10 unchanged since STEP 0": bool({arm: sha256_f32(META10[arm]) for arm in ARMS} == META10_SHA256),
        "META10 age_imputed == (age - FRP mean) / FRP scale": bool(
            np.array_equal(META10["signal_plus_all"][:, 0], _age_frp)
            and np.array_equal(META10["signal_plus_physio"][:, 0], _age_frp)),
        "no folds 1-8 object (*18) exists": not [n for n in globals() if n.endswith("18")],
    }
    failed = [k for k, v in checks.items() if not v]
    assert not failed, "IDENTITY GUARD ({}) FAILED: {} - aborting.".format(where, failed)
    print("identity guard ({}): {} checks PASS".format(where, len(checks)))
    return {"where": where, "checks": checks}


# =====================================================================================
# GPU SETUP, MODELS (verbatim), INFERENCE FUNCTION, MODEL LOADING
# =====================================================================================
hdr("GPU SETUP AND MODELS")
# >>> verbatim from v0.8 (device)
DEVICE = torch.device("cuda:0")
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (seeding + cuDNN flags)
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_all_seeds(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (gpu line)
print("gpu: {} | torch {} | CUDA {} | cuDNN {} | cudnn.deterministic=True, benchmark=False".format(
    GPU_NAME, torch.__version__, torch.version.cuda, CUDNN_VERSION))
# <<< verbatim from v0.8


# >>> verbatim from v0.8 (MODELS)
# >>> MODELS
# SOURCE FOLLOWED: Strodthoff, Wagner, Schaeffter, Samek (2021) "Deep Learning for ECG
# Analysis: Benchmarks and Insights from PTB-XL", reference code repository
# helme/ecg_ptbxl_benchmarking: code/models/resnet1d.py (resnet1d_wang, BasicBlock1d,
# ResNet1d), code/models/inception1d.py (Inception1d, InceptionBlock1d, Shortcut1d),
# code/models/basic_conv1d.py (create_head1d, bn_drop_lin, AdaptiveConcatPool1d), and
# code/models/fastai_model.py (how the benchmark instantiates them). Re-implemented here
# from memory of that code in plain PyTorch (no fastai); the repository was NOT
# consulted at write time.
#
# DETAILS I AM UNSURE OF (should be checked against the repository):
#  1) resnet1d_wang kernel sizes: factory default kernel_size=[5,3], stem kernel 7,
#     stride_stem=1, no stem pooling, inplanes=128, layers [1,1,1], fix_feature_dim=True
#     (all stages 128 channels), stride 2 in stages 2 and 3. If fastai_model.py passes
#     the int kernel_size=5, BasicBlock1d expands it to [5, 5//2+1] = [5,3] -> same.
#  2) inception1d as used by the benchmark: kernel_size = 8*5 = 40 -> branch kernels
#     [39,19,9], depth 6, bottleneck 32, nb_filters 32 (output 4*32 = 128 channels),
#     residual shortcut every 3 blocks; all convs bias=False.
#  3) Head (create_head1d) with lin_ftrs_head=[128], ps_head=0.5, bn=True, relu,
#     concat pooling: AdaptiveConcatPool1d [max, avg] -> 2*nf -> BN, Dropout(0.25),
#     Linear(2*nf,128), ReLU -> BN, Dropout(0.5), Linear(128, 5). I believe the
#     benchmark default is lin_ftrs_head=[128]; with lin_ftrs_head=None the head would be
#     a single Linear(2*nf, 5) instead.
#  4) Order inside AdaptiveConcatPool1d is [max, avg] (fastai convention).
#  5) The reference aggregates sliding-window predictions with aggregate_fn="max" by
#     default in fastai_model.py (I am fairly but not fully sure); the project's dev
#     protocol uses the MEAN over windows (Part F). Flagged, not changed.
#  6) MaxPool1d(3, stride, padding=1) on the inception max-pool branch.
#
# METADATA FUSION (v0.4 PATCH FIX 1 -- sealed text is binding: "concatenated to the
# network's global-pooled embedding before the final linear classification head").
# The global-pooled embedding is the 2*nf = 256-dim AdaptiveConcatPool1d output; the
# "head" is the whole MLP head. The head's first block keeps the reference order
# (BN, Dropout(0.25), Linear, ReLU), split as: BN + Dropout applied to the embedding
# (`pool_norm`, identical across arms), then the RAW metadata (no BN, no dropout) is
# concatenated, then Linear(256 + meta_dim, 128) -> ReLU -> BN -> Dropout(0.5) ->
# Linear(128, 5). Backbone, pooling and pool_norm are identical across arms; the head's
# first Linear differs in input width by design.


def conv1d_same(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=(3, 3), downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]
        self.conv1 = conv1d_same(inplanes, planes, kernel_size=kernel_size[0], stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d_same(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


def resnet1d_wang_backbone(input_channels=12, inplanes=128, kernel_size=(5, 3), kernel_size_stem=7,
                           stride_stem=1, layers=(1, 1, 1), stride=2):
    mods = [nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem,
                      padding=(kernel_size_stem - 1) // 2, bias=False),
            nn.BatchNorm1d(inplanes), nn.ReLU(inplace=True)]          # no stem pooling
    cur = inplanes
    for i, n_blocks in enumerate(layers):
        st = 1 if i == 0 else stride
        planes = inplanes                                             # fix_feature_dim=True
        downsample = None
        if st != 1 or cur != planes:
            downsample = nn.Sequential(nn.Conv1d(cur, planes, kernel_size=1, stride=st, bias=False),
                                       nn.BatchNorm1d(planes))
        blocks = [BasicBlock1d(cur, planes, st, list(kernel_size), downsample)]
        cur = planes
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1d(cur, planes))
        mods.append(nn.Sequential(*blocks))
    return nn.Sequential(*mods), cur


class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv1d_same(ni, bottleneck_size, 1, stride) if bottleneck_size > 0 else nn.Identity()
        bi = bottleneck_size if bottleneck_size > 0 else ni
        self.convs = nn.ModuleList([conv1d_same(bi, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv1d_same(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss) + 1) * nb_filters), nn.ReLU())

    def forward(self, x):
        b = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(b) for c in self.convs] + [self.conv_bottle(x)], dim=1))


class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn = nn.ReLU(True)
        self.conv = conv1d_same(ni, nf, 1)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels=12, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32,
                 use_residual=True):
        super().__init__()
        assert kernel_size >= 40 and depth % 3 == 0
        kss = [k - 1 if k % 2 == 0 else k for k in [kernel_size, kernel_size // 2, kernel_size // 4]]
        n_ks = len(kss) + 1
        self.depth, self.use_residual = depth, use_residual
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d == 0 else n_ks * nb_filters,
                                                  nb_filters, kss, bottleneck_size=bottleneck_size)
                                 for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters)
                                 for d in range(depth // 3)])
        self.out_channels = n_ks * nb_filters
        self.kss = kss

    def forward(self, x):
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = self.sk[d // 3](input_res, x)
                input_res = x
        return x


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp = nn.AdaptiveMaxPool1d(1)
        self.ap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1).flatten(1)


class ECGNet(nn.Module):
    """Backbone + reference head. Metadata (meta_dim >= 0) is concatenated, raw, to the
    global-pooled embedding (after its BN + Dropout) and enters the head's first Linear.
    backbone / pool / pool_norm are built in the same order with the same shapes for
    every arm, so with the same seed they are initialised identically (checked below)."""

    def __init__(self, arch, meta_dim, n_classes=5, input_channels=12, lin_ftrs=128, ps_head=0.5):
        super().__init__()
        if arch == "resnet1d_wang":
            self.backbone, nf = resnet1d_wang_backbone(input_channels=input_channels)
        elif arch == "inception1d":
            self.backbone = InceptionBackbone(input_channels=input_channels, kernel_size=40)
            nf = self.backbone.out_channels
        else:
            raise ValueError(arch)
        self.pool = AdaptiveConcatPool1d()
        # embedding pre-processing: identical across arms
        self.pool_norm = nn.Sequential(nn.BatchNorm1d(2 * nf), nn.Dropout(ps_head / 2))
        self.embedding_dim = 2 * nf
        self.meta_dim = meta_dim
        # head: receives [embedding (+ raw metadata)]
        self.hidden = nn.Sequential(nn.Linear(2 * nf + meta_dim, lin_ftrs), nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(lin_ftrs), nn.Dropout(ps_head))
        self.final = nn.Linear(lin_ftrs, n_classes)

    def forward(self, x, meta=None):
        z = self.pool_norm(self.pool(self.backbone(x)))
        if self.meta_dim > 0:
            z = torch.cat([z, meta], dim=1)   # raw metadata: no BN, no dropout
        return self.final(self.hidden(z))
# <<< MODELS
# <<< verbatim from v0.8


# NOTE (v0.9): the "256 / 259 / 338" widths in this locked text are the folds 1-8 development widths (the A2 models);
# for the 18 v0.8 models final_run_protocol encoder_width_note governs (256 / 259 / 339).
# >>> verbatim from v0.8 (METADATA_FUSION_DESCRIPTION)
METADATA_FUSION_DESCRIPTION = (
    "location: the metadata vector is concatenated to the network's global-pooled embedding (the "
    "2*nf = 256-dim AdaptiveConcatPool1d [max, avg] output, after that embedding's BatchNorm1d + "
    "Dropout(0.25)) and the concatenation is the input of the head's first Linear "
    "(256 + meta_dim -> 128), followed by ReLU, BatchNorm1d(128), Dropout(0.5), Linear(128 -> 5); "
    "metadata enters RAW (age_imputed standardized with training-fold statistics, all other columns "
    "0/1): no BatchNorm and no dropout is applied to the metadata; identical across arms: backbone, "
    "global pooling and the embedding's BatchNorm + Dropout (same seed -> identical initial weights, "
    "asserted in v0.4); differs across arms by design: only the input width of the head's first "
    "Linear (256 / 259 / 338 for signal_only / signal_plus_physio / signal_plus_all)")
assert METADATA_FUSION_DESCRIPTION == DEV["metadata_fusion"], \
    "ECGNet fusion description differs from the one locked in dev_protocol.json - aborting."
print("ECGNet metadata fusion == dev_protocol.json 'metadata_fusion': PASS")
# <<< verbatim from v0.8
# >>> verbatim from v0.8 (encoder width note)
assert "256 / 259 / 339" in FRP["encoder_width_note"]
# <<< verbatim from v0.8
print("head first-Linear widths: v0.8 models {} (FRP encoder_width_note) | development models {} (A2)".format(
    EXPECTED_HEAD_IN, EXPECTED_HEAD_IN18))


# NOTE (v0.9): v0.7 predict_fold9 with the globals X, eval_rows, META[arm], LEAD_MEAN_F32 and LEAD_STD_F32 turned into
# parameters whose names keep the body unchanged except 'meta_all = meta'; the docstring is verbatim ("fold-9"). No
# default arguments; every call passes keyword arguments only.
# >>> adapted from v0.7 (predict_fold9) | literal: def predict_fold9(model, arm): => def predict_records(model, X, eval_rows, meta, LEAD_MEAN_F32, LEAD_STD_F32): ;; meta_all = META[arm] => meta_all = meta
@torch.no_grad()
def predict_records(model, X, eval_rows, meta, LEAD_MEAN_F32, LEAD_STD_F32):
    """Mean over the 7 windows of per-window sigmoid probabilities, per fold-9 record."""
    model.eval()
    probs = np.zeros((len(eval_rows), len(SUPERCLASSES)), dtype=np.float64)
    meta_all = meta
    for st in WINDOW_STARTS:
        for b in range(0, len(eval_rows), EVAL_BATCH_SIZE):
            rows = eval_rows[b:b + EVAL_BATCH_SIZE]
            xw = (X[rows, :, st:st + CROP] - LEAD_MEAN_F32) / LEAD_STD_F32
            xb = torch.from_numpy(np.ascontiguousarray(xw, dtype=np.float32)).to(DEVICE)
            mb = torch.from_numpy(meta_all[rows]).to(DEVICE)
            probs[b:b + len(rows)] += torch.sigmoid(model(xb, mb)).double().cpu().numpy()
    return probs / len(WINDOW_STARTS)
# <<< adapted from v0.7


# >>> verbatim from v0.7 (load_saved_probs; A2 only)
def load_saved_probs(paths):
    pdf = pd.read_csv(paths["probs"], encoding="utf-8-sig")
    assert (pdf["ecg_id"].to_numpy() == current_index["ecg_id"].to_numpy()[eval_rows]).all()
    return pdf[["prob_" + sc for sc in SUPERCLASSES]].to_numpy(dtype=np.float64)
# <<< verbatim from v0.7


# >>> adapted from v0.7 (load_saved_probs -> fold-10 frozen probability CSVs) | literal: def load_saved_probs(paths): => def load_saved_probs10(path): ;; pd.read_csv(paths["probs"], => pd.read_csv(path, ;; current_index["ecg_id"].to_numpy()[eval_rows] => F10_IDS[:, 0]
def load_saved_probs10(path):
    pdf = pd.read_csv(path, encoding="utf-8-sig")
    assert list(pdf.columns) == ["ecg_id", "patient_id"] + ["prob_" + sc for sc in SUPERCLASSES] and len(pdf) == N_F10  # [v0.9 added]
    assert (pdf["patient_id"].to_numpy() == F10_IDS[:, 1]).all()  # [v0.9 added]
    assert (pdf["ecg_id"].to_numpy() == F10_IDS[:, 0]).all()
    return pdf[["prob_" + sc for sc in SUPERCLASSES]].to_numpy(dtype=np.float64)
# <<< adapted from v0.7


def load_model_verified(path, expected_sha256, arch, meta_dim):
    """Hard rule 1: sha256 first, torch.load(path, map_location=DEVICE, weights_only=True), strict load. No training."""
    got = sha256_of(path)
    assert got == expected_sha256, "{}: sha256 {} != expected {} - aborting.".format(path, got, expected_sha256)
    sd = torch.load(path, map_location=DEVICE, weights_only=True)
    model = ECGNet(arch, meta_dim)
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE)


def fold10_prob_frame(probs):
    pdf = pd.DataFrame({"ecg_id": F10_IDS[:, 0], "patient_id": F10_IDS[:, 1]})
    for k, sc in enumerate(SUPERCLASSES):
        pdf["prob_" + sc] = probs[:, k]
    return pdf


def permuted_meta(r):
    """FRP acquisition_permutation: row i of the device + site + nurse block = block of record PERM_IDX[r, i];
    the physio columns stay fixed."""
    base = META10["signal_plus_all"]
    out = base.copy()
    out[:, BLOCK_IDX] = base[PERM_IDX[r]][:, BLOCK_IDX]
    assert np.array_equal(out[:, PHYSIO_IDX], base[:, PHYSIO_IDX]), "permutation changed a physio column"
    assert np.array_equal(out[:, BLOCK_IDX], base[:, BLOCK_IDX][PERM_IDX[r]]), "permuted block != block[PERM_IDX[r]]"
    return out


# =====================================================================================
# PART A - PRE-OPENING VALIDATION (no fold-10 signal, no fold-10 label, no metric)
# =====================================================================================
hdr("PART A - PRE-OPENING VALIDATION (no fold-10 signal, no label, no metric)")


# NOTE (v0.9): the v0.4 record-reading loop as one function; reads the given record files in the given order.
# >>> adapted from v0.4 (wfdb record reader) | literal: EXPECTED_N_RECORDS_FOLDS1TO9 => len(filenames) | renames: FILES_TO_OPEN->filenames
def read_records(filenames):  # [v0.9 added]
    N_REC = len(filenames)  # [v0.9 added]
    N_FILES_OPENED = 0  # [v0.9 added]
    X = np.empty((N_REC, N_LEADS, N_SAMPLES), dtype=np.float32)
    LEAD_NAMES = None
    t0 = time.time()
    for i, fn in enumerate(filenames):
        sig, fields = wfdb.rdsamp(_record_path(fn))
        N_FILES_OPENED += 1
        assert int(fields["fs"]) == FS, "fs != 100 for " + fn
        assert sig.shape == (N_SAMPLES, N_LEADS), "shape {} != (1000, 12) for {}".format(sig.shape, fn)
        names = list(fields["sig_name"])
        if LEAD_NAMES is None:
            LEAD_NAMES = names
        assert names == LEAD_NAMES, "lead names/order differ for {}: {}".format(fn, names)
        X[i] = sig.T.astype(np.float32)
        if (i + 1) % 2000 == 0 or i + 1 == N_REC:
            print("  read {:6d}/{} records ({:.0f}s)".format(i + 1, N_REC, time.time() - t0))
    assert N_FILES_OPENED == len(filenames)
    return X, LEAD_NAMES  # [v0.9 added]
# <<< adapted from v0.4


# >>> verbatim from v0.4 (record path)
def _record_path(fn):
    return os.path.join(DATA_ROOT, fn)
# <<< verbatim from v0.4


# --- A5: h3_verdict truth table (the nine v0.6 cases; synthetic CI bounds, no data). It runs in EVERY state, so
#     it also protects Part D on a post-freeze re-execution. ---
# >>> verbatim from v0.6 (h3_verdict truth-table self-test)
# --- H3 verdict function: truth-table self-test (synthetic CI bounds, not data) ---
_cases = [  # (lo95, hi95, lo90, hi90) -> expected verdict, a, b, material
    ((0.002, 0.008, 0.003, 0.007), "statistically detectable, below the materiality threshold", True, True, False),
    ((0.012, 0.030, 0.014, 0.028), "acquisition metadata materially changes performance (sign: +)", True, False, True),
    ((-0.030, -0.012, -0.028, -0.014), "acquisition metadata materially changes performance (sign: -)", True, False, True),
    ((0.002, 0.020, 0.004, 0.018), "acquisition metadata changes performance (sign: +); magnitude relative to SESOI undetermined", True, False, False),
    ((-0.005, 0.005, -0.004, 0.004), "no material change", False, True, False),
    ((-0.012, 0.009, -0.010, 0.010), "no material change", False, True, False),      # closed interval
    ((-0.005, 0.020, -0.003, 0.018), "inconclusive", False, False, False),
    ((0.0, 0.010, 0.001, 0.009), "no material change", False, True, False),           # lo95 == 0 -> a fails
    ((0.010, 0.030, 0.012, 0.028), "acquisition metadata changes performance (sign: +); magnitude relative to SESOI undetermined", True, False, False),
]
for (lo95, hi95, lo90, hi90), v, a, b, m in _cases:
    r = REF["h3_verdict"](lo95, hi95, lo90, hi90, SESOI_H3)
    assert (r["verdict"], r["a_holds"], r["b_holds"], r["material"]) == (v, a, b, m), (lo95, hi95, lo90, hi90, r)
print("h3_verdict truth-table self-test ({} cases): PASS".format(len(_cases)))
# <<< verbatim from v0.6
A5 = {"n_cases": len(_cases), "pass": True, "function": "REF['h3_verdict']", "sesoi": SESOI_H3}

PRE_OPENING_VALIDATION = None
if RUN_PARTS_AB:
    # --- A1: the reader reproduces the v0.4 cache bitwise (chunks of A1_CHUNK records) ---
    t0 = time.time()
    _n_chunks = 0
    for _s in range(0, N_REC, A1_CHUNK):
        _Xc, _names = read_records(FILES9[_s:_s + A1_CHUNK])
        _ref = X[_s:_s + A1_CHUNK]
        assert _names == LQ_LN["lead_order"], "A1: sig_name {} != FRP lead order - aborting; fold 10 not opened.".format(_names)
        assert _Xc.shape == _ref.shape and _Xc.dtype == _ref.dtype == np.float32 and np.array_equal(_Xc, _ref) and \
            _Xc.tobytes() == _ref.tobytes(), \
            "A1 FAILED: records {}-{} re-read from records100 differ from the v0.4 cache - aborting; fold 10 not opened.".format(
                _s, _s + len(_Xc) - 1)
        _n_chunks += 1
    del _Xc, _ref
    A1 = {"n_records": N_REC, "chunk": A1_CHUNK, "n_chunks": _n_chunks, "bitwise_equal_to_v0.4_cache": True,
          "lead_order": LQ_LN["lead_order"], "fs": FS, "seconds": round(time.time() - t0, 1)}
    print("A1: {} folds 1-9 records re-read with the v0.4 reader in {} chunks ({:.0f}s): bitwise == v0.4 cache, "
          "sig_name == FRP lead order, fs == {}: PASS".format(N_REC, _n_chunks, A1["seconds"], FS))

    # --- A2: predict_records reproduces the saved fold-9 probabilities of the 18 development cells ---
    t0 = time.time()
    A2 = []
    for arch in ARCHS:
        for arm in ARMS:
            for seed in SEEDS:
                lr = SELECTED_LR[arch]
                if (arch, arm, seed) in REUSED_CELLS:
                    _paths, _src = v05_run_paths(arch, arm, seed, lr), "v0.5"
                    _exp = {k: REUSED_FILES[os.path.basename(p)]["sha256"] for k, p in _paths.items()}
                else:
                    _paths, _src = v07_run_paths(arch, arm, seed, lr), "v0.7"
                    _exp = {k: manifest_entry(V07_MAN, os.path.basename(p))["sha256"] for k, p in _paths.items()}
                _got_probs_sha = sha256_of(_paths["probs"])
                assert _got_probs_sha == _exp["probs"], "A2: {} differs from its manifest - aborting.".format(_paths["probs"])
                _model = load_model_verified(_paths["state"], _exp["state"], arch, META18[arm].shape[1])
                assert _model.hidden[0].in_features == EXPECTED_HEAD_IN18[arm]
                _p = predict_records(model=_model, X=X, eval_rows=eval_rows, meta=META18[arm],
                                     LEAD_MEAN_F32=LEAD_MEAN_F32_18, LEAD_STD_F32=LEAD_STD_F32_18)
                _saved = load_saved_probs(_paths)
                _d = float(np.max(np.abs(_p - _saved)))
                A2.append({"arch": arch, "arm": arm, "seed": seed, "source": _src, "max_abs_diff": _d, "tolerance": A2_TOL,
                           "state_dict_sha256": _exp["state"], "probs_sha256": _exp["probs"]})
                print("  A2 {:14s} {:19s} seed {:4d} ({}): max |p - saved| = {:.3e}".format(arch, arm, seed, _src, _d))
                assert _d <= A2_TOL, "A2 FAILED: {} {} seed {}: max |p - saved| {} > {} - aborting; fold 10 not opened.".format(
                    arch, arm, seed, _d, A2_TOL)
                del _model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print("A2: 18 development cells reproduced with predict_records (max diff {:.3e} <= {}, {:.0f}s): PASS".format(
        max(r["max_abs_diff"] for r in A2), A2_TOL, time.time() - t0))

    # --- A3: encode_with_categories reproduces the folds 1-8 dummies and META18 of the fold-9 rows ---
    _enc9 = {col: encode_with_categories(LEVELS[col][mask_fold9], CATS18[col], col) for col in ACQ_COLS}
    for col in ACQ_COLS:
        _ref9 = DUMMIES18[col].loc[mask_fold9]
        assert list(_enc9[col].columns) == list(_ref9.columns) and _enc9[col].index.equals(_ref9.index) and \
            list(_enc9[col].dtypes) == list(_ref9.dtypes) and np.array_equal(_enc9[col].to_numpy(), _ref9.to_numpy()), \
            "A3 FAILED: {} dummies of the fold-9 rows differ from the folds 1-8 encoder - aborting.".format(col)
    _n_a3_cols = sum(_enc9[col].shape[1] for col in ACQ_COLS)
    assert _n_a3_cols == sum(EXPECTED_ONEHOT_COUNTS18.values())
    _mf9 = pd.concat([age_imputed_std18[mask_fold9], age_censored[mask_fold9].rename("age_censored"),
                      sex[mask_fold9].rename("sex"), _enc9["device"], _enc9["site"], _enc9["nurse"]], axis=1)
    for arm in ARMS:
        _cols = ARM_COLUMNS18[arm]
        _arr = _mf9[_cols].to_numpy(dtype=np.float32) if _cols else np.zeros((len(eval_rows), 0), dtype=np.float32)
        assert np.array_equal(_arr, META18[arm][eval_rows]), "A3 FAILED: META18 {} of the fold-9 rows not reproduced".format(arm)
    A3 = {"n_onehot_columns": _n_a3_cols, "dummies_equal": True, "meta18_fold9_equal": True, "n_fold9_rows": int(len(eval_rows))}
    print("A3: encode_with_categories(fold-9 levels, CATS18) == folds 1-8 dummies ({} columns, order and values) and "
          "META18 of the {} fold-9 rows (all arms): PASS".format(_n_a3_cols, len(eval_rows)))

    # --- A4: the 18 v0.8 state_dicts load strict (no forward pass) ---
    A4 = []
    for arch, arm, seed in RUN_ORDER:
        _paths = v08_run_paths(arch, arm, seed, SELECTED_LR[arch])
        _model = load_model_verified(_paths["state"], V08_STATE_SHA[(arch, arm, seed)], arch, META10[arm].shape[1])
        assert _model.hidden[0].in_features == EXPECTED_HEAD_IN[arm] == LQ["head_first_linear_in_features"][arm]
        A4.append({"arch": arch, "arm": arm, "seed": seed, "in_features": int(_model.hidden[0].in_features),
                   "state_dict_sha256": V08_STATE_SHA[(arch, arm, seed)]})
        del _model
    print("A4: 18 v0.8 state_dicts sha256-verified and loaded with strict=True; head first-Linear in_features {}: PASS".format(
        sorted({r["in_features"] for r in A4})))

    PRE_OPENING_VALIDATION = {"A1_reader": A1, "A2_inference": A2, "A3_metadata_encoding": A3,
                              "A4_final_models_load": A4, "A5_h3_verdict_truth_table": A5, "all_pass": True}
    print("PART A: all pre-opening checks PASS")
else:
    print("PART A1-A4 skipped (state 'post_freeze'): the pre-opening validation of the frozen run is in the freeze file.")

# --- every folds 1-8 object is deleted, whatever the state (supervisor answer 3) ---
_FOLDS18_DELETED = sorted(n for n in list(globals()) if n.endswith("18"))
for _n in _FOLDS18_DELETED:
    del globals()[_n]
del eval_rows, X
gc.collect()
assert not [n for n in globals() if n.endswith("18")], "a folds 1-8 object survived - aborting."
print("folds 1-8 objects deleted ({}): {}; folds 1-9 signal cache released".format(len(_FOLDS18_DELETED), _FOLDS18_DELETED))


# =====================================================================================
# PART B - OPENING FOLD 10 -> INFERENCE (18 models + permutation passes) -> FREEZE
# =====================================================================================
hdr("PART B - OPENING FOLD 10 -> INFERENCE -> FREEZE")
if RUN_PARTS_AB:
    os.makedirs(OUT_DIR, exist_ok=True)                 # the first write of this cell
    ATTEMPT_LOG = read_attempt_log()
    _n_started = sum(e.get("event") == "started" for e in ATTEMPT_LOG)
    assert _n_started < MAX_ATTEMPTS
    _leftover = [n for n in sorted(os.listdir(OUT_DIR))
                 if n in PROB_NAMES + PERM_NAMES or n in [_partial(x) for x in PROB_NAMES + PERM_NAMES]
                 or n == _partial(os.path.basename(FREEZE_PATH))]
    if _leftover:
        for _n in _leftover:
            os.remove(os.path.join(OUT_DIR, _n))
        log_event(RUN_LABEL, _n_started, "deleted_partial_files", files=_leftover)
        note("deleted the leftover files of the previous pre-freeze attempt: {}".format(_leftover))
    ATTEMPT = _n_started + 1
    GUARD_BEFORE_OPENING = identity_guard("immediately before the first fold-10 file is opened")
    log_event(RUN_LABEL, ATTEMPT, "started", state=STATE, code_revision=CODE_REVISION)
    try:
        t0 = time.time()
        X10, LEAD_NAMES10 = read_records(FILES10)          # FOLD 10 IS OPENED HERE (A1-validated reader)
        assert LEAD_NAMES10 == LQ_LN["lead_order"], "fold-10 sig_name {} != FRP lead order".format(LEAD_NAMES10)
        assert X10.shape == (N_F10, N_LEADS, N_SAMPLES) and X10.dtype == np.float32
        assert np.isfinite(X10).all(), "non-finite values in the fold-10 signals - aborting."
        _allzero = (X10 == 0).all(axis=2)
        _mean10, _std10 = lead_norm_two_pass(X10, ROWS10)
        _std_ratio = _std10 / FRP_LEAD_STD
        FOLD10_SIGNAL_CHECKS = {"n_records": N_F10, "n_nonfinite_values": 0, "n_allzero_leads": int(_allzero.sum()),
                                "n_records_with_allzero_lead": int(_allzero.any(axis=1).sum()),
                                "std_ratio_to_frp_folds1to9": {LQ_LN["lead_order"][i]: float(_std_ratio[i]) for i in range(N_LEADS)},
                                "std_ratio_warn_range": list(STD_RATIO_WARN_RANGE), "seconds_to_read": round(time.time() - t0, 1)}
        print("fold 10 opened: {} records read in {:.0f}s | {} {} | all finite | all-zero leads {} (in {} records) | "
              "per-lead std / FRP std: min {:.3f} max {:.3f}".format(
                  N_F10, FOLD10_SIGNAL_CHECKS["seconds_to_read"], X10.shape, X10.dtype, FOLD10_SIGNAL_CHECKS["n_allzero_leads"],
                  FOLD10_SIGNAL_CHECKS["n_records_with_allzero_lead"], float(_std_ratio.min()), float(_std_ratio.max())))
        _bad = [LQ_LN["lead_order"][i] for i in range(N_LEADS)
                if not (STD_RATIO_WARN_RANGE[0] <= _std_ratio[i] <= STD_RATIO_WARN_RANGE[1])]
        if _bad:
            warn_part_b("fold-10 per-lead std / FRP folds 1-9 std outside {} for leads {} (WARN-only; no consequence)".format(
                STD_RATIO_WARN_RANGE, _bad))
        X10_SHA256 = sha256_f32(X10)

        PROB_FILES = {}
        PERM_PROBS = {arch: np.zeros((N_PERM, len(SEEDS), N_F10, len(SUPERCLASSES)), dtype=np.float64) for arch in ARCHS}
        PERM_FILES = {}
        for arch in ARCHS:
            for i_seed, seed in enumerate(SEEDS):
                for arm in ARMS:
                    t1 = time.time()
                    _paths = v08_run_paths(arch, arm, seed, SELECTED_LR[arch])
                    _model = load_model_verified(_paths["state"], V08_STATE_SHA[(arch, arm, seed)], arch, META10[arm].shape[1])
                    assert _model.hidden[0].in_features == EXPECTED_HEAD_IN[arm]
                    _p = predict_records(model=_model, X=X10, eval_rows=ROWS10, meta=META10[arm],
                                         LEAD_MEAN_F32=LEAD_MEAN_F32, LEAD_STD_F32=LEAD_STD_F32)
                    assert _p.shape == (N_F10, len(SUPERCLASSES)) and np.isfinite(_p).all() and (_p >= 0).all() and (_p <= 1).all()
                    _pp = prob_path(arch, arm, seed, SELECTED_LR[arch])
                    _sha = write_csv_replace(fold10_prob_frame(_p), _pp)
                    PROB_FILES[os.path.basename(_pp)] = {"sha256": _sha, "bytes": os.path.getsize(_pp), "arch": arch,
                                                         "arm": arm, "seed": seed,
                                                         "produced_from_v0.8_state_dict_sha256": V08_STATE_SHA[(arch, arm, seed)]}
                    _msg = ""
                    if arm == "signal_plus_all":
                        for _r in range(N_PERM):
                            PERM_PROBS[arch][_r, i_seed] = predict_records(
                                model=_model, X=X10, eval_rows=ROWS10, meta=permuted_meta(_r),
                                LEAD_MEAN_F32=LEAD_MEAN_F32, LEAD_STD_F32=LEAD_STD_F32)
                        _msg = " + {} permutation passes".format(N_PERM)
                    print("  {:14s} {:19s} seed {:4d}: fold-10 probabilities{} ({:.0f}s) -> {}".format(
                        arch, arm, seed, _msg, time.time() - t1, os.path.basename(_pp)))
                    del _model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            assert np.isfinite(PERM_PROBS[arch]).all() and (PERM_PROBS[arch] >= 0).all() and (PERM_PROBS[arch] <= 1).all()
            _np = perm_path(arch)
            _sha = write_npy_replace(PERM_PROBS[arch], _np)
            PERM_FILES[os.path.basename(_np)] = {
                "sha256": _sha, "bytes": os.path.getsize(_np), "arch": arch, "arm": "signal_plus_all",
                "shape": list(PERM_PROBS[arch].shape), "dtype": "float64",
                "axes": "(repetition r = 0..{}, seed in {}, record in fold-10 file order, class {})".format(
                    N_PERM - 1, SEEDS, SUPERCLASSES),
                "produced_from_v0.8_state_dict_sha256": [V08_STATE_SHA[(arch, "signal_plus_all", s)] for s in SEEDS]}
            print("  {}: permutation probabilities {} saved -> {}".format(arch, PERM_PROBS[arch].shape, os.path.basename(_np)))
        GUARD_BEFORE_FREEZE = identity_guard("before FREEZE")
        FREEZE = {
            "document": "v0.9 frozen fold-10 model outputs (write-once; written before any label is read or any metric is computed)",
            "timestamp": datetime.now().astimezone().isoformat(),
            "code_version": CODE_VERSION, "code_revision": CODE_REVISION, "attempt": ATTEMPT,
            "statement": FREEZE_STATEMENT,
            "outputs": {**PROB_FILES, **PERM_FILES},
            "lock_hashes": LOCK_HASHES,
            "x10_sha256": X10_SHA256,
            "x10_serialization": "X10 ({}, {}, {}) as '<f4' C-order bytes, fold-10 file order".format(N_F10, N_LEADS, N_SAMPLES),
            "meta10_sha256": META10_SHA256, "meta10_serialization": "META10[arm] as '<f4' C-order bytes",
            "meta10_columns": META10_COLUMNS,
            "fold10_signal_checks": FOLD10_SIGNAL_CHECKS,
            "pre_opening_validation": PRE_OPENING_VALIDATION,
            "identity_guard": [GUARD_BEFORE_OPENING, GUARD_BEFORE_FREEZE],
            "part_b_warnings": PART_B_WARNINGS,
            "v0.8_manifest_sha256": V08_MAN_SHA,
            "final_run_protocol": {"self_sha256": DOC_SELF["final_run_protocol"], "file_sha256": DOC_FILE["final_run_protocol"]},
            "reference_implementation_sha256": REFERENCE_CODE_SHA256,
            "interpretation_post_fold10_rule_2": INTERPRETATION_POST_FOLD10_RULE_2,
            "post_opening_policy": POST_OPENING_POLICY,
        }
        assert not _find_nonfinite(plain(FREEZE)), "non-finite value in the freeze document: {}".format(
            _find_nonfinite(plain(FREEZE)))
        FREEZE_SELF_SHA = write_once_json_atomic(FREEZE_PATH, FREEZE)
    except BaseException as exc:
        log_event(RUN_LABEL, ATTEMPT, "failed", error="{}: {}".format(type(exc).__name__, exc),
                  freeze_file_exists=os.path.isfile(FREEZE_PATH))
        print("Part B attempt {} FAILED -> logged; {}".format(
            ATTEMPT, "re-run the cell for attempt {} (pre-freeze)".format(ATTEMPT + 1) if ATTEMPT < MAX_ATTEMPTS
            else "STOP and report (no further pre-freeze attempt allowed)"))
        raise
    log_event(RUN_LABEL, ATTEMPT, "frozen", freeze_self_sha256=FREEZE_SELF_SHA, freeze_file_sha256=sha256_of(FREEZE_PATH))
    print("[saved] {} (write-once, self_sha256 {}) - FROZEN: {} files; no label has been read, no metric computed".format(
        FREEZE_PATH, FREEZE_SELF_SHA, len(FREEZE["outputs"])))
    del X10
    gc.collect()
else:
    print("PART B skipped (state 'post_freeze'): no fold-10 record is read, no model is loaded.")


# =====================================================================================
# PART C - LABELS (only after FREEZE)
# =====================================================================================
hdr("PART C - LABELS (only after the freeze file verifies)")
FROZEN = verify_frozen_outputs()
assert FROZEN["lock_hashes"] == LOCK_HASHES and FROZEN["meta10_sha256"] == META10_SHA256 and \
    FROZEN["meta10_columns"] == META10_COLUMNS, "the freeze file does not match this execution's locks / META10 - aborting."
FREEZE_SELF_SHA = FROZEN["self_sha256"]
FREEZE_FILE_SHA = sha256_of(FREEZE_PATH)
print("freeze file verified: self_sha256 {} | {} frozen files unchanged | statement: '{}'".format(
    FREEZE_SELF_SHA, len(FROZEN["outputs"]), FROZEN["statement"]))
ATTEMPT_LOG = read_attempt_log()
FROZEN_ATTEMPT = int(FROZEN["attempt"])
if STATE == "post_freeze":
    if not any(e.get("event") == "frozen" for e in ATTEMPT_LOG):
        warn("the freeze file is valid but the attempt log has no 'frozen' entry (crash between the write and the log "
             "line) -> 'frozen_recovered'")
        log_event(RUN_LABEL, FROZEN_ATTEMPT, "frozen_recovered", freeze_self_sha256=FREEZE_SELF_SHA)
    log_event(RUN_LABEL, FROZEN_ATTEMPT, "post_freeze_started", code_revision=CODE_REVISION)
_stale = [n for n in sorted(os.listdir(OUT_DIR)) if n in [_partial(x) for x in WRITE_ONCE_NAMES]
          or (n.endswith(".partial") and REEXEC_MANIFEST_RE.match(n[:-len(".partial")]))]
if _stale:
    for _n in _stale:
        os.remove(os.path.join(OUT_DIR, _n))
    log_event(RUN_LABEL, FROZEN_ATTEMPT, "deleted_partial_files", files=_stale)
    note("deleted stale '.partial' files of write-once outputs: {}".format(_stale))

_db_sha = sha256_of(DB_CSV)
assert _db_sha == EXPECTED_INPUT_HASHES["ptbxl_database.csv"] == INPUT_HASHES["ptbxl_database.csv"], \
    "ptbxl_database.csv changed since STEP 0 - aborting."
df_lab = pd.read_csv(DB_CSV, index_col="ecg_id")         # all columns: the first read of any label column
_f10 = df_lab.loc[df_lab["strat_fold"] == 10]
_f9 = df_lab.loc[df_lab["strat_fold"] == 9]
_pid10 = _f10["patient_id"].to_numpy()
assert np.all(_pid10 == np.round(_pid10))
assert np.array_equal(np.stack([_f10.index.to_numpy().astype(np.int64), _pid10.astype(np.int64)], axis=1), F10_IDS), \
    "fold-10 (ecg_id, patient_id) of the full read != F10_IDS - aborting."
assert np.array_equal(_f9.index.to_numpy(), df9.index.to_numpy()[mask_fold9.to_numpy()]), "fold-9 rows differ - aborting."
LAB9 = derive_labels(_f9)
LAB10 = derive_labels(_f10)
assert LAB10.index.equals(_f10.index) and LAB9.index.equals(_f9.index)
FOLD9_POSITIVES = {c: int(LAB9[c].sum()) for c in SUPERCLASSES}
FOLD10_POSITIVES = {c: int(LAB10[c].sum()) for c in SUPERCLASSES}
assert FOLD9_POSITIVES == EXPECTED_FOLD9_POSITIVES, \
    "LABEL-CODE CHECK FAILED: fold-9 positives {} != v0.2.1 {} - aborting before any metric.".format(
        FOLD9_POSITIVES, EXPECTED_FOLD9_POSITIVES)
assert FOLD10_POSITIVES == EXPECTED_FOLD10_POSITIVES, \
    "LABEL-CODE CHECK FAILED: fold-10 positives {} != counts disclosed in FRP {} - aborting before any metric.".format(
        FOLD10_POSITIVES, EXPECTED_FOLD10_POSITIVES)
print("labels (v0.2.1 code, after FREEZE): fold-9 positives {} == v0.2.1 | fold-10 positives {} == FRP disclosure: PASS".format(
    FOLD9_POSITIVES, FOLD10_POSITIVES))
LABELS10 = pd.DataFrame({"ecg_id": F10_IDS[:, 0], "patient_id": F10_IDS[:, 1],
                         **{c: LAB10[c].to_numpy().astype(np.int64) for c in SUPERCLASSES}})
del df_lab, _f9, _f10, LAB9
gc.collect()
LABELS_SHA, _ = compare_or_write_csv(LABELS_PATH, LABELS10, "v0.9_fold10_labels.csv")
Y10 = LABELS10[SUPERCLASSES].to_numpy(dtype=np.int64)
assert Y10.shape == (N_F10, len(SUPERCLASSES))


# =====================================================================================
# PART D - ANALYSIS (REF only; FRP closed list; CPU; deterministic)
# =====================================================================================
hdr("PART D - ANALYSIS (REF only, FRP closed list)")
t0 = time.time()
P = {}
for arch, arm, seed in RUN_ORDER:
    _pp = prob_path(arch, arm, seed, SELECTED_LR[arch])
    assert sha256_of(_pp) == FROZEN["outputs"][os.path.basename(_pp)]["sha256"]
    P[(arch, arm, seed)] = load_saved_probs10(_pp)
ENS = {(arch, arm): np.mean([P[(arch, arm, s)] for s in SEEDS], axis=0) for arch in ARCHS for arm in ARMS}
PERM = {}
for arch in ARCHS:
    _np = perm_path(arch)
    assert sha256_of(_np) == FROZEN["outputs"][os.path.basename(_np)]["sha256"]
    PERM[arch] = np.load(_np)
    assert PERM[arch].shape == (N_PERM, len(SEEDS), N_F10, len(SUPERCLASSES)) and PERM[arch].dtype == np.float64
_pat10, _rows10 = REF["patient_rows_in_file_order"](F10_IDS[:, 1])
assert np.array_equal(_pat10, F10_PATIENTS) and len(_rows10) == N_F10_PATIENTS and \
    all(np.array_equal(a, b) for a, b in zip(_rows10, F10_ROWS))
assert sha256_bytes(np.ascontiguousarray(BOOT_IDX.astype("<i8")).tobytes()) == LOCKED_HASHES_FRP["boot_idx_sha256"]
print("inputs: 18 frozen probability files (hash-verified, ecg_id order == F10_IDS) | seed ensembles = arithmetic mean "
      "over seeds {} | 2 permutation arrays | {} patients | BOOT_IDX {} (hash == FRP)".format(
          SEEDS, len(_pat10), BOOT_IDX.shape))

# --- D0: fast_auc (REF) vs sklearn.metrics.roc_auc_score, once (the locked 'metric' text) ---
from sklearn.metrics import roc_auc_score              # D0 only
_d0_input = ENS[(ARCHS[0], "signal_plus_physio")]
D0 = {"input": "{} signal_plus_physio seed ensemble".format(ARCHS[0]), "tolerance": SKLEARN_CHECK_TOL, "abs_diff": {}}
for k, sc in enumerate(SUPERCLASSES):
    _fa = REF["fast_auc"](Y10[:, k], _d0_input[:, k])
    _sk = float(roc_auc_score(Y10[:, k], _d0_input[:, k]))
    D0["abs_diff"][sc] = abs(_fa - _sk)
    assert abs(_fa - _sk) <= SKLEARN_CHECK_TOL, "D0: fast_auc != sklearn roc_auc_score for " + sc
D0["max_abs_diff"] = max(D0["abs_diff"].values())
print("D0: REF fast_auc == sklearn roc_auc_score (max |diff| {:.1e} <= {}): PASS".format(D0["max_abs_diff"], SKLEARN_CHECK_TOL))


def paired_delta(P_A, P_B):
    """REF only: delta = macro_auroc(B) - macro_auroc(A); paired patient-level bootstrap on BOOT_IDX."""
    md, cd = REF["paired_bootstrap"](Y10, P_A, P_B, _rows10, BOOT_IDX)
    mac_a = REF["macro_auroc_strict"](Y10, P_A)
    mac_b = REF["macro_auroc_strict"](Y10, P_B)
    S = REF["summarize_deltas"](mac_b - mac_a, md, NAN_FLAG_FRACTION)
    return md, cd, mac_a, mac_b, S


BOOT_COLUMNS = {"replicate": np.arange(N_BOOT)}
H3_ARMS = ("signal_plus_physio", "signal_plus_all")
SECTION = {}
CD_ENS = {}
for i_arch, arch in enumerate(ARCHS):                  # D1 PRIMARY (resnet1d_wang), D2 ROBUSTNESS (inception1d)
    _md, _cd, _ma, _mb, _S = paired_delta(ENS[(arch, H3_ARMS[0])], ENS[(arch, H3_ARMS[1])])
    _V = REF["h3_verdict"](_S["lo95"], _S["hi95"], _S["lo90"], _S["hi90"], SESOI_H3)
    SECTION[arch] = {"arch": arch, "input": "seed ensemble (arithmetic mean over seeds {})".format(SEEDS),
                     "arm_A": H3_ARMS[0], "arm_B": H3_ARMS[1], "macro_auroc_A": _ma, "macro_auroc_B": _mb, **_S,
                     "h3_verdict": _V,
                     "sealed_rules_holding_texts": {r: SEALED_DECISION[r] for r in _V["sealed_rules_holding"]},
                     "sealed_decision_rules_verbatim": SEALED_DECISION}
    CD_ENS[arch] = _cd
    BOOT_COLUMNS["D{}_{}_ensemble".format(i_arch + 1, arch)] = _md
    if _S["nan_flag"]:
        warn_analysis("{} H3 ensemble: {} of {} bootstrap draws are NaN (> {:.0%})".format(arch, _S["n_nan"], N_BOOT, NAN_FLAG_FRACTION))

D3 = []                                                # SECONDARY per seed (same BOOT_IDX, same seed in both arms)
for arch in ARCHS:
    for seed in SEEDS:
        _md, _cd, _ma, _mb, _S = paired_delta(P[(arch, H3_ARMS[0], seed)], P[(arch, H3_ARMS[1], seed)])
        _V = REF["h3_verdict"](_S["lo95"], _S["hi95"], _S["lo90"], _S["hi90"], SESOI_H3)
        D3.append({"arch": arch, "seed": seed, "arm_A": H3_ARMS[0], "arm_B": H3_ARMS[1], "macro_auroc_A": _ma,
                   "macro_auroc_B": _mb, **_S, "h3_verdict_secondary": {**_V, "label": SECONDARY_VERDICT_LABEL}})
        BOOT_COLUMNS["D3_{}_seed{}".format(arch, seed)] = _md
        if _S["nan_flag"]:
            warn_analysis("{} seed {} H3: {} of {} bootstrap draws are NaN".format(arch, seed, _S["n_nan"], N_BOOT))

D4 = []                                                # DISAGREEMENTS (reported, never resolved)
_v_ens = {arch: SECTION[arch]["h3_verdict"]["verdict"] for arch in ARCHS}
_sign_ens = {arch: int(np.sign(SECTION[arch]["point_delta"])) for arch in ARCHS}
D4.append({"item": "(i)", "comparison": "H3 verdict: {} seed ensemble vs {} seed ensemble".format(ARCHS[0], ARCHS[1]),
           "value_1": _v_ens[ARCHS[0]], "value_2": _v_ens[ARCHS[1]], "disagreement": _v_ens[ARCHS[0]] != _v_ens[ARCHS[1]]})
for _r in D3:
    D4.append({"item": "(ii)", "comparison": "verdict: {} seed {} ({}) vs {} seed ensemble".format(
                   _r["arch"], _r["seed"], SECONDARY_VERDICT_LABEL, _r["arch"]),
               "value_1": _r["h3_verdict_secondary"]["verdict"], "value_2": _v_ens[_r["arch"]],
               "disagreement": _r["h3_verdict_secondary"]["verdict"] != _v_ens[_r["arch"]]})
for _r in D3:
    _s = int(np.sign(_r["point_delta"]))
    D4.append({"item": "(iii)", "comparison": "sign of the point estimate: {} seed {} vs {} seed ensemble".format(
                   _r["arch"], _r["seed"], _r["arch"]),
               "value_1": _s, "value_2": _sign_ens[_r["arch"]], "disagreement": _s != _sign_ens[_r["arch"]]})
D4.append({"item": "(iv)", "comparison": "sign of the point estimate: {} seed ensemble vs {} seed ensemble".format(
               ARCHS[0], ARCHS[1]),
           "value_1": _sign_ens[ARCHS[0]], "value_2": _sign_ens[ARCHS[1]],
           "disagreement": _sign_ens[ARCHS[0]] != _sign_ens[ARCHS[1]]})

E1 = []                                                # EXPLORATORY: signal_plus_physio - signal_only
for arch in ARCHS:
    _md, _cd, _ma, _mb, _S = paired_delta(ENS[(arch, "signal_only")], ENS[(arch, "signal_plus_physio")])
    E1.append({"arch": arch, "input": "seed ensemble", "arm_A": "signal_only", "arm_B": "signal_plus_physio",
               "macro_auroc_A": _ma, "macro_auroc_B": _mb, **_S, "label": "EXPLORATORY; no decision label"})
    BOOT_COLUMNS["E1_{}_ensemble".format(arch)] = _md
    if _S["nan_flag"]:
        warn_analysis("E1 {}: {} of {} bootstrap draws are NaN".format(arch, _S["n_nan"], N_BOOT))

E2 = []                                                # EXPLORATORY: per-class deltas of the H3 pair (from D1 / D2)
for arch in ARCHS:
    _pa = REF["per_class_auroc"](Y10, ENS[(arch, H3_ARMS[0])])
    _pb = REF["per_class_auroc"](Y10, ENS[(arch, H3_ARMS[1])])
    _per = {}
    for k, sc in enumerate(SUPERCLASSES):
        _cs = REF["summarize_deltas"](_pb[k] - _pa[k], CD_ENS[arch][:, k], NAN_FLAG_FRACTION)
        _per[sc] = {"point_delta": _cs["point_delta"], "lo95": _cs["lo95"], "hi95": _cs["hi95"], "n_nan": _cs["n_nan"],
                    "auroc_A": float(_pa[k]), "auroc_B": float(_pb[k])}
        BOOT_COLUMNS["E2_{}_{}".format(arch, sc)] = CD_ENS[arch][:, k]
    E2.append({"arch": arch, "input": "seed ensemble", "arm_A": H3_ARMS[0], "arm_B": H3_ARMS[1], "per_class": _per,
               "label": "EXPLORATORY; point and 95% CI only; no multiplicity correction; no decision label"})

E3 = []                                                # EXPLORATORY: acquisition permutation (inference only)
for arch in ARCHS:
    _unperm = REF["macro_auroc_strict"](Y10, ENS[(arch, "signal_plus_all")])
    _perm_auc = [REF["macro_auroc_strict"](Y10, PERM[arch][_r].mean(axis=0)) for _r in range(N_PERM)]
    _drops = [_unperm - v for v in _perm_auc]
    E3.append({"arch": arch, "arm": "signal_plus_all", "n_perm": N_PERM, "perm_seed": PERM_SEED,
               "unpermuted_macro_auroc": _unperm, "permuted_macro_auroc": _perm_auc, "drop": _drops,
               "drop_mean": float(np.mean(_drops)), "drop_min": float(np.min(_drops)), "drop_max": float(np.max(_drops)),
               "label": "EXPLORATORY; same permutation for the three seed models, probabilities averaged, then macro-AUROC; "
                        "summary = mean and range of drop; no CI; no decision label"})

_so = [e for e in E1 if e["arch"] == ARCHS[0]][0]["macro_auroc_A"]   # D7 sanity (WARN-only)
D7 = {"resnet1d_wang_signal_only_ensemble_macro_auroc": _so, "threshold": SANITY_GATE_MIN_AUROC,
      "below": bool(_so < SANITY_GATE_MIN_AUROC), "note": "WARN-only; results are reported as they are"}
if D7["below"]:
    warn_analysis("D7: resnet1d_wang signal_only seed-ensemble macro-AUROC {} below the development sanity gate".format(_so))
print("Part D computed in {:.0f}s (10 paired bootstraps x {} draws)".format(time.time() - t0, N_BOOT))

BOOT_DF = pd.DataFrame(BOOT_COLUMNS)
assert BOOT_DF.shape == (N_BOOT, 21)
BOOT_SHA, _ = compare_or_write_csv(BOOT_CSV_PATH, BOOT_DF, "v0.9_bootstrap_deltas.csv")

RESULTS = {
    "document": "v0.9 fold-10 results (write-once; written LAST, after the freeze file, labels and bootstrap deltas)",
    "timestamp": datetime.now().astimezone().isoformat(),
    "code_version": CODE_VERSION, "code_revision": CODE_REVISION,
    "frozen_outputs_self_sha256": FREEZE_SELF_SHA,
    "reference_implementation_sha256": REFERENCE_CODE_SHA256,
    "final_run_protocol": {"self_sha256": DOC_SELF["final_run_protocol"], "file_sha256": DOC_FILE["final_run_protocol"]},
    "sealed_decision_rules_verbatim": SEALED_DECISION,
    "H3_decision_verbatim": H3_DECISION,
    "reporting_order_verbatim": REPORTING_ORDER,
    "settings": {"SESOI_H3": SESOI_H3, "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED, "N_PERM": N_PERM, "PERM_SEED": PERM_SEED,
                 "nan_flag_fraction": NAN_FLAG_FRACTION, "seeds": SEEDS,
                 "ci": "percentile, np.percentile method='linear' (REF summarize_deltas); 95% = [2.5, 97.5], 90% = [5, 95]"},
    "fold10": {"n_records": N_F10, "n_patients": N_F10_PATIENTS, "label_positives": FOLD10_POSITIVES,
               "labels_csv_sha256": LABELS_SHA},
    "D0_metric_check": D0,
    "1_primary": SECTION[ARCHS[0]],
    "2_robustness": SECTION[ARCHS[1]],
    "3_secondary_per_seed": D3,
    "4_disagreements": {"note": "reported as they are; none is resolved by selection (FRP reporting_order 4)", "table": D4},
    "5_exploratory": {"label": "EXPLORATORY (FRP exploratory_fold10_comparisons, closed list)",
                      "E1_physio_minus_signal_only": E1, "E2_per_class_deltas_H3_pair": E2,
                      "E3_acquisition_permutation": E3},
    "D6_descriptive_components": "macro-AUROC of each arm: macro_auroc_A / macro_auroc_B in sections 1-3 and E1; per-class "
                                 "AUROC of signal_plus_physio / signal_plus_all (seed ensemble): E2 auroc_A / auroc_B",
    "D7_sanity": D7,
    "bootstrap_deltas_csv_sha256": BOOT_SHA,
    "probability_files_sha256": {n: r["sha256"] for n, r in FROZEN["outputs"].items()},
    "versions": VERSIONS, "wfdb": WFDB_VERSION, "gpu": GPU_NAME, "cudnn": CUDNN_VERSION,
    "warnings": ANALYSIS_WARNINGS,
    "interpretation_post_fold10_rule_2": INTERPRETATION_POST_FOLD10_RULE_2,
    "post_opening_policy": POST_OPENING_POLICY,
}
RESULTS_SELF_SHA, _written = compare_or_write_json(RESULTS_PATH, RESULTS, "v0.9_results.json")
log_event(RUN_LABEL, FROZEN_ATTEMPT, "results_written" if _written else "results_verified",
          results_self_sha256=RESULTS_SELF_SHA, results_file_sha256=sha256_of(RESULTS_PATH))


# --- printing in FRP reporting_order: 1 primary, 2 robustness, 3 per-seed, 4 disagreements, 5 exploratory ---
def _fmt_ci(r):
    return "point {:+.4f} | 95% CI [{:+.4f}, {:+.4f}] | 90% CI [{:+.4f}, {:+.4f}] | valid {} NaN {}{}".format(
        r["point_delta"], r["lo95"], r["hi95"], r["lo90"], r["hi90"], r["n_valid"], r["n_nan"],
        " (NaN FLAG)" if r["nan_flag"] else "")


hdr("RESULTS (FRP reporting_order; full precision in v0.9_results.json)")
for _title, _key in (("1. PRIMARY - H3 verdict", "1_primary"), ("2. ROBUSTNESS - same procedure, same verdict table", "2_robustness")):
    _r = RESULTS[_key]
    print("\n{}: {} | delta = macro-AUROC({}) - macro-AUROC({}), seed ensemble".format(_title, _r["arch"], _r["arm_B"], _r["arm_A"]))
    print("  macro-AUROC A {:.4f} | B {:.4f}".format(_r["macro_auroc_A"], _r["macro_auroc_B"]))
    print("  " + _fmt_ci(_r))
    print("  a_holds {} | b_holds {} | material {} | sign {}".format(
        _r["h3_verdict"]["a_holds"], _r["h3_verdict"]["b_holds"], _r["h3_verdict"]["material"], _r["h3_verdict"]["sign"]))
    print("  VERDICT: {}".format(_r["h3_verdict"]["verdict"]))
    print("  sealed rules holding: {}".format(_r["h3_verdict"]["sealed_rules_holding"]))
    for _k in ("a", "b", "c"):
        print("    sealed rule {}: {}".format(_k, SEALED_DECISION[_k]))
print("\n3. SECONDARY - per seed (same BOOT_IDX, same seed in both arms):")
for _r in D3:
    print("  {:14s} seed {:4d}: A {:.4f} B {:.4f} | {} | {}: {}".format(
        _r["arch"], _r["seed"], _r["macro_auroc_A"], _r["macro_auroc_B"], _fmt_ci(_r), SECONDARY_VERDICT_LABEL,
        _r["h3_verdict_secondary"]["verdict"]))
print("\n4. DISAGREEMENTS (reported, never resolved by selection):")
for _r in D4:
    print("  {:5s} {} : {} vs {} -> {}".format(_r["item"], _r["comparison"], _r["value_1"], _r["value_2"],
                                              "DISAGREE" if _r["disagreement"] else "agree"))
print("\n5. EXPLORATORY (closed list; no decision label):")
for _r in E1:
    print("  E1 {:14s} signal_plus_physio - signal_only: A {:.4f} B {:.4f} | {}".format(
        _r["arch"], _r["macro_auroc_A"], _r["macro_auroc_B"], _fmt_ci(_r)))
for _r in E2:
    print("  E2 {:14s} per-class delta (all - physio), point [95% CI]: ".format(_r["arch"]) + " | ".join(
        "{} {:+.4f} [{:+.4f}, {:+.4f}]".format(sc, _r["per_class"][sc]["point_delta"], _r["per_class"][sc]["lo95"],
                                               _r["per_class"][sc]["hi95"]) for sc in SUPERCLASSES))
for _r in E3:
    print("  E3 {:14s} acquisition permutation ({} reps): unpermuted {:.4f} | drop mean {:+.4f} [min {:+.4f}, max {:+.4f}]".format(
        _r["arch"], _r["n_perm"], _r["unpermuted_macro_auroc"], _r["drop_mean"], _r["drop_min"], _r["drop_max"]))
print("\nD0 fast_auc vs sklearn: max |diff| {:.1e} | D7 sanity: resnet1d_wang signal_only ensemble macro-AUROC {:.4f}{}".format(
    D0["max_abs_diff"], D7["resnet1d_wang_signal_only_ensemble_macro_auroc"], " (BELOW the development sanity gate)" if D7["below"] else ""))


# =====================================================================================
# MANIFEST (write-once first manifest; later executions: v0.9_manifest_reexec_<n>.json)
# =====================================================================================
hdr("MANIFEST")
assert prereg_snapshot() == PREREG_SNAPSHOT_START, "PREREG_DIR CHANGED during v0.9 - aborting."
assert dir_snapshot(V08_DIR) == V08_SNAPSHOT_START, "outputs\\v0.8 CHANGED during v0.9 - aborting."
_files_now = sorted(os.listdir(OUT_DIR))
_allowed_final = set([os.path.basename(ATTEMPT_LOG_PATH)] + PROB_NAMES + PERM_NAMES + WRITE_ONCE_NAMES)
_extra = [f for f in _files_now if f not in _allowed_final and not REEXEC_MANIFEST_RE.match(f)
          and not f.endswith(".ots") and not f.endswith(".ots.bak")]
assert not _extra, "outputs\\v0.9 contains files outside the closed output list: {} - aborting.".format(_extra)
assert sorted(k for k in RESULTS if k[0].isdigit()) == ["1_primary", "2_robustness", "3_secondary_per_seed",
                                                         "4_disagreements", "5_exploratory"]
if not os.path.exists(MANIFEST_PATH):
    _manifest_target = MANIFEST_PATH
else:
    _ns = [int(REEXEC_MANIFEST_RE.match(f).group(1)) for f in _files_now if REEXEC_MANIFEST_RE.match(f)]
    _manifest_target = os.path.join(OUT_DIR, "{}manifest_reexec_{}.json".format(PREFIX, max(_ns, default=0) + 1))
ATTEMPT_LOG = read_attempt_log()
_outputs = [{"file": os.path.join(OUT_DIR, f), "bytes": os.path.getsize(os.path.join(OUT_DIR, f)),
             "sha256": sha256_of(os.path.join(OUT_DIR, f))} for f in _files_now]
MANIFEST = {
    "code_version": CODE_VERSION, "code_revision": CODE_REVISION, "state_at_start": STATE,
    "run_started": RUN_STARTED, "run_finished": datetime.now().astimezone().isoformat(),
    "versions": VERSIONS, "wfdb": WFDB_VERSION, "gpu": GPU_NAME, "cudnn": CUDNN_VERSION,
    "parent_documents": {k: {"self_sha256": DOC_SELF[k], "file_sha256": DOC_FILE[k]} for k in DOC_PATHS},
    "ots_files": OTS_FILES, "ots_note": OTS_NOTE, "v0.8_manifest_ots": V08_MANIFEST_OTS_INFO,
    "earlier_manifests_sha256": {"v0.3_manifest.json (from final_run_protocol, not opened)": V03_MAN_SHA_FROM_FRP,
                                 "v0.4_manifest.json": V04_MAN_SHA, "v0.5_manifest.json": V05_MAN_SHA,
                                 "v0.6_manifest.json": V06_MAN_SHA, "v0.7_manifest.json": V07_MAN_SHA,
                                 "v0.8_manifest.json": V08_MAN_SHA},
    "input_files_sha256": INPUT_HASHES, "v0.4_signal_cache_sha256": CACHE_HASHES,
    "reference_implementation_sha256": REFERENCE_CODE_SHA256,
    "v0.8_state_dict_sha256": {"{}_{}_seed{}".format(a, m, s): V08_STATE_SHA[(a, m, s)] for a, m, s in RUN_ORDER},
    "prereg_snapshot": PREREG_SNAPSHOT_START, "outputs_v0.8_snapshot": V08_SNAPSHOT_START,
    "fold10_access": FOLD10_ACCESS_LINE,
    "locked_quantity_checks": LOCKED_CHECKS,
    "lock_hashes": LOCK_HASHES,
    "pre_opening_validation": FROZEN["pre_opening_validation"],
    "identity_guard": FROZEN["identity_guard"],
    "fold10_signal_checks": FROZEN["fold10_signal_checks"],
    "frozen_outputs": {"self_sha256": FREEZE_SELF_SHA, "file_sha256": FREEZE_FILE_SHA, "attempt": FROZEN_ATTEMPT},
    "results": {"self_sha256": RESULTS_SELF_SHA, "file_sha256": sha256_of(RESULTS_PATH)},
    "labels_csv_sha256": LABELS_SHA, "bootstrap_deltas_csv_sha256": BOOT_SHA,
    "crash_and_rerun_rule_v0.9": CRASH_RULE_V09, "post_fold10_rules": POST_FOLD10_RULES,
    "interpretation_post_fold10_rule_2": INTERPRETATION_POST_FOLD10_RULE_2,
    "post_opening_policy": POST_OPENING_POLICY,
    "attempt_log_entries": ATTEMPT_LOG,
    "warnings": WARNINGS, "part_b_warnings_from_freeze_file": FROZEN["part_b_warnings"],
    "flags": {"fold10_opened": True, "training_performed": False, "labels_read_only_after_freeze": True,
              "prereg_dir_unchanged": True, "outputs_v0.8_unchanged": True, "closed_output_list_respected": True},
    "outputs": _outputs,
}
MANIFEST_SELF_SHA = write_once_json_atomic(_manifest_target, MANIFEST)
print("[saved] {} (write-once, self_sha256 {})".format(_manifest_target, MANIFEST_SELF_SHA))
print("manifest sha256:", sha256_of(_manifest_target))
for _o in _outputs:
    print("  {}  {}".format(_o["sha256"], os.path.basename(_o["file"])))


# =====================================================================================
# FINAL SUMMARY
# =====================================================================================
hdr("SUMMARY - " + CODE_VERSION + " " + CODE_REVISION)
print("state at start: {} | freeze self_sha256 {} | results self_sha256 {}".format(STATE, FREEZE_SELF_SHA, RESULTS_SELF_SHA))
print("PRIMARY ({}): {}".format(ARCHS[0], RESULTS["1_primary"]["h3_verdict"]["verdict"]))
print("ROBUSTNESS ({}): {}".format(ARCHS[1], RESULTS["2_robustness"]["h3_verdict"]["verdict"]))
print("\nwarnings ({}):".format(len(WARNINGS)))
for w in WARNINGS:
    print("  - " + w)
assert prereg_snapshot() == PREREG_SNAPSHOT_START, "PREREG_DIR CHANGED during v0.9 - aborting."
assert dir_snapshot(V08_DIR) == V08_SNAPSHOT_START, "outputs\\v0.8 CHANGED during v0.9 - aborting."
print("\nv0.9: fold 10 opened once. 18 frozen model outputs evaluated with the locked protocol. No training.")
