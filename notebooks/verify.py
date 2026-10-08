# =====================================================================================
# Project: PTB-XL Metadata Audit  -  verify.py: read-only reproduction of every number in the paper
# Code version: verify-1.0 (VERIFY_VERSION)
# Date: 2026-10-07
#
# Purpose: every number that appears in the paper is either RECOMPUTED from the frozen artifacts and found identical to
#   the stored value, or READ from a hash-verified locked document. The run writes paper_numbers.json, the only source
#   the figures and tables may use.
#
# Run:  python verify.py --root D:\ptbxl-metadata-audit      (exit code 1 if any check FAILS)
#       or paste the whole file into ONE Jupyter cell          (ROOT defaults to D:\ptbxl-metadata-audit; a FAIL banner
#                                                              is printed instead of raising SystemExit)
#
# READ-ONLY: every input is opened in read mode; the only writes go to a NEW folder
#   ROOT\outputs\verify\run_<YYYYmmdd-HHMMSS>\ (exclusive create; the run STOPS if that folder exists). The sha256 of
#   every file in preregistration\, outputs\v0.2.1, v0.3, v0.8, v0.9, ph1, ph1_failed_*, notebooks\v0.9_cell.py and
#   notebooks\ph1_cell.py is taken at the start (V0) and compared at the end (V15).
# No ECG signal is opened; torch and wfdb are never imported. CPU only.
#
# Sections (each check: id, section, description, kind, expected, observed, max_abs_diff, tolerance, status):
#   V0  environment (versions WARN-only), cpu_count, threadpool_info; source of this code
#   V1  dataset input hashes                       V2  the five locked documents, parent chain, REF (.ots: INFO)
#   V3  v0.2.1 and v0.3 chains                     V4  v0.8 chain (hashes only; nothing is loaded)
#   V5  v0.9 chain                                 VP  provenance: every block copied from ph1 / v0.9 occurs verbatim
#   V6  fold-10 rows and labels                    V7  BOOT_IDX / PERM_IDX
#   V8  H3 + exploratory results (v0.9 Part D code, verbatim) deep-compared with v0.9_results.json
#   V9  descriptive per-arm macro-AUROC + 95% CI (descriptive, not pre-specified)
#   V10 metadata-only fold 9 (H1 / H2) with the approved ph1 functions; ph1 r2 tolerances
#   V11 ph1 (post hoc, fold 10) and the r1 attempt in outputs\ph1_failed_1
#   V12 locked settings (read)                     V13 dataset descriptives (v0.2.1 label code)
#   V14 provenance (INFO; Bitcoin anchors checked manually)      V15 end snapshot
#
# CODE PROVENANCE MARKERS: '# >>> verbatim from ph1 (label)' ... '# <<< verbatim from ph1' is byte-identical to a
#   contiguous excerpt of notebooks\ph1_cell.py (approved r2, sha256 asserted); '# >>> verbatim from v0.9 (label)' ...
#   likewise for notebooks\v0.9_cell.py; '# >>> adapted from v0.9 (label) | indent: N' ... '# <<< adapted from v0.9' is
#   identical to an excerpt after removing N leading spaces from every non-blank line (no other change). Markers of
#   other sources nested inside a copied block are part of the copied text. VP re-checks every block at run time.
#   Everything without a marker is new verify code.
# =====================================================================================

import os, sys, io, re, json, ast, math, time, hashlib, platform, traceback, ntpath, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

VERIFY_VERSION = "verify-1.0"
DEFAULT_ROOT = r"D:\ptbxl-metadata-audit"

# >>> VERIFY CONSTANTS (the synthetic test harness replaces exactly this block; the delivered values are the real ones)
INPUT_SHA256 = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}
DOC_EXPECTED = {
    "pre_registration": {"name": "pre_registration.json",
                         "file_sha256": "11dd446272654fc80d65b2fdd40259fa674bbb419a44fde84e589b6e297cde32",
                         "self_sha256": "5183f54569f3358a5339b16a2c3c3be49f5c7798eabe6b0bb25946119c2ba469"},
    "dev_protocol": {"name": "dev_protocol.json",
                     "file_sha256": "babf101c1d69b322daef6907175fdc4c17b5a53755a68f13a998eefa0148dc2d",
                     "self_sha256": "c967ac38c8cc9382bf41fdd3785c8bbd730db196e349846b8052d856bbac5fc6"},
    "training_details": {"name": "v0.5_training_details.json",
                         "file_sha256": "6480c191aeaed9ff1e6b1da5caa0c1e46908d453d30106a32beb7cf83678be11",
                         "self_sha256": "c12c33a2f649984fb1b70a46445823c46b57d1b024d87655e16661789f04b46f"},
    "lr_selection": {"name": "lr_selection.json",
                     "file_sha256": "358d783b37bba6fcf61237a43817c1fd8594fe192f8ecdfca2c25b871a21415b",
                     "self_sha256": "c70cbfa65e0ab75321dfef9c64714a64705c60a967685d79a878a1d40450749d"},
    "final_run_protocol": {"name": "final_run_protocol.json",
                           "file_sha256": "e1a5260e72d2e8b0d27dc709639468b7f9a48a7c8e2f8af2d663e105dbd329d4",
                           "self_sha256": "834acddec288862d0418e9ee70204898c9d371788eed7eb362f4c0d63d305c6b"},
}
REFERENCE_CODE_SHA256_EXPECTED = "48ac837c07880f0bd88a7bd01475e7da3edbeab46019f8288976884ad7662856"
V021_DECISION_TABLE_SHA256 = "20bffb2c6df30e16a070184f91f7211bddc14ede9bc2de39273b5be1cc32359b"
V021_MANIFEST_SHA256_PRINTED = "37ca9b785ac2c1d78dd3e28f729dc64d289d42ae80803adacbe8a406122b23f6"   # INFO / WARN only
V03_MANIFEST_SHA256 = "c7e1bec283d11a3d4cd94a3f2509640651e8069d26424cfd14b5d6b975053e35"
V08_MANIFEST_SHA256 = "9725a0ee502784ffc267b1021ec3cb427966708205db9a09ce7e8c1fc8f30415"
V09_CELL_SHA256 = "9af1773dba86039cfa2b191f2aaf0e54c02f9ccb90880b60471074e2c6fe4c7b"
APPROVED_PH1_CELL_SHA256 = "6136c633d52f6b0dc75260342f2d6da5fe74ed167f09d94579b4afaab437fe09"
V09_DOCS = {
    "v0.9_frozen_outputs.json": {"file_sha256": "133ca5fe9c9faf299cec5270483eb6ac2abf568a1c6fcc6bb5998d351528867f",
                                 "self_sha256": "8dfa1ebdfede5b3a9a3360bcac1c161be475758ffe7cd7219854ee3b6f9ee95f"},
    "v0.9_results.json": {"file_sha256": "8996ac975f7091fd07d0127521d8413e14e957f878a930334fdf69983bbc506f",
                          "self_sha256": "77c9cae439edf7e0b31757c392853354033470c3fc154578d953b5f7a4b8e3ba"},
    "v0.9_manifest.json": {"file_sha256": "710e3519bbb750fc427cf9bafe2f6075806610401d1f52f7682f1670417cdca6",
                           "self_sha256": "2007bee22651c8dc8cc9c9de6f47a5eb79d7bdfdbf523c377fd799fb2984c612"},
}
V09_FILES = {
    "v0.9_fold10_labels.csv": "29051bd87676a59b70b3a79a91075f70a7cddd028b2c297461c7bbae3f15b3b0",
    "v0.9_bootstrap_deltas.csv": "8ecee19cb0b2fa9393fee74636af1682c6b3692cfbf81da56d8d21ee88b6a5e9",
    "v0.9_resnet1d_wang_perm_probs.npy": "2a6bfa0d093053483404a81ba3ec8b1eeb59cdfc38fd212800f92b077777645c",
    "v0.9_inception1d_perm_probs.npy": "0f9473d12cc991b4e5e53b7c73e7d467ea8e95f525c0cf49d78d830f78cae823",
    "v0.9_attempt_log.jsonl": "a83b1f03d36ec5a68a1f7808836372ed18d9951532b3fdc020bdfa7be2833dc6",
}
PH1_DOCS = {
    "ph1_analysis_plan.json": {"file_sha256": "aaa18d93c5c18813cea2a31dcfc35d7cc604335dc2627cdb069e82228a52f20a",
                               "self_sha256": "34effc0fda5cb2b7fda1bce1fc43de0e17aa55f566a0902de75455e33a31c5c0"},
    "ph1_frozen_outputs.json": {"file_sha256": "1fdfc9bf06cd9aedd77fc265c72151707de12309af1772adb9cd72a657542381",
                                "self_sha256": "a3183532cfb48ada58e299ce77df2484df59c3e607181d9c5012f59acb7d8fff"},
    "ph1_results.json": {"file_sha256": "1947d0779c3f2f1a01cd86eddb0b51f1552aeda03a12f16270350d3043b94b1d",
                         "self_sha256": "f096bf9390c2cf5ee0d43a453482049ac07051ea4aeeb24b5aa01a6315f23b94"},
    "ph1_manifest.json": {"file_sha256": "f056cb2c225f46e18be0e0ccf1e718ae8335d4c59f3341684635ab59f57c4d87",
                          "self_sha256": "406006ac40bdebc83d6a0cccbcf07fab1c2faa59f390a83667fe93b26b6a9780"},
}
PH1_FILES = {
    "ph1_fold10_probs.csv": "4519b99c90c6e9e575f12b15819c42ca25cf391c3577760e0a4f3a60ea30f29b",
    "ph1_decision_table.csv": "517a9482264f6d234d2fb435565d176f7e60139e737016b774470d5804bbeb3e",
    "ph1_bootstrap_deltas.csv": "139933ef584d83034082b19f941af918f078e15a75f548f4feadedba9f946165",
    "ph1_metric_rows.csv": "f4eef7720ef29e1e7c13dce90ca371031e25fd315678d3fc09d52c2a596d75fb",
}
PH1_R1_DOCS = {
    "ph1_analysis_plan.json": {"file_sha256": "0b1ad3f68f0eb925e5ec2bf86db575198ecba6839d3a0c9e26be6705cbb72252",
                               "self_sha256": "bacbd0b2a56f366add7416e38ff9181d7b50b2c05220a6eb84918a7078f0311f"},
    "ph1_partA_failure.json": {"file_sha256": "d7b7d5e1bbdb07f02ca58664a116b83ac1c7109c023f5d6b11f28e916d6e9415",
                               "self_sha256": "d48b8c77ccbdc957d54f0907503e7463197fb2570e6ec26c7bb67e2c40c77807"},
}
PH1_PLAN_UTF8_SHA256 = "bb9669de4e54b0530db3eef0b42fe4fd8378da4a084d142df6750bc61b182c17"
PH1_AMENDMENT_R2_UTF8_SHA256 = "e0255ebbd7588f84523577bd5bb3d480fe52d8d0bbada4668e2e59d57f651094"
N_PATIENTS_ALL_EXPECTED = 18869
N_RECORDS_ALL_EXPECTED = 21799
# <<< VERIFY CONSTANTS

REFERENCE_PUBLISHED = {"resnet1d_wang": 0.930, "inception1d": 0.921,
                       "source": "Strodthoff et al., IEEE JBHI 2021, PTB-XL diagnostic superclasses "
                                 "(helme/ecg_ptbxl_benchmarking README)"}
BITCOIN_ANCHORS = [
    {"block": 969577, "time": "2026-10-02", "time_precision": "date",
     "documents": ["pre_registration.json", "dev_protocol.json", "v0.5_training_details.json", "lr_selection.json"]},
    {"block": 969584, "time": "2026-10-02 15:21:18 +03:30", "time_precision": "second",
     "documents": ["final_run_protocol.json"]},
    {"block": 969763, "time": "2026-10-04 00:48:05 +03:30", "time_precision": "second",
     "documents": ["v0.8_manifest.json"]},
    {"block": 969863, "time": "2026-10-04", "time_precision": "date",
     "documents": ["v0.9_frozen_outputs.json", "v0.9_results.json", "v0.9_manifest.json"]},
]
PROVENANCE_ROLES = {
    "pre_registration.json": "sealed pre-registration: H1-H3, fold-10 analysis plan, SESOI",
    "dev_protocol.json": "development protocol: training recipe, lr grid, sanity gate",
    "v0.5_training_details.json": "training / evaluation implementation details",
    "lr_selection.json": "learning-rate selection on fold 9 (signal_plus_physio, seed 42)",
    "final_run_protocol.json": "final-run and fold-10 analysis protocol (bootstrap, H3 decision, closed list)",
    "v0.8_manifest.json": "final models: state_dict hashes, written before fold 10 was opened",
    "v0.9_frozen_outputs.json": "fold-10 model outputs frozen before any label was read",
    "v0.9_results.json": "fold-10 H3 and exploratory results",
    "v0.9_manifest.json": "v0.9 run manifest (every v0.9 output file hash)",
}
METADATA_ONLY_NOTE = ("ci95 = mean of the three per-seed marginal 95% percentile CIs (v0.2.1 convention); for "
                      "LogisticRegression the three seeds are identical")
ABS_TOL = 1e-12
WARNINGS = []                 # filled by the ph1 helper warn() (messages of the copied ph1 / v0.2.1 code)
REF = {}                      # the reference implementation stored in final_run_protocol.json (exec'd in V2)

# ------------------------------------- copied constants and helpers -------------------------------------
# >>> verbatim from ph1 (constants: classes, seeds, learners, blocks, A2_TOLERANCE, environment)
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
# <<< verbatim from ph1
# >>> verbatim from ph1 (fold-9 / fold-10 counts and split sizes)
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
# <<< verbatim from ph1
# >>> verbatim from ph1 (helpers: sha256_of, hdr, warn, note, _json_default, fast_auc, to_level_string, encode_categorical, canonical JSON, verify_self_sha256, plain, manifest_entry, write-once writers, csv_bytes)
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
# <<< verbatim from ph1
# >>> verbatim from ph1 (abs_diff_matrix, cell_tolerance)
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

# <<< verbatim from ph1
# >>> verbatim from v0.9 (sklearn check tolerance)
SKLEARN_CHECK_TOL = 1e-9
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (architectures, arms)
ARCHS = ["resnet1d_wang", "inception1d"]
ARMS = ["signal_only", "signal_plus_physio", "signal_plus_all"]
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (NaN flag fraction)
NAN_FLAG_FRACTION = 0.01               # FRP bootstrap.nan_rule: "if > 1% of draws are NaN this is flagged"
# <<< verbatim from v0.9
# >>> verbatim from v0.9 (sanity gate, secondary verdict label)
SANITY_GATE_MIN_AUROC = 0.90           # D7 WARN-only (dev_protocol pipeline_sanity_gate threshold)
SECONDARY_VERDICT_LABEL = "secondary (per-seed); not the H3 verdict"
# <<< verbatim from v0.9


# ------------------------------------- the approved ph1 functions (a) and (b) -------------------------------------
# >>> verbatim from ph1 (fit_predict and metrics_from_probs)
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
# <<< verbatim from ph1


# ------------------------------------- v0.9 Part D (fold-10 analysis) -------------------------------------
# >>> verbatim from v0.9 (load_saved_probs10)
def load_saved_probs10(path):
    pdf = pd.read_csv(path, encoding="utf-8-sig")
    assert list(pdf.columns) == ["ecg_id", "patient_id"] + ["prob_" + sc for sc in SUPERCLASSES] and len(pdf) == N_F10  # [v0.9 added]
    assert (pdf["patient_id"].to_numpy() == F10_IDS[:, 1]).all()  # [v0.9 added]
    assert (pdf["ecg_id"].to_numpy() == F10_IDS[:, 0]).all()
    return pdf[["prob_" + sc for sc in SUPERCLASSES]].to_numpy(dtype=np.float64)
# <<< verbatim from v0.9


ANALYSIS_WARNINGS = []


def warn_analysis(msg):
    """v0.9 Part D reports NaN flags and the sanity gate with warn_analysis; verify records them as WARN checks."""
    ANALYSIS_WARNINGS.append(msg)
    print("[WARN] " + msg)


def v09_part_d(Y10, _rows10, BOOT_IDX, ENS, P, PERM, N_PERM, PERM_SEED, SESOI_H3, SEALED_DECISION):
    """v0.9 Part D (D0 .. bootstrap-delta frame), copied verbatim; only the indentation differs."""
    t0 = time.time()
# >>> adapted from v0.9 (Part D: D0, D1-D4, E1-E3, D7, bootstrap-delta frame) | indent: 4
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
# <<< adapted from v0.9
    return {"D0": D0, "SECTION": SECTION, "D3": D3, "D4": D4, "E1": E1, "E2": E2, "E3": E3, "D7": D7,
            "BOOT_DF": BOOT_DF, "CD_ENS": CD_ENS}


def v09_labels_frame(F10_IDS, LAB10):
    """The v0.9 Part C fold-10 label frame (LABELS10) and Y10."""
# >>> adapted from v0.9 (Part C: fold-10 label frame) | indent: 4
    LABELS10 = pd.DataFrame({"ecg_id": F10_IDS[:, 0], "patient_id": F10_IDS[:, 1],
                             **{c: LAB10[c].to_numpy().astype(np.int64) for c in SUPERCLASSES}})
# <<< adapted from v0.9
# >>> adapted from v0.9 (Part C: Y10) | indent: 4
    Y10 = LABELS10[SUPERCLASSES].to_numpy(dtype=np.int64)
# <<< adapted from v0.9
    return LABELS10, Y10


def v09_seed_ensembles(P, ARCHS, ARMS, SEEDS):
# >>> adapted from v0.9 (Part D: seed ensembles) | indent: 4
    ENS = {(arch, arm): np.mean([P[(arch, arm, s)] for s in SEEDS], axis=0) for arch in ARCHS for arm in ARMS}
# <<< adapted from v0.9
    return ENS


# =====================================================================================
# check bookkeeping
# =====================================================================================
CHECKS = []
STATE = {}
STATUS_ORDER = ("PASS", "FAIL", "WARN", "SKIP", "INFO")


class Skip(Exception):
    """A dependency of a check is missing (an earlier check failed or an input is absent)."""


def _jsonable(o):
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def record(section, cid, description, kind, status, expected=None, observed=None, max_abs_diff=None, tolerance=None,
           **extra):
    assert status in STATUS_ORDER, status
    rec = {"id": cid, "section": section, "description": description, "kind": kind, "expected": expected,
           "observed": observed, "max_abs_diff": max_abs_diff, "tolerance": tolerance, "status": status}
    rec.update(extra)
    rec = _jsonable(json.loads(json.dumps(rec, default=_json_default, allow_nan=True)))
    CHECKS.append(rec)
    line = "  [{:4s}] {:4s} {} - {}".format(status, section, cid, description)
    if max_abs_diff is not None:
        line += " | max |diff| {}".format(max_abs_diff)
    if status in ("FAIL", "WARN", "SKIP"):
        line += " | expected {} | observed {}".format(str(rec["expected"])[:160], str(rec["observed"])[:160])
        if extra.get("note"):
            line += " | " + str(extra["note"])[:200]
    print(line)
    return rec


def check_eq(section, cid, description, observed, expected, kind="hash", **extra):
    """Exact equality (strings, booleans, hashes, integers, structures)."""
    return record(section, cid, description, kind, "PASS" if observed == expected else "FAIL", expected=expected,
                  observed=observed, **extra)


def check_true(section, cid, description, ok, kind="read_locked", **extra):
    return record(section, cid, description, kind, "PASS" if ok else "FAIL", **extra)


def info(section, cid, description, observed, **extra):
    return record(section, cid, description, "informational", "INFO", observed=observed, **extra)


def _is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def deep_compare(got, exp, tol):
    """Leaf-by-leaf comparison of two JSON-like structures. Dict key sets and list lengths must be equal; numbers within
    tol (NaN only matches NaN); strings / booleans / None exactly. Returns (n_leaves, max_abs_diff, mismatches)."""
    mism, maxd, n = [], [0.0], [0]

    def rec(g, e, p):
        if isinstance(e, dict):
            if not isinstance(g, dict):
                mism.append({"path": p, "expected": "dict", "observed": type(g).__name__})
                return
            for k in sorted(set(e) | set(g), key=str):
                if k not in g or k not in e:
                    mism.append({"path": "{}.{}".format(p, k), "expected": "present" if k in e else "absent",
                                 "observed": "present" if k in g else "absent"})
                    continue
                rec(g[k], e[k], "{}.{}".format(p, k))
        elif isinstance(e, (list, tuple)):
            if not isinstance(g, (list, tuple)) or len(g) != len(e):
                mism.append({"path": p, "expected": "list of {}".format(len(e)),
                             "observed": "list of {}".format(len(g)) if isinstance(g, (list, tuple)) else type(g).__name__})
                return
            for i, (gi, ei) in enumerate(zip(g, e)):
                rec(gi, ei, "{}[{}]".format(p, i))
        elif _is_num(e) and _is_num(g):
            n[0] += 1
            g_, e_ = float(g), float(e)
            if math.isnan(g_) and math.isnan(e_):
                return
            d = abs(g_ - e_)
            if not math.isfinite(d) or d > tol:
                mism.append({"path": p, "expected": e, "observed": g, "abs_diff": d})
            if math.isfinite(d):
                maxd[0] = max(maxd[0], d)
        else:
            n[0] += 1
            if type(g) != type(e) or g != e:
                mism.append({"path": p, "expected": e, "observed": g})
    rec(got, exp, "$")
    return n[0], maxd[0], mism


def check_struct(section, cid, description, got, exp, tol=ABS_TOL, kind="recomputed"):
    n, md, mism = deep_compare(plain(got), plain(exp), tol)
    return record(section, cid, description, kind, "PASS" if not mism else "FAIL", max_abs_diff=md, tolerance=tol,
                  n_leaves=n, n_mismatches=len(mism), mismatches=mism[:20],
                  expected="(structure)" if not mism else mism[0].get("expected"),
                  observed="(structure)" if not mism else mism[0].get("observed"))


def check_frame(section, cid, description, got, exp, id_cols, num_cols, tol=ABS_TOL, kind="recomputed"):
    """Same columns, rows, identifier columns exactly; numeric columns within tol (NaN only where expected is NaN)."""
    shape_ok = list(got.columns) == list(exp.columns) and len(got) == len(exp)
    ids_ok = shape_ok and all(np.array_equal(got[c].to_numpy(), exp[c].to_numpy()) for c in id_cols)
    if not ids_ok:
        return record(section, cid, description, kind, "FAIL", expected={"columns": list(exp.columns), "rows": len(exp)},
                      observed={"columns": list(got.columns), "rows": len(got), "identifiers_equal": False},
                      tolerance=tol)
    D = abs_diff_matrix(got, exp, num_cols)
    md = float(D.max()) if D.size else 0.0
    per_col = {c: float(D[:, j].max()) if len(D) else 0.0 for j, c in enumerate(num_cols)}
    return record(section, cid, description, kind, "PASS" if md <= tol else "FAIL", max_abs_diff=md, tolerance=tol,
                  observed=per_col, expected="identical within tolerance")


def need(*keys):
    missing = [k for k in keys if k not in STATE]
    if missing:
        raise Skip("dependency not available (an earlier check failed or an input is missing): {}".format(missing))


def run_section(code, title, fn):
    hdr("{} - {}".format(code, title))
    t0 = time.time()
    try:
        fn()
    except Skip as e:
        record(code, "{}_skipped".format(code), "{}: remaining checks of this section".format(title), "informational",
               "SKIP", note=str(e))
    except Exception as e:
        record(code, "{}_exception".format(code), "{}: the section raised an exception".format(title), "informational",
               "FAIL", observed="{}: {}".format(type(e).__name__, e), traceback=traceback.format_exc()[-4000:])
    print("  ({} done in {:.1f}s)".format(code, time.time() - t0))


def base(p):
    return re.split(r"[\\/]", str(p))[-1]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def self_and_file(section, cid, path, exp_file, exp_self, label):
    """File sha256 and self_sha256 (stored == recomputed == expected) of a self-hashed JSON document; returns the dict."""
    fs = sha256_of(path)
    check_eq(section, cid + "_file_sha256", "{} file sha256".format(label), fs, exp_file)
    s_, r_, d = verify_self_sha256(path)
    record(section, cid + "_self_sha256", "{} self_sha256 (stored == recomputed == expected)".format(label), "hash",
           "PASS" if s_ == r_ == exp_self else "FAIL", expected=exp_self, observed={"stored": s_, "recomputed": r_})
    return d


def manifest_outputs(section, cid, manifest, dirs, label):
    """Every entry of manifest['outputs'] exists in one of dirs (by basename) with the listed sha256 (and bytes)."""
    n_ok = 0
    for o in manifest["outputs"]:
        name = base(o["file"])
        hit = [os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))]
        if not hit:
            record(section, "{}_{}".format(cid, name), "{} entry {} exists".format(label, name), "hash", "FAIL",
                   expected=o["sha256"], observed="missing")
            continue
        got = sha256_of(hit[0])
        ok = got == o["sha256"] and ("bytes" not in o or os.path.getsize(hit[0]) == o["bytes"])
        n_ok += ok
        record(section, "{}_{}".format(cid, name), "{} entry {} == file on disk".format(label, name), "hash",
               "PASS" if ok else "FAIL", expected=o["sha256"], observed=got, stored_path=o["file"])
    return n_ok


def signed(x):
    return "{:+.4f}".format(x).replace("-", "\u2212")


def unsigned(x):
    return "{:.4f}".format(x).replace("-", "\u2212")


# =====================================================================================
# ROOT, paths, the source of this code
# =====================================================================================
def in_ipython():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


IN_IPYTHON = in_ipython()
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--root", default=DEFAULT_ROOT)
ROOT = _ap.parse_known_args()[0].root
DATA_ROOT = os.path.join(ROOT, "data", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
PREREG_DIR = os.path.join(ROOT, "preregistration")
NB_DIR = os.path.join(ROOT, "notebooks")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
V021_DIR, V03_DIR, V08_DIR, V09_DIR = (os.path.join(OUTPUTS_DIR, v) for v in ("v0.2.1", "v0.3", "v0.8", "v0.9"))
PH1_DIR = os.path.join(OUTPUTS_DIR, "ph1")
PH1_R1_DIR = os.path.join(OUTPUTS_DIR, "ph1_failed_1")
DB_CSV = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_CSV = os.path.join(DATA_ROOT, "scp_statements.csv")
V09_CELL_PATH = os.path.join(NB_DIR, "v0.9_cell.py")
PH1_CELL_PATH = os.path.join(NB_DIR, "ph1_cell.py")
VERIFY_FALLBACK_PATH = os.path.join(NB_DIR, "verify.py")
RUN_STARTED_AT = datetime.now()
RUN_ID = "run_" + RUN_STARTED_AT.strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join(OUTPUTS_DIR, "verify", RUN_ID)
SNAPSHOT_FOLDERS = ["preregistration", os.path.join("outputs", "v0.2.1"), os.path.join("outputs", "v0.3"),
                    os.path.join("outputs", "v0.8"), os.path.join("outputs", "v0.9"), os.path.join("outputs", "ph1")]


def own_source():
    """(text, route): this code as it runs. Script: __file__; Jupyter: the current cell (IPython _ih); otherwise
    ROOT\\notebooks\\verify.py, accepted only if every function compiled from it has the bytecode of the running one and
    its VERIFY CONSTANTS block evaluates to the running constants (the sha256 printed in V0 identifies the text)."""
    f = globals().get("__file__")
    if f and os.path.isfile(f):
        return read_bytes(f).decode("utf-8"), "script ({})".format(f)
    if IN_IPYTHON:
        try:
            ih = get_ipython().user_ns.get("_ih") or []  # noqa: F821
            for src in reversed(ih):
                if "VERIFY_VERSION = " in src and "def own_source" in src:
                    return src, "jupyter cell (_ih)"
        except Exception:
            pass
    if os.path.isfile(VERIFY_FALLBACK_PATH):
        text = read_bytes(VERIFY_FALLBACK_PATH).decode("utf-8")
        mod = compile(text, VERIFY_FALLBACK_PATH, "exec")
        bad = []
        for co in mod.co_consts:
            if hasattr(co, "co_code") and co.co_name in globals() and hasattr(globals()[co.co_name], "__code__"):
                if globals()[co.co_name].__code__.co_code != co.co_code:
                    bad.append(co.co_name)
        lf = text.replace("\r\n", "\n")
        i0, i1 = lf.find("\n# >>> VERIFY CONSTANTS"), lf.find("\n# <<< VERIFY CONSTANTS")
        consts = {}
        if i0 < 0 or i1 < 0:
            bad.append("VERIFY CONSTANTS block not found")
        else:
            exec(lf[i0:i1], {}, consts)
            bad += ["constant " + k for k, v in consts.items() if globals().get(k) != v]
        if bad:
            raise RuntimeError("notebooks\\verify.py differs from the running code in: {}".format(bad))
        return text, "fallback {} (bytecode of every function and every VERIFY constant == running code)".format(
            VERIFY_FALLBACK_PATH)
    raise RuntimeError("the source of the running code cannot be obtained (__file__, _ih, notebooks\\verify.py)")


def tree_files(folder):
    out = {}
    if not os.path.isdir(folder):
        return None
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for nm in sorted(filenames):
            p = os.path.join(dirpath, nm)
            out[os.path.relpath(p, ROOT).replace(os.sep, "/")] = sha256_of(p)
    return out


def snapshot():
    snap = {}
    folders = list(SNAPSHOT_FOLDERS)
    if os.path.isdir(OUTPUTS_DIR):
        folders += [os.path.join("outputs", n) for n in sorted(os.listdir(OUTPUTS_DIR)) if n.startswith("ph1_failed_")]
    for f in folders:
        snap[f.replace(os.sep, "/")] = tree_files(os.path.join(ROOT, f))
    for p in (V09_CELL_PATH, PH1_CELL_PATH):
        snap[os.path.relpath(p, ROOT).replace(os.sep, "/")] = sha256_of(p) if os.path.isfile(p) else None
    return snap


# =====================================================================================
# V0 - environment
# =====================================================================================
def sec_v0():
    vers = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    STATE["versions"] = dict(vers, os=platform.platform())
    for k, v in EXPECTED_VERSIONS.items():
        record("V0", "V0_version_" + k, "{} version == the v0.4-v0.9 / ph1 environment".format(k), "informational",
               "PASS" if vers[k] == v else "WARN", expected=v, observed=vers[k],
               note=None if vers[k] == v else "BOOT_IDX streams and HistGradientBoosting refits may differ across "
                                              "versions; the numeric checks below decide")
    info("V0", "V0_environment_note", "v0.2.1 and v0.3 originally ran with scipy 1.11.4; their reproduction under scipy "
         "1.15.3 is established by ph1 Part A (amendment_r2)", STATE["versions"])
    try:
        from threadpoolctl import threadpool_info
        tpi = plain(threadpool_info())
    except Exception as e:
        tpi = "unavailable: {}".format(e)
    info("V0", "V0_cpu_threads", "os.cpu_count() and threadpoolctl.threadpool_info() (recorded; no thread setting is "
         "changed)", {"cpu_count": os.cpu_count(), "threadpool_info": tpi})
    STATE["cpu_count"], STATE["threadpool_info"] = os.cpu_count(), tpi
    info("V0", "V0_root", "project root and run folder", {"root": ROOT, "run_dir": RUN_DIR, "jupyter": IN_IPYTHON})
    text, route = own_source()
    STATE["own_source"] = text
    sha = sha256_bytes(text.encode("utf-8"))
    STATE["own_source_sha256"] = sha
    info("V0", "V0_own_source", "source of the running verify code (sha256 of its UTF-8 text, LF line ends; compare with "
         "the sha256 delivered with verify.py)", {"route": route, "sha256": sha, "n_lines": text.count("\n")})


# =====================================================================================
# V1 - dataset inputs
# =====================================================================================
def sec_v1():
    for name, exp in INPUT_SHA256.items():
        p = os.path.join(DATA_ROOT, name)
        got = sha256_of(p) if os.path.isfile(p) else "missing"
        check_eq("V1", "V1_" + name, "{} sha256".format(name), got, exp)
        if got == exp:
            STATE["input_ok_" + name] = True


# =====================================================================================
# V2 - locked documents, reference implementation
# =====================================================================================
def sec_v2():
    docs = {}
    for key, e in DOC_EXPECTED.items():
        p = os.path.join(PREREG_DIR, e["name"])
        if not os.path.isfile(p):
            record("V2", "V2_{}_exists".format(key), "{} exists".format(e["name"]), "hash", "FAIL", observed="missing")
            continue
        docs[key] = self_and_file("V2", "V2_" + key, p, e["file_sha256"], e["self_sha256"], e["name"])
    need_keys = [k for k in DOC_EXPECTED if k not in docs]
    if need_keys:
        raise Skip("locked documents missing: {}".format(need_keys))
    STATE["DOCS"] = docs
    frp, seal = docs["final_run_protocol"], docs["pre_registration"]
    STATE["FRP"], STATE["SEAL"] = frp, seal
    check_struct("V2", "V2_frp_parent_hashes", "final_run_protocol.parent_hashes == the four parent documents (self + "
                 "file sha256 constants)", frp["parent_hashes"],
                 {k: {"self_sha256": DOC_EXPECTED[k]["self_sha256"], "file_sha256": DOC_EXPECTED[k]["file_sha256"]}
                  for k in ("pre_registration", "dev_protocol", "training_details", "lr_selection")}, kind="hash")
    check_eq("V2", "V2_dev_protocol_parent", "dev_protocol.parent_seal_self_sha256 == pre_registration self_sha256",
             docs["dev_protocol"]["parent_seal_self_sha256"], DOC_EXPECTED["pre_registration"]["self_sha256"])
    check_eq("V2", "V2_training_details_parents", "v0.5_training_details parents == pre_registration / dev_protocol "
             "self_sha256", [docs["training_details"]["parent_seal_self_sha256"],
                             docs["training_details"]["parent_dev_protocol_self_sha256"]],
             [DOC_EXPECTED["pre_registration"]["self_sha256"], DOC_EXPECTED["dev_protocol"]["self_sha256"]])
    check_eq("V2", "V2_lr_selection_parents", "lr_selection.parent_hashes == the three parent self_sha256",
             docs["lr_selection"]["parent_hashes"],
             {"pre_registration": DOC_EXPECTED["pre_registration"]["self_sha256"],
              "dev_protocol": DOC_EXPECTED["dev_protocol"]["self_sha256"],
              "training_details": DOC_EXPECTED["training_details"]["self_sha256"]})
    check_eq("V2", "V2_frp_input_hashes", "final_run_protocol.input_files_sha256 == the dataset input constants",
             frp["input_files_sha256"], INPUT_SHA256)
    code = frp["reference_implementation"]["code"]
    sha = sha256_bytes(code.encode("utf-8"))
    ok = sha == frp["reference_implementation"]["sha256_of_utf8_code"] == REFERENCE_CODE_SHA256_EXPECTED
    record("V2", "V2_reference_implementation", "reference implementation code (UTF-8) sha256 == FRP field == constant",
           "hash", "PASS" if ok else "FAIL", expected=REFERENCE_CODE_SHA256_EXPECTED,
           observed={"code": sha, "frp_field": frp["reference_implementation"]["sha256_of_utf8_code"]})
    if ok:
        ns = {}
        exec(code, ns)
        REF.clear()
        REF.update(ns)
        STATE["REF"] = True
    ots = {}
    for folder in (PREREG_DIR, V08_DIR, V09_DIR, PH1_DIR, NB_DIR):
        if os.path.isdir(folder):
            for nm in sorted(os.listdir(folder)):
                if nm.endswith(".ots"):
                    p = os.path.join(folder, nm)
                    ots[os.path.relpath(p, ROOT).replace(os.sep, "/")] = {"sha256": sha256_of(p),
                                                                         "bytes": os.path.getsize(p)}
    info("V2", "V2_ots_files", "OpenTimestamps proofs present (listed only, never asserted)", ots)


# =====================================================================================
# V3 - v0.2.1 and v0.3 chains
# =====================================================================================
def sec_v3():
    man_p = os.path.join(V021_DIR, "v0.2.1_manifest.json")
    msha = sha256_of(man_p)
    record("V3", "V3_v021_manifest_sha256", "v0.2.1_manifest.json sha256 (the v0.2.1 run printed a hash starting "
           "37ca9b78; recorded, WARN only)", "informational", "INFO" if msha == V021_MANIFEST_SHA256_PRINTED else "WARN",
           expected=V021_MANIFEST_SHA256_PRINTED, observed=msha)
    man = read_json(man_p)
    STATE["V021_MAN"] = man
    dt = os.path.join(V021_DIR, "v0.2.1_decision_table.csv")
    mf = os.path.join(V021_DIR, "v0.2.1_metrics_full.csv")
    dsha, fsha = sha256_of(dt), sha256_of(mf)
    check_eq("V3", "V3_v021_decision_table_sha256", "v0.2.1_decision_table.csv sha256 == constant", dsha,
             V021_DECISION_TABLE_SHA256)
    check_eq("V3", "V3_v021_decision_table_manifest", "v0.2.1_decision_table.csv sha256 == its v0.2.1 manifest entry",
             dsha, manifest_entry(man, "v0.2.1_decision_table.csv")["sha256"])
    check_eq("V3", "V3_v021_metrics_full_manifest", "v0.2.1_metrics_full.csv sha256 == its v0.2.1 manifest entry", fsha,
             manifest_entry(man, "v0.2.1_metrics_full.csv")["sha256"])
    if dsha == V021_DECISION_TABLE_SHA256 == manifest_entry(man, "v0.2.1_decision_table.csv")["sha256"] and \
            fsha == manifest_entry(man, "v0.2.1_metrics_full.csv")["sha256"]:
        STATE["V021_OK"] = True
    v03p = os.path.join(V03_DIR, "v0.3_manifest.json")
    s = sha256_of(v03p)
    check_eq("V3", "V3_v03_manifest_sha256", "v0.3_manifest.json sha256 == constant", s, V03_MANIFEST_SHA256)
    v03 = read_json(v03p)
    n = manifest_outputs("V3", "V3_v03_output", v03, [V03_DIR, PREREG_DIR], "v0.3 manifest")
    if s == V03_MANIFEST_SHA256 and n == len(v03["outputs"]):
        STATE["V03_OK"] = True


# =====================================================================================
# V4 - v0.8 chain (hashes only)
# =====================================================================================
def sec_v4():
    p = os.path.join(V08_DIR, "v0.8_manifest.json")
    s = sha256_of(p)
    check_eq("V4", "V4_v08_manifest_sha256", "v0.8_manifest.json sha256 == constant", s, V08_MANIFEST_SHA256)
    man = read_json(p)
    n = manifest_outputs("V4", "V4_v08_output", man, [V08_DIR], "v0.8 manifest")
    info("V4", "V4_v08_outputs_count", "v0.8 manifest outputs verified (state_dicts, loss CSVs, logs; hashed only, never "
         "loaded)", {"listed": len(man["outputs"]), "verified": n})


# =====================================================================================
# V5 - v0.9 chain
# =====================================================================================
def sec_v5():
    s = sha256_of(V09_CELL_PATH) if os.path.isfile(V09_CELL_PATH) else "missing"
    check_eq("V5", "V5_v09_cell_sha256", "notebooks\\v0.9_cell.py sha256 == constant", s, V09_CELL_SHA256)
    if s == V09_CELL_SHA256:
        STATE["V09_CELL_TEXT"] = read_bytes(V09_CELL_PATH).decode("utf-8")
    d = {}
    for name, e in V09_DOCS.items():
        d[name] = self_and_file("V5", "V5_" + name.replace(".json", ""), os.path.join(V09_DIR, name), e["file_sha256"],
                                e["self_sha256"], name)
    frozen, res, man = d["v0.9_frozen_outputs.json"], d["v0.9_results.json"], d["v0.9_manifest.json"]
    STATE["V09_RESULTS"], STATE["V09_FROZEN"] = res, frozen
    for name, exp in V09_FILES.items():
        p = os.path.join(V09_DIR, name)
        check_eq("V5", "V5_" + name, "{} sha256 == constant".format(name), sha256_of(p) if os.path.isfile(p) else "missing",
                 exp)
    fro_list = {n: r["sha256"] for n, r in frozen["outputs"].items()}
    check_eq("V5", "V5_frozen_eq_results_list", "probability / permutation files: frozen list == results "
             "probability_files_sha256", fro_list, res["probability_files_sha256"])
    n_csv = sum(n.endswith("_fold10_probs.csv") for n in fro_list)
    check_eq("V5", "V5_n_probability_csvs", "18 probability CSVs + 2 permutation arrays in the frozen list",
             [n_csv, len(fro_list)], [18, 20], kind="read_locked")
    all_ok = True
    for n, h in sorted(fro_list.items()):
        p = os.path.join(V09_DIR, n)
        got = sha256_of(p) if os.path.isfile(p) else "missing"
        all_ok &= got == h
        check_eq("V5", "V5_prob_file_" + n, "{} on disk == frozen list".format(n), got, h)
    for k, n in (("perm_resnet", "v0.9_resnet1d_wang_perm_probs.npy"), ("perm_inception", "v0.9_inception1d_perm_probs.npy")):
        check_eq("V5", "V5_frozen_" + k, "{} frozen hash == constant".format(n), fro_list.get(n), V09_FILES[n])
    check_eq("V5", "V5_results_frozen_self", "results.frozen_outputs_self_sha256 == frozen self_sha256 constant",
             res["frozen_outputs_self_sha256"], V09_DOCS["v0.9_frozen_outputs.json"]["self_sha256"])
    check_eq("V5", "V5_results_frp", "results.final_run_protocol == FRP constants", res["final_run_protocol"],
             {"self_sha256": DOC_EXPECTED["final_run_protocol"]["self_sha256"],
              "file_sha256": DOC_EXPECTED["final_run_protocol"]["file_sha256"]})
    check_eq("V5", "V5_results_ref", "results.reference_implementation_sha256 == constant",
             res["reference_implementation_sha256"], REFERENCE_CODE_SHA256_EXPECTED)
    check_eq("V5", "V5_results_labels_bootstrap", "results labels / bootstrap CSV hashes == constants",
             [res["fold10"]["labels_csv_sha256"], res["bootstrap_deltas_csv_sha256"]],
             [V09_FILES["v0.9_fold10_labels.csv"], V09_FILES["v0.9_bootstrap_deltas.csv"]])
    check_eq("V5", "V5_manifest_frozen_results", "v0.9 manifest frozen_outputs / results hashes == constants",
             [man["frozen_outputs"]["self_sha256"], man["frozen_outputs"]["file_sha256"], man["results"]["self_sha256"],
              man["results"]["file_sha256"]],
             [V09_DOCS["v0.9_frozen_outputs.json"]["self_sha256"], V09_DOCS["v0.9_frozen_outputs.json"]["file_sha256"],
              V09_DOCS["v0.9_results.json"]["self_sha256"], V09_DOCS["v0.9_results.json"]["file_sha256"]])
    manifest_outputs("V5", "V5_v09_manifest_output", man, [V09_DIR], "v0.9 manifest")
    if all_ok:
        STATE["V09_PROB_OK"] = True
    STATE["V09_PROB_NAMES"] = sorted(fro_list)


# =====================================================================================
# VP - provenance of the copied code
# =====================================================================================
_OPEN_MARK = re.compile(r"^# >>> (verbatim|adapted) from (ph1|v0\.9) \((.*)\)(.*)$")


def provenance_blocks(text):
    """Outer blocks copied from ph1 / v0.9 in 'text' (nested markers of other sources are part of the copied text)."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = _OPEN_MARK.match(lines[i])
        if not m:
            i += 1
            continue
        kind, src, label, spec = m.groups()
        close = "# <<< {} from {}".format(kind, src)
        j = i + 1
        while j < len(lines) and lines[j] != close:
            j += 1
        body = lines[i + 1:j]
        indent, spec_ok = 0, True
        if kind == "verbatim":
            spec_ok = spec.strip() == ""
        else:
            m2 = re.match(r"^ \| indent: (\d+)$", spec)
            spec_ok = bool(m2)
            indent = int(m2.group(1)) if m2 else 0
            spec_ok = spec_ok and all(not l.strip() or l.startswith(" " * indent) for l in body)
            body = [l[indent:] if l.strip() else l for l in body]
        out.append({"kind": kind, "source": src, "label": label, "line": i + 1, "closed": j < len(lines),
                    "spec_ok": spec_ok, "indent": indent, "text": "\n".join(body) + "\n", "n_lines": len(body)})
        i = j + 1
    return out


def sec_vp():
    s = sha256_of(PH1_CELL_PATH) if os.path.isfile(PH1_CELL_PATH) else "missing"
    check_eq("VP", "VP_ph1_cell_sha256", "notebooks\\ph1_cell.py sha256 == APPROVED_PH1_CELL_SHA256 (r2)", s,
             APPROVED_PH1_CELL_SHA256)
    sources = {}
    if s == APPROVED_PH1_CELL_SHA256:
        sources["ph1"] = read_bytes(PH1_CELL_PATH).decode("utf-8").replace("\r\n", "\n")
    if "V09_CELL_TEXT" in STATE:
        sources["v0.9"] = STATE["V09_CELL_TEXT"].replace("\r\n", "\n")
    need("own_source")
    blocks = provenance_blocks(STATE["own_source"].replace("\r\n", "\n"))
    counts = {"ph1": 0, "v0.9": 0}
    for k, b in enumerate(blocks):
        cid = "VP_block_{:02d}_{}".format(k + 1, b["source"])
        desc = "{} from {} ({}; {} lines{}) occurs in {}".format(b["kind"], b["source"], b["label"], b["n_lines"],
                                                                 ", indent {}".format(b["indent"]) if b["indent"] else "",
                                                                 "ph1_cell.py" if b["source"] == "ph1" else "v0.9_cell.py")
        if b["source"] not in sources:
            record("VP", cid, desc, "hash", "FAIL", observed="source file missing or its sha256 differs")
            continue
        src = sources[b["source"]]
        found = b["closed"] and b["spec_ok"] and (src.startswith(b["text"]) or ("\n" + b["text"]) in src)
        counts[b["source"]] += found
        record("VP", cid, desc, "hash", "PASS" if found else "FAIL", expected="verbatim excerpt",
               observed="found" if found else "NOT found", line_in_verify=b["line"])
    check_eq("VP", "VP_blocks_present", "every copied block of this code was found in its source (5 from ph1, 9 from "
             "v0.9)", {"blocks": len(blocks), "verified": counts}, {"blocks": 14, "verified": {"ph1": 5, "v0.9": 9}})


# =====================================================================================
# V6 - fold-10 rows and labels (v0.2.1 label code, through the approved ph1 fit_predict)
# =====================================================================================
class _LabelsOnly(Exception):
    """Raised by the label-only gate: fit_predict ran the v0.2.1 label code and preprocessing, no model is fit."""

    def __init__(self, info_):
        super().__init__("labels only")
        self.info = info_


def _labels_gate(info_):
    raise _LabelsOnly(info_)


def sec_v6():
    need("input_ok_ptbxl_database.csv", "input_ok_scp_statements.csv", "FRP")
    scp = pd.read_csv(SCP_CSV, index_col=0)
    df_all = pd.read_csv(DB_CSV, index_col="ecg_id")
    assert df_all.index.is_unique and set(df_all["strat_fold"].unique()) == set(range(1, 11))
    STATE["SCP"], STATE["DF_ALL"] = scp, df_all
    try:
        fit_predict(df_all, scp, 9, 10, _labels_gate)
        raise RuntimeError("the label-only gate was not called")
    except _LabelsOnly as e:
        inf = e.info
    STATE["LAB_ALL"] = inf["lab"]
    STATE["PREPROC_ALL"] = {"n_age_sentinel": inf["n_age_sentinel"], "n_year_missing": inf["n_year_missing"]}
    df10 = df_all[df_all["strat_fold"] == 10]
    pid = df10["patient_id"].to_numpy()
    assert np.all(pid == np.round(pid))
    F10_IDS = np.ascontiguousarray(np.stack([df10.index.to_numpy().astype(np.int64), pid.astype(np.int64)],
                                            axis=1).astype("<i8"))
    locked = STATE["FRP"]["bootstrap"]["locked_hashes"]
    h_ids = sha256_bytes(F10_IDS.tobytes(order="C"))
    pats = np.unique(F10_IDS[:, 1])
    h_pat = sha256_bytes(np.ascontiguousarray(pats.astype("<i8")).tobytes())
    check_eq("V6", "V6_fold10_ids_sha256", "fold-10 (ecg_id, patient_id) in file order, FRP serialization == FRP locked "
             "hash", h_ids, locked["fold10_ids_file_order_sha256"], kind="recomputed")
    check_eq("V6", "V6_fold10_patients_sha256", "fold-10 patients (sorted, '<i8') == FRP locked hash", h_pat,
             locked["fold10_patients_sorted_sha256"], kind="recomputed")
    STATE["F10_IDS"] = F10_IDS
    LAB10 = inf["lab"].loc[df10.index]
    LABELS10, Y10 = v09_labels_frame(F10_IDS, LAB10)
    STATE["LAB10_DF"] = LAB10[SUPERCLASSES]
    STATE["Y10"], STATE["LABELS10"] = Y10, LABELS10
    lp = os.path.join(V09_DIR, "v0.9_fold10_labels.csv")
    stored = pd.read_csv(lp, encoding="utf-8-sig")
    eq = list(stored.columns) == list(LABELS10.columns) and len(stored) == len(LABELS10) and all(
        np.array_equal(stored[c].to_numpy(), LABELS10[c].to_numpy()) for c in LABELS10.columns)
    n_diff = int((stored[SUPERCLASSES].to_numpy() != LABELS10[SUPERCLASSES].to_numpy()).sum()) \
        if list(stored.columns) == list(LABELS10.columns) and len(stored) == len(LABELS10) else None
    record("V6", "V6_labels_eq_v09_file", "fold-10 labels (v0.2.1 label code) == v0.9_fold10_labels.csv (rows, order, "
           "values)", "recomputed", "PASS" if eq else "FAIL", expected=0, observed={"n_differing_values": n_diff})
    lb = csv_bytes(LABELS10)
    check_eq("V6", "V6_labels_bytes", "v0.9_fold10_labels.csv == csv_bytes(recomputed label frame), byte for byte "
             "(v0.9 writer csv_bytes, verbatim)", sha256_bytes(lb), sha256_of(lp), kind="recomputed")
    STATE["LABELS10_SHA"] = sha256_bytes(lb)
    pos = {c: int(LAB10[c].sum()) for c in SUPERCLASSES}
    check_eq("V6", "V6_fold10_positives", "fold-10 positives == {963, 550, 521, 496, 262}", pos,
             EXPECTED_FOLD10_POSITIVES, kind="recomputed")
    if eq and pos == EXPECTED_FOLD10_POSITIVES and h_ids == locked["fold10_ids_file_order_sha256"]:
        STATE["V6_OK"] = True


# =====================================================================================
# V7 - BOOT_IDX and PERM_IDX
# =====================================================================================
def sec_v7():
    need("REF", "F10_IDS", "FRP")
    locked = STATE["FRP"]["bootstrap"]["locked_hashes"]
    pats, rows = REF["patient_rows_in_file_order"](STATE["F10_IDS"][:, 1])
    BOOT_IDX = REF["build_boot_idx"](len(pats), 2000, 7)
    PERM_IDX = REF["build_perm_idx"](len(STATE["F10_IDS"]), 20, 42)
    hb = sha256_bytes(np.ascontiguousarray(BOOT_IDX.astype("<i8")).tobytes())
    hp = sha256_bytes(np.ascontiguousarray(PERM_IDX.astype("<i8")).tobytes())
    check_eq("V7", "V7_boot_idx_sha256", "REF.build_boot_idx(1904, 2000, 7) == FRP locked hash", [len(pats), hb],
             [1904, locked["boot_idx_sha256"]], kind="recomputed")
    check_eq("V7", "V7_perm_idx_sha256", "REF.build_perm_idx(2198, 20, 42) == FRP locked hash", hp,
             locked["perm_idx_sha256"], kind="recomputed")
    if hb == locked["boot_idx_sha256"]:
        STATE["BOOT_IDX"], STATE["F10_PATIENTS"], STATE["F10_ROWS"] = BOOT_IDX, pats, rows


# =====================================================================================
# V8 - H3 and exploratory results (v0.9 Part D, verbatim)
# =====================================================================================
def _v09_prob_name(arch, arm, seed, lr):
    return "v0.9_{}_{}_seed{}_lr{:g}_fold10_probs.csv".format(arch, arm, seed, lr)


def sec_v8():
    need("REF", "Y10", "BOOT_IDX", "F10_IDS", "V09_RESULTS", "DOCS", "LABELS10_SHA")
    global N_F10, F10_IDS
    F10_IDS = STATE["F10_IDS"]
    N_F10 = len(F10_IDS)
    sel = STATE["DOCS"]["lr_selection"]["selected_lr"]
    frp = STATE["FRP"]
    m = re.search(r"N_PERM = (\d+) sequential rng\.permutation\((\d+)\) calls on one default_rng\(PERM_SEED = (\d+)\)",
                  frp["exploratory_fold10_comparisons"]["acquisition_permutation"]["repetitions"])
    n_perm, perm_seed = int(m.group(1)), int(m.group(3))
    P = {}
    for arch in ARCHS:
        for seed in SEEDS:
            for arm in ARMS:
                P[(arch, arm, seed)] = load_saved_probs10(os.path.join(V09_DIR, _v09_prob_name(arch, arm, seed, sel[arch])))
    ENS = v09_seed_ensembles(P, ARCHS, ARMS, SEEDS)
    PERM = {}
    for arch in ARCHS:
        PERM[arch] = np.load(os.path.join(V09_DIR, "v0.9_{}_perm_probs.npy".format(arch)), allow_pickle=False)
        assert PERM[arch].shape == (n_perm, len(SEEDS), N_F10, len(SUPERCLASSES)) and PERM[arch].dtype == np.float64
    _rows10 = STATE["F10_ROWS"]
    out = v09_part_d(STATE["Y10"], _rows10, STATE["BOOT_IDX"], ENS, P, PERM, n_perm, perm_seed,
                     float(frp["H3_decision"]["SESOI_H3"]), frp["sealed_H3_protocol_verbatim"]["decision"])
    STATE["ENS"], STATE["V8"] = ENS, out
    res = STATE["V09_RESULTS"]
    rebuilt = {
        "fold10": {"n_records": N_F10, "n_patients": int(len(STATE["F10_PATIENTS"])),
                   "label_positives": {c: int(STATE["LAB10_DF"][c].sum()) for c in SUPERCLASSES},
                   "labels_csv_sha256": STATE["LABELS10_SHA"]},
        "D0_metric_check": out["D0"],
        "1_primary": out["SECTION"][ARCHS[0]],
        "2_robustness": out["SECTION"][ARCHS[1]],
        "3_secondary_per_seed": out["D3"],
        "4_disagreements": {"note": "reported as they are; none is resolved by selection (FRP reporting_order 4)",
                            "table": out["D4"]},
        "5_exploratory": {"label": "EXPLORATORY (FRP exploratory_fold10_comparisons, closed list)",
                          "E1_physio_minus_signal_only": out["E1"], "E2_per_class_deltas_H3_pair": out["E2"],
                          "E3_acquisition_permutation": out["E3"]},
        "D7_sanity": out["D7"],
    }
    for k, v in rebuilt.items():
        check_struct("V8", "V8_results_" + k, "v0.9_results.json['{}'] == recomputation (REF, BOOT_IDX, frozen "
                     "probabilities; every leaf)".format(k), v, res[k])
    check_eq("V8", "V8_results_warnings", "v0.9 results warnings == the warnings of the recomputation", ANALYSIS_WARNINGS,
             res["warnings"], kind="recomputed")
    bp = os.path.join(V09_DIR, "v0.9_bootstrap_deltas.csv")
    stored = pd.read_csv(bp, encoding="utf-8-sig", float_precision="round_trip")
    bdf = out["BOOT_DF"]
    check_frame("V8", "V8_bootstrap_deltas", "21 per-draw delta columns == v0.9_bootstrap_deltas.csv", bdf, stored,
                ["replicate"], [c for c in bdf.columns if c != "replicate"])
    check_eq("V8", "V8_bootstrap_deltas_bytes", "v0.9_bootstrap_deltas.csv == csv_bytes(recomputed frame), byte for byte",
             sha256_bytes(csv_bytes(bdf)), sha256_of(bp), kind="recomputed")


# =====================================================================================
# V9 - descriptive per-arm macro-AUROC with a percentile bootstrap CI
# =====================================================================================
def sec_v9():
    need("ENS", "Y10", "BOOT_IDX", "F10_ROWS", "V8")
    out = {}
    Y10, rows, boot = STATE["Y10"], STATE["F10_ROWS"], STATE["BOOT_IDX"]
    for arch in ARCHS:
        out[arch] = {}
        for arm in ARMS:
            Pm = STATE["ENS"][(arch, arm)]
            point = REF["macro_auroc_strict"](Y10, Pm)
            draws = np.array([REF["macro_auroc_strict"](Y10, Pm, REF["boot_rows"](rows, boot[b]))
                              for b in range(boot.shape[0])])
            valid = draws[~np.isnan(draws)]
            lo, hi = np.percentile(valid, [2.5, 97.5], method="linear")
            out[arch][arm] = {"auroc": float(point), "ci95_descriptive": [float(lo), float(hi)],
                              "n_valid": int(len(valid)), "n_nan": int(len(draws) - len(valid))}
            info("V9", "V9_{}_{}".format(arch, arm), "descriptive, not pre-specified: {} {} seed-ensemble macro-AUROC "
                 "and 95% percentile CI over the BOOT_IDX draws".format(arch, arm), out[arch][arm])
    sec = STATE["V8"]
    pts = {(a, "signal_plus_physio"): sec["SECTION"][a]["macro_auroc_A"] for a in ARCHS}
    pts.update({(a, "signal_plus_all"): sec["SECTION"][a]["macro_auroc_B"] for a in ARCHS})
    pts.update({(e["arch"], "signal_only"): e["macro_auroc_A"] for e in sec["E1"]})
    md = max(abs(out[a][m]["auroc"] - pts[(a, m)]) for (a, m) in pts)
    record("V9", "V9_points_eq_v8", "descriptive point estimates == the arm macro-AUROC values of V8", "recomputed",
           "PASS" if md <= ABS_TOL else "FAIL", max_abs_diff=md, tolerance=ABS_TOL)
    STATE["V9"] = out


# =====================================================================================
# V10 - metadata-only, fold 9 (H1 / H2) with the approved ph1 functions
# =====================================================================================
def _gate_a_factory(section, lq_enc):
    def gate(info_):
        pos9 = {c: int(info_["labels_eval"][c].sum()) for c in SUPERCLASSES}
        check_eq(section, section + "_A1_fold9_positives", "fold-9 positives (v0.2.1 label code) == v0.2.1 counts", pos9,
                 EXPECTED_FOLD9_POSITIVES, kind="recomputed")
        split = {k: info_[k] for k in ("n_train", "n_eval", "n_train_patients", "n_eval_patients")}
        check_eq(section, section + "_A1_split_sizes", "train folds 1-8 / eval fold 9 records and patients", split,
                 EXPECTED_SPLITS["A"], kind="recomputed")
        exp18 = {c: [v for v in lq_enc["categories"][c] if v not in lq_enc["new_levels_vs_folds1to8"][c]] for c in ACQ_COLS}
        widths = {c: len(info_["categories"][c]) for c in ACQ_COLS}
        check_true(section, section + "_A1_encoder_folds1to8", "folds 1-8 categories == FRP categories minus "
                   "new_levels_vs_folds1to8 (widths == widths_folds1to8_development)",
                   info_["categories"] == exp18 and widths == lq_enc["widths_folds1to8_development"], kind="recomputed",
                   expected=lq_enc["widths_folds1to8_development"], observed=widths)
    return gate


def _frame_with_tolerance(section, cid, label, got, exp, id_cols, num_cols, tol_spec):
    """Strict 1e-12 check of every non-relaxed cell (one check per frame) + one check per relaxed cell (ph1 r2)."""
    shape_ok = list(got.columns) == list(exp.columns) and len(got) == len(exp)
    ids_ok = shape_ok and all(np.array_equal(got[c].to_numpy(), exp[c].to_numpy()) for c in id_cols)
    if not ids_ok:
        record(section, cid + "_identifiers", "{}: columns, rows and identifier columns".format(label), "recomputed",
               "FAIL", expected=list(exp.columns), observed=list(got.columns))
        return None
    D = abs_diff_matrix(got, exp, num_cols)
    row_ids = [{c: exp[c].iloc[i] for c in id_cols} for i in range(len(exp))]
    T = np.array([[cell_tolerance(tol_spec, ids, c) for c in num_cols] for ids in row_ids], dtype=np.float64)
    strict = T == tol_spec["default"]
    md_strict = float(D[strict].max()) if strict.any() else 0.0
    record(section, cid + "_strict", "{}: identifiers exact; every cell outside the ph1 r2 relaxed cells within "
           "1e-12".format(label), "recomputed", "PASS" if md_strict <= tol_spec["default"] else "FAIL",
           max_abs_diff=md_strict, tolerance=tol_spec["default"], n_cells=int(strict.sum()))
    for i, j in zip(*np.nonzero(~strict)):
        ids = row_ids[i]
        tag = "_".join(str(ids[c]) for c in id_cols)
        record(section, "{}_relaxed_{}_{}".format(cid, tag, num_cols[j]),
               "{}: {} {} (relaxed cell)".format(label, ids, num_cols[j]),
               "recomputed (relaxed tolerance, ph1 r2 amendment)", "PASS" if D[i, j] <= T[i, j] else "FAIL",
               max_abs_diff=float(D[i, j]), tolerance=float(T[i, j]),
               expected=float(exp[num_cols[j]].iloc[i]), observed=float(got[num_cols[j]].iloc[i]),
               note="scipy 1.11.4 -> 1.15.3, ph1_analysis_plan.json amendment_r2")
    return {c: float(D[:, j].max()) for j, c in enumerate(num_cols)}


def sec_v10():
    need("DF_ALL", "SCP", "REF", "FRP", "SEAL")
    df_all, seal, frp = STATE["DF_ALL"], STATE["SEAL"], STATE["FRP"]
    df9 = df_all[df_all["strat_fold"] <= 9].copy()
    A_FIT = fit_predict(df9, STATE["SCP"], 8, 9, _gate_a_factory("V10", frp["locked_quantities"]["categorical_encoder"]))
    boot9 = REF["build_boot_idx"](EXPECTED_SPLITS["A"]["n_eval_patients"], N_BOOT, BOOT_SEED)
    A_MET = metrics_from_probs(A_FIT["probs"], A_FIT["labels_eval"], boot9, float(seal["SESOI_H2"]),
                               float(seal["MDE"]["z_alpha_2"]), float(seal["MDE"]["z_power"]))
    STATE["V10_MET"] = A_MET
    v021_dt = pd.read_csv(os.path.join(V021_DIR, "v0.2.1_decision_table.csv"), encoding="utf-8-sig",
                          float_precision="round_trip")
    v021_mf = pd.read_csv(os.path.join(V021_DIR, "v0.2.1_metrics_full.csv"), encoding="utf-8-sig",
                          float_precision="round_trip")
    STATE["V021_DT"], STATE["V021_MF"] = v021_dt, v021_mf
    a_dt = A_MET["decision_table"].reset_index()
    dt_cols = _frame_with_tolerance("V10", "V10_decision_table", "decision table vs v0.2.1_decision_table.csv (12 rows x "
                                    "5 columns)", a_dt, v021_dt, DT_ID_COLUMNS, DT_NUMERIC_COLUMNS, A2_TOLERANCE)
    mf_cols = _frame_with_tolerance("V10", "V10_metric_rows", "metric rows vs v0.2.1_metrics_full.csv (216 rows)",
                                    A_MET["metric_rows"], v021_mf, MF_ID_COLUMNS, MF_NUMERIC_COLUMNS, A2_TOLERANCE)
    STATE["V10_DIFFS"] = {"decision_table": dt_cols, "metric_rows": mf_cols}
    a2b = []
    for frame_, exp_, ids_, col_ in ((a_dt, v021_dt, DT_ID_COLUMNS, "macro_auroc_mean"),
                                     (A_MET["metric_rows"], v021_mf, MF_ID_COLUMNS, "auroc")):
        for i in range(len(exp_)):
            if exp_["learner"].iloc[i] == "LogisticRegression":
                e, g = float(exp_[col_].iloc[i]), float(frame_[col_].iloc[i])
                if e != g:
                    a2b.append(dict({c: exp_[c].iloc[i] for c in ids_}, column=col_, v021_4dp="{:.4f}".format(e),
                                    recomputed_4dp="{:.4f}".format(g)))
    info("V10", "V10_A2b_LR_4dp", "LogisticRegression rows that are not exact keep their 4-decimal AUROC (ph1 A2b; "
         "reported, not gating)", {"rows_not_exact": len(a2b),
                                   "rows_4dp_changed": [r for r in a2b if r["v021_4dp"] != r["recomputed_4dp"]]})
    # H1 on fold 9
    v021_man = STATE.get("V021_MAN") or read_json(os.path.join(V021_DIR, "v0.2.1_manifest.json"))
    check_struct("V10", "V10_h1_verdict_vs_v021", "H1 verdict (B3 >= 0.60 per learner, values) == v0.2.1 manifest "
                 "verdict", A_MET["h1_verdict"], v021_man["verdict"])
    h1v = seal["H1"]["values"]
    rec4 = {"LogisticRegression_B3": "{:.4f}".format(A_MET["h1_verdict"]["LogisticRegression"]["macro_auroc_mean"]),
            "HistGradientBoosting_B3_3seed_mean": "{:.4f}".format(
                A_MET["h1_verdict"]["HistGradientBoosting"]["macro_auroc_mean"])}
    check_eq("V10", "V10_h1_vs_prereg", "H1 B3 macro-AUROC, 4 decimals == pre_registration.json H1.values (stored rounded "
             "to 4 decimals)", rec4, {k: "{:.4f}".format(v) for k, v in h1v.items()}, kind="recomputed")
    check_eq("V10", "V10_h1_pass_vs_prereg", "H1 passes for both learners; pre_registration H1.status 'already PASSED'",
             [A_MET["h1_verdict"]["overall_pass"], seal["H1"]["status"]], [True, "already PASSED in v0.2.1"],
             kind="recomputed")
    # H2 on fold 9
    rpl = {l: {k: v for k, v in A_MET["results_per_learner"][l].items() if k != "n_patients"} for l in LEARNERS}
    check_struct("V10", "V10_h2_results_per_learner", "H2 results_per_learner == pre_registration.json H2 (every field)",
                 rpl, seal["H2"]["results_per_learner"])
    check_eq("V10", "V10_h2_n_patients", "H2 on 1942 fold-9 patients",
             [A_MET["results_per_learner"][l]["n_patients"] for l in LEARNERS], [1942, 1942], kind="recomputed")
    check_eq("V10", "V10_h2_learner_disagreement", "learner disagreement list == pre_registration.json (exact strings)",
             A_MET["learner_disagreement"], seal["H2"]["learner_disagreement"], kind="recomputed")
    v03d = pd.read_csv(os.path.join(V03_DIR, "v0.3_bootstrap_deltas.csv"), encoding="utf-8-sig",
                       float_precision="round_trip")
    check_frame("V10", "V10_h2_per_draw_deltas", "2000 per-draw deltas (LR, HGB) == v0.3_bootstrap_deltas.csv (byte "
                "comparison not applicable: v0.3 wrote it with its own save_csv)", A_MET["boot_df"], v03d, ["replicate"],
                ["delta_LR", "delta_HGB"])
    STATE["V10_DELTA_DIFFS"] = {c: float(np.max(np.abs(A_MET["boot_df"][c].to_numpy() - v03d[c].to_numpy())))
                                for c in ("delta_LR", "delta_HGB")}
    check_true("V10", "V10_boot_index_matrix", "v0.2.1 marginal-bootstrap index matrix on fold 9 == "
               "REF.build_boot_idx(1942, 2000, 7)", np.array_equal(A_MET["boot_samp"], boot9), kind="recomputed")
    STATE["V10_OK"] = True


# =====================================================================================
# V11 - ph1 (post hoc, fold 10) and the r1 attempt
# =====================================================================================
def _gate_b_factory(section, lq_enc, f10_ids, v09_labels, df9):
    def gate(info_):
        m_tr, m_ev, frame = info_["mask_train"], info_["mask_eval"], info_["frame"]
        ev = np.stack([frame.index[m_ev].to_numpy().astype(np.int64),
                       frame.loc[m_ev, "patient_id"].to_numpy().astype(np.int64)], axis=1)
        check_true(section, section + "_B0_eval_rows_file_order", "evaluation rows == fold-10 (ecg_id, patient_id) in "
                   "file order", np.array_equal(ev, f10_ids), kind="recomputed")
        lab10 = info_["labels_eval"]
        cols_ok = list(v09_labels.columns) == ["ecg_id", "patient_id"] + SUPERCLASSES
        same = cols_ok and len(v09_labels) == len(lab10) and \
            np.array_equal(v09_labels["ecg_id"].to_numpy(), lab10.index.to_numpy()) and \
            np.array_equal(v09_labels["patient_id"].to_numpy(), ev[:, 1])
        n_diff = int((v09_labels[SUPERCLASSES].to_numpy(dtype=np.int64) !=
                      lab10[SUPERCLASSES].to_numpy(dtype=np.int64)).sum()) if same else None
        check_true(section, section + "_B0_labels_eq_v09", "fold-10 labels (v0.2.1 code) == v0.9_fold10_labels.csv",
                   bool(same) and n_diff == 0, kind="recomputed", observed={"n_differing_label_values": n_diff})
        check_eq(section, section + "_B0_fold10_positives", "fold-10 positives == FRP disclosure",
                 {c: int(lab10[c].sum()) for c in SUPERCLASSES}, EXPECTED_FOLD10_POSITIVES, kind="recomputed")
        check_true(section, section + "_B0_encoder_eq_FRP", "device / site / nurse categories fit on folds 1-9 == FRP "
                   "locked categories", info_["categories"] == lq_enc["categories"], kind="recomputed")
        n_unknown = {c: int(info_["dummies"][c].loc[m_ev, "{}=__UNKNOWN__".format(c)].sum()) for c in ACQ_COLS}
        check_eq(section, section + "_B0_fold10_unknown_zero", "fold-10 records mapped to __UNKNOWN__", n_unknown,
                 {c: 0 for c in ACQ_COLS}, kind="recomputed")
        allzero = sorted(col for c in ACQ_COLS for col in info_["dummies"][c].columns
                         if int(info_["dummies"][c].loc[m_tr, col].sum()) == 0)
        on_allzero = sum(int(info_["dummies"][c].loc[m_ev, [col for col in allzero if col.startswith(c + "=")]]
                             .to_numpy().sum()) for c in ACQ_COLS)
        check_eq(section, section + "_B0_untrained_columns", "all-zero training columns == FRP list, 0 fold-10 records "
                 "on them", [allzero, on_allzero], [sorted(lq_enc["allzero_training_columns_folds1to9"]), 0],
                 kind="recomputed")
        same_levels = {c: bool(info_["levels"][c].loc[df9.index].equals(to_level_string(df9[c]))) for c in ACQ_COLS}
        check_true(section, section + "_B0_level_strings_unchanged", "folds 1-9 level strings unchanged by the fold-10 "
                   "values", all(same_levels.values()), kind="recomputed", observed=same_levels)
        split = {k: info_[k] for k in ("n_train", "n_eval", "n_train_patients", "n_eval_patients")}
        check_eq(section, section + "_B0_split_sizes", "train folds 1-9 / eval fold 10 records and patients", split,
                 EXPECTED_SPLITS["B"], kind="recomputed")
    return gate


def sec_v11():
    if not os.path.isfile(os.path.join(PH1_DIR, "ph1_results.json")):
        STATE["PH1_MISSING"] = True
        record("V11", "V11_ph1_missing", "outputs\\ph1\\ph1_results.json exists", "hash", "SKIP", observed="missing",
               note="paper cannot be finalised: ph1 missing")
        print("\n" + "!" * 100 + "\n!!! PAPER CANNOT BE FINALISED: ph1 missing (outputs\\ph1\\ph1_results.json not found)\n"
              + "!" * 100)
        return
    d = {}
    for name, e in PH1_DOCS.items():
        d[name] = self_and_file("V11", "V11_" + name.replace(".json", ""), os.path.join(PH1_DIR, name), e["file_sha256"],
                                e["self_sha256"], "ph1 " + name)
    plan, frozen, res, man = (d["ph1_analysis_plan.json"], d["ph1_frozen_outputs.json"], d["ph1_results.json"],
                              d["ph1_manifest.json"])
    STATE["PH1_RESULTS"] = res
    for name, exp in PH1_FILES.items():
        p = os.path.join(PH1_DIR, name)
        check_eq("V11", "V11_" + name, "ph1 {} sha256 == constant".format(name), sha256_of(p) if os.path.isfile(p)
                 else "missing", exp)
    info("V11", "V11_ph1_manifest_file_sha256_listed", "ph1_manifest.json file sha256 (recorded and listed)",
         sha256_of(os.path.join(PH1_DIR, "ph1_manifest.json")))
    listed = sorted(base(o["file"]) for o in man["outputs"])
    check_eq("V11", "V11_manifest_list", "ph1 manifest lists the 7 output files", listed,
             sorted(["ph1_analysis_plan.json", "ph1_fold10_probs.csv", "ph1_frozen_outputs.json",
                     "ph1_decision_table.csv", "ph1_bootstrap_deltas.csv", "ph1_metric_rows.csv", "ph1_results.json"]))
    manifest_outputs("V11", "V11_manifest_output", man, [PH1_DIR], "ph1 manifest")
    check_eq("V11", "V11_manifest_self_links", "ph1 manifest plan / frozen / results self_sha256 == constants",
             [man["plan_self_sha256"], man["frozen_outputs_self_sha256"], man["results_self_sha256"]],
             [PH1_DOCS[n]["self_sha256"] for n in ("ph1_analysis_plan.json", "ph1_frozen_outputs.json",
                                                   "ph1_results.json")])
    check_eq("V11", "V11_folder_contents", "outputs\\ph1 holds exactly the 8 write-once files",
             sorted(n for n in os.listdir(PH1_DIR) if not n.endswith(".ots")), sorted(listed + ["ph1_manifest.json"]))
    # (ii) plan
    check_eq("V11", "V11_plan_text_sha256", "ph1 plan text (UTF-8) sha256 == locked plan hash",
             [sha256_bytes(plan["plan"].encode("utf-8")), plan["plan_utf8_sha256"]],
             [PH1_PLAN_UTF8_SHA256, PH1_PLAN_UTF8_SHA256])
    check_eq("V11", "V11_amendment_r2_sha256", "plan amendment_r2 (UTF-8) sha256 == constant; results amendment_r2 == "
             "plan amendment_r2", [sha256_bytes(plan["amendment_r2"].encode("utf-8")), res["amendment_r2"]],
             [PH1_AMENDMENT_R2_UTF8_SHA256, plan["amendment_r2"]])
    check_eq("V11", "V11_revision_labels", "plan / frozen / results / manifest: code_version ph1, code_revision r2",
             [(x["code_version"], x["code_revision"]) for x in (plan, frozen, res, man)], [("ph1", "r2")] * 4)
    # (iii) frozen
    check_eq("V11", "V11_frozen_links", "frozen: probability-file hash == constant, plan_self_sha256 == plan, lock hashes "
             "== FRP", [frozen["outputs"]["ph1_fold10_probs.csv"]["sha256"], frozen["plan_self_sha256"],
                        frozen["fold10_lock_hashes"]],
             [PH1_FILES["ph1_fold10_probs.csv"], PH1_DOCS["ph1_analysis_plan.json"]["self_sha256"],
              {k: STATE["FRP"]["bootstrap"]["locked_hashes"][k] for k in frozen["fold10_lock_hashes"]}]
             if "FRP" in STATE else None)
    check_eq("V11", "V11_results_links", "results.plan / frozen_outputs / outputs_before_results == the file hashes",
             [res["plan"], res["frozen_outputs"], res["outputs_before_results"]],
             [{"file": "ph1_analysis_plan.json", "self_sha256": PH1_DOCS["ph1_analysis_plan.json"]["self_sha256"],
               "file_sha256": PH1_DOCS["ph1_analysis_plan.json"]["file_sha256"],
               "plan_utf8_sha256": PH1_PLAN_UTF8_SHA256, "amendment_r2_utf8_sha256": PH1_AMENDMENT_R2_UTF8_SHA256},
              {"file": "ph1_frozen_outputs.json", "self_sha256": PH1_DOCS["ph1_frozen_outputs.json"]["self_sha256"],
               "file_sha256": PH1_DOCS["ph1_frozen_outputs.json"]["file_sha256"]},
              dict({"ph1_analysis_plan.json": PH1_DOCS["ph1_analysis_plan.json"]["file_sha256"],
                    "ph1_frozen_outputs.json": PH1_DOCS["ph1_frozen_outputs.json"]["file_sha256"]}, **PH1_FILES)])
    cur = {"ptbxl_database.csv": INPUT_SHA256["ptbxl_database.csv"],
           "scp_statements.csv": INPUT_SHA256["scp_statements.csv"],
           "pre_registration.json": {"file_sha256": DOC_EXPECTED["pre_registration"]["file_sha256"],
                                     "self_sha256": DOC_EXPECTED["pre_registration"]["self_sha256"]},
           "final_run_protocol.json": {"file_sha256": DOC_EXPECTED["final_run_protocol"]["file_sha256"],
                                       "self_sha256": DOC_EXPECTED["final_run_protocol"]["self_sha256"]},
           "reference_implementation_code_utf8": REFERENCE_CODE_SHA256_EXPECTED,
           "v0.2.1_decision_table.csv": V021_DECISION_TABLE_SHA256,
           "v0.2.1_metrics_full.csv": sha256_of(os.path.join(V021_DIR, "v0.2.1_metrics_full.csv")),
           "v0.2.1_manifest.json": sha256_of(os.path.join(V021_DIR, "v0.2.1_manifest.json")),
           "v0.3_bootstrap_deltas.csv": sha256_of(os.path.join(V03_DIR, "v0.3_bootstrap_deltas.csv")),
           "v0.3_manifest.json": V03_MANIFEST_SHA256,
           "v0.9_fold10_labels.csv": V09_FILES["v0.9_fold10_labels.csv"]}
    check_eq("V11", "V11_input_hashes", "ph1 results.input_files_sha256 == the current input files / constants",
             res["input_files_sha256"], cur)
    # (vi) ph1's own checks
    bad = [c["id"] for c in res["checks"] if c["status"] != "PASS"]
    check_eq("V11", "V11_ph1_checks_pass", "every check recorded in ph1_results.json is PASS", bad, [], kind="read_locked",
             n_checks=len(res["checks"]))
    info("V11", "V11_ph1_warnings", "warnings recorded by ph1", res["warnings"])
    # (vii) r1 attempt
    _sec_v11_r1(res, plan)
    # (iv) recompute from the probability file
    need("REF", "BOOT_IDX", "LAB10_DF", "F10_IDS", "SEAL", "DF_ALL", "SCP", "FRP")
    seal = STATE["SEAL"]
    probs = pd.read_csv(os.path.join(PH1_DIR, "ph1_fold10_probs.csv"), encoding="utf-8-sig", float_precision="round_trip")
    check_eq("V11", "V11_probs_columns_rows", "ph1_fold10_probs.csv columns and 79,128 rows",
             [list(probs.columns), len(probs)], [PROBS_CSV_COLUMNS, len(CONFIGS) * EXPECTED_N_FOLD10_RECORDS],
             kind="read_locked")
    B_MET = metrics_from_probs(probs, STATE["LAB10_DF"], STATE["BOOT_IDX"], float(seal["SESOI_H2"]),
                               float(seal["MDE"]["z_alpha_2"]), float(seal["MDE"]["z_power"]))
    STATE["V11_MET"] = B_MET
    b_dt = B_MET["decision_table"].reset_index()
    for name, frame_, ids_, nums_ in (("ph1_decision_table.csv", b_dt, DT_ID_COLUMNS, DT_NUMERIC_COLUMNS),
                                      ("ph1_metric_rows.csv", B_MET["metric_rows"], MF_ID_COLUMNS, MF_NUMERIC_COLUMNS),
                                      ("ph1_bootstrap_deltas.csv", B_MET["boot_df"], ["replicate"],
                                       ["delta_LR", "delta_HGB"])):
        p = os.path.join(PH1_DIR, name)
        stored = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
        tag = name.replace(".csv", "")
        check_frame("V11", "V11_" + tag, "{} == recomputation from ph1_fold10_probs.csv (function b)".format(name), frame_,
                    stored, ids_, nums_)
        check_eq("V11", "V11_" + tag + "_bytes", "{} == csv_bytes(recomputed frame), byte for byte (ph1 writer, verbatim)"
                 .format(name), sha256_bytes(csv_bytes(frame_)), sha256_of(p), kind="recomputed")
    check_struct("V11", "V11_results_decision_table", "ph1_results.decision_table == recomputation",
                 b_dt.to_dict(orient="records"), res["decision_table"])
    check_struct("V11", "V11_results_H1_posthoc", "ph1_results.H1_posthoc.verdict == recomputation", B_MET["h1_verdict"],
                 res["H1_posthoc"]["verdict"])
    check_struct("V11", "V11_results_H2_posthoc", "ph1_results.H2_posthoc (results_per_learner, learner_disagreement, "
                 "ref_summaries) == recomputation",
                 {"results_per_learner": B_MET["results_per_learner"], "learner_disagreement": B_MET["learner_disagreement"],
                  "ref_summaries": B_MET["ref_summaries"]},
                 {k: res["H2_posthoc"][k] for k in ("results_per_learner", "learner_disagreement", "ref_summaries")})
    check_true("V11", "V11_B2_index_matrix", "v0.2.1 marginal-bootstrap index matrix on fold 10 == BOOT_IDX; patient "
               "grouping == REF", np.array_equal(B_MET["boot_samp"], STATE["BOOT_IDX"]) and
               len(B_MET["v021_patient_row_list"]) == len(STATE["F10_ROWS"]) and
               all(np.array_equal(a, b) for a, b in zip(B_MET["v021_patient_row_list"], STATE["F10_ROWS"])),
               kind="recomputed")
    # (x) part_A cross-check with V10
    if "V10_DIFFS" in STATE and STATE["V10_DIFFS"]["decision_table"] is not None:
        pa = res["part_A"]
        got = {"decision_table": STATE["V10_DIFFS"]["decision_table"], "metric_rows": STATE["V10_DIFFS"]["metric_rows"],
               "per_draw_deltas": STATE["V10_DELTA_DIFFS"]}
        exp = {"decision_table": pa["decision_table_max_abs_diff_per_column"],
               "metric_rows": pa["metric_rows_max_abs_diff_per_column"],
               "per_draw_deltas": pa["per_draw_deltas_max_abs_diff_per_column"]}
        check_struct("V11", "V11_partA_vs_V10", "ph1_results.part_A max |diff| per column == the V10 recomputation "
                     "(within 1e-15)", got, exp, tol=1e-15)
    else:
        record("V11", "V11_partA_vs_V10", "ph1_results.part_A vs V10", "recomputed", "SKIP", note="V10 did not complete")
    # (v) refit folds 1-9 -> fold 10
    df_all = STATE["DF_ALL"]
    df9, df10 = df_all[df_all["strat_fold"] <= 9].copy(), df_all[df_all["strat_fold"] == 10].copy()
    v09_labels = pd.read_csv(os.path.join(V09_DIR, "v0.9_fold10_labels.csv"), encoding="utf-8-sig")
    lq_enc = STATE["FRP"]["locked_quantities"]["categorical_encoder"]
    B_FIT = fit_predict(pd.concat([df9, df10]), STATE["SCP"], 9, 10,
                        _gate_b_factory("V11", lq_enc, STATE["F10_IDS"], v09_labels, df9))
    check_frame("V11", "V11_refit_probabilities", "refit of all 36 configurations (folds 1-9 -> fold 10) == "
                "ph1_fold10_probs.csv", B_FIT["probs"], probs, ["ecg_id", "patient_id", "block", "learner", "seed"],
                PROB_COLUMNS)
    sc_rows = {b: int(s.n_samples_seen_) for b, s in B_FIT["scalers"].items()}
    check_eq("V11", "V11_B1_scalers_rows", "every LR scaler fit on the 19,601 training rows", sc_rows,
             {b: EXPECTED_SPLITS["B"]["n_train"] for b in ("B1_physio", "B3_physio_plus_acquisition", "B4_temporal",
                                                           "B5_all")}, kind="recomputed")
    lq_age = STATE["FRP"]["locked_quantities"]["age_scaler"]
    age = B_FIT["scalers"]["B1_physio"]
    diffs = {"mean": abs(float(age.mean_[0]) - lq_age["mean"]), "scale": abs(float(age.scale_[0]) - lq_age["scale"]),
             "var": abs(float(age.var_[0]) - lq_age["var"])}
    record("V11", "V11_B1_age_scaler_eq_FRP", "B1 age_imputed scaler (folds 1-9) == FRP locked age_scaler", "recomputed",
           "PASS" if all(diffs[k] <= AGE_SCALER_TOL[k] for k in diffs) else "FAIL", max_abs_diff=max(diffs.values()),
           tolerance=AGE_SCALER_TOL, observed=diffs)
    STATE["V11_OK"] = True


def _sec_v11_r1(res, plan):
    r1 = res["r1_attempt"]
    if not os.path.isdir(PH1_R1_DIR):
        record("V11", "V11_r1_folder", "outputs\\ph1_failed_1 exists", "hash", "FAIL", observed="missing")
        return
    files = sorted(n for n in os.listdir(PH1_R1_DIR) if not n.endswith(".ots"))
    check_eq("V11", "V11_r1_folder_contents", "outputs\\ph1_failed_1 holds exactly the r1 plan and the Part A failure "
             "file (no probability file)", files, ["ph1_analysis_plan.json", "ph1_partA_failure.json"])
    d = {}
    for name, e in PH1_R1_DOCS.items():
        d[name] = self_and_file("V11", "V11_r1_" + name.replace(".json", ""), os.path.join(PH1_R1_DIR, name),
                                e["file_sha256"], e["self_sha256"], "r1 " + name)
    r1plan, fail = d["ph1_analysis_plan.json"], d["ph1_partA_failure.json"]
    check_eq("V11", "V11_r1_files_eq_record", "ph1_results.r1_attempt.files == the files of outputs\\ph1_failed_1 "
             "(name, sha256)", r1["files"],
             [{"file": n, "sha256": sha256_of(os.path.join(PH1_R1_DIR, n))} for n in files])
    check_eq("V11", "V11_r1_fields", "r1_attempt fields == the failure file (code_revision, timestamp, exception, "
             "failed_checks, plan_self_sha256)",
             [r1["code_revision"], r1["timestamp"], r1["outcome"], r1["failed_checks"], r1["plan_self_sha256"]],
             [fail["code_revision"], fail["timestamp"], fail["exception"], fail["failed_checks"], fail["plan_self_sha256"]])
    check_eq("V11", "V11_r1_hashes", "r1_attempt failure-file hashes == constants",
             [r1["partA_failure_self_sha256"], r1["partA_failure_file_sha256"]],
             [PH1_R1_DOCS["ph1_partA_failure.json"]["self_sha256"], PH1_R1_DOCS["ph1_partA_failure.json"]["file_sha256"]])
    check_eq("V11", "V11_r1_plan_link", "r1 plan self_sha256 == r1_attempt.plan_self_sha256 == failure-file "
             "plan_self_sha256", [r1plan["self_sha256"], r1["plan_self_sha256"]],
             [fail["plan_self_sha256"], fail["plan_self_sha256"]])
    check_eq("V11", "V11_r1_plan_text", "r1 plan text == r2 plan text; the r1 plan has no amendment_r2; r1 revision 'r1'",
             [r1plan["plan"], "amendment_r2" in r1plan, r1plan["code_revision"]], [plan["plan"], False, "r1"])
    check_eq("V11", "V11_r1_previous_attempts", "ph1_results.previous_attempts == [ph1_failed_1 with these files]",
             res["previous_attempts"], [{"folder": "ph1_failed_1", "files": r1["files"]}])


# =====================================================================================
# V12 - locked settings (read)
# =====================================================================================
def sec_v12():
    need("DOCS")
    docs = STATE["DOCS"]
    frp, seal, dev, lrs = docs["final_run_protocol"], docs["pre_registration"], docs["dev_protocol"], docs["lr_selection"]
    m = re.search(r"N_PERM = (\d+) sequential rng\.permutation\((\d+)\) calls on one default_rng\(PERM_SEED = (\d+)\)",
                  frp["exploratory_fold10_comparisons"]["acquisition_permutation"]["repetitions"])
    v021_man = STATE.get("V021_MAN") or read_json(os.path.join(V021_DIR, "v0.2.1_manifest.json"))
    L = {
        "lr_grid": {"value": lrs["lr_grid"], "source_document": "lr_selection.json", "json_path": "lr_grid"},
        "lr_grid_dev_protocol": {"value": dev["tuning_grid"]["learning_rate"], "source_document": "dev_protocol.json",
                                 "json_path": "tuning_grid.learning_rate"},
        "selected_lr": {"value": lrs["selected_lr"], "source_document": "lr_selection.json", "json_path": "selected_lr"},
        "epochs": {"value": dev["training_recipe"]["epochs"], "source_document": "dev_protocol.json",
                   "json_path": "training_recipe.epochs"},
        "SESOI_H2": {"value": seal["SESOI_H2"], "source_document": "pre_registration.json", "json_path": "SESOI_H2"},
        "SESOI_H3": {"value": frp["H3_decision"]["SESOI_H3"], "source_document": "final_run_protocol.json",
                     "json_path": "H3_decision.SESOI_H3"},
        "N_BOOT": {"value": frp["bootstrap"]["N_BOOT"], "source_document": "final_run_protocol.json",
                   "json_path": "bootstrap.N_BOOT"},
        "BOOT_SEED": {"value": frp["bootstrap"]["BOOT_SEED"], "source_document": "final_run_protocol.json",
                      "json_path": "bootstrap.BOOT_SEED"},
        "seeds_signal_models": {"value": frp["sealed_H3_protocol_verbatim"]["seeds"],
                                "source_document": "final_run_protocol.json", "json_path": "sealed_H3_protocol_verbatim.seeds"},
        "seeds_metadata_only": {"value": v021_man["seeds"], "source_document": "v0.2.1_manifest.json",
                                "json_path": "seeds", "flag": "INFO: not in a locked document; the ph1 plan says "
                                                              "'seeds as in v0.2.1'"},
        "N_PERM": {"value": int(m.group(1)), "source_document": "final_run_protocol.json",
                   "json_path": "exploratory_fold10_comparisons.acquisition_permutation.repetitions (text)"},
        "PERM_SEED": {"value": int(m.group(3)), "source_document": "final_run_protocol.json",
                      "json_path": "exploratory_fold10_comparisons.acquisition_permutation.repetitions (text)"},
    }
    for k, v in L.items():
        info("V12", "V12_" + k, "locked setting {} ({} : {})".format(k, v["source_document"], v["json_path"]), v["value"])
    check_eq("V12", "V12_lr_grid_equal", "lr grid: lr_selection.json lr_grid == dev_protocol.json tuning_grid",
             L["lr_grid"]["value"], L["lr_grid_dev_protocol"]["value"], kind="read_locked")
    check_eq("V12", "V12_sesoi_h3_sealed", "SESOI_H3: H3_decision == sealed H3 protocol == pre_registration H3_protocol",
             [frp["H3_decision"]["SESOI_H3"], frp["sealed_H3_protocol_verbatim"]["SESOI_H3"],
              seal["H3_protocol"]["SESOI_H3"]], [L["SESOI_H3"]["value"]] * 3, kind="read_locked")
    if "V09_RESULTS" in STATE:
        s = STATE["V09_RESULTS"]["settings"]
        check_eq("V12", "V12_v09_settings", "v0.9 results settings == the locked values",
                 [s["SESOI_H3"], s["N_BOOT"], s["BOOT_SEED"], s["N_PERM"], s["PERM_SEED"], s["seeds"]],
                 [L[k]["value"] for k in ("SESOI_H3", "N_BOOT", "BOOT_SEED", "N_PERM", "PERM_SEED", "seeds_signal_models")],
                 kind="read_locked")
    STATE["LOCKED_SETTINGS"] = L


# =====================================================================================
# V13 - dataset descriptives
# =====================================================================================
def sec_v13():
    need("DF_ALL", "LAB_ALL", "FRP")
    df, lab, frp = STATE["DF_ALL"], STATE["LAB_ALL"], STATE["FRP"]
    lab = lab.loc[df.index]
    folds = df["strat_fold"]

    def split(mask):
        return {"n_records": int(mask.sum()), "n_patients": int(df.loc[mask, "patient_id"].nunique())}
    splits = {"folds_1_8": split(folds <= 8), "fold_9": split(folds == 9), "fold_10": split(folds == 10),
              "folds_1_9": split(folds <= 9)}
    pos = lambda mask: {c: int(lab.loc[mask, c].sum()) for c in SUPERCLASSES}
    no_sc = int((lab[SUPERCLASSES].sum(axis=1) == 0).sum())
    sent = df["age"] == 300
    D = {"n_records": int(len(df)), "n_patients": int(df["patient_id"].nunique()),
         "superclass_counts": pos(folds > 0), "n_without_superclass": no_sc, "splits": splits,
         "positives_fold_9": pos(folds == 9), "positives_fold_10": pos(folds == 10),
         "age_sentinel_300": {"all": int(sent.sum()), "folds_1_9": int(sent[folds <= 9].sum()),
                              "fold_10": int(sent[folds == 10].sum())}}
    STATE["DATASET"] = D
    info("V13", "V13_dataset", "dataset descriptives (v0.2.1 label code)", D)
    check_eq("V13", "V13_n_records_patients", "21,799 records, 18,869 patients", [D["n_records"], D["n_patients"]],
             [N_RECORDS_ALL_EXPECTED, N_PATIENTS_ALL_EXPECTED], kind="recomputed")
    disjoint = int((df.groupby("patient_id")["strat_fold"].nunique() > 1).sum())
    check_eq("V13", "V13_patient_disjoint_folds", "every patient is in exactly one fold", disjoint, 0, kind="recomputed")
    check_eq("V13", "V13_splits_vs_ph1_constants", "split sizes == the v0.2.1 / v0.3 (A) and FRP (B) split sizes",
             [splits["folds_1_8"], splits["fold_9"], splits["folds_1_9"], splits["fold_10"]],
             [{"n_records": EXPECTED_SPLITS["A"]["n_train"], "n_patients": EXPECTED_SPLITS["A"]["n_train_patients"]},
              {"n_records": EXPECTED_SPLITS["A"]["n_eval"], "n_patients": EXPECTED_SPLITS["A"]["n_eval_patients"]},
              {"n_records": EXPECTED_SPLITS["B"]["n_train"], "n_patients": EXPECTED_SPLITS["B"]["n_train_patients"]},
              {"n_records": EXPECTED_SPLITS["B"]["n_eval"], "n_patients": EXPECTED_SPLITS["B"]["n_eval_patients"]}],
             kind="recomputed")
    v021 = STATE.get("V021_MAN")
    if v021 is not None:
        sp = v021["split_protocol"]
        check_eq("V13", "V13_splits_vs_v021_manifest", "folds 1-8 / fold 9 / fold 10 sizes == v0.2.1 manifest "
                 "split_protocol", [splits["folds_1_8"]["n_records"], splits["fold_9"]["n_records"],
                                    splits["folds_1_8"]["n_patients"], splits["fold_9"]["n_patients"],
                                    splits["fold_10"]["n_records"]],
                 [sp["n_train_records"], sp["n_eval_records"], sp["n_train_patients"], sp["n_eval_patients"],
                  sp["n_fold10_records_in_source_untouched"]], kind="recomputed")
        check_eq("V13", "V13_fold9_positives_vs_v021", "fold-9 positives == v0.2.1 fold9_chance_baseline n_positive",
                 D["positives_fold_9"], {r["superclass"]: r["n_positive"] for r in v021["fold9_chance_baseline"]},
                 kind="recomputed")
    disc = frp["disclosures_fold10_exposure"]
    m10 = re.search(r"\(NORM (\d+), MI (\d+), STTC (\d+), CD (\d+), HYP (\d+) of (\d+)\)", disc["fold10_labels"][0])
    check_eq("V13", "V13_fold10_vs_frp", "fold-10 positives and size == FRP disclosure",
             [D["positives_fold_10"], splits["fold_10"]["n_records"]],
             [dict(zip(SUPERCLASSES, map(int, m10.groups()[:5]))), int(m10.group(6))], kind="recomputed")
    mall = re.search(r"superclass counts \(NORM (\d+), MI (\d+), STTC (\d+), CD (\d+), HYP (\d+)\), (\d+) records "
                     r"without superclass", disc["fold10_labels"][2])
    check_eq("V13", "V13_whole_dataset_vs_frp", "whole-dataset superclass counts and records without superclass == FRP "
             "disclosure (v0.1.x audit)", [D["superclass_counts"], no_sc],
             [dict(zip(SUPERCLASSES, map(int, mall.groups()[:5]))), int(mall.group(6))], kind="recomputed")
    ms = re.search(r"sentinel 300: (\d+) records overall vs (\d+) in folds 1-9", disc["whole_dataset_metadata"][0])
    check_eq("V13", "V13_age_sentinel_vs_frp", "age-300 sentinel counts (all, folds 1-9) == FRP disclosure",
             [D["age_sentinel_300"]["all"], D["age_sentinel_300"]["folds_1_9"]], [int(ms.group(1)), int(ms.group(2))],
             kind="recomputed")
    if "PH1_RESULTS" in STATE:
        r = STATE["PH1_RESULTS"]
        check_eq("V13", "V13_ph1_split_positives", "ph1_results.split and fold10_positives == V13",
                 [r["split"], r["fold10_positives"]],
                 [{"train": "folds 1-9", "eval": "fold 10", "n_train": splits["folds_1_9"]["n_records"],
                   "n_eval": splits["fold_10"]["n_records"], "n_train_patients": splits["folds_1_9"]["n_patients"],
                   "n_eval_patients": splits["fold_10"]["n_patients"]}, D["positives_fold_10"]], kind="recomputed")


# =====================================================================================
# V14 - provenance (INFO)
# =====================================================================================
def sec_v14():
    for a in BITCOIN_ANCHORS:
        info("V14", "V14_anchor_{}".format(a["block"]), "Bitcoin anchor (verified manually on opentimestamps.org; not "
             "machine-checked)", a)


# =====================================================================================
# paper_numbers.json and tables
# =====================================================================================
def _dt_dict(dt):
    out = {}
    for _, r in dt.iterrows():
        out.setdefault(r["block"], {})[r["learner"]] = {
            "auroc": float(r["macro_auroc_mean"]), "ci95": [float(r["ci_low_mean"]), float(r["ci_high_mean"])],
            "std_across_seeds": float(r["macro_auroc_std_across_seeds"])}
    return out


def _h2_dict(rpl):
    return {l: {"auroc_B1": r["point_macro_auroc_B1"], "auroc_B3": r["point_macro_auroc_B3"],
                "point_delta": r["point_delta"], "se": r["SE_delta"], "mde": r["MDE"], "ci95": r["ci95_percentile"],
                "ci90": r["ci90_percentile"], "difference_test_verdict": r["difference_test_verdict"],
                "tost_verdict": r["tost_verdict"], "achieved_equivalence_bound": r["achieved_equivalence_bound"]}
            for l, r in rpl.items()}


def _h3_dict(s):
    v = s["h3_verdict"]
    return {"arch": s["arch"], "auroc_A": s["macro_auroc_A"], "auroc_B": s["macro_auroc_B"], "point_delta": s["point_delta"],
            "ci95": [s["lo95"], s["hi95"]], "ci90": [s["lo90"], s["hi90"]], "a_holds": v["a_holds"],
            "b_holds": v["b_holds"], "material": v["material"], "sign": v["sign"], "verdict": v["verdict"],
            "sealed_rules_holding": v["sealed_rules_holding"]}


def build_paper_numbers(summary):
    pn = {"schema_version": "1", "verify": summary}
    pn["dataset"] = STATE.get("DATASET")
    fold9 = _dt_dict(STATE["V021_DT"]) if "V021_DT" in STATE and "V10_OK" in STATE else None
    p1 = os.path.join(PH1_DIR, "ph1_decision_table.csv")
    fold10 = _dt_dict(pd.read_csv(p1, encoding="utf-8-sig", float_precision="round_trip")) \
        if "V11_OK" in STATE else None
    pn["metadata_only"] = {"fold9": fold9, "fold10_posthoc": fold10, "note": METADATA_ONLY_NOTE}
    h1 = {"threshold": VERDICT_THRESHOLD, "fold9": None, "fold10_posthoc": None}
    if fold9:
        h1["fold9"] = {l: {"auroc_B3": fold9["B3_physio_plus_acquisition"][l]["auroc"],
                           "pass": bool(fold9["B3_physio_plus_acquisition"][l]["auroc"] >= VERDICT_THRESHOLD)}
                       for l in LEARNERS}
    if "V11_OK" in STATE:
        v = STATE["PH1_RESULTS"]["H1_posthoc"]["verdict"]
        h1["fold10_posthoc"] = {l: {"auroc_B3": v[l]["macro_auroc_mean"], "pass": v[l]["passes_threshold"]}
                                for l in LEARNERS}
    pn["h1"] = h1
    pn["h2"] = {"fold9": _h2_dict(STATE["SEAL"]["H2"]["results_per_learner"]) if "V10_OK" in STATE else None,
                "fold10_posthoc": _h2_dict(STATE["PH1_RESULTS"]["H2_posthoc"]["results_per_learner"])
                if "V11_OK" in STATE else None,
                "sesoi_h2": STATE["SEAL"]["SESOI_H2"] if "SEAL" in STATE else None}
    pn["ecg_fold10"] = STATE.get("V9")
    if "V8" in STATE:
        res = STATE["V09_RESULTS"]
        pn["h3"] = {"sesoi_h3": res["settings"]["SESOI_H3"], "primary": _h3_dict(res["1_primary"]),
                    "robustness": _h3_dict(res["2_robustness"]),
                    "per_seed": [{"arch": r["arch"], "seed": r["seed"], "point_delta": r["point_delta"],
                                  "ci95": [r["lo95"], r["hi95"]], "ci90": [r["lo90"], r["hi90"]],
                                  "verdict": r["h3_verdict_secondary"]["verdict"],
                                  "label": r["h3_verdict_secondary"]["label"]} for r in res["3_secondary_per_seed"]]}
        ex = res["5_exploratory"]
        pn["exploratory"] = {
            "E1_physio_minus_signal_only": {e["arch"]: {"point_delta": e["point_delta"], "ci95": [e["lo95"], e["hi95"]],
                                                        "ci90": [e["lo90"], e["hi90"]]}
                                            for e in ex["E1_physio_minus_signal_only"]},
            "E2_per_class_h3_pair": {e["arch"]: {c: {"point_delta": e["per_class"][c]["point_delta"],
                                                     "ci95": [e["per_class"][c]["lo95"], e["per_class"][c]["hi95"]]}
                                                 for c in SUPERCLASSES} for e in ex["E2_per_class_deltas_H3_pair"]},
            "E3_acquisition_permutation": {e["arch"]: {"unpermuted": e["unpermuted_macro_auroc"],
                                                       "drop_mean": e["drop_mean"], "drop_min": e["drop_min"],
                                                       "drop_max": e["drop_max"], "drops": e["drop"]}
                                           for e in ex["E3_acquisition_permutation"]}}
    else:
        pn["h3"], pn["exploratory"] = None, None
    pn["reference_published"] = REFERENCE_PUBLISHED
    pn["locked_settings"] = STATE.get("LOCKED_SETTINGS")
    docs = []
    blocks = {d: a for a in BITCOIN_ANCHORS for d in a["documents"]}
    for name, file_sha, self_sha in (
            [(e["name"], e["file_sha256"], e["self_sha256"]) for e in DOC_EXPECTED.values()] +
            [("v0.8_manifest.json", V08_MANIFEST_SHA256, None)] +
            [(n, e["file_sha256"], e["self_sha256"]) for n, e in V09_DOCS.items()]):
        a = blocks[name]
        docs.append({"document": name, "role": PROVENANCE_ROLES[name], "file_sha256": file_sha, "self_sha256": self_sha,
                     "bitcoin_block": a["block"], "block_time": a["time"]})
    pn["provenance"] = {"documents": docs, "bitcoin_anchors": BITCOIN_ANCHORS,
                        "note": "anchors verified manually on opentimestamps.org; not machine-checked"}
    return pn


def _ci(lohi, sign=False):
    f = signed if sign else unsigned
    return "[{}, {}]".format(f(lohi[0]), f(lohi[1]))


def build_tables(pn):
    """Tables generated only from paper_numbers.json. Returns {file name: DataFrame}."""
    T = {}
    rows = []
    mo = pn["metadata_only"]
    for b in BLOCK_NAMES:
        r = {"block": b}
        for fold, key in (("fold 9", "fold9"), ("fold 10 (post hoc)", "fold10_posthoc")):
            for l, short in (("LogisticRegression", "LR"), ("HistGradientBoosting", "HGB")):
                v = (mo[key] or {}).get(b, {}).get(l)
                r["{} {}".format(fold, short)] = "n/a" if v is None else "{} {}".format(unsigned(v["auroc"]), _ci(v["ci95"]))
        rows.append(r)
    T["table_1_metadata_only.csv"] = pd.DataFrame(rows)
    rows = []
    for fold, key in (("fold 9", "fold9"), ("fold 10 (post hoc)", "fold10_posthoc")):
        for l in LEARNERS:
            v = (pn["h2"][key] or {}).get(l)
            if v is None:
                rows.append({"fold": fold, "learner": l, "delta_B3_minus_B1": "n/a", "ci95": "n/a", "ci90": "n/a",
                             "difference_test": "n/a", "tost": "n/a", "achieved_bound": "n/a"})
                continue
            rows.append({"fold": fold, "learner": l, "delta_B3_minus_B1": signed(v["point_delta"]),
                         "ci95": _ci(v["ci95"], True), "ci90": _ci(v["ci90"], True),
                         "difference_test": v["difference_test_verdict"], "tost": v["tost_verdict"],
                         "achieved_bound": unsigned(v["achieved_equivalence_bound"])})
    T["table_2_h2.csv"] = pd.DataFrame(rows)
    rows = []
    if pn["h3"] is not None:
        for arch, key in (("resnet1d_wang", "primary"), ("inception1d", "robustness")):
            h, e = pn["h3"][key], pn["ecg_fold10"][arch] if pn["ecg_fold10"] else None
            r = {"architecture": arch}
            for arm in ARMS:
                r["auroc " + arm] = "n/a" if e is None else "{} {}".format(unsigned(e[arm]["auroc"]),
                                                                          _ci(e[arm]["ci95_descriptive"]))
            r.update({"H3 delta": signed(h["point_delta"]), "H3 ci95": _ci(h["ci95"], True), "H3 ci90": _ci(h["ci90"], True),
                      "a / b / material": "{} / {} / {}".format(h["a_holds"], h["b_holds"], h["material"]),
                      "H3 verdict": h["verdict"]})
            e1 = pn["exploratory"]["E1_physio_minus_signal_only"][arch]
            e3 = pn["exploratory"]["E3_acquisition_permutation"][arch]
            r["E1 delta [ci95]"] = "{} {}".format(signed(e1["point_delta"]), _ci(e1["ci95"], True))
            r["E3 drop mean [min, max]"] = "{} [{}, {}]".format(signed(e3["drop_mean"]), signed(e3["drop_min"]),
                                                               signed(e3["drop_max"]))
            rows.append(r)
    T["table_3_ecg_h3.csv"] = pd.DataFrame(rows)
    rows = []
    for r in (pn["h3"] or {}).get("per_seed", []):
        rows.append({"architecture": r["arch"], "seed": r["seed"], "delta": signed(r["point_delta"]),
                     "ci95": _ci(r["ci95"], True), "ci90": _ci(r["ci90"], True), "verdict": r["verdict"],
                     "label": r["label"]})
    T["table_4_h3_per_seed.csv"] = pd.DataFrame(rows)
    rows = []
    for d in pn["provenance"]["documents"]:
        rows.append({"document": d["document"], "role": d["role"], "file_sha256_12": d["file_sha256"][:12],
                     "self_sha256_12": "\u2014" if d["self_sha256"] is None else d["self_sha256"][:12],
                     "bitcoin_block": d["bitcoin_block"], "block_time": d["block_time"]})
    T["table_5_provenance.csv"] = pd.DataFrame(rows)
    return T


def tables_markdown(T):
    out = ["# Paper tables (generated by {} from paper_numbers.json only)".format(VERIFY_VERSION), ""]
    for name, df in T.items():
        out.append("## " + name.replace(".csv", ""))
        out.append("")
        cols = list(df.columns)
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join("---" for _ in cols) + "|")
        for _, r in df.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        out.append("")
    return "\n".join(out)


# =====================================================================================
# main
# =====================================================================================
hdr("verify {} | root {} | run {}".format(VERIFY_VERSION, ROOT, RUN_ID))
T_START = time.time()
SNAPSHOT_START = snapshot()
print("start snapshot: {} entries".format({k: (None if v is None else (len(v) if isinstance(v, dict) else 1))
                                           for k, v in SNAPSHOT_START.items()}))
for _code, _title, _fn in (("V0", "Environment", sec_v0), ("V1", "Dataset inputs", sec_v1),
                           ("V2", "Locked documents", sec_v2), ("V3", "v0.2.1 and v0.3 chains", sec_v3),
                           ("V4", "v0.8 chain", sec_v4), ("V5", "v0.9 chain", sec_v5),
                           ("VP", "Provenance of the copied code", sec_vp),
                           ("V6", "Fold-10 rows and labels", sec_v6), ("V7", "BOOT_IDX and PERM_IDX", sec_v7),
                           ("V8", "H3 and exploratory results (v0.9 Part D)", sec_v8),
                           ("V9", "Descriptive per-arm macro-AUROC (not pre-specified)", sec_v9),
                           ("V10", "Metadata-only, fold 9 (H1 / H2)", sec_v10),
                           ("V11", "ph1 (post hoc, fold 10) and the r1 attempt", sec_v11),
                           ("V12", "Locked settings", sec_v12), ("V13", "Dataset descriptives", sec_v13),
                           ("V14", "Provenance anchors", sec_v14)):
    run_section(_code, _title, _fn)
if WARNINGS:
    record("V0", "V0_code_warnings", "warnings raised by the copied ph1 / v0.2.1 code during the recomputation",
           "informational", "WARN", observed=list(WARNINGS))
for _w in ANALYSIS_WARNINGS:
    pass   # recorded through V8_results_warnings (compared with the stored v0.9 warnings)

hdr("V15 - End snapshot")
_end = snapshot()
_changed = {k: v for k, v in _end.items() if SNAPSHOT_START.get(k) != v}
check_eq("V15", "V15_snapshot_unchanged", "every protected file is unchanged since the start of the run",
         sorted(_changed), [], kind="hash", changed={k: [SNAPSHOT_START.get(k) is None, v is None] for k, v in _changed.items()})

_ids = [c["id"] for c in CHECKS]
_dup = sorted({i for i in _ids if _ids.count(i) > 1})
record("V15", "V15_unique_check_ids", "every check id is unique", "informational", "PASS" if not _dup else "FAIL",
       expected=[], observed=_dup)
N_FAIL = sum(c["status"] == "FAIL" for c in CHECKS)
N_SKIP = sum(c["status"] == "SKIP" for c in CHECKS)
SUMMARY = {"version": VERIFY_VERSION, "run_id": RUN_ID, "started": RUN_STARTED_AT.astimezone().isoformat(),
           "finished": datetime.now().astimezone().isoformat(), "all_checks_passed": N_FAIL == 0,
           "n_checks": len(CHECKS), "n_failed": N_FAIL, "n_skipped": N_SKIP,
           "paper_final": N_FAIL == 0 and N_SKIP == 0, "versions": STATE.get("versions")}

hdr("OUTPUTS -> " + RUN_DIR)
try:
    os.makedirs(RUN_DIR, exist_ok=False)
except FileExistsError:
    raise RuntimeError("STOP: {} already exists - a run folder is never overwritten".format(RUN_DIR))
_written = {}
PAPER_NUMBERS = build_paper_numbers(SUMMARY)
_written["paper_numbers.json"] = write_once_bytes(os.path.join(RUN_DIR, "paper_numbers.json"), json.dumps(
    plain(PAPER_NUMBERS), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
TABLES = build_tables(json.loads(json.dumps(plain(PAPER_NUMBERS))))
for _n, _df in TABLES.items():
    _written[_n] = write_once_bytes(os.path.join(RUN_DIR, _n), csv_bytes(_df))
_written["tables.md"] = write_once_bytes(os.path.join(RUN_DIR, "tables.md"), tables_markdown(TABLES).encode("utf-8"))
_report = {"document": "verify report", "summary": SUMMARY, "checks": CHECKS,
           "cpu_count": STATE.get("cpu_count"), "threadpool_info": STATE.get("threadpool_info"),
           "own_source_sha256": STATE.get("own_source_sha256"), "self_sha256_procedure": SELF_SHA256_PROCEDURE}
_written["verify_report.json"] = write_once_json_atomic(os.path.join(RUN_DIR, "verify_report.json"), _report)
_txt = ["verify {} | {} | root {}".format(VERIFY_VERSION, RUN_ID, ROOT),
        "checks {} | FAIL {} | SKIP {} | WARN {} | paper_final {}".format(
            len(CHECKS), N_FAIL, N_SKIP, sum(c["status"] == "WARN" for c in CHECKS), SUMMARY["paper_final"]), ""]
for _c in CHECKS:
    _txt.append("[{:4s}] {:4s} {} - {}{}".format(_c["status"], _c["section"], _c["id"], _c["description"],
                                                 "" if _c["max_abs_diff"] is None else " | max |diff| {}".format(
                                                     _c["max_abs_diff"])))
    if _c["status"] in ("FAIL", "WARN", "SKIP"):
        _txt.append("         expected {} | observed {}{}".format(str(_c["expected"])[:300], str(_c["observed"])[:300],
                                                                  " | " + str(_c["note"]) if _c.get("note") else ""))
_written["verify_report.txt"] = write_once_bytes(os.path.join(RUN_DIR, "verify_report.txt"),
                                                 ("\n".join(_txt) + "\n").encode("utf-8"))
_man = {"document": "verify manifest", "version": VERIFY_VERSION, "run_id": RUN_ID, "root": ROOT,
        "own_source_sha256": STATE.get("own_source_sha256"),
        "outputs": [{"file": n, "bytes": os.path.getsize(os.path.join(RUN_DIR, n)), "sha256": sha256_of(os.path.join(RUN_DIR, n))}
                    for n in sorted(os.listdir(RUN_DIR))],
        "summary": SUMMARY, "self_sha256_procedure": SELF_SHA256_PROCEDURE}
_man_self = write_once_json_atomic(os.path.join(RUN_DIR, "verify_manifest.json"), _man)
print("files written: {} | verify_manifest.json self_sha256 {}".format(sorted(os.listdir(RUN_DIR)), _man_self))

hdr("RESULT")
if N_FAIL == 0:
    print("ALL CHECKS PASSED ({}) | SKIP {} | paper_final {} | {:.0f}s".format(len(CHECKS), N_SKIP, SUMMARY["paper_final"],
                                                                          time.time() - T_START))
    if not SUMMARY["paper_final"]:
        print("!!! paper cannot be finalised: {} check(s) were skipped".format(N_SKIP))
else:
    print("#" * 100)
    print("FAIL: {} of {} checks failed".format(N_FAIL, len(CHECKS)))
    for _c in CHECKS:
        if _c["status"] == "FAIL":
            print("  - {} : {}".format(_c["id"], _c["description"]))
    print("#" * 100)
    if not IN_IPYTHON:
        sys.exit(1)
