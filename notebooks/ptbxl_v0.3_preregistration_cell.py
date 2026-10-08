# =====================================================================================
# Project: PTB-XL Metadata Audit  -  v0.3 MDE, equivalence test, sealed pre-registration
# Code version: v0.3
# Date: 2026-10-01
# Purpose: (A) refit B1_physio and B3_physio_plus_acquisition on folds 1-8 (seed 42),
#          sanity-check them against v0.2.1, and run a PAIRED patient-level bootstrap of
#          delta = macro-AUROC(B3) - macro-AUROC(B1) on fold 9 for each learner;
#          (B) report the minimum detectable effect (MDE, descriptive only);
#          (C) test H2 two ways (difference test + TOST equivalence test, SESOI 0.02);
#          (D) write the sealed pre-registration document, ONCE.
#
# EPISTEMIC STATUS (stated here and inside the sealed document):
#   H1 and H2 are CONFIRMATORY-AFTER-THE-FACT / POST-HOC: their statistics were already
#   seen in v0.2.1 (the week-1 death experiment was a read-only feasibility gate run
#   before this document existed; inherent design limitation). Only H3 and the one-time
#   fold-10 analysis are genuinely prospective; locking THOSE is the purpose of this
#   document.
#
# REUSE RULE: data loading, label derivation, preprocessing (age sentinel,
# encode_categorical with __MISSING__/__UNKNOWN__, mixed scaling for
# LogisticRegression), feature blocks, model hyperparameters, fast_auc and BOOT_ROWS
# construction are copied VERBATIM from v0.2.1. v0.2.1 did not persist per-row eval
# scores, so the two blocks are refit here.
#
# SPLIT PROTOCOL: train on strat_fold in {1..8}, evaluate on strat_fold == 9.
# strat_fold == 10 is never assembled into any array used by this cell (same runtime
# asserts as v0.2.1).
#
# STEP 0 (seal check) runs before any model fit and before any file write. If
# pre_registration.json already exists, the cell prints it and stops.
#
# CPU only. No deep model. No plot. Nothing is written inside the dataset folder.
#
# Outputs:
#   OUT_DIR:
#     1) v0.3_bootstrap_deltas.csv   (replicate, delta_LR, delta_HGB)
#     2) v0.3_h2_summary.csv         (per learner: point delta, mean, SE, MDE, CIs, verdicts)
#     3) v0.3_manifest.json          (sha256 of files 1, 2 and of SEAL_PATH; not of itself)
#   SEAL_PATH:
#     4) pre_registration.json       (written once, exclusive-create; self_sha256 inside)
# =====================================================================================

import os, sys, json, ast, hashlib, platform, time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 220)

# ------------------------------- configuration ---------------------------------------
DATA_ROOT = r"D:\ptbxl-metadata-audit\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT_DIR   = r"D:\ptbxl-metadata-audit\outputs\v0.3"
SEAL_DIR  = r"D:\ptbxl-metadata-audit\preregistration"     # version-independent
SEAL_PATH = os.path.join(SEAL_DIR, "pre_registration.json")
CODE_VERSION = "v0.3"
PREFIX = CODE_VERSION + "_"
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

SEED = 42
N_BOOT = 2000            # reuses v0.2.1's exact scheme (original plan: 10,000 -> logged deviation)
BOOT_SEED = 7
LEARNERS = ["LogisticRegression", "HistGradientBoosting"]
PRIMARY_LEARNER = "LogisticRegression"
ROBUSTNESS_LEARNER = "HistGradientBoosting"
SHORT = {"LogisticRegression": "LR", "HistGradientBoosting": "HGB"}
COMPARED_BLOCKS = ["B1_physio", "B3_physio_plus_acquisition"]

ALPHA = 0.05
POWER = 0.80
Z_ALPHA_2 = 1.959964
Z_POWER = 0.841621
SESOI_H2 = 0.02
SESOI_H3 = 0.01
SANITY_TOL = 2e-4

# v0.2.1 input hashes (must match exactly)
EXPECTED_INPUT_HASHES = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}

# v0.2.1 seed-42 fold-9 macro-AUROC (source: v0.2.1 Section 3+4 console output,
# rows "B1_physio"/"B3_physio_plus_acquisition", seed=42)
V021_SEED42 = {
    "LogisticRegression":   {"B1_physio": 0.6890, "B3_physio_plus_acquisition": 0.6972, "delta": +0.0082},
    "HistGradientBoosting": {"B1_physio": 0.6724, "B3_physio_plus_acquisition": 0.6923, "delta": +0.0199},
}
# v0.2.1 seed-42 marginal 95% bootstrap CIs (same source). Additional check: if the
# resampled row sets are identical to v0.2.1's, these reproduce.
V021_SEED42_CI = {
    "LogisticRegression":   {"B1_physio": (0.6743, 0.7042), "B3_physio_plus_acquisition": (0.6825, 0.7119)},
    "HistGradientBoosting": {"B1_physio": (0.6568, 0.6879), "B3_physio_plus_acquisition": (0.6776, 0.7072)},
}
# v0.2.1 context-only constants (Section 5 decision table)
V021_HGB_3SEED_MEAN_DELTA = 0.0212
V021_H1 = {"LogisticRegression_B3": 0.6972, "HistGradientBoosting_B3_3seed_mean": 0.6937}
V021_B5_VS_B3 = {"LR_B5": 0.6969, "LR_B3": 0.6972, "HGB_B5_3seed_mean": 0.6935, "HGB_B3_3seed_mean": 0.6937}

_root_abs = os.path.abspath(DATA_ROOT)
for _p in (OUT_DIR, SEAL_DIR):
    assert not os.path.abspath(_p).lower().startswith(_root_abs.lower()), \
        "output path is inside the dataset folder - aborting: " + _p

RUN_STARTED = datetime.now().astimezone().isoformat()
OUTPUT_FILES = []
WARNINGS = []


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_csv(obj, name, index=True):
    p = os.path.join(OUT_DIR, name)
    obj.to_csv(p, index=index, encoding="utf-8-sig")
    OUTPUT_FILES.append(name)
    print("[saved] " + p)
    return p


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


# =====================================================================================
# STEP 0 - SEAL CHECK (before any model fit and before any file write of any kind)
# =====================================================================================
if os.path.exists(SEAL_PATH):
    hdr("STEP 0 - SEAL CHECK")
    with open(SEAL_PATH, "r", encoding="utf-8") as f:
        _sealed = json.load(f)
    print(json.dumps(_sealed, ensure_ascii=False, indent=2))
    print("\nALREADY SEALED (self_sha256={}) -- nothing was recomputed or overwritten. "
          "Delete the file by hand outside this notebook to redo this stage.".format(
              _sealed.get("self_sha256", "<missing>")))
    raise SystemExit
print("STEP 0 - seal check: no pre_registration.json at " + SEAL_PATH + " -> proceeding.")

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================================
# LOAD + PROVENANCE (verbatim from v0.2.1, plus the hash equality assert)
# =====================================================================================
hdr("LOAD + PROVENANCE")

print("run started  :", RUN_STARTED)
print("code version :", CODE_VERSION)
print("python       :", sys.version.replace("\n", " "))
print("platform     :", platform.platform())
print("numpy        :", np.__version__, "| pandas:", pd.__version__)

DB_CSV = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_CSV = os.path.join(DATA_ROOT, "scp_statements.csv")
for p in (DB_CSV, SCP_CSV):
    assert os.path.isfile(p), "file not found: " + p

INPUT_HASHES = {os.path.basename(p): sha256_of(p) for p in (DB_CSV, SCP_CSV)}
for k, v in INPUT_HASHES.items():
    match = (v == EXPECTED_INPUT_HASHES[k])
    print("{:26s} {}  matches v0.2.1: {}".format(k, v, "YES" if match else "NO"))
    assert match, "input file hash differs from v0.2.1 for " + k + " - aborting."

df_all = pd.read_csv(DB_CSV, index_col="ecg_id")
print("full dataset shape:", df_all.shape)

# --- Working set: folds 1-9 ONLY. Fold 10 is never assembled into any array below. ---
df9 = df_all[df_all["strat_fold"] <= 9].copy()
mask_train = df9["strat_fold"] <= 8
mask_eval = df9["strat_fold"] == 9
print("folds 1-8 (train) records:", int(mask_train.sum()), "| fold 9 (eval) records:", int(mask_eval.sum()))

# --- Required runtime assertions ---
assert (df9["strat_fold"] == 10).sum() == 0, \
    "fold 10 leaked into the working set - aborting."
n_fold10_in_source = int((df_all["strat_fold"] == 10).sum())
print("fold 10 records in the source file (untouched):", n_fold10_in_source,
      "| fold 10 records in the working set: 0 (by construction, verified by assert)")

train_patients = set(df9.loc[mask_train, "patient_id"])
eval_patients = set(df9.loc[mask_eval, "patient_id"])
assert train_patients.isdisjoint(eval_patients), \
    "patient-level leakage between train (folds 1-8) and eval (fold 9) - aborting."
print("train patients:", len(train_patients), "| eval patients:", len(eval_patients),
      "| disjoint: PASS")


# =====================================================================================
# LABEL DERIVATION (verbatim from v0.2.1)
# =====================================================================================
scp = pd.read_csv(SCP_CSV, index_col=0)
diag = scp[scp["diagnostic"].fillna(0) == 1]
code2class = {k: v for k, v in diag["diagnostic_class"].items() if isinstance(v, str)}
parsed = df9["scp_codes"].apply(ast.literal_eval)
rec_classes = parsed.apply(lambda d: {code2class[k] for k in d if k in code2class})
lab = pd.DataFrame({c: rec_classes.apply(lambda s, c=c: int(c in s)) for c in SUPERCLASSES},
                   index=df9.index)
for c in SUPERCLASSES:
    n_pos_train = int(lab.loc[mask_train, c].sum())
    n_pos_eval = int(lab.loc[mask_eval, c].sum())
    if n_pos_train == 0 or n_pos_eval == 0:
        warn("{}: zero positive examples in {} set - this class cannot be modeled.".format(
            c, "train" if n_pos_train == 0 else "eval"))


# =====================================================================================
# PREPROCESSING + FEATURE BLOCKS (verbatim from v0.2.1)
# =====================================================================================
hdr("PREPROCESSING + FEATURE BLOCKS (verbatim from v0.2.1)")

age_raw = df9["age"]
age_sentinel_mask = (age_raw == 300)
n_age_sentinel = int(age_sentinel_mask.sum())
age_clean = age_raw.where(~age_sentinel_mask, np.nan)
age_censored = age_sentinel_mask.astype(float)
age_imputed = age_clean.fillna(90.0)
print("age: sentinel 300 affects", n_age_sentinel, "records; flagged via age_censored, imputed with 90.")

sex = df9["sex"].astype(float)

rd = pd.to_datetime(df9["recording_date"], errors="coerce")
recording_year = rd.dt.year.astype(float)
n_year_missing = int(recording_year.isna().sum())
if n_year_missing > 0:
    train_median_year = float(recording_year[mask_train].median())
    recording_year = recording_year.fillna(train_median_year)
    warn("{} records had an unparsable recording_date; recording_year imputed with the "
         "training-set median ({}).".format(n_year_missing, train_median_year))

device_level = to_level_string(df9["device"])
site_level = to_level_string(df9["site"])
nurse_level = to_level_string(df9["nurse"])

device_dummies, device_categories = encode_categorical(device_level, mask_train, "device")
site_dummies, site_categories = encode_categorical(site_level, mask_train, "site")
nurse_dummies, nurse_categories = encode_categorical(nurse_level, mask_train, "nurse")

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
for name in COMPARED_BLOCKS:
    print("  {:28s} train=({}, {})  eval=({}, {})".format(
        name, int(mask_train.sum()), len(BLOCKS[name]), int(mask_eval.sum()), len(BLOCKS[name])))


# =====================================================================================
# PART A - REFIT (seed 42) + SANITY CHECK + PAIRED BOOTSTRAP
# =====================================================================================
hdr("PART A - REFIT (seed {}) + HARD SANITY CHECK vs v0.2.1".format(SEED))

y_train_full = lab.loc[mask_train]
y_eval_full = lab.loc[mask_eval]
Y_EVAL = {sc: y_eval_full[sc].to_numpy() for sc in SUPERCLASSES}

SCORES = {learner: {} for learner in LEARNERS}   # SCORES[learner][block][class] -> eval probas
POINT_MACRO = {learner: {} for learner in LEARNERS}

t_fit = time.time()
for block_name in COMPARED_BLOCKS:
    cols = BLOCKS[block_name]
    Xtr_raw = FEATURES.loc[mask_train, cols].to_numpy(dtype=float)
    Xev_raw = FEATURES.loc[mask_eval, cols].to_numpy(dtype=float)

    # Only continuous columns are standardized for LogisticRegression (verbatim v0.2.1).
    block_cols = cols
    continuous_cols = [c for c in block_cols if c in ("age_imputed", "recording_year")]
    if continuous_cols:
        cont_idx = [block_cols.index(c) for c in continuous_cols]
        rest_idx = [i for i in range(len(block_cols)) if i not in cont_idx]

        scaler = StandardScaler().fit(Xtr_raw[:, cont_idx])
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
        Xtr_lr = Xtr_raw
        Xev_lr = Xev_raw

    for learner in LEARNERS:
        scores_eval = {}
        per_class_auroc = []
        for sc in SUPERCLASSES:
            ytr = y_train_full[sc].to_numpy()
            if learner == "LogisticRegression":
                model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=SEED)
                model.fit(Xtr_lr, ytr)
                pos_col = list(model.classes_).index(1)
                proba = model.predict_proba(Xev_lr)[:, pos_col]
            else:
                model = HistGradientBoostingClassifier(random_state=SEED)
                model.fit(Xtr_raw, ytr)
                pos_col = list(model.classes_).index(1)
                proba = model.predict_proba(Xev_raw)[:, pos_col]
            scores_eval[sc] = proba
            per_class_auroc.append(fast_auc(Y_EVAL[sc], proba))
        SCORES[learner][block_name] = scores_eval
        POINT_MACRO[learner][block_name] = float(np.nanmean(per_class_auroc))
print("refit time: {:.1f}s".format(time.time() - t_fit))

SANITY_ROWS = []
for learner in LEARNERS:
    b1 = POINT_MACRO[learner]["B1_physio"]
    b3 = POINT_MACRO[learner]["B3_physio_plus_acquisition"]
    ref = V021_SEED42[learner]
    for label, got, exp in [("B1", b1, ref["B1_physio"]),
                            ("B3", b3, ref["B3_physio_plus_acquisition"]),
                            ("delta", b3 - b1, ref["delta"])]:
        ok = abs(got - exp) <= SANITY_TOL
        SANITY_ROWS.append({"learner": learner, "quantity": label, "refit": got,
                            "v0.2.1": exp, "abs_diff": abs(got - exp), "pass": ok})
        print("  [{:20s}] {:5s} refit={:+.6f}  v0.2.1={:+.4f}  |diff|={:.6f}  -> {}".format(
            learner, label, got, exp, abs(got - exp), "PASS" if ok else "FAIL"))
for r in SANITY_ROWS:
    assert r["pass"], ("SANITY CHECK FAILED ({} {}): refit {:.6f} vs v0.2.1 {:.4f}. "
                       "The bootstrap below would not be trustworthy - stopping.".format(
                           r["learner"], r["quantity"], r["refit"], r["v0.2.1"]))
print("HARD SANITY CHECK: all 6 values reproduce v0.2.1 within abs tolerance {}.".format(SANITY_TOL))
note("Context only: v0.2.1 HGB 3-seed-mean delta (B3-B1) = +{:.4f}. This stage uses the "
     "seed-42 refit (HGB seed-42 delta = {:+.4f}).".format(
         V021_HGB_3SEED_MEAN_DELTA, POINT_MACRO["HistGradientBoosting"]["B3_physio_plus_acquisition"]
         - POINT_MACRO["HistGradientBoosting"]["B1_physio"]))

# --- bootstrap resamples: verbatim v0.2.1 construction (same order, same RNG) ---
hdr("PART A - PAIRED PATIENT-LEVEL BOOTSTRAP (N_BOOT={}, BOOT_SEED={})".format(N_BOOT, BOOT_SEED))

eval_patient_arr = df9.loc[mask_eval, "patient_id"].to_numpy()
unique_eval_patients = np.unique(eval_patient_arr)
n_eval_patients = len(unique_eval_patients)
patient_row_list = [np.where(eval_patient_arr == p)[0] for p in unique_eval_patients]

t0 = time.time()
rng_boot = np.random.default_rng(BOOT_SEED)
BOOT_ROWS = []
for _ in range(N_BOOT):
    samp = rng_boot.integers(0, n_eval_patients, size=n_eval_patients)
    BOOT_ROWS.append(np.concatenate([patient_row_list[i] for i in samp]))
print("precomputed {} patient-level bootstrap resamples of the fold-9 eval set "
      "({} patients, {} records) in {:.2f}s".format(N_BOOT, n_eval_patients, int(mask_eval.sum()),
                                                     time.time() - t0))
DEVIATION_NOTE = ("N_BOOT=2000 reuses v0.2.1's exact scheme; the original plan specified "
                  "10,000 -- stated, logged deviation.")
note(DEVIATION_NOTE)


def macro_on_rows(scores, rows):
    """Macro-AUROC on a row subset, same nan-skipping rule as v0.2.1's bootstrap."""
    accs = []
    for sc in SUPERCLASSES:
        a = fast_auc(Y_EVAL[sc][rows], scores[sc][rows])
        if not np.isnan(a):
            accs.append(a)
    return np.mean(accs) if accs else np.nan


# PAIRED: B1 and B3 are always evaluated on the SAME rows_b.
BOOT_B1 = {learner: np.empty(N_BOOT) for learner in LEARNERS}
BOOT_B3 = {learner: np.empty(N_BOOT) for learner in LEARNERS}
DELTAS = {learner: np.empty(N_BOOT) for learner in LEARNERS}
t0 = time.time()
for bi, rows in enumerate(BOOT_ROWS):
    for learner in LEARNERS:
        m1 = macro_on_rows(SCORES[learner]["B1_physio"], rows)
        m3 = macro_on_rows(SCORES[learner]["B3_physio_plus_acquisition"], rows)
        BOOT_B1[learner][bi] = m1
        BOOT_B3[learner][bi] = m3
        DELTAS[learner][bi] = m3 - m1
print("paired bootstrap time: {:.1f}s".format(time.time() - t0))

# Additional check: marginal CIs must reproduce v0.2.1's seed-42 CIs (confirms identical row sets).
for learner in LEARNERS:
    for block_name, arr in [("B1_physio", BOOT_B1[learner]), ("B3_physio_plus_acquisition", BOOT_B3[learner])]:
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        exp_lo, exp_hi = V021_SEED42_CI[learner][block_name]
        ok = abs(lo - exp_lo) <= SANITY_TOL and abs(hi - exp_hi) <= SANITY_TOL
        print("  [{:20s}] {:28s} marginal 95%CI [{:.4f},{:.4f}]  v0.2.1 [{:.4f},{:.4f}] -> {}".format(
            learner, block_name, lo, hi, exp_lo, exp_hi, "PASS" if ok else "FAIL"))
        assert ok, "marginal bootstrap CI does not reproduce v0.2.1 - resampling differs; stopping."

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


# =====================================================================================
# PART B - MDE (descriptive only)
# =====================================================================================
hdr("PART B - MINIMUM DETECTABLE EFFECT (alpha={} two-sided, power={})".format(ALPHA, POWER))
for learner in LEARNERS:
    mde = round((Z_ALPHA_2 + Z_POWER) * H2_STATS[learner]["SE_delta"], 4)
    H2_STATS[learner]["MDE"] = mde
    print("  [{:20s}] MDE = ({} + {}) * {:.6f} = {:.4f}".format(
        learner, Z_ALPHA_2, Z_POWER, H2_STATS[learner]["SE_delta"], mde))
MDE_NOTE = ("MDE is a power quantity (smallest true effect detectable at 80% power on this eval "
            "set). It is NOT a decision threshold; see Part C.")
print(MDE_NOTE)


# =====================================================================================
# PART C - H2: DIFFERENCE TEST + EQUIVALENCE TEST (TOST), per learner
# =====================================================================================
hdr("PART C - H2 TESTS (SESOI_H2 = {:.2f} macro-AUROC)".format(SESOI_H2))

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
    tost_pass = (lo90 >= -SESOI_H2) and (hi90 <= SESOI_H2)
    tost_verdict = "practically negligible" if tost_pass else "inconclusive / not shown negligible"
    s.update({
        "difference_significant": (not zero_in_ci), "difference_verdict": diff_verdict,
        "tost_pass": tost_pass, "tost_verdict": tost_verdict,
        "achieved_equivalence_bound": float(max(abs(lo90), abs(hi90))),
    })
    print("  [{:20s}] difference test: 95%CI[{:+.4f},{:+.4f}] -> {}".format(learner, lo95, hi95, diff_verdict))
    print("  [{:20s}] TOST (+/-{:.2f}): 90%CI[{:+.4f},{:+.4f}] -> {}".format(
        learner, SESOI_H2, lo90, hi90, tost_verdict))
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

n_fold9_records = int(mask_eval.sum())
NOTE_FOLD9_PROXY = ("Fold 9 is a STATED PROXY for fold 10's sampling variability (fold 9: {:,} records / "
                    "{:,} patients; fold 10: {:,} records).".format(
                        n_fold9_records, len(eval_patients), n_fold10_in_source))
NOTE_TEMPORAL = ("Temporal confounding: v0.2.1 B5_all (B3 + recording_year) ~= B3 (LR {:.4f} vs {:.4f}; "
                 "HGB 3-seed mean {:.4f} vs {:.4f}) -> the temporal covariate adds nothing material on "
                 "the metadata-only side; this does NOT address H3.".format(
                     V021_B5_VS_B3["LR_B5"], V021_B5_VS_B3["LR_B3"],
                     V021_B5_VS_B3["HGB_B5_3seed_mean"], V021_B5_VS_B3["HGB_B3_3seed_mean"]))
print()
note(NOTE_FOLD9_PROXY)
note(NOTE_TEMPORAL)


# =====================================================================================
# OUTPUTS 1-2 (OUT_DIR)
# =====================================================================================
hdr("OUTPUTS")

boot_df = pd.DataFrame({
    "replicate": np.arange(N_BOOT),
    "delta_LR": DELTAS["LogisticRegression"],
    "delta_HGB": DELTAS["HistGradientBoosting"],
})
save_csv(boot_df, PREFIX + "bootstrap_deltas.csv", index=False)

summary_rows = []
for learner in LEARNERS:
    s = H2_STATS[learner]
    summary_rows.append({
        "learner": learner, "role": "primary" if learner == PRIMARY_LEARNER else "robustness",
        "point_B1": s["point_B1"], "point_B3": s["point_B3"],
        "point_delta": s["point_delta"], "mean_delta": s["mean_delta"], "SE_delta": s["SE_delta"],
        "MDE": s["MDE"],
        "ci95_low": s["ci95"][0], "ci95_high": s["ci95"][1],
        "ci90_low": s["ci90"][0], "ci90_high": s["ci90"][1],
        "difference_verdict": s["difference_verdict"], "tost_verdict": s["tost_verdict"],
        "achieved_equivalence_bound": s["achieved_equivalence_bound"],
        "n_boot_valid": s["n_boot_valid"], "n_boot_invalid": s["n_boot_invalid"],
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.round(4).to_string(index=False))
save_csv(summary_df, PREFIX + "h2_summary.csv", index=False)


# =====================================================================================
# PART D - pre_registration.json (written ONCE, exclusive-create, after A-C succeeded)
# =====================================================================================
hdr("PART D - SEALED PRE-REGISTRATION")


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


prereg = {
    "timestamp": datetime.now().astimezone().isoformat(),
    "code_version": CODE_VERSION,
    "input_files_sha256": INPUT_HASHES,
    "framing": ("H1 and H2 are CONFIRMATORY-AFTER-THE-FACT / POST-HOC: their statistics were already "
                "seen in v0.2.1 (the week-1 death experiment was a read-only feasibility gate run before "
                "this document existed; inherent design limitation). Only H3 and the one-time fold-10 "
                "analysis are genuinely prospective; locking THOSE is the purpose of this document."),
    "epistemic_status": {
        "H1": "confirmatory-after-the-fact",
        "H2": "confirmatory-after-the-fact",
        "H3": "prospective, pending signal model",
        "fold10_final_analysis": "prospective, sealed until week 8",
    },
    "H1": {
        "statement": "B3 fold-9 macro-AUROC >= 0.60",
        "status": "already PASSED in v0.2.1",
        "values": {"LogisticRegression_B3": V021_H1["LogisticRegression_B3"],
                   "HistGradientBoosting_B3_3seed_mean": V021_H1["HistGradientBoosting_B3_3seed_mean"]},
    },
    "H2": {
        "statement": ("after controlling for age/sex, adding device/site/nurse does not change fold-9 "
                      "macro-AUROC by a practically relevant amount"),
        "comparison": "B3_physio_plus_acquisition vs B1_physio, delta = B3 - B1, seed 42 refit",
        "tests": {
            "difference_test": "0 inside the 95% percentile CI of the paired delta -> no statistically "
                               "significant difference; otherwise significant difference (sign, magnitude)",
            "equivalence_test_TOST": "'practically negligible' only if the ENTIRE 90% percentile CI of the "
                                     "paired delta lies within [-SESOI_H2, +SESOI_H2]; otherwise "
                                     "'inconclusive / not shown negligible'",
            "achieved_equivalence_bound": "max(|lower_90|, |upper_90|): smallest margin at which TOST passes",
        },
        "results_per_learner": {learner: _learner_block(learner) for learner in LEARNERS},
        "learner_disagreement": DISAGREEMENTS,
        "sanity_check_vs_v0.2.1": {"tolerance_abs": SANITY_TOL, "rows": SANITY_ROWS, "all_pass": True},
    },
    "SESOI_H2": SESOI_H2,
    "SESOI_H2_note": ("chosen by the researcher on 2026-10-01 AFTER v0.2.1's deltas were visible, "
                      "consistent with H2's confirmatory-after-the-fact status; lenient relative to the "
                      "~0.01 spread between published architectures on PTB-XL superclasses "
                      "(0.926-0.937); achieved_equivalence_bound is reported so readers can apply their "
                      "own margin"),
    "primary_learner": PRIMARY_LEARNER,
    "robustness_learner": ROBUSTNESS_LEARNER,
    "learner_rule": "disagreement is reported, never resolved by selection",
    "MDE": {
        "values": {learner: H2_STATS[learner]["MDE"] for learner in LEARNERS},
        "formula": "MDE = (z_{1-alpha/2} + z_{power}) * SE_delta",
        "alpha": ALPHA, "alpha_sided": "two-sided", "power": POWER,
        "z_alpha_2": Z_ALPHA_2, "z_power": Z_POWER,
        "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED, "deviation_note": DEVIATION_NOTE,
        "blocks": "B3_physio_plus_acquisition vs B1_physio", "model_seed": SEED,
        "note": MDE_NOTE,
    },
    "stopping_rule": ("decisions use the CI of the PAIRED delta distribution, never overlap of two "
                      "separate CIs"),
    "kill_criterion_week3": {
        "verbatim": ("metadata-only macro-AUROC <= 0.55 AND acquisition contribution after age/sex "
                     "< MDE"),
        "annotation": ("first condition already false (H1 passed, 0.697); second condition is "
                       "superseded by the TOST rule and is no longer a decision criterion anywhere in "
                       "the project; recorded for completeness"),
    },
    "H3": {
        "statement": ("adding acquisition metadata (device, site, nurse) to a signal model that already "
                      "receives age and sex does not materially CHANGE -- inflate OR deflate -- held-out "
                      "macro-AUROC"),
        "status": "pending, not yet tested",
    },
    "H3_protocol": {
        "comparison": "signal_plus_physio [age, sex] vs signal_plus_all [age, sex, device, site, nurse]",
        "metadata_fusion": ("metadata vector, encoded exactly as in v0.2.1 (encoder fit on training folds "
                            "only), concatenated to the network's global-pooled embedding before the final "
                            "linear classification head"),
        "primary_architecture": "resnet1d_wang on 100 Hz",
        "robustness_architecture": "inception1d",
        "architecture_rule": "disagreement is reported, never resolved by selection",
        "development": ("weeks 4-5: train folds 1-8, validate fold 9; all hyperparameters AND number of "
                        "epochs fixed there"),
        "final_run": ("train folds 1-9 with those fixed settings, no early stopping, no tuning; fold 10 "
                      "evaluated exactly once"),
        "seeds": [42, 123, 2024],
        "primary_estimate": ("per-record probabilities averaged over the 3 seeds, then macro-AUROC; "
                             "per-seed results secondary"),
        "test": ("paired patient-level bootstrap on fold 10, N_BOOT=2000, BOOT_SEED=7, delta = "
                 "macro_auroc(signal_plus_all) - macro_auroc(signal_plus_physio)"),
        "decision": {
            "a": "95% CI excludes 0 -> 'acquisition metadata changes performance' (report sign; either "
                 "direction counts)",
            "b": "90% CI entirely within [-0.01, +0.01] -> 'no material change'",
            "c": "otherwise -> 'inconclusive'",
        },
        "SESOI_H3": SESOI_H3,
        "SESOI_H3_note": ("comparable to the spread between published architectures on PTB-XL "
                          "superclasses; an effect this size could reorder them. Fixed BEFORE any signal "
                          "model is trained."),
        "exploratory": ("any analysis not specified here (ablations, per-class results, other block "
                        "comparisons) is EXPLORATORY"),
    },
    "notes": {
        "fold9_as_proxy": NOTE_FOLD9_PROXY,
        "temporal_confounding": NOTE_TEMPORAL,
    },
    "self_sha256_procedure": SELF_SHA256_PROCEDURE,
    "sealed": True,
}

# Normalize to plain JSON types (numpy scalars -> Python) so the hash computed now equals the
# hash a verifier computes from the loaded file. self_sha256 is ABSENT at this point.
prereg = json.loads(json.dumps(prereg, ensure_ascii=False, default=_json_default))
assert "self_sha256" not in prereg
self_sha = hashlib.sha256(canonical_json_bytes(prereg)).hexdigest()
prereg["self_sha256"] = self_sha

os.makedirs(SEAL_DIR, exist_ok=True)
with open(SEAL_PATH, "x", encoding="utf-8") as f:     # "x": refuses to overwrite an existing file
    json.dump(prereg, f, ensure_ascii=False, indent=2)
print("[saved] " + SEAL_PATH)

# verify the written file with the documented procedure
with open(SEAL_PATH, "r", encoding="utf-8") as f:
    _check = json.load(f)
_stored = _check.pop("self_sha256")
assert hashlib.sha256(canonical_json_bytes(_check)).hexdigest() == _stored == self_sha, \
    "self_sha256 verification of the written file FAILED."
print("self_sha256 verified from disk: PASS")


# =====================================================================================
# OUTPUT 3 - MANIFEST (OUT_DIR; sha256 of files 1, 2 and of SEAL_PATH, not of itself)
# =====================================================================================
manifest = {
    "code_version": CODE_VERSION,
    "run_started": RUN_STARTED,
    "run_finished": datetime.now().astimezone().isoformat(),
    "dataset_root": DATA_ROOT,
    "output_dir": OUT_DIR,
    "seal_path": SEAL_PATH,
    "input_files_sha256": INPUT_HASHES,
    "input_files_match_v0.2.1": True,
    "split_protocol": {
        "train_folds": "1-8", "eval_fold": 9, "fold10_touched": False,
        "n_fold10_records_in_source_untouched": n_fold10_in_source,
        "n_train_records": int(mask_train.sum()), "n_eval_records": int(mask_eval.sum()),
        "n_train_patients": len(train_patients), "n_eval_patients": len(eval_patients),
    },
    "learners": {"primary": PRIMARY_LEARNER, "robustness": ROBUSTNESS_LEARNER},
    "blocks_compared": COMPARED_BLOCKS,
    "model_seed": SEED, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
    "deviations": [DEVIATION_NOTE],
    "sanity_check": {"tolerance_abs": SANITY_TOL, "rows": SANITY_ROWS,
                     "marginal_ci_reproduced_v0.2.1": True},
    "h2_results": {learner: _learner_block(learner) for learner in LEARNERS},
    "learner_disagreement": DISAGREEMENTS,
    "pre_registration_self_sha256": self_sha,
    "checks": {"fold10_excluded_by_construction": True, "train_eval_patient_disjoint": True},
    "warnings": WARNINGS,
    "outputs": [],
}
for name in OUTPUT_FILES:
    p = os.path.join(OUT_DIR, name)
    manifest["outputs"].append({"file": name, "bytes": os.path.getsize(p), "sha256": sha256_of(p)})
manifest["outputs"].append({"file": SEAL_PATH, "bytes": os.path.getsize(SEAL_PATH),
                            "sha256": sha256_of(SEAL_PATH)})

MANIFEST_PATH = os.path.join(OUT_DIR, PREFIX + "manifest.json")
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)
print("[saved] " + MANIFEST_PATH)
print("manifest sha256:", sha256_of(MANIFEST_PATH))


# =====================================================================================
# FINAL SUMMARY
# =====================================================================================
hdr("SUMMARY - " + CODE_VERSION)
print("fold 10 touched            : No (0 records; assert verified)")
print("sanity check vs v0.2.1     : PASS (6 point values + 4 marginal CIs)")
for learner in LEARNERS:
    s = H2_STATS[learner]
    print("[{:20s}] delta={:+.4f} SE={:.4f} MDE={:.4f} | {} | TOST: {} | achieved bound={:.4f}".format(
        learner, s["point_delta"], s["SE_delta"], s["MDE"], s["difference_verdict"],
        s["tost_verdict"], s["achieved_equivalence_bound"]))
print("learner disagreement       :", "; ".join(DISAGREEMENTS) if DISAGREEMENTS else "none")
print("\nwarnings ({}):".format(len(WARNINGS)))
for w in WARNINGS:
    print("  - " + w)

print("\nSEALED FILE : " + SEAL_PATH)
print("self_sha256 : " + self_sha)
print("Commit and tag pre_registration.json in git (or register it on OSF) before week 4 starts "
      "-- a local timestamp proves nothing externally.")
print("\n" + CODE_VERSION + " finished successfully.")
