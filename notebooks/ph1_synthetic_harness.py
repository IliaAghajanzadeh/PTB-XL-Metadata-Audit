# =====================================================================================
# PTB-XL Metadata Audit - ph1 SYNTHETIC TEST HARNESS (separate from the delivered cell; never run on real data)
# Cell under test: ph1_cell.py (CODE_VERSION ph1, CODE_REVISION r1); the notebook copy must be identical (P1).
#
# Replica: a FULL-SIZE synthetic PTB-XL in tempfile.mkdtemp() with the real folder layout, file names, column names and
#   dtypes: 21,799 records (folds 1-8 = 17,418: 2175/2181/2192/2174/2174/2173/2176/2173; fold 9 = 2,183 records /
#   1,942 patients; fold 10 = 2,198 records / 1,904 patients; folds 1-8 = 15,023 patients), the real fold-9 / fold-10
#   label counts, 259 age sentinels in folds 1-9, and device / site / nurse levels that reproduce the FRP locked
#   categories (site 049 only twice in fold 9 and once in fold 10). Labels depend on the metadata (moderate signal).
#   The Part A reference files are produced by running the ACTUAL v0.2.1 and v0.3 cells (the main code cell of each
#   notebook) on the replica, with declared path / hash substitutions only. Locked documents: final_run_protocol.json
#   is the REAL document with the data-dependent values replaced (input hashes, pre_registration parent hashes, v0.3
#   manifest hash, fold-10 id / patient lock hashes, folds 1-9 age scaler; categories, BOOT_IDX hash and every text are
#   the real ones); pre_registration.json is the one the v0.3 cell wrote; v0.9_fold10_labels.csv is written in the v0.9
#   format from an independent implementation of the label rule. The real project is never touched.
# How the cell is run: the delivered source is exec'd UNCHANGED except for an explicit list of substitutions (the
#   project root and the sha256 constants of the replica's files); each must match exactly once; the line diff is
#   printed (P5). Negative tests add declared substitutions / file edits and nothing else.
# Spies: builtins.open, os.rename, os.makedirs, pandas.read_csv, scipy.stats.rankdata (AUROC kernel of fast_auc and REF),
#   sklearn.metrics.average_precision_score, LogisticRegression.fit, HistGradientBoostingClassifier.fit (inherited),
#   StandardScaler.fit; every event gets a sequence number (ordering checks).
# Test ids: T = replica, P = static, M = main run, H = independent recomputation (harness REF + sklearn), X = cross-check
#   with the ACTUAL v0.2.1 / v0.3 cells run on a replica whose strat_fold is shifted (9 -> 8, 10 -> 9), D = crash
#   protocol / determinism, S = STOP paths, N = single-code-path guard.
# Inputs (read-only; paths can be overridden with the PH1_* environment variables): ph1_cell.py and the notebook next to
#   this file, the uploaded final_run_protocol.json copy, the v0.2.1 and v0.3 notebooks, v0.9_cell.py and prompt 1.
# =====================================================================================
import os, sys, io, json, ast, copy, time, shutil, hashlib, tempfile, platform, contextlib, traceback, re, tokenize
import builtins
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

T_ALL = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = "/root/.claude/uploads/7db4f938-afa7-5ec3-9f2e-2ebfd21ed61e"
CELL_PATH = os.environ.get("PH1_CELL", os.path.join(HERE, "ph1_cell.py"))
NOTEBOOK_PATH = os.environ.get("PH1_NOTEBOOK", os.path.join(HERE, "ptbxl_ph1_posthoc_metadata_fold10.ipynb"))
FRP_COPY_PATH = os.environ.get("PH1_FRP_COPY", os.path.join(UPLOADS, "d642f871-final_run_protocol.json"))
PROMPT_PATH = os.environ.get("PH1_PROMPT", os.path.join(UPLOADS, "43ff062d-attachment.txt"))
SOURCE_NOTEBOOKS = {"v0.2.1": os.environ.get("PH1_NB_V021", os.path.join(UPLOADS, "0c03b2a9-ptbxl_death_experiment_v0.2.1.ipynb")),
                    "v0.3": os.environ.get("PH1_NB_V03", os.path.join(UPLOADS, "c9d8e0c5-ptbxl_v0.3_preregistration.ipynb"))}
V09_CELL_PATH = os.environ.get("PH1_V09_CELL", os.path.join(UPLOADS, "6791f18d-v0.9_cell.py"))
TMP = tempfile.mkdtemp(prefix="ph1_harness_")
assert os.path.abspath(TMP).startswith(os.path.abspath(tempfile.gettempdir())) and "ptbxl-metadata-audit" not in TMP
LOGS = os.path.join(TMP, "logs")
os.makedirs(LOGS)
REPORT = []
DATASET_DIR = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ACQ_COLS = ["device", "site", "nurse"]
LEARNERS = ["LogisticRegression", "HistGradientBoosting"]
SEEDS = [42, 123, 2024]
BLOCK_NAMES = ["B0_intercept", "B1_physio", "B2_acquisition", "B3_physio_plus_acquisition", "B4_temporal", "B5_all"]
FOLD_SIZES = {1: 2175, 2: 2181, 3: 2192, 4: 2174, 5: 2174, 6: 2173, 7: 2176, 8: 2173, 9: 2183, 10: 2198}
N_PAT_18, N_PAT_9, N_PAT_10 = 15023, 1942, 1904
POS9 = {"NORM": 955, "MI": 540, "STTC": 528, "CD": 495, "HYP": 268}
POS10 = {"NORM": 963, "MI": 550, "STTC": 521, "CD": 496, "HYP": 262}
N_SENT_19, N_SENT_10 = 259, 34
CLASS_CODES = {"NORM": ["NORM"], "MI": ["IMI", "AMI"], "STTC": ["NDT", "ISC_"], "CD": ["CLBBB", "IRBBB"], "HYP": ["LVH", "RVH"]}
DB_COLUMNS = ["patient_id", "age", "sex", "height", "weight", "nurse", "site", "device", "recording_date", "report",
              "scp_codes", "heart_axis", "infarction_stadium1", "infarction_stadium2", "validated_by", "second_opinion",
              "initial_autogenerated_report", "validated_by_human", "baseline_drift", "static_noise", "burst_noise",
              "electrodes_problems", "extra_beats", "pacemaker", "strat_fold", "filename_lr", "filename_hr"]
POSTHOC_LABEL = "POST-HOC (ph1): not pre-registered; reported separately from pre-registered results"
OUT_FILES = ["ph1_analysis_plan.json", "ph1_bootstrap_deltas.csv", "ph1_decision_table.csv", "ph1_fold10_probs.csv",
             "ph1_frozen_outputs.json", "ph1_manifest.json", "ph1_metric_rows.csv", "ph1_results.json"]
# the real values in the delivered cell (each is substituted by the replica's value; each occurs exactly once)
REAL_CONST = {
    "db": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
    "prereg_self": "5183f54569f3358a5339b16a2c3c3be49f5c7798eabe6b0bb25946119c2ba469",
    "prereg_file": "11dd446272654fc80d65b2fdd40259fa674bbb419a44fde84e589b6e297cde32",
    "frp_self": "834acddec288862d0418e9ee70204898c9d371788eed7eb362f4c0d63d305c6b",
    "frp_file": "e1a5260e72d2e8b0d27dc709639468b7f9a48a7c8e2f8af2d663e105dbd329d4",
    "v021_dt": "20bffb2c6df30e16a070184f91f7211bddc14ede9bc2de39273b5be1cc32359b",
    "v021_man": "37ca9b785ac2c1d78dd3e28f729dc64d289d42ae80803adacbe8a406122b23f6",
    "v03_man": "c7e1bec283d11a3d4cd94a3f2509640651e8069d26424cfd14b5d6b975053e35",
    "v09_labels": "29051bd87676a59b70b3a79a91075f70a7cddd028b2c297461c7bbae3f15b3b0",
}
REAL_REF_SHA = "48ac837c07880f0bd88a7bd01475e7da3edbeab46019f8288976884ad7662856"


# ----------------------------- helpers (copies of the project helpers) -----------------------------
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


def canonical_json_bytes(d):
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default).encode("utf-8")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with REAL_FN["open"](path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_bytes(path, data):
    if os.path.lexists(path):
        os.unlink(path)
    with REAL_FN["open"](path, "wb") as f:
        f.write(data)


def write_json(path, doc):
    if os.path.lexists(path):
        os.unlink(path)
    with REAL_FN["open"](path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def write_self_hashed(path, doc):
    doc = json.loads(json.dumps(doc, ensure_ascii=False, default=_json_default))
    doc.pop("self_sha256", None)
    s = sha256_bytes(canonical_json_bytes(doc))
    doc["self_sha256"] = s
    write_json(path, doc)
    return s, sha256_of(path)


def load_json(path):
    with REAL_FN["open"](path, "r", encoding="utf-8") as f:
        return json.load(f)


def self_sha_ok(path):
    if not os.path.isfile(path):
        return False, {}
    d = load_json(path)
    s = d.pop("self_sha256", None)
    ok = s == sha256_bytes(canonical_json_bytes(d))
    d["self_sha256"] = s
    return ok, d


def csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def to_level_string(series, missing="__MISSING__"):
    if pd.api.types.is_numeric_dtype(series):
        nn_ = series.dropna()
        integral = bool(len(nn_) == 0 or np.all(np.isclose(nn_.values, np.round(nn_.values))))
        if integral:
            width = max(3, len(str(int(nn_.max()))) if len(nn_) else 3)
            return series.apply(lambda v: missing if pd.isna(v) else str(int(v)).zfill(width))
    return series.apply(lambda v: missing if pd.isna(v) else str(v).strip())


def encode_categorical(level_series, train_mask, col_name):
    train_vals = level_series[train_mask]
    train_categories = sorted(set(train_vals.unique().tolist()) | {"__MISSING__"})
    known = set(train_categories)
    remapped = level_series.where(level_series.isin(known), other="__UNKNOWN__")
    categories = train_categories + ["__UNKNOWN__"]
    dummies = pd.get_dummies(pd.Categorical(remapped, categories=categories))
    dummies.columns = ["{}={}".format(col_name, c) for c in dummies.columns]
    dummies.index = level_series.index
    return dummies, categories


def harness_labels(db, scp):
    """Independent implementation of the v0.2.1 label rule (every scp_codes key mapped via diagnostic == 1)."""
    diag = scp[scp["diagnostic"].fillna(0) == 1]
    c2c = {k: v for k, v in diag["diagnostic_class"].items() if isinstance(v, str)}
    lab = np.zeros((len(db), 5), dtype=np.int64)
    for i, s in enumerate(db["scp_codes"].tolist()):
        cls = {c2c[k] for k in ast.literal_eval(s) if k in c2c}
        lab[i] = [int(c in cls) for c in SUPERCLASSES]
    return pd.DataFrame(lab, index=db.index, columns=SUPERCLASSES)


def record(test, ok, detail):
    REPORT.append((test, "PASS" if ok else "FAIL", detail))
    print("[{}] {} - {}".format("PASS" if ok else "FAIL", test, detail), flush=True)


def harness_exception(group):
    """A test group that raised (in the harness or because the cell misbehaved) is a FAIL; the other groups still run."""
    record("HARNESS EXCEPTION in group '{}'".format(group), False, traceback.format_exc()[-1500:].replace("\n", " | "))


def short(e, n=160):
    return None if e is None else (e if len(e) <= n else e[:n] + "...")


def main_code_cell(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return max(codes, key=len)


# ----------------------------- inputs of the harness -----------------------------
REAL_FN = {"open": builtins.open, "rename": os.rename, "makedirs": os.makedirs, "read_csv": pd.read_csv,
           "rankdata": scipy.stats.rankdata, "ap": sklearn.metrics.average_precision_score,
           "lr_fit": LogisticRegression.fit, "hgb_fit": HistGradientBoostingClassifier.fit, "sc_fit": StandardScaler.fit}
CELL_SRC = open(CELL_PATH, encoding="utf-8").read()
REAL = json.load(open(FRP_COPY_PATH, encoding="utf-8"))
assert sha256_of(FRP_COPY_PATH) == REAL_CONST["frp_file"], "the FRP copy is not the locked document"
REF_CODE = REAL["reference_implementation"]["code"]
assert sha256_bytes(REF_CODE.encode("utf-8")) == REAL_REF_SHA
HREF = {}
exec(REF_CODE, HREF)
DEVICES = REAL["locked_quantities"]["categorical_encoder"]["categories"]["device"][:-2]       # the 11 real strings
SRC_V021 = main_code_cell(SOURCE_NOTEBOOKS["v0.2.1"])
SRC_V03 = main_code_cell(SOURCE_NOTEBOOKS["v0.3"])
SRC_V09 = open(V09_CELL_PATH, encoding="utf-8").read()
assert sha256_bytes(SRC_V09.encode("utf-8")) == "9af1773dba86039cfa2b191f2aaf0e54c02f9ccb90880b60471074e2c6fe4c7b"
PROMPT_TEXT = open(PROMPT_PATH, encoding="utf-8").read()


# ----------------------------- the replica: ptbxl_database.csv + scp_statements.csv -----------------------------
def split_counts(total, sizes):
    raw = np.array(sizes, dtype=float) * total / float(sum(sizes))
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[: total - out.sum()]:
        out[i] += 1
    assert out.sum() == total
    return out


def build_database(data_root, seed=20261005):
    rng = np.random.default_rng(seed)
    os.makedirs(data_root, exist_ok=True)
    folds = list(range(1, 11))
    n_pat = dict(zip(range(1, 9), split_counts(N_PAT_18, [FOLD_SIZES[f] for f in range(1, 9)])))
    n_pat[9], n_pat[10] = N_PAT_9, N_PAT_10
    rec_fold, rec_pat = [], []
    for f in folds:                                              # patients have 1..k records (heavy tail)
        n_r, n_p = FOLD_SIZES[f], n_pat[f]
        w = rng.gamma(0.35, 1.0, n_p)
        extra = rng.choice(n_p, size=n_r - n_p, p=w / w.sum())
        mult = np.ones(n_p, dtype=int) + np.bincount(extra, minlength=n_p)
        for p_i, m in enumerate(mult):
            rec_fold += [f] * int(m)
            rec_pat += [p_i] * int(m)
    rec_fold, rec_pat = np.array(rec_fold), np.array(rec_pat)
    N = len(rec_fold)
    assert N == 21799
    order = rng.permutation(N)                                   # file order: folds interleaved, patients scattered
    rec_fold, rec_pat = rec_fold[order], rec_pat[order]
    ecg_id = np.sort(rng.choice(np.arange(1, 21838), size=N, replace=False))
    pid_pool = rng.choice(np.arange(1, 21798), size=sum(n_pat.values()), replace=False)
    pid_of, k = {}, 0
    for f in folds:
        for p_i in range(n_pat[f]):
            pid_of[(f, p_i)] = int(pid_pool[k])
            k += 1
    patient_id = np.array([pid_of[(int(f), int(p))] for f, p in zip(rec_fold, rec_pat)], dtype=np.int64)
    pat_keys = sorted(pid_of.values())
    pat_sex = dict(zip(pat_keys, rng.integers(0, 2, len(pat_keys))))
    pat_age = dict(zip(pat_keys, np.clip(rng.normal(61.0, 17.0, len(pat_keys)), 2, 89)))
    sex = np.array([pat_sex[p] for p in patient_id], dtype=np.int64)
    true_age = np.clip(np.round(np.array([pat_age[p] for p in patient_id]) + rng.integers(0, 3, N)), 0, 89)
    age = true_age.astype(float)
    idx19, idx10 = np.where(rec_fold <= 9)[0], np.where(rec_fold == 10)[0]
    sent = np.concatenate([rng.choice(idx19, N_SENT_19, replace=False), rng.choice(idx10, N_SENT_10, replace=False)])
    age[sent] = 300.0                                            # ages above 89 (v1.0.2 sentinel)
    true_age[sent] = 93.0
    dev_w = np.array([3, 1, 1, 2, 2, 1, 1, 2, 6, 5, 9], dtype=float)
    device = np.array(DEVICES, dtype=object)[rng.choice(len(DEVICES), size=N, p=dev_w / dev_w.sum())]
    levels_site = [s for s in range(51) if s != 49]
    site_w = rng.gamma(1.2, 1.0, len(levels_site))
    site = np.array(levels_site, dtype=float)[rng.choice(len(levels_site), size=N, p=site_w / site_w.sum())]
    idx18 = np.where(rec_fold <= 8)[0]
    for j, s in enumerate(levels_site):                          # every level present in folds 1-8
        site[idx18[j * 7:(j * 7) + 3]] = s
    site[rng.choice(np.where(rec_fold == 9)[0], 2, replace=False)] = 49.0
    site[rng.choice(idx10, 1, replace=False)] = 49.0
    site[rng.random(N) < 0.012] = np.nan
    for f, r in ((1, 3), (5, 2), (9, 1), (10, 1)):
        site[rng.choice(np.where(rec_fold == f)[0], r, replace=False)] = np.nan
    nw = np.r_[np.full(6, 0.11), np.full(6, 0.0567)]
    nurse = rng.choice(12, size=N, p=nw / nw.sum()).astype(float)
    nurse[rng.random(N) < 0.06] = np.nan
    year = np.clip(np.round(1984 + 17 * rng.beta(2.2, 2.0, N) + (np.isin(device, DEVICES[8:]) * 2.5)), 1984, 2001).astype(int)
    rec_date = ["{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(y, int(m), int(d), int(h), int(mi), int(s))
                for y, m, d, h, mi, s in zip(year, rng.integers(1, 13, N), rng.integers(1, 29, N), rng.integers(7, 19, N),
                                             rng.integers(0, 60, N), rng.integers(0, 60, N))]
    z = (true_age - 60.0) / 17.0                                 # labels from a metadata model (moderate signal)
    di = np.array([DEVICES.index(d) for d in device])
    si = np.where(np.isnan(site), 51, np.nan_to_num(site, nan=51)).astype(int)
    ni = np.where(np.isnan(nurse), 12, np.nan_to_num(nurse, nan=12)).astype(int)
    target_prev = {c: (POS9[c] + POS10[c]) / float(FOLD_SIZES[9] + FOLD_SIZES[10]) for c in SUPERCLASSES}
    slopes = {"NORM": (-1.25, -0.15), "MI": (0.65, 0.45), "STTC": (0.75, -0.1), "CD": (0.5, 0.3), "HYP": (0.55, 0.4)}
    L = np.zeros((N, 5), dtype=bool)
    for k_c, c in enumerate(SUPERCLASSES):
        ba, bs = slopes[c]
        de, se, ne = rng.normal(0, 0.35, len(DEVICES)), rng.normal(0, 0.3, 52), rng.normal(0, 0.15, 13)
        lin = ba * z + bs * sex + de[di] + se[si] + ne[ni] + 0.025 * (year - 1992) * (1 if k_c % 2 else -1)
        lo, hi = -8.0, 8.0                                       # intercept calibrated to the fold-9/10 prevalence
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.mean(1.0 / (1.0 + np.exp(-(mid + lin)))) < target_prev[c] * (1.25 if c != "NORM" else 1.0) else (lo, mid)
        L[:, k_c] = rng.random(N) < 1.0 / (1.0 + np.exp(-(0.5 * (lo + hi) + lin)))
    L[L[:, 0] & (rng.random(N) < 0.9), 1:] = False               # NORM mostly exclusive
    scp = []
    for i in range(N):
        d = {}
        for k_c, c in enumerate(SUPERCLASSES):
            if L[i, k_c]:
                codes = CLASS_CODES[c]
                pick = codes if (len(codes) > 1 and rng.random() < 0.12) else [codes[int(rng.integers(len(codes)))]]
                for code in pick:                                # likelihood 0.0 included (no likelihood threshold)
                    d[code] = float(rng.choice([0.0, 15.0, 35.0, 50.0, 80.0, 100.0], p=[0.06, 0.06, 0.08, 0.2, 0.2, 0.4]))
        if rng.random() < 0.85:
            d["SR"] = 0.0
        if rng.random() < 0.07:
            d["PVC"] = 100.0
        if rng.random() < 0.05:
            d["AFIB"] = 0.0
        scp.append(d)

    def classes_of(dct):
        return {c for c in SUPERCLASSES if any(code in dct for code in CLASS_CODES[c])}

    for f, target in ((9, POS9), (10, POS10)):                   # exact real label counts in folds 9 and 10
        rows = np.where(rec_fold == f)[0]
        for c in SUPERCLASSES:
            have = [r for r in rows if c in classes_of(scp[r])]
            lack = [r for r in rows if c not in classes_of(scp[r])]
            diff = len(have) - target[c]
            if diff > 0:
                for r in rng.choice(have, diff, replace=False):
                    for code in CLASS_CODES[c]:
                        scp[r].pop(code, None)
            elif diff < 0:
                for r in rng.choice(lack, -diff, replace=False):
                    scp[r][CLASS_CODES[c][0]] = 100.0
        assert {c: sum(c in classes_of(scp[r]) for r in rows) for c in SUPERCLASSES} == target
    choice = lambda vals, p, n: np.array(vals, dtype=object)[rng.choice(len(vals), size=n, p=p)]
    na = np.nan
    db = pd.DataFrame({
        "ecg_id": ecg_id, "patient_id": patient_id.astype(float), "age": age, "sex": sex,
        "height": np.where(rng.random(N) < 0.32, np.round(rng.normal(168, 9, N)), na),
        "weight": np.where(rng.random(N) < 0.43, np.round(rng.normal(77, 14, N)), na),
        "nurse": nurse, "site": site, "device": device, "recording_date": rec_date,
        "report": ["synthetic report {} sinusrhythmus".format(i) for i in range(N)],
        "scp_codes": [str(d) for d in scp],
        "heart_axis": choice([na, "MID", "LAD", "RAD", "ALAD"], [0.39, 0.33, 0.2, 0.03, 0.05], N),
        "infarction_stadium1": choice([na, "Stadium I", "Stadium II", "unknown"], [0.74, 0.1, 0.1, 0.06], N),
        "infarction_stadium2": choice([na, "Stadium III"], [0.995, 0.005], N),
        "validated_by": np.where(rng.random(N) < 0.57, rng.integers(0, 12, N).astype(float), na),
        "second_opinion": rng.random(N) < 0.002, "initial_autogenerated_report": rng.random(N) < 0.08,
        "validated_by_human": rng.random(N) < 0.73,
        "baseline_drift": choice([na, ", I-AVR,"], [0.93, 0.07], N), "static_noise": choice([na, " , I-V1,"], [0.85, 0.15], N),
        "burst_noise": choice([na, "alles"], [0.97, 0.03], N), "electrodes_problems": choice([na, "V6"], [0.999, 0.001], N),
        "extra_beats": choice([na, "1ES"], [0.91, 0.09], N), "pacemaker": choice([na, "ja, pacemaker"], [0.987, 0.013], N),
        "strat_fold": rec_fold.astype(np.int64),
        "filename_lr": ["records100/{:05d}/{:05d}_lr".format((e // 1000) * 1000, e) for e in ecg_id],
        "filename_hr": ["records500/{:05d}/{:05d}_hr".format((e // 1000) * 1000, e) for e in ecg_id],
    }).set_index("ecg_id")
    assert list(db.columns) == DB_COLUMNS
    db.to_csv(os.path.join(data_root, "ptbxl_database.csv"))
    rows = [(code, "{} statement".format(code), 1.0, na, na, c, code) for c in SUPERCLASSES for code in CLASS_CODES[c]]
    rows += [("SR", "sinus rhythm", na, na, 1.0, na, na), ("AFIB", "atrial fibrillation", na, na, 1.0, na, na),
             ("PVC", "ventricular premature complex", na, 1.0, na, na, na)]
    scp_df = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows],
                          columns=["description", "diagnostic", "form", "rhythm", "diagnostic_class", "diagnostic_subclass"])
    scp_df["Statement Category"] = "synthetic"
    scp_df["SCP-ECG Statement Description"] = scp_df["description"]
    for col in ("AHA code", "aECG REFID", "CDISC Code", "DICOM Code"):
        scp_df[col] = na
    scp_df.to_csv(os.path.join(data_root, "scp_statements.csv"))


# ----------------------------- running the ACTUAL earlier cells (declared substitutions only) -----------------------------
def apply_subs(src, subs):
    for old, new in subs:
        n = src.count(old)
        assert n == 1, "substitution matched {} times (must be exactly 1): {!r}".format(n, old[:100])
        src = src.replace(old, new)
    return src


def run_source(src, label):
    ns = {"__name__": "__main__"}
    buf = io.StringIO()
    t0 = time.time()
    exc = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, label, "exec"), ns)
    except BaseException as e:                                   # SystemExit included
        exc = "{}: {}\n{}".format(type(e).__name__, e, traceback.format_exc())
    with REAL_FN["open"](os.path.join(LOGS, label + ".log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue() + ("\n\nEXCEPTION: " + exc if exc else ""))
    if exc:
        raise RuntimeError("{} failed: {}".format(label, exc[-2000:]))
    print("    ran {:34s} {:6.1f}s".format(label, time.time() - t0), flush=True)
    return ns


V021_DATA_ROOT_LINE = 'DATA_ROOT = r"D:\\ptbxl-metadata-audit\\data\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"'


def run_v021(root, label):
    subs = [(V021_DATA_ROOT_LINE, 'DATA_ROOT = r"{}"'.format(os.path.join(root, "data", DATASET_DIR))),
            ('OUT_DIR   = r"D:\\ptbxl-metadata-audit\\outputs\\v0.2.1"',
             'OUT_DIR   = r"{}"'.format(os.path.join(root, "outputs", "v0.2.1")))]
    return run_source(apply_subs(SRC_V021, subs), label)


def run_v03(root, label):
    data_root = os.path.join(root, "data", DATASET_DIR)
    m = pd.read_csv(os.path.join(root, "outputs", "v0.2.1", "v0.2.1_metrics_full.csv"), encoding="utf-8-sig",
                    float_precision="round_trip")
    mm = m[(m["class"] == "MACRO") & (m["seed"] == 42)]
    pt, ci = {}, {}
    for l in LEARNERS:
        b1 = mm[(mm["block"] == "B1_physio") & (mm["learner"] == l)].iloc[0]
        b3 = mm[(mm["block"] == "B3_physio_plus_acquisition") & (mm["learner"] == l)].iloc[0]
        pt[l] = (round(float(b1["auroc"]), 4), round(float(b3["auroc"]), 4), round(float(b3["auroc"]) - float(b1["auroc"]), 4))
        ci[l] = (round(float(b1["ci_low"]), 4), round(float(b1["ci_high"]), 4), round(float(b3["ci_low"]), 4),
                 round(float(b3["ci_high"]), 4))
    seed_old = SRC_V03[SRC_V03.index("V021_SEED42 = {"):SRC_V03.index("# v0.2.1 seed-42 marginal 95% bootstrap CIs")]
    seed_new = "V021_SEED42 = {\n" + "".join(
        '    "{}": {{"B1_physio": {:.4f}, "B3_physio_plus_acquisition": {:.4f}, "delta": {:+.4f}}},\n'.format(l, *pt[l])
        for l in LEARNERS) + "}\n"
    ci_old = SRC_V03[SRC_V03.index("V021_SEED42_CI = {"):SRC_V03.index("# v0.2.1 context-only constants")]
    ci_new = "V021_SEED42_CI = {\n" + "".join(
        '    "{}": {{"B1_physio": ({:.4f}, {:.4f}), "B3_physio_plus_acquisition": ({:.4f}, {:.4f})}},\n'.format(l, *ci[l])
        for l in LEARNERS) + "}\n"
    subs = [(V021_DATA_ROOT_LINE, 'DATA_ROOT = r"{}"'.format(data_root)),
            ('OUT_DIR   = r"D:\\ptbxl-metadata-audit\\outputs\\v0.3"', 'OUT_DIR   = r"{}"'.format(os.path.join(root, "outputs", "v0.3"))),
            ('SEAL_DIR  = r"D:\\ptbxl-metadata-audit\\preregistration"',
             'SEAL_DIR  = r"{}"'.format(os.path.join(root, "preregistration"))),
            ('"ptbxl_database.csv": "{}"'.format(REAL_CONST["db"]),
             '"ptbxl_database.csv": "{}"'.format(sha256_of(os.path.join(data_root, "ptbxl_database.csv")))),
            ('"scp_statements.csv": "{}"'.format(REAL_CONST["scp"]),
             '"scp_statements.csv": "{}"'.format(sha256_of(os.path.join(data_root, "scp_statements.csv")))),
            (seed_old, seed_new), (ci_old, ci_new)]
    return run_source(apply_subs(SRC_V03, subs), label)


# ----------------------------- locked documents and the v0.9 labels of the replica -----------------------------
def frp_rewrite(root, mutate):
    p = os.path.join(root, "preregistration", "final_run_protocol.json")
    frp = load_json(p)
    frp.pop("self_sha256", None)
    mutate(frp)
    return write_self_hashed(p, frp)


def manifest_set(path, basename, new_sha):
    man = load_json(path)
    hits = [o for o in man["outputs"] if os.path.basename(o["file"].replace("\\", "/")) == basename]
    assert len(hits) == 1, basename
    hits[0]["sha256"] = new_sha
    write_json(path, man)
    return sha256_of(path)


def make_docs(root):
    data_root = os.path.join(root, "data", DATASET_DIR)
    prereg = os.path.join(root, "preregistration")
    db = pd.read_csv(os.path.join(data_root, "ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(os.path.join(data_root, "scp_statements.csv"), index_col=0)
    df9, f10 = db[db["strat_fold"] <= 9], db[db["strat_fold"] == 10]
    m19, m18 = pd.Series(True, index=df9.index), df9["strat_fold"] <= 8
    lq_enc = REAL["locked_quantities"]["categorical_encoder"]
    cats19, cats18, enc19 = {}, {}, {}
    for col in ACQ_COLS:
        lv = to_level_string(df9[col])
        enc19[col], cats19[col] = encode_categorical(lv, m19, col)
        _, cats18[col] = encode_categorical(lv, m18, col)
    allzero = sorted(c for col in ACQ_COLS for c in enc19[col].columns if int(enc19[col][c].sum()) == 0)
    assert cats19 == lq_enc["categories"] and allzero == sorted(lq_enc["allzero_training_columns_folds1to9"]), \
        "replica categories differ from the locked FRP categories"
    assert {c: len(cats18[c]) for c in ACQ_COLS} == lq_enc["widths_folds1to8_development"]
    sent = df9["age"] == 300
    age_imp = df9["age"].where(~sent, np.nan).fillna(90.0)
    sc19 = StandardScaler().fit(age_imp.to_numpy().reshape(-1, 1))
    pid10 = f10["patient_id"].to_numpy().astype(np.int64)
    ids = np.ascontiguousarray(np.stack([f10.index.to_numpy().astype(np.int64), pid10], axis=1).astype("<i8"))
    pats, _ = HREF["patient_rows_in_file_order"](pid10)
    boot = HREF["build_boot_idx"](len(pats), 2000, 7)
    assert sha256_bytes(np.ascontiguousarray(boot.astype("<i8")).tobytes()) == REAL["bootstrap"]["locked_hashes"]["boot_idx_sha256"]
    # pre_registration.json was written by the v0.3 cell; placeholders for the other three documents; .ots bytes
    seal_path = os.path.join(prereg, "pre_registration.json")
    ok, seal = self_sha_ok(seal_path)
    assert ok
    seal_self, seal_file = seal["self_sha256"], sha256_of(seal_path)
    others = {}
    for key, name in (("dev_protocol", "dev_protocol.json"), ("training_details", "v0.5_training_details.json"),
                      ("lr_selection", "lr_selection.json")):
        others[key] = write_self_hashed(os.path.join(prereg, name), {"document": "synthetic placeholder ({})".format(key)})
    g = np.random.default_rng(54321)
    for name, size in (("pre_registration.json", 516), ("dev_protocol.json", 516), ("v0.5_training_details.json", 446),
                       ("lr_selection.json", 481), ("final_run_protocol.json", 656)):
        write_bytes(os.path.join(prereg, name + ".ots"), g.bytes(size))
    inputs = {nm: sha256_of(os.path.join(data_root, nm)) for nm in ("ptbxl_database.csv", "scp_statements.csv")}
    v03_man_sha = sha256_of(os.path.join(root, "outputs", "v0.3", "v0.3_manifest.json"))
    frp = copy.deepcopy(REAL)
    frp.pop("self_sha256")
    frp["input_files_sha256"] = inputs
    frp["parent_hashes"] = {"pre_registration": {"self_sha256": seal_self, "file_sha256": seal_file},
                            **{k: {"self_sha256": v[0], "file_sha256": v[1]} for k, v in others.items()}}
    frp["earlier_manifests_sha256"]["v0.3_manifest.json"] = v03_man_sha
    frp["bootstrap"]["locked_hashes"]["fold10_ids_file_order_sha256"] = sha256_bytes(ids.tobytes(order="C"))
    frp["bootstrap"]["locked_hashes"]["fold10_patients_sorted_sha256"] = sha256_bytes(np.ascontiguousarray(pats.astype("<i8")).tobytes())
    frp["locked_quantities"]["age_scaler"].update({"mean": float(sc19.mean_[0]), "scale": float(sc19.scale_[0]),
                                                   "var": float(sc19.var_[0]), "n_samples_seen": int(sc19.n_samples_seen_)})
    write_self_hashed(os.path.join(prereg, "final_run_protocol.json"), frp)
    # outputs\v0.9: the labels file (v0.9 format) + decoys that must never be parsed; outputs\v0.8: a decoy
    v09 = os.path.join(root, "outputs", "v0.9")
    os.makedirs(v09, exist_ok=True)
    lab10 = harness_labels(f10, scp)
    assert {c: int(lab10[c].sum()) for c in SUPERCLASSES} == POS10
    labels = pd.DataFrame({"ecg_id": ids[:, 0], "patient_id": ids[:, 1], **{c: lab10[c].to_numpy() for c in SUPERCLASSES}})
    write_bytes(os.path.join(v09, "v0.9_fold10_labels.csv"), csv_bytes(labels))
    for name in ("v0.9_results.json", "v0.9_manifest.json", "v0.9_frozen_outputs.json"):
        write_bytes(os.path.join(v09, name), b'{"DECOY": "must never be parsed by ph1"}\n')
    for name in ("v0.9_bootstrap_deltas.csv", "v0.9_resnet1d_wang_signal_plus_all_seed42_lr0.01_fold10_probs.csv"):
        write_bytes(os.path.join(v09, name), b"DECOY,must never be parsed by ph1\n")
    write_bytes(os.path.join(v09, "v0.9_attempt_log.jsonl"), b'{"event": "DECOY"}\n')
    os.makedirs(os.path.join(root, "outputs", "v0.8"), exist_ok=True)
    write_bytes(os.path.join(root, "outputs", "v0.8", "v0.8_manifest.json"), b'{"DECOY": "never opened by ph1"}\n')
    return {"labels10": labels}


def facts(root):
    """sha256 values of the replica's files that the cell holds as constants (computed from disk)."""
    P = lambda *a: os.path.join(root, *a)
    ok_s, seal = self_sha_ok(P("preregistration", "pre_registration.json"))
    ok_f, frp = self_sha_ok(P("preregistration", "final_run_protocol.json"))
    return {"root": root,
            "db": sha256_of(P("data", DATASET_DIR, "ptbxl_database.csv")),
            "scp": sha256_of(P("data", DATASET_DIR, "scp_statements.csv")),
            "prereg_self": seal.get("self_sha256"), "prereg_file": sha256_of(P("preregistration", "pre_registration.json")),
            "frp_self": frp.get("self_sha256"), "frp_file": sha256_of(P("preregistration", "final_run_protocol.json")),
            "v021_dt": sha256_of(P("outputs", "v0.2.1", "v0.2.1_decision_table.csv")),
            "v021_man": sha256_of(P("outputs", "v0.2.1", "v0.2.1_manifest.json")),
            "v03_man": sha256_of(P("outputs", "v0.3", "v0.3_manifest.json")),
            "v09_labels": sha256_of(P("outputs", "v0.9", "v0.9_fold10_labels.csv"))}


def substitutions(F, extra=()):
    subs = [('PROJECT_ROOT = r"D:\\ptbxl-metadata-audit"', 'PROJECT_ROOT = r"{}"'.format(F["root"]))]
    for k, real in REAL_CONST.items():
        subs.append(('"{}"'.format(real), '"{}"'.format(F[k])))
    return subs + list(extra)


def src_for(F, extra=()):
    return apply_subs(CELL_SRC, substitutions(F, extra))


def make_template(root):
    t0 = time.time()
    for d in ("preregistration", os.path.join("outputs", "v0.2.1"), os.path.join("outputs", "v0.3")):
        os.makedirs(os.path.join(root, d))
    build_database(os.path.join(root, "data", DATASET_DIR))
    run_v021(root, "template_v0.2.1")
    run_v03(root, "template_v0.3")
    extra = make_docs(root)
    F = facts(root)
    F.update(extra)
    F["seconds"] = time.time() - t0
    return F


def clone(src, dst):
    shutil.copytree(src, dst)
    return dst


# ----------------------------- spies and the runner -----------------------------
CUR = None
SEQ = [0]


def nseq():
    SEQ[0] += 1
    return SEQ[0]


def _under(path, folder):
    p, f = os.path.abspath(path), os.path.abspath(folder)
    return p == f or p.startswith(f + os.sep)


def spy_open(file, *a, **k):
    c = CUR
    if c is not None and isinstance(file, (str, os.PathLike)):
        p = os.fspath(file)
        mode = a[0] if a else k.get("mode", "r")
        c["opens"].append({"path": p, "mode": mode, "seq": nseq()})
    return REAL_FN["open"](file, *a, **k)


def spy_rename(src, dst, *a, **k):
    c = CUR
    if c is not None:
        s = nseq()
        b = os.path.basename(os.fspath(dst))
        c["renames"].append({"src": os.fspath(src), "dst": os.fspath(dst), "seq": s})
        out = REAL_FN["rename"](src, dst, *a, **k)
        inj = c["inject"].get("modify_protected_at")
        if inj and inj[0] == b:                                  # a locked / read-only file changes DURING the run
            with REAL_FN["open"](os.path.join(c["root"], *inj[1]), "ab") as f:
                f.write(b"\n")
        if c["inject"].get("crash_at_rename") == b:
            raise RuntimeError("INJECTED CRASH right after writing " + b)
        return out
    return REAL_FN["rename"](src, dst, *a, **k)


def spy_makedirs(name, *a, **k):
    if CUR is not None:
        CUR["makedirs"].append({"path": os.fspath(name), "seq": nseq()})
    return REAL_FN["makedirs"](name, *a, **k)


def spy_read_csv(fp, *a, **k):
    c = CUR
    if c is not None:
        name = os.path.basename(os.fspath(fp)) if isinstance(fp, (str, os.PathLike)) else repr(fp)
        c["read_csv"].append({"file": name, "seq": nseq(), "float_precision": k.get("float_precision")})
        if c["inject"].get("crash_on_read") == name:
            raise RuntimeError("INJECTED CRASH while reading " + name)
    return REAL_FN["read_csv"](fp, *a, **k)


def spy_rankdata(*a, **k):
    if CUR is not None:
        CUR["rank_seq"].append(nseq())
    return REAL_FN["rankdata"](*a, **k)


def spy_ap(*a, **k):
    if CUR is not None:
        CUR["ap_seq"].append(nseq())
    return REAL_FN["ap"](*a, **k)


def spy_lr_fit(self, X, y, *a, **k):
    if CUR is not None:
        CUR["fits"].append({"kind": "LR", "rows": int(np.asarray(X).shape[0]), "cols": int(np.asarray(X).shape[1]),
                            "seq": nseq(), "id": id(self)})
    return REAL_FN["lr_fit"](self, X, y, *a, **k)


def spy_hgb_fit(self, X, y, *a, **k):
    if CUR is not None:
        CUR["fits"].append({"kind": "HGB", "rows": int(np.asarray(X).shape[0]), "cols": int(np.asarray(X).shape[1]),
                            "seq": nseq(), "id": id(self)})
    return REAL_FN["hgb_fit"](self, X, y, *a, **k)


def spy_sc_fit(self, X, *a, **k):
    if CUR is not None:
        CUR["fits"].append({"kind": "SC", "rows": int(np.asarray(X).shape[0]), "cols": int(np.asarray(X).shape[1]),
                            "seq": nseq(), "id": id(self)})
    return REAL_FN["sc_fit"](self, X, *a, **k)


def protected_snapshot(root):
    out = {}
    for sub in ("preregistration", os.path.join("outputs", "v0.2.1"), os.path.join("outputs", "v0.3"),
                os.path.join("outputs", "v0.9"), os.path.join("outputs", "v0.8"), os.path.join("data", DATASET_DIR)):
        d = os.path.join(root, sub)
        for dirpath, _, files in os.walk(d):
            for nm in files:
                out[os.path.relpath(os.path.join(dirpath, nm), root)] = sha256_of(os.path.join(dirpath, nm))
    for nm in os.listdir(os.path.join(root, "outputs")):
        if nm.startswith("ph1_failed_"):
            for dirpath, _, files in os.walk(os.path.join(root, "outputs", nm)):
                for f in files:
                    out[os.path.relpath(os.path.join(dirpath, f), root)] = sha256_of(os.path.join(dirpath, f))
    return out


def out_state(root):
    d = os.path.join(root, "outputs", "ph1")
    return None if not os.path.isdir(d) else sorted(os.listdir(d))


def run_cell(src, label, root, inject=None):
    global CUR
    c = {"label": label, "root": root, "inject": inject or {}, "opens": [], "renames": [], "makedirs": [], "read_csv": [],
         "rank_seq": [], "ap_seq": [], "fits": [], "out_dir": os.path.join(root, "outputs", "ph1")}
    snap0 = protected_snapshot(root)
    ns = {"__name__": "__main__"}
    buf = io.StringIO()
    exc = tb = None
    t0 = time.time()
    CUR = c
    builtins.open, os.rename, os.makedirs, pd.read_csv = spy_open, spy_rename, spy_makedirs, spy_read_csv
    scipy.stats.rankdata, sklearn.metrics.average_precision_score = spy_rankdata, spy_ap
    LogisticRegression.fit, HistGradientBoostingClassifier.fit, StandardScaler.fit = spy_lr_fit, spy_hgb_fit, spy_sc_fit
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "ph1_cell", "exec"), ns)
    except Exception as e:
        exc = "{}: {}".format(type(e).__name__, e)
        tb = traceback.format_exc()
    finally:
        builtins.open, os.rename, os.makedirs, pd.read_csv = REAL_FN["open"], REAL_FN["rename"], REAL_FN["makedirs"], REAL_FN["read_csv"]
        scipy.stats.rankdata, sklearn.metrics.average_precision_score = REAL_FN["rankdata"], REAL_FN["ap"]
        LogisticRegression.fit, StandardScaler.fit = REAL_FN["lr_fit"], REAL_FN["sc_fit"]
        del HistGradientBoostingClassifier.fit                  # back to the inherited BaseHistGradientBoosting.fit
        CUR = None
    assert HistGradientBoostingClassifier.fit is REAL_FN["hgb_fit"]
    c.update({"ok": exc is None, "exc": exc, "tb": tb, "out": buf.getvalue(), "seconds": time.time() - t0, "ns": ns,
              "protected_unchanged": protected_snapshot(root) == snap0, "out_files": out_state(root)})
    with REAL_FN["open"](os.path.join(LOGS, label + ".log"), "w", encoding="utf-8") as f:
        f.write(c["out"] + ("\n\nEXCEPTION: {}\n{}".format(exc, tb) if exc else ""))
    print("    run {:34s} {:6.1f}s  {}".format(label, c["seconds"], "ok" if exc is None else "STOPPED: " + short(exc, 120)),
          flush=True)
    return c


def writes_ok(c):
    """Every write: '<OUT_DIR>/<name>.partial' opened 'x'/'xb', then renamed to '<name>' inside OUT_DIR."""
    bad = [o for o in c["opens"] if any(m in o["mode"] for m in "wxa+") and
           not (_under(o["path"], c["out_dir"]) and o["path"].endswith(".partial") and "x" in o["mode"])]
    bad += [r for r in c["renames"] if not (_under(r["src"], c["out_dir"]) and r["src"] == r["dst"] + ".partial")]
    bad += [m for m in c["makedirs"] if os.path.abspath(m["path"]) != os.path.abspath(c["out_dir"])]
    return bad


def fit_summary(c):
    s = {}
    for f in c["fits"]:
        s.setdefault((f["kind"], f["rows"]), 0)
        s[(f["kind"], f["rows"])] += 1
    return s


def rename_seq(c, name):
    hits = [r["seq"] for r in c["renames"] if os.path.basename(r["dst"]) == name]
    return hits[0] if hits else None


def read_seq(c, name):
    hits = [r["seq"] for r in c["read_csv"] if r["file"] == name]
    return hits[0] if hits else None


def load_out(root, name):
    return load_json(os.path.join(root, "outputs", "ph1", name))


def read_out_csv(root, name):
    return pd.read_csv(os.path.join(root, "outputs", "ph1", name), encoding="utf-8-sig", float_precision="round_trip")


def frame_maxdiff(a, b, id_cols, num_cols):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, float("inf")
    if not all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in id_cols):
        return False, float("inf")
    m = 0.0
    for c in num_cols:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return False, float("inf")
        if len(x):
            m = max(m, float(np.nanmax(np.abs(np.where(np.isnan(x), 0.0, x - y)))))
    return True, m


# ----------------------------- provenance checker (independent of the build tools) -----------------------------
PROV_OPEN_RE = re.compile(r"^# >>> (verbatim|adapted) from (v0\.\d(?:\.\d)?)\b(.*)$")
PROV_SOURCES = {"v0.2.1": SRC_V021, "v0.3": SRC_V03, "v0.9": SRC_V09}
PROV_SOURCE_NAMES = {"v0.2.1": "ptbxl_death_experiment_v0.2.1.ipynb, main code cell (cell 2 of 2)",
                     "v0.3": "ptbxl_v0.3_preregistration.ipynb, main code cell (cell 2 of 2)",
                     "v0.9": "v0.9_cell.py (sha256 9af1773d...)"}
IDENT = "A-Za-z0-9_"


def provenance_check(cell_text):
    def parse(rest):
        m = re.match(r"^ \(([^)]*)\)(.*)$", rest)
        assert m, "marker without (label): " + rest
        label, spec = m.group(1), m.group(2).strip()
        renames, literals, indent = [], [], 0
        if spec:
            assert spec.startswith("| ")
            for seg in [x.strip() for x in spec[2:].split(" | ") if x.strip()]:
                if seg.startswith("renames:"):
                    renames += [tuple(t.strip() for t in pair.split("->")) for pair in seg[len("renames:"):].split(",")]
                elif seg.startswith("literal:"):
                    literals += [tuple(t.strip() for t in pair.split(" => ")) for pair in seg[len("literal:"):].split(" ;; ")]
                elif seg.startswith("indent:"):
                    indent = int(seg[len("indent:"):])
                else:
                    raise AssertionError("unknown spec segment " + seg)
        return label, renames, literals, indent

    def excerpt(text, src):
        if not text.endswith("\n"):
            return None
        start = 0
        while True:
            idx = src.find(text, start)
            if idx < 0:
                return None
            if idx == 0 or src[idx - 1] == "\n":
                first = src.count("\n", 0, idx) + 1
                return first, first + text.count("\n") - 1
            start = idx + 1

    lines = cell_text.splitlines(keepends=True)
    res, i = [], 0
    while i < len(lines):
        m = PROV_OPEN_RE.match(lines[i].rstrip("\n"))
        if not m:
            assert not lines[i].startswith("# <<< verbatim") and not lines[i].startswith("# <<< adapted"), i + 1
            i += 1
            continue
        kind, ver, rest = m.groups()
        close = "# <<< {} from {}".format(kind, ver)
        j = i + 1
        while lines[j].rstrip("\n") != close:
            assert not PROV_OPEN_RE.match(lines[j].rstrip("\n")), "nested marker at line {}".format(j + 1)
            j += 1
        block = lines[i + 1:j]
        label, renames, literals, indent = parse(rest)
        if kind == "verbatim":
            assert not (renames or literals or indent) and not any("# [ph1 added]" in l for l in block)
            span = excerpt("".join(block), PROV_SOURCES[ver])
            n_added = 0
        else:
            kept = [l for l in block if not l.rstrip("\n").endswith("# [ph1 added]")]
            n_added = len(block) - len(kept)
            if indent:
                assert all(l.startswith(" " * indent) for l in kept if l.strip())
                kept = [l[indent:] if l.strip() else l for l in kept]
            text = "".join(kept)
            for old, new in literals:
                assert new in text
                text = text.replace(new, old)
            for old, new in renames:
                pat = r"(?<![{0}]){1}(?![{0}])".format(IDENT, re.escape(new))
                assert re.search(pat, text)
                text = re.sub(pat, old, text)
            span = excerpt(text, PROV_SOURCES[ver])
        res.append({"cell_lines": (i + 2, j), "kind": kind, "source": ver, "label": label, "source_lines": span,
                    "n_added": n_added, "spec": rest[rest.find(")") + 1:].strip()})
        i = j + 1
    return res


def code_names_and_strings(src):
    names, strings = set(), []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME:
            names.add(tok.string)
        elif tok.type == tokenize.STRING:
            strings.append(tok.string)
    return names, strings


def results_ok(root):
    ok, d = self_sha_ok(os.path.join(root, "outputs", "ph1", "ph1_results.json"))
    return ok, d


# =====================================================================================
# RUN
# =====================================================================================
print("=" * 110)
print("ph1 SYNTHETIC HARNESS | python {} | numpy {} | pandas {} | scipy {} | sklearn {} | {}".format(
    sys.version.split()[0], np.__version__, pd.__version__, scipy.__version__, sklearn.__version__, platform.platform()))
print("cell under test: {} (sha256 {}, {} lines)".format(CELL_PATH, sha256_bytes(CELL_SRC.encode("utf-8")),
                                                        len(CELL_SRC.splitlines())))
print("temp dir: {}".format(TMP))
print("=" * 110, flush=True)

print("\n--- T: full-size replica template (data, ACTUAL v0.2.1 + v0.3 runs, documents) ---", flush=True)
TEMPLATE = os.path.join(TMP, "template")
F0 = make_template(TEMPLATE)
try:
    _db = pd.read_csv(os.path.join(TEMPLATE, "data", DATASET_DIR, "ptbxl_database.csv"), index_col="ecg_id")
    _scp = pd.read_csv(os.path.join(TEMPLATE, "data", DATASET_DIR, "scp_statements.csv"), index_col=0)
    _lab = harness_labels(_db, _scp)
    _sizes = {f: int((_db["strat_fold"] == f).sum()) for f in range(1, 11)}
    _pats = {"1-8": int(_db.loc[_db["strat_fold"] <= 8, "patient_id"].nunique()),
             "9": int(_db.loc[_db["strat_fold"] == 9, "patient_id"].nunique()),
             "10": int(_db.loc[_db["strat_fold"] == 10, "patient_id"].nunique())}
    _pos = {f: {c: int(_lab.loc[_db["strat_fold"] == f, c].sum()) for c in SUPERCLASSES} for f in (9, 10)}
    _disjoint = len(set(_db.groupby("patient_id")["strat_fold"].nunique().unique()) - {1}) == 0
    record("T1 replica layout", list(_db.columns) == DB_COLUMNS and _db.shape == (21799, 27) and _sizes == FOLD_SIZES
           and _pats == {"1-8": N_PAT_18, "9": N_PAT_9, "10": N_PAT_10} and _pos == {9: POS9, 10: POS10} and _disjoint
           and int((_db.loc[_db["strat_fold"] <= 9, "age"] == 300).sum()) == N_SENT_19,
           "shape {} | fold sizes {} | patients {} | fold-9/10 positives {} | patient-disjoint folds {} | dtypes {}".format(
               _db.shape, list(_sizes.values()), _pats, _pos, _disjoint,
               dict(_db.dtypes.astype(str).value_counts())))
    _v021_out = sorted(os.listdir(os.path.join(TEMPLATE, "outputs", "v0.2.1")))
    _v03_out = sorted(os.listdir(os.path.join(TEMPLATE, "outputs", "v0.3")))
    record("T2 Part A references made by the ACTUAL v0.2.1 and v0.3 cells", len(_v021_out) == 6 and len(_v03_out) == 3 and
           os.path.isfile(os.path.join(TEMPLATE, "preregistration", "pre_registration.json")),
           "outputs\\v0.2.1 {} | outputs\\v0.3 {} | pre_registration.json written by v0.3 (exclusive create) | {:.0f}s".format(
               _v021_out, _v03_out, F0["seconds"]))
    _frp = load_json(os.path.join(TEMPLATE, "preregistration", "final_run_protocol.json"))
    _changed = sorted(k for k in REAL if k != "self_sha256" and json.dumps(REAL[k], sort_keys=True) != json.dumps(_frp[k], sort_keys=True))
    record("T3 replica FRP = the real document with data-dependent values replaced", _changed == [
        "bootstrap", "earlier_manifests_sha256", "input_files_sha256", "locked_quantities", "parent_hashes"] and
        _frp["reference_implementation"] == REAL["reference_implementation"] and
        _frp["locked_quantities"]["categorical_encoder"] == REAL["locked_quantities"]["categorical_encoder"] and
        _frp["bootstrap"]["locked_hashes"]["boot_idx_sha256"] == REAL["bootstrap"]["locked_hashes"]["boot_idx_sha256"],
        "top-level keys that differ: {} (categorical_encoder, reference_implementation, BOOT_IDX hash: real)".format(_changed))
except Exception:
    harness_exception("T")

# ---------------------------------------- P: static ----------------------------------------
print("\n--- P: static checks ---", flush=True)
try:
    nb = json.load(open(NOTEBOOK_PATH, encoding="utf-8"))
    cells = nb["cells"]
    record("P1 notebook == ph1_cell.py (one code cell, no outputs)", len(cells) == 1 and cells[0]["cell_type"] == "code"
           and cells[0]["outputs"] == [] and cells[0]["execution_count"] is None and "".join(cells[0]["source"]) == CELL_SRC,
           "cells {} | outputs {} | source sha256 {}".format(len(cells), cells[0].get("outputs"),
                                                              sha256_bytes("".join(cells[0]["source"]).encode("utf-8"))))
except Exception:
    harness_exception("P1")
try:
    _i = PROMPT_TEXT.index("> ph1") + 2
    PLAN_FROM_PROMPT = PROMPT_TEXT[_i:PROMPT_TEXT.index("\n", _i)]
    _tree = ast.parse(CELL_SRC)
    _plan = [ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "PH1_PLAN" for t in n.targets)]
    _exp_sha = [ast.literal_eval(n.value) for n in _tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "EXPECTED_PH1_PLAN_SHA256" for t in n.targets)]
    record("P2 PH1_PLAN == prompt section 3 text, verbatim", len(_plan) == 1 and _plan[0] == PLAN_FROM_PROMPT and
           _exp_sha == [sha256_bytes(PLAN_FROM_PROMPT.encode("utf-8"))],
           "{} chars, non-ASCII {} | sha256 {}".format(len(PLAN_FROM_PROMPT), sorted({hex(ord(ch)) for ch in PLAN_FROM_PROMPT
                                                                                       if ord(ch) > 127}),
                                                    sha256_bytes(PLAN_FROM_PROMPT.encode("utf-8"))))
except Exception:
    harness_exception("P2")
try:
    PROV = provenance_check(CELL_SRC)
    _bad = [r for r in PROV if r["source_lines"] is None]
    print("    provenance list (cell lines -> source lines):")
    for r in PROV:
        print("      {:8s} cell {:>9s}  <- {:6s} lines {:>9s}  {}{}".format(
            r["kind"], "{}-{}".format(*r["cell_lines"]), r["source"],
            "{}-{}".format(*r["source_lines"]) if r["source_lines"] else "NOT FOUND", r["label"],
            " " + r["spec"] if r["spec"] else ""))
    record("P3 provenance: every verbatim / adapted block is an excerpt of its source (after undoing the declared spec)",
           not _bad and len(PROV) > 0, "{} blocks ({} verbatim, {} adapted), {} mismatches; sources: v0.2.1 / v0.3 main cells "
           "read from the notebooks, v0.9_cell.py".format(len(PROV), sum(r["kind"] == "verbatim" for r in PROV),
                                                         sum(r["kind"] == "adapted" for r in PROV), len(_bad)))
except Exception:
    harness_exception("P3")
try:
    _names, _strings = code_names_and_strings(CELL_SRC)
    _imports = set()
    for n in ast.walk(ast.parse(CELL_SRC)):
        if isinstance(n, ast.Import):
            _imports |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            _imports.add(n.module)
    _allowed_imports = {"os", "sys", "json", "ast", "hashlib", "platform", "time", "datetime", "numpy", "pandas",
                        "scipy.stats", "sklearn.linear_model", "sklearn.ensemble", "sklearn.preprocessing", "sklearn.metrics",
                        "re", "math", "ntpath", "dis", "types", "builtins", "scipy", "sklearn", "threadpoolctl"}
    record("P4 no ECG signal access: no wfdb / torch name, no records100 string, imports within the allowed set",
           not ({"wfdb", "torch"} & _names) and not any("records100" in s for s in _strings) and _imports <= _allowed_imports,
           "imports {} | wfdb/torch names: {} | records100 in strings: {}".format(sorted(_imports), sorted({"wfdb", "torch"} & _names),
                                                                              any("records100" in s for s in _strings)))
except Exception:
    harness_exception("P4")
try:
    _test_src = src_for(F0)
    _a, _b = CELL_SRC.splitlines(), _test_src.splitlines()
    _subs = substitutions(F0)
    _diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(_a, _b)) if x != y]
    _explained = all(sum(1 for o, n in _subs if o in x and x.replace(o, n) == y) == 1 for _, x, y in _diff)
    print("    test cell = delivered cell + these substitutions ({} lines differ):".format(len(_diff)))
    for ln, x, y in _diff:
        print("      line {:5d}: {}\n                -> {}".format(ln, x.strip()[:150], y.strip()[:150]))
    record("P5 the test cell differs from the delivered cell only by the declared substitutions", len(_a) == len(_b) and
           _explained and len(_diff) == len(_subs), "{} substitutions, each matched exactly once".format(len(_subs)))
except Exception:
    harness_exception("P5")
try:
    record("P6 version constants", 'CODE_VERSION = "ph1"' in CELL_SRC and 'CODE_REVISION = "r1"' in CELL_SRC and
           CELL_SRC.splitlines()[2].startswith("# Code version: ph1 | Code revision: r1"),
           "CODE_VERSION ph1, CODE_REVISION r1, header line 3: {!r}".format(CELL_SRC.splitlines()[2]))
except Exception:
    harness_exception("P6")

# ---------------------------------------- M: main run ----------------------------------------
print("\n--- M: main run on a fresh clone of the replica ---", flush=True)
M_ROOT = clone(TEMPLATE, os.path.join(TMP, "main"))
M = run_cell(src_for(dict(F0, root=M_ROOT)), "M_main", M_ROOT)
try:
    record("M1 the cell runs end to end; outputs\\ph1 holds exactly the 8 write-once files", M["ok"] and
           M["out_files"] == OUT_FILES, "exception {} | files {}".format(short(M["exc"]), M["out_files"]))
    _bad = writes_ok(M)
    record("M2 writes: only '<outputs\\ph1>\\<name>.partial' (exclusive create) + exclusive rename; read-only folders, "
           "input CSVs, decoys and earlier outputs unchanged", not _bad and M["protected_unchanged"],
           "violations {} | protected snapshot unchanged: {} | {} renames".format(_bad[:3], M["protected_unchanged"],
                                                                              len(M["renames"])))
    _docs = {n: self_sha_ok(os.path.join(M_ROOT, "outputs", "ph1", n)) for n in
             ("ph1_analysis_plan.json", "ph1_frozen_outputs.json", "ph1_results.json", "ph1_manifest.json")}
    _man = _docs["ph1_manifest.json"][1]
    _man_ok = sorted(o["file"] for o in _man["outputs"]) == sorted(set(OUT_FILES) - {"ph1_manifest.json"}) and all(
        o["sha256"] == sha256_of(os.path.join(M_ROOT, "outputs", "ph1", o["file"])) and
        o["bytes"] == os.path.getsize(os.path.join(M_ROOT, "outputs", "ph1", o["file"])) for o in _man["outputs"])
    _fro = _docs["ph1_frozen_outputs.json"][1]
    _fro_ok = _fro["outputs"]["ph1_fold10_probs.csv"]["sha256"] == sha256_of(os.path.join(M_ROOT, "outputs", "ph1", "ph1_fold10_probs.csv"))
    record("M3 self_sha256 of the four JSON files verifies; manifest / frozen file hashes == disk", all(v[0] for v in _docs.values())
           and _man_ok and _fro_ok, "self_sha256 ok {} | manifest ok {} | frozen ok {}".format(
               {k: v[0] for k, v in _docs.items()}, _man_ok, _fro_ok))
    RES = _docs["ph1_results.json"][1]
    PLAN = _docs["ph1_analysis_plan.json"][1]
    _ids = [r["id"] for r in RES["checks"]]
    _need = ["A1_fold9_positives", "A1_split_sizes", "A1_encoder_folds1to8", "A2_decision_table", "A2_metric_rows",
             "A3_per_draw_deltas", "A3_learner_disagreement", "A3_v021_index_matrix_eq_REF", "B0_fold10_ids_file_order_sha256",
             "B0_fold10_patients_sorted_sha256", "B0_boot_idx_sha256", "B0_eval_rows_file_order", "B0_labels_eq_v09",
             "B0_fold10_positives", "B0_encoder_eq_FRP", "B0_fold10_unknown_zero", "B0_untrained_columns",
             "B0_level_strings_unchanged", "B0_split_sizes", "B1_no_partA_object_reused", "B1_scalers_fit_on_training_rows",
             "B1_age_scaler_eq_FRP", "B1_probs_roundtrip", "B2_v021_index_matrix_eq_BOOT_IDX", "B2_patient_rows_eq_REF",
             "OUT_snapshots_before_results"]
    _h2f = [r for r in RES["checks"] if r["section"] == "A3" and r["id"].startswith("A3_Logistic") or r["id"].startswith("A3_Hist")]
    record("M4 ph1_results.json: label, every check PASS, Part A max |diff| per column, all required check ids",
           RES["label"] == POSTHOC_LABEL and all(r["status"] == "PASS" for r in RES["checks"]) and set(_need) <= set(_ids)
           and RES["part_A"]["all_pass"] and len(_h2f) == 26,
           "{} checks ({} H2 field checks) | Part A max |diff|: decision table {} | metric rows {} | H2 fields {} | deltas {} "
           "| any AUROC changed {}".format(len(_ids), len(_h2f), RES["part_A"]["decision_table_max_abs_diff_per_column"],
                                           RES["part_A"]["metric_rows_max_abs_diff_per_column"],
                                           RES["part_A"]["h2_fields_max_abs_diff"],
                                           RES["part_A"]["per_draw_deltas_max_abs_diff_per_column"],
                                           RES["part_A"]["metric_rows_any_auroc_changed"]))
    _first_fit = min(f["seq"] for f in M["fits"])
    record("M5 ph1_analysis_plan.json: PH1_PLAN verbatim, plan sha256, POST-HOC label, input hashes; written before any "
           "model was fit", PLAN["plan"] == PLAN_FROM_PROMPT and PLAN["plan_utf8_sha256"] == sha256_bytes(PLAN_FROM_PROMPT.encode())
           and PLAN["label"] == POSTHOC_LABEL and len(PLAN["input_files_sha256"]) == 11 and
           rename_seq(M, "ph1_analysis_plan.json") < _first_fit,
           "plan rename seq {} < first fit seq {} | keys {}".format(rename_seq(M, "ph1_analysis_plan.json"), _first_fit,
                                                                    sorted(PLAN)))
    _fs = fit_summary(M)
    _A = [f for f in M["fits"] if f["rows"] == 17418]
    _B = [f for f in M["fits"] if f["rows"] == 19601]
    _seq_lab = read_seq(M, "v0.9_fold10_labels.csv")
    _seq_v03 = read_seq(M, "v0.3_bootstrap_deltas.csv")
    _seq_probs = rename_seq(M, "ph1_fold10_probs.csv")
    _seq_frozen = rename_seq(M, "ph1_frozen_outputs.json")
    _seq_back = [r["seq"] for r in M["read_csv"] if r["file"] == "ph1_fold10_probs.csv"]
    _metric_between = [s for s in M["rank_seq"] + M["ap_seq"] if min(f["seq"] for f in _B) < s < _seq_frozen]
    record("M6 order: Part A fits (17,418 rows) -> Part A comparisons -> v0.9 labels parsed -> Part B fits (19,601 rows) -> "
           "probabilities + freeze file -> metrics only after the freeze (no AUROC / AUPRC call in between) -> read-back",
           _fs == {("LR", 17418): 90, ("HGB", 17418): 90, ("SC", 17418): 4, ("LR", 19601): 90, ("HGB", 19601): 90,
                   ("SC", 19601): 4} and max(f["seq"] for f in _A) < _seq_v03 < _seq_lab < min(f["seq"] for f in _B)
           and max(f["seq"] for f in _B) < _seq_probs < _seq_frozen and not _metric_between and
           all(f["seq"] < _seq_frozen for f in M["fits"]) and len(_seq_back) == 1 and _seq_back[0] > _seq_frozen and
           [r["float_precision"] for r in M["read_csv"] if r["file"] == "ph1_fold10_probs.csv"] == ["round_trip"],
           "fits {} | seq: last A fit {} < v0.3 deltas read {} < v0.9 labels read {} < first B fit {} | last B fit {} < probs {} "
           "< frozen {} < read-back {} | AUROC/AUPRC calls between first B fit and freeze: {}".format(
               {"{}:{}".format(*k): v for k, v in sorted(_fs.items())}, max(f["seq"] for f in _A), _seq_v03, _seq_lab,
               min(f["seq"] for f in _B), max(f["seq"] for f in _B), _seq_probs, _seq_frozen, _seq_back, len(_metric_between)))
    _reads = [r["file"] for r in M["read_csv"]]
    record("M7 files parsed with pandas.read_csv (exactly, in this order)", _reads == [
        "scp_statements.csv", "ptbxl_database.csv", "v0.2.1_decision_table.csv", "v0.2.1_metrics_full.csv",
        "v0.3_bootstrap_deltas.csv", "v0.9_fold10_labels.csv", "ph1_fold10_probs.csv"], str(_reads))
    _txt = sorted({os.path.basename(o["path"]) for o in M["opens"] if o["mode"] in ("r", "rt") and _under(o["path"], M_ROOT)})
    _own = ["ph1_analysis_plan.json", "ph1_frozen_outputs.json", "ph1_results.json", "ph1_manifest.json"]
    _exp_txt = sorted(["final_run_protocol.json", "pre_registration.json", "v0.2.1_manifest.json", "v0.3_manifest.json",
                       "ptbxl_database.csv", "scp_statements.csv", "v0.2.1_decision_table.csv", "v0.2.1_metrics_full.csv",
                       "v0.3_bootstrap_deltas.csv", "v0.9_fold10_labels.csv", "ph1_fold10_probs.csv"] + _own +
                      [n + ".partial" for n in _own])
    _decoys = {os.path.abspath(os.path.join(M_ROOT, "outputs", "v0.9", n)) for n in os.listdir(os.path.join(M_ROOT, "outputs", "v0.9"))
               if n != "v0.9_fold10_labels.csv"}
    _decoy = [o for o in M["opens"] if os.path.abspath(o["path"]) in _decoys and o["mode"] != "rb"]
    _v08 = [o for o in M["opens"] if os.sep + os.path.join("outputs", "v0.8") + os.sep in o["path"]]
    record("M8 files opened for reading in text mode (project folder): exactly the inputs + the cell's own outputs; decoys "
           "in outputs\\v0.9 only hashed ('rb'); outputs\\v0.8 never opened", _txt == _exp_txt and not _decoy and not _v08,
           "text-mode opens {} | decoy opened other than 'rb': {} | outputs\\v0.8 opens: {}".format(_txt, len(_decoy), len(_v08)))
    record("M9 no ECG library imported", "torch" not in sys.modules and "wfdb" not in sys.modules,
           "torch in sys.modules: {} | wfdb: {}".format("torch" in sys.modules, "wfdb" in sys.modules))
    _o = M["out"]
    record("M10 printout: bytecode guard, Part A PASSED, POST-HOC label, summary", "read only constants, helper functions "
           "and REF (bytecode check): PASS" in _o and "PART A PASSED" in _o and "ph1 r1 finished successfully. " + POSTHOC_LABEL
           in _o and RES["previous_attempts"] == [] and RES["warnings"] == [],
           "previous attempts {} | warnings {}".format(RES["previous_attempts"], RES["warnings"]))
except Exception:
    harness_exception("M")

# ---------------------------------------- H: independent recomputation ----------------------------------------
print("\n--- H: independent recomputation from the files on disk (harness REF, sklearn, independent label rule) ---", flush=True)
try:
    PR = read_out_csv(M_ROOT, "ph1_fold10_probs.csv")
    MR = read_out_csv(M_ROOT, "ph1_metric_rows.csv")
    DT = read_out_csv(M_ROOT, "ph1_decision_table.csv")
    BD = read_out_csv(M_ROOT, "ph1_bootstrap_deltas.csv")
    _f10 = _db[_db["strat_fold"] == 10]
    Y10 = harness_labels(_f10, _scp)[SUPERCLASSES].to_numpy(dtype=np.int64)
    IDS10 = np.stack([_f10.index.to_numpy(), _f10["patient_id"].to_numpy().astype(np.int64)], axis=1)
    _cfg = [(b, l, s) for b in BLOCK_NAMES for l in LEARNERS for s in SEEDS]
    n10 = len(_f10)
    _order_ok = len(PR) == 36 * n10 and all(
        (PR["block"].iloc[k * n10:(k + 1) * n10] == b).all() and (PR["learner"].iloc[k * n10:(k + 1) * n10] == l).all() and
        (PR["seed"].iloc[k * n10:(k + 1) * n10] == s).all() and
        np.array_equal(PR[["ecg_id", "patient_id"]].iloc[k * n10:(k + 1) * n10].to_numpy(), IDS10) for k, (b, l, s) in enumerate(_cfg))
    _pv = PR[["p_" + c for c in SUPERCLASSES]].to_numpy()
    record("H1 ph1_fold10_probs.csv: 79,128 rows, block / learner / seed order, fold-10 rows in file order, columns",
           _order_ok and list(PR.columns) == ["ecg_id", "patient_id", "block", "learner", "seed"] + ["p_" + c for c in SUPERCLASSES]
           and str(PR["patient_id"].dtype) == "int64" and np.all((_pv >= 0) & (_pv <= 1)),
           "rows {} | dtypes {}".format(len(PR), dict(PR.dtypes.astype(str))))
    P = {cfg: PR.iloc[k * n10:(k + 1) * n10][["p_" + c for c in SUPERCLASSES]].to_numpy(dtype=np.float64)
         for k, cfg in enumerate(_cfg)}
    _md = 0.0
    _rows = []
    for (b, l, s) in _cfg:
        per = [float(sklearn.metrics.roc_auc_score(Y10[:, k], P[(b, l, s)][:, k])) for k in range(5)]
        for k, c in enumerate(SUPERCLASSES):
            r = MR[(MR["block"] == b) & (MR["learner"] == l) & (MR["seed"] == s) & (MR["class"] == c)].iloc[0]
            _md = max(_md, abs(per[k] - r["auroc"]))
        r = MR[(MR["block"] == b) & (MR["learner"] == l) & (MR["seed"] == s) & (MR["class"] == "MACRO")].iloc[0]
        _md = max(_md, abs(float(np.mean(per)) - r["auroc"]))
        _rows.append((b, l, s, float(np.mean(per)), r["ci_low"], r["ci_high"]))
    _prev = Y10.mean(axis=0)
    _chance_ok = all(abs(MR[(MR["class"] == c)]["auprc_chance"] - _prev[k]).max() == 0 for k, c in enumerate(SUPERCLASSES))
    record("H2 metric rows: per-class and macro AUROC == sklearn roc_auc_score on the probability file + independent labels; "
           "AUPRC chance == fold-10 prevalence", _md <= 1e-12 and len(MR) == 216 and _chance_ok,
           "max |diff| {:.2e} over 216 rows".format(_md))
    _rec = MR[MR["class"] == "MACRO"].groupby(["block", "learner"], sort=False).agg(
        macro_auroc_mean=("auroc", "mean"), ci_low_mean=("ci_low", "mean"), ci_high_mean=("ci_high", "mean")).reset_index()
    _ok_dt, _mdt = frame_maxdiff(DT[["block", "learner", "macro_auroc_mean", "ci_low_mean", "ci_high_mean"]], _rec,
                                 ["block", "learner"], ["macro_auroc_mean", "ci_low_mean", "ci_high_mean"])
    record("H3 decision table == recomputation from the metric rows", _ok_dt and _mdt <= 1e-15 and len(DT) == 12,
           "max |diff| {:.2e}".format(_mdt))
    _pats, _prow = HREF["patient_rows_in_file_order"](IDS10[:, 1])
    BOOT = HREF["build_boot_idx"](len(_pats), 2000, 7)
    _bh = sha256_bytes(np.ascontiguousarray(BOOT.astype("<i8")).tobytes())
    _mc = 0.0
    for (b, l, s, mac, lo, hi) in [_rows[_cfg.index(("B3_physio_plus_acquisition", "LogisticRegression", 42))],
                                   _rows[_cfg.index(("B1_physio", "HistGradientBoosting", 2024))]]:
        Pm = P[(b, l, s)]
        bm = np.empty(2000)
        for bi in range(2000):
            rows = HREF["boot_rows"](_prow, BOOT[bi])
            acc = [HREF["fast_auc"](Y10[rows, k], Pm[rows, k]) for k in range(5)]
            acc = [a for a in acc if not np.isnan(a)]
            bm[bi] = np.mean(acc) if acc else np.nan
        q = np.nanpercentile(bm, [2.5, 97.5])
        _mc = max(_mc, abs(q[0] - lo), abs(q[1] - hi))
    record("H4 v0.2.1 marginal CIs == recomputation on the locked BOOT_IDX (harness REF boot_rows, per-class nan-skip)",
           _mc == 0.0 and _bh == REAL["bootstrap"]["locked_hashes"]["boot_idx_sha256"],
           "2 configurations | max |diff| {} | BOOT_IDX sha256 == FRP lock: {}".format(_mc, _bh == REAL["bootstrap"]["locked_hashes"]["boot_idx_sha256"]))
    _h2 = RES["H2_posthoc"]["results_per_learner"]
    _hd, _strings_ok = 0.0, True
    for l, col in (("LogisticRegression", "delta_LR"), ("HistGradientBoosting", "delta_HGB")):
        A, B = P[("B1_physio", l, 42)], P[("B3_physio_plus_acquisition", l, 42)]
        md, _ = HREF["paired_bootstrap"](Y10, A, B, _prow, BOOT)
        ma, mb = HREF["macro_auroc_strict"](Y10, A), HREF["macro_auroc_strict"](Y10, B)
        S = HREF["summarize_deltas"](mb - ma, md)
        _hd = max(_hd, float(np.max(np.abs(md - BD[col].to_numpy()))))
        r = _h2[l]
        se = float(np.std(md[~np.isnan(md)], ddof=1))
        exp_fields = {"point_macro_auroc_B1": ma, "point_macro_auroc_B3": mb, "point_delta": mb - ma,
                      "mean_delta": float(np.mean(md[~np.isnan(md)])), "SE_delta": se,
                      "ci95_percentile": [S["lo95"], S["hi95"]], "ci90_percentile": [S["lo90"], S["hi90"]],
                      "n_boot_valid": S["n_valid"], "n_boot_invalid": S["n_nan"], "MDE": round((1.959964 + 0.841621) * se, 4),
                      "achieved_equivalence_bound": max(abs(S["lo90"]), abs(S["hi90"])), "n_patients": len(_pats)}
        for kf, v in exp_fields.items():
            _hd = max(_hd, max(abs(np.array(r[kf], dtype=float) - np.array(v, dtype=float)).ravel()))
        dv = ("no statistically significant difference between B1 and B3" if S["lo95"] <= 0 <= S["hi95"] else
              "significant difference: B3 {} than B1 by {:+.4f} (point delta)".format("higher" if mb - ma > 0 else "lower", mb - ma))
        tv = "practically negligible" if (S["lo90"] >= -0.02 and S["hi90"] <= 0.02) else "inconclusive / not shown negligible"
        _strings_ok &= r["difference_test_verdict"] == dv and r["tost_verdict"] == tv
    record("H5 H2-style results == harness REF paired bootstrap on the locked BOOT_IDX + independently re-typed v0.3 rules "
           "(deltas, points, mean, SE, CIs, MDE, bound, verdict strings)", _hd <= 1e-15 and _strings_ok and
           list(BD.columns) == ["replicate", "delta_LR", "delta_HGB"] and np.array_equal(BD["replicate"].to_numpy(), np.arange(2000)),
           "max |diff| {:.2e} | verdict strings equal: {} | LR: {} / {} | HGB: {} / {}".format(
               _hd, _strings_ok, _h2["LogisticRegression"]["difference_test_verdict"], _h2["LogisticRegression"]["tost_verdict"],
               _h2["HistGradientBoosting"]["difference_test_verdict"], _h2["HistGradientBoosting"]["tost_verdict"]))
    _v = RES["H1_posthoc"]["verdict"]
    record("H6 H1 rule (B3 >= 0.60) == recomputation from the decision table", all(
        _v[l]["passes_threshold"] == bool(DT[(DT["block"] == "B3_physio_plus_acquisition") & (DT["learner"] == l)]
                                          ["macro_auroc_mean"].iloc[0] >= 0.60) for l in LEARNERS),
           str({l: (round(_v[l]["macro_auroc_mean"], 4), _v[l]["passes_threshold"]) for l in LEARNERS}))
except Exception:
    harness_exception("H")


# ---------------------------------------- X: cross-check with the ACTUAL v0.2.1 / v0.3 cells ----------------------------------------
print("\n--- X: the ACTUAL v0.2.1 and v0.3 cells on a replica whose strat_fold is shifted (9 -> 8, 10 -> 9): their train = "
      "folds 1-9, eval = fold 10 -> must equal ph1 Part B ---", flush=True)
try:
    X_ROOT = os.path.join(TMP, "shifted")
    for d in ("preregistration", os.path.join("outputs", "v0.2.1"), os.path.join("outputs", "v0.3"), os.path.join("data", DATASET_DIR)):
        os.makedirs(os.path.join(X_ROOT, d))
    _dbx = pd.read_csv(os.path.join(TEMPLATE, "data", DATASET_DIR, "ptbxl_database.csv"), index_col="ecg_id")
    _dbx["strat_fold"] = _dbx["strat_fold"].replace({9: 8, 10: 9})
    _dbx.to_csv(os.path.join(X_ROOT, "data", DATASET_DIR, "ptbxl_database.csv"))
    shutil.copy2(os.path.join(TEMPLATE, "data", DATASET_DIR, "scp_statements.csv"), os.path.join(X_ROOT, "data", DATASET_DIR))
    run_v021(X_ROOT, "X_v0.2.1_shifted")
    run_v03(X_ROOT, "X_v0.3_shifted")
    _xdt = pd.read_csv(os.path.join(X_ROOT, "outputs", "v0.2.1", "v0.2.1_decision_table.csv"), encoding="utf-8-sig", float_precision="round_trip")
    _xmf = pd.read_csv(os.path.join(X_ROOT, "outputs", "v0.2.1", "v0.2.1_metrics_full.csv"), encoding="utf-8-sig", float_precision="round_trip")
    _xbd = pd.read_csv(os.path.join(X_ROOT, "outputs", "v0.3", "v0.3_bootstrap_deltas.csv"), encoding="utf-8-sig", float_precision="round_trip")
    _xpr = load_json(os.path.join(X_ROOT, "preregistration", "pre_registration.json"))
    _ok1, _m1 = frame_maxdiff(DT, _xdt, ["block", "learner"], ["macro_auroc_mean", "macro_auroc_std_across_seeds", "ci_low_mean",
                                                                "ci_high_mean", "delta_vs_B1_physio"])
    record("X1 ph1_decision_table.csv == decision table of the actual v0.2.1 cell trained on folds 1-9, evaluated on fold 10",
           _ok1 and _m1 == 0.0, "identifiers equal: {} | max |diff| {}".format(_ok1, _m1))
    _ok2, _m2 = frame_maxdiff(MR, _xmf, ["block", "learner", "seed", "class"],
                              ["auroc", "auroc_chance", "auprc", "auprc_chance", "ci_low", "ci_high"])
    record("X2 ph1_metric_rows.csv == all 216 metric rows of that v0.2.1 run", _ok2 and _m2 == 0.0,
           "identifiers equal: {} | max |diff| {}".format(_ok2, _m2))
    _ok3, _m3 = frame_maxdiff(BD, _xbd, ["replicate"], ["delta_LR", "delta_HGB"])
    record("X3 ph1_bootstrap_deltas.csv == per-draw deltas of the actual v0.3 cell on the same split", _ok3 and _m3 == 0.0,
           "max |diff| {}".format(_m3))
    _xr = _xpr["H2"]["results_per_learner"]
    _m4, _s4 = 0.0, True
    for l in LEARNERS:
        for kf, v in _xr[l].items():
            if isinstance(v, str):
                _s4 &= RES["H2_posthoc"]["results_per_learner"][l][kf] == v
            else:
                _m4 = max(_m4, float(np.max(np.abs(np.array(RES["H2_posthoc"]["results_per_learner"][l][kf], dtype=float) -
                                                   np.array(v, dtype=float)))))
    _s4 &= RES["H2_posthoc"]["learner_disagreement"] == _xpr["H2"]["learner_disagreement"]
    record("X4 ph1 H2-style results == pre_registration.json H2 block written by that v0.3 run (every field, verdict strings, "
           "disagreement list)", _m4 == 0.0 and _s4, "max |diff| {} | strings equal {}".format(_m4, _s4))
    _xman = load_json(os.path.join(X_ROOT, "outputs", "v0.2.1", "v0.2.1_manifest.json"))
    record("X5 H1 verdict == that v0.2.1 run's verdict", _xman["verdict"] == RES["H1_posthoc"]["verdict"], str(_xman["verdict"]))
except Exception:
    harness_exception("X")


# ---------------------------------------- D: crash protocol and determinism ----------------------------------------
print("\n--- D: crash protocol (rename, never delete) and determinism ---", flush=True)
try:
    D_ROOT = clone(TEMPLATE, os.path.join(TMP, "crash"))
    _src = src_for(dict(F0, root=D_ROOT))
    D1 = run_cell(_src, "D1_crash_after_plan", D_ROOT, inject={"crash_on_read": "ptbxl_database.csv"})
    record("D1 a crash after STEP 0 leaves outputs\\ph1 with the plan only (nothing else written, nothing fitted)",
           not D1["ok"] and "INJECTED CRASH" in D1["exc"] and D1["out_files"] == ["ph1_analysis_plan.json"] and not D1["fits"]
           and D1["protected_unchanged"], "exception {} | files {}".format(short(D1["exc"], 80), D1["out_files"]))
    _before = {n: sha256_of(os.path.join(D_ROOT, "outputs", "ph1", n)) for n in D1["out_files"]}
    D2 = run_cell(_src, "D2_rerun_without_rename", D_ROOT)
    _after = {n: sha256_of(os.path.join(D_ROOT, "outputs", "ph1", n)) for n in os.listdir(os.path.join(D_ROOT, "outputs", "ph1"))}
    record("D2 re-running without renaming the folder STOPS in STEP 0 (write-once), the folder is untouched",
           not D2["ok"] and "already exists" in D2["exc"] and _after == _before and not D2["renames"] and not D2["fits"],
           "exception {}".format(short(D2["exc"], 110)))
    os.rename(os.path.join(D_ROOT, "outputs", "ph1"), os.path.join(D_ROOT, "outputs", "ph1_failed_1"))
    os.makedirs(os.path.join(D_ROOT, "outputs", "ph1_failed_2"))
    _pb = open(os.path.join(M_ROOT, "outputs", "ph1", "ph1_fold10_probs.csv"), "rb").read()
    write_bytes(os.path.join(D_ROOT, "outputs", "ph1_failed_2", "ph1_fold10_probs.csv"), _pb[:-2] + b"9\n")
    os.makedirs(os.path.join(D_ROOT, "outputs", "ph1_failed_3"))
    write_bytes(os.path.join(D_ROOT, "outputs", "ph1_failed_3", "ph1_fold10_probs.csv"), _pb)
    D3 = run_cell(_src, "D3_rerun_after_rename", D_ROOT)
    _okd, RD = results_ok(D_ROOT)
    _csv_same = all(open(os.path.join(D_ROOT, "outputs", "ph1", n), "rb").read() == open(os.path.join(M_ROOT, "outputs", "ph1", n), "rb").read()
                    for n in ("ph1_fold10_probs.csv", "ph1_decision_table.csv", "ph1_bootstrap_deltas.csv", "ph1_metric_rows.csv"))
    _skip = {"timestamp", "run_started", "self_sha256", "previous_attempts", "previous_attempts_probability_files", "warnings",
             "plan", "frozen_outputs", "outputs_before_results"}
    _json_same = {k: RD[k] == RES[k] for k in RES if k not in _skip}
    record("D3 after RENAMING to outputs\\ph1_failed_1 the re-run completes; the 4 CSV outputs are byte-identical to the main run "
           "and the results equal it field by field (except timestamps / attempt records)",
           D3["ok"] and _okd and _csv_same and all(_json_same.values()), "csv identical {} | differing result fields {}".format(
               _csv_same, [k for k, v in _json_same.items() if not v]))
    _pc = {a["folder"]: a["identical_bytes"] for a in RD["previous_attempts_probability_files"]}
    record("D4 every earlier attempt is recorded; their probability files are compared byte for byte (WARN if different)",
           [a["folder"] for a in RD["previous_attempts"]] == ["ph1_failed_1", "ph1_failed_2", "ph1_failed_3"] and
           _pc == {"ph1_failed_1": None, "ph1_failed_2": False, "ph1_failed_3": True} and len(RD["warnings"]) == 1 and
           "ph1_failed_2" in RD["warnings"][0] and D3["protected_unchanged"],
           "comparison {} | warnings {}".format(_pc, RD["warnings"]))
except Exception:
    harness_exception("D")


# ---------------------------------------- S: STOP paths ----------------------------------------
print("\n--- S: every STOP path fires ---", flush=True)


def stop_test(name, prepare=None, extra=(), inject=None, consistent=True):
    """Clone the template, apply 'prepare' (file edits), run with substitutions computed from the edited files
    (consistent=True: the edit is taken as the locked state) or from the template (consistent=False: the cell's
    constants no longer match the edited file)."""
    root = clone(TEMPLATE, os.path.join(TMP, name))
    if prepare:
        prepare(root)
    F = facts(root) if consistent else dict(F0, root=root)
    return run_cell(src_for(F, extra), name, root, inject), root


def partA_failure(root):
    ok, d = self_sha_ok(os.path.join(root, "outputs", "ph1", "ph1_partA_failure.json"))
    return ok, d


def _append(path):
    with REAL_FN["open"](path, "ab") as f:
        f.write(b"\n")


def _db_edit(root, fn):
    p = os.path.join(root, "data", DATASET_DIR, "ptbxl_database.csv")
    db = pd.read_csv(p, index_col="ecg_id")
    fn(db)
    db.to_csv(p)
    h = sha256_of(p)
    frp_rewrite(root, lambda frp: frp["input_files_sha256"].__setitem__("ptbxl_database.csv", h))


S_EXPECT = []
try:
    c, r = stop_test("S1_wrong_input_hash", lambda root: _append(os.path.join(root, "data", DATASET_DIR, "ptbxl_database.csv")),
                     consistent=False)
    record("S1 wrong input hash (ptbxl_database.csv changed) -> STOP in STEP 0, nothing written", not c["ok"] and
           "STOP: input CSV hashes" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S1")
try:
    def _reindent(root):
        p = os.path.join(root, "preregistration", "pre_registration.json")
        d = load_json(p)
        with REAL_FN["open"](p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    c, r = stop_test("S2_prereg_file_hash", _reindent, consistent=False)
    record("S2 pre_registration.json bytes changed (self_sha256 still valid) -> STOP in STEP 0 (file sha256)", not c["ok"] and
           "STOP: pre_registration: file sha256" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S2")
try:
    c, r = stop_test("S3_reference_code", lambda root: frp_rewrite(root, lambda frp: frp["reference_implementation"].__setitem__(
        "code", frp["reference_implementation"]["code"] + "\n# changed\n")))
    record("S3 reference implementation code changed (documents otherwise consistent) -> STOP before exec", not c["ok"] and
           "reference implementation sha256 mismatch" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S3")
try:
    c, r = stop_test("S4_v021_decision_table_hash", lambda root: _append(os.path.join(root, "outputs", "v0.2.1",
                                                                                      "v0.2.1_decision_table.csv")),
                     consistent=False)
    record("S4 v0.2.1_decision_table.csv != the core chat's copy / manifest -> STOP in STEP 0", not c["ok"] and
           "STOP: v0.2.1_decision_table.csv differs" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S4")
try:
    c, r = stop_test("S5_v03_deltas_vs_manifest", lambda root: _append(os.path.join(root, "outputs", "v0.3",
                                                                                    "v0.3_bootstrap_deltas.csv")))
    record("S5 v0.3_bootstrap_deltas.csv != its v0.3 manifest entry -> STOP in STEP 0", not c["ok"] and
           "differs from the v0.3 manifest" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S5")
try:
    c, r = stop_test("S6_v09_labels_hash", lambda root: _append(os.path.join(root, "outputs", "v0.9", "v0.9_fold10_labels.csv")),
                     consistent=False)
    record("S6 v0.9_fold10_labels.csv sha256 != expected -> STOP in STEP 0", not c["ok"] and
           "STOP: v0.9_fold10_labels.csv sha256" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S6")
try:
    c, r = stop_test("S7_version", extra=[('"scipy": "1.15.3"', '"scipy": "1.15.2"')])
    record("S7 a library version differs from the required environment -> STOP in STEP 0 (declared substitution of the expected "
           "scipy version)", not c["ok"] and "STOP: environment differs" in c["exc"] and c["out_files"] is None, short(c["exc"]))
except Exception:
    harness_exception("S7")
try:
    def _mk_out(root):
        os.makedirs(os.path.join(root, "outputs", "ph1"))
        write_bytes(os.path.join(root, "outputs", "ph1", "user_file.txt"), b"left by an earlier run\n")
    c, r = stop_test("S8_out_dir_exists", _mk_out)
    record("S8 outputs\\ph1 already exists -> STOP in STEP 0 (folder untouched)", not c["ok"] and "already exists" in c["exc"]
           and c["out_files"] == ["user_file.txt"] and not c["renames"] and not c["fits"], short(c["exc"]))
except Exception:
    harness_exception("S8")
try:
    c, r = stop_test("S9_locked_file_changed_during_run",
                     inject={"modify_protected_at": ("ph1_frozen_outputs.json", ["preregistration", "pre_registration.json.ots"])})
    _have = c["out_files"] or []
    record("S9 a file in preregistration\\ changes during the run -> the re-check before the results FAILS; no results, no "
           "manifest", not c["ok"] and "OUT_snapshots_before_results" in c["exc"] and "ph1_results.json" not in _have and
           "ph1_manifest.json" not in _have and "ph1_metric_rows.csv" in _have, "{} | files {}".format(short(c["exc"], 90), _have))
except Exception:
    harness_exception("S9")
try:
    c, r = stop_test("S9b_file_changed_after_results",
                     inject={"modify_protected_at": ("ph1_results.json", ["outputs", "v0.3", "v0.3_h2_summary.csv"])})
    record("S9b a file in outputs\\v0.3 changes after the results were written -> the re-check after the manifest FAILS",
           not c["ok"] and "FAIL: read-only folders / input CSVs changed during the run" in c["exc"] and
           "outputs\\\\v0.3" in c["exc"] and c["out_files"] == OUT_FILES, "{} | files {}".format(short(c["exc"], 120), c["out_files"]))
except Exception:
    harness_exception("S9b")
try:
    def _perturb_dt(root):
        p = os.path.join(root, "outputs", "v0.2.1", "v0.2.1_decision_table.csv")
        dt = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
        dt.loc[7, "ci_low_mean"] = dt.loc[7, "ci_low_mean"] + 1e-9
        dt.to_csv(p, index=False, encoding="utf-8-sig")
        manifest_set(os.path.join(root, "outputs", "v0.2.1", "v0.2.1_manifest.json"), "v0.2.1_decision_table.csv", sha256_of(p))
    c, r = stop_test("S10_partA_decision_table_1e-9", _perturb_dt)
    ok, d = partA_failure(r)
    _dtd = d.get("diagnostics", {}).get("decision_table", {})
    record("S10 one cell of the v0.2.1 decision table + 1e-9 -> Part A STOPS; outputs\\ph1 = plan + ph1_partA_failure.json "
           "only; diagnostics per column / per row; no fold-10 model fit", not c["ok"] and "Part A validation gate FAILED" in c["exc"]
           and c["out_files"] == ["ph1_analysis_plan.json", "ph1_partA_failure.json"] and ok and
           d["failed_checks"] == ["A2_decision_table"] and 0.5e-9 < _dtd["max_abs_diff_per_column"]["ci_low_mean"] < 2e-9 and
           sum(r_["max_abs_diff"] > 0 for r_ in _dtd["max_abs_diff_per_row"]) == 1 and
           not any(f["rows"] == 19601 for f in c["fits"]) and c["protected_unchanged"],
           "failed {} | max |diff| per column {} | rows with diff > 0: {} | Part B fits {} | v0.9 labels parsed: {}".format(
               d.get("failed_checks"), _dtd.get("max_abs_diff_per_column"),
               [r_ for r_ in _dtd.get("max_abs_diff_per_row", []) if r_["max_abs_diff"] > 0],
               sum(f["rows"] == 19601 for f in c["fits"]), read_seq(c, "v0.9_fold10_labels.csv") is not None))
except Exception:
    harness_exception("S10")
try:
    def _perturb_mf(root):
        p = os.path.join(root, "outputs", "v0.2.1", "v0.2.1_metrics_full.csv")
        mf = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
        mf.loc[100, "auroc"] = mf.loc[100, "auroc"] + 1e-9
        mf.to_csv(p, index=False, encoding="utf-8-sig")
        manifest_set(os.path.join(root, "outputs", "v0.2.1", "v0.2.1_manifest.json"), "v0.2.1_metrics_full.csv", sha256_of(p))
    c, r = stop_test("S11_partA_metrics_full_1e-9", _perturb_mf)
    ok, d = partA_failure(r)
    _mfd = d.get("diagnostics", {}).get("metrics_full", {})
    record("S11 one value of v0.2.1_metrics_full.csv + 1e-9 -> Part A STOPS; outputs\\ph1 = plan + ph1_partA_failure.json only; "
           "the diagnostics say an AUROC changed", not c["ok"] and c["out_files"] == ["ph1_analysis_plan.json",
                                                                                     "ph1_partA_failure.json"]
           and ok and d["failed_checks"] == ["A2_metric_rows"] and _mfd.get("any_auroc_changed") is True and
           _mfd.get("n_auroc_changed") == 1 and not any(f["rows"] == 19601 for f in c["fits"]),
           "failed {} | max |diff| per column {} | any AUROC changed {} ({} rows)".format(
               d.get("failed_checks"), _mfd.get("max_abs_diff_per_column"), _mfd.get("any_auroc_changed"),
               _mfd.get("n_auroc_changed")))
except Exception:
    harness_exception("S11")
try:
    def _perturb_deltas(root):
        p = os.path.join(root, "outputs", "v0.3", "v0.3_bootstrap_deltas.csv")
        bd = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
        bd.loc[1234, "delta_HGB"] = bd.loc[1234, "delta_HGB"] + 1e-9
        bd.to_csv(p, index=False, encoding="utf-8-sig")
        h = manifest_set(os.path.join(root, "outputs", "v0.3", "v0.3_manifest.json"), "v0.3_bootstrap_deltas.csv", sha256_of(p))
        frp_rewrite(root, lambda frp: frp["earlier_manifests_sha256"].__setitem__("v0.3_manifest.json", h))
    c, r = stop_test("S12_partA_v03_delta_1e-9", _perturb_deltas)
    ok, d = partA_failure(r)
    record("S12 one per-draw delta of v0.3_bootstrap_deltas.csv + 1e-9 -> Part A (A3) STOPS", not c["ok"] and
           c["out_files"] == ["ph1_analysis_plan.json", "ph1_partA_failure.json"] and ok and
           d["failed_checks"] == ["A3_per_draw_deltas"], "failed {} | {}".format(
               d.get("failed_checks"), d.get("diagnostics", {}).get("bootstrap_deltas", {}).get("max_abs_diff_per_column")))
except Exception:
    harness_exception("S12")
try:
    def _a1(db):
        rows = db.index[(db["strat_fold"] == 9) & ~db["scp_codes"].str.contains("LVH|RVH")]
        d = ast.literal_eval(db.at[rows[0], "scp_codes"])
        d["LVH"] = 100.0
        db.at[rows[0], "scp_codes"] = str(d)
    c, r = stop_test("S13_partA_A1_label_counts", lambda root: _db_edit(root, _a1))
    ok, d = partA_failure(r)
    record("S13 fold-9 label counts differ (A1) -> STOP inside fit_predict BEFORE any model is fit; failure file written",
           not c["ok"] and c["out_files"] == ["ph1_analysis_plan.json", "ph1_partA_failure.json"] and ok and
           d["failed_checks"] == ["A1_fold9_positives"] and not c["fits"], "{} | fits {}".format(short(c["exc"], 100), len(c["fits"])))
except Exception:
    harness_exception("S13")
try:
    def _flip(root):
        p = os.path.join(root, "outputs", "v0.9", "v0.9_fold10_labels.csv")
        lab = pd.read_csv(p, encoding="utf-8-sig")
        lab.loc[17, "STTC"] = 1 - lab.loc[17, "STTC"]
        write_bytes(p, csv_bytes(lab))
    c, r = stop_test("S14_label_mismatch", _flip)
    record("S14 one fold-10 label differs from v0.9_fold10_labels.csv -> STOP in B0 (Part A passed, no fold-10 model fit)",
           not c["ok"] and "B0_labels_eq_v09" in c["exc"] and c["out_files"] == ["ph1_analysis_plan.json"] and
           not any(f["rows"] == 19601 for f in c["fits"]) and sum(f["rows"] == 17418 for f in c["fits"]) == 184,
           short(c["exc"], 140))
except Exception:
    harness_exception("S14")
try:
    def _cat(root):
        def m(frp):
            enc = frp["locked_quantities"]["categorical_encoder"]
            enc["categories"]["site"].insert(enc["categories"]["site"].index("__MISSING__"), "777")
            enc["new_levels_vs_folds1to8"]["site"].append("777")
        frp_rewrite(root, m)
    c, r = stop_test("S15_encoder_category_mismatch", _cat)
    record("S15 folds 1-9 encoder categories != FRP locked categories -> STOP in B0 (no fold-10 model fit)",
           not c["ok"] and "B0_encoder_eq_FRP" in c["exc"] and c["out_files"] == ["ph1_analysis_plan.json"] and
           not any(f["rows"] == 19601 for f in c["fits"]), short(c["exc"], 140))
except Exception:
    harness_exception("S15")
try:
    c, r = stop_test("S16_boot_idx_hash", lambda root: frp_rewrite(
        root, lambda frp: frp["bootstrap"]["locked_hashes"].__setitem__("boot_idx_sha256", "0" * 64)))
    record("S16 wrong locked BOOT_IDX hash -> STOP in B0 (no fold-10 model fit)", not c["ok"] and "B0_boot_idx_sha256" in c["exc"]
           and c["out_files"] == ["ph1_analysis_plan.json"] and not any(f["rows"] == 19601 for f in c["fits"]), short(c["exc"], 140))
except Exception:
    harness_exception("S16")
try:
    def _unk(db):
        db.at[db.index[db["strat_fold"] == 10][5], "site"] = 77.0
    c, r = stop_test("S17_fold10_unknown_category", lambda root: _db_edit(root, _unk))
    record("S17 a fold-10 record maps to __UNKNOWN__ -> STOP in B0 (no fold-10 model fit)", not c["ok"] and
           "B0_fold10_unknown_zero" in c["exc"] and c["out_files"] == ["ph1_analysis_plan.json"] and
           not any(f["rows"] == 19601 for f in c["fits"]), short(c["exc"], 140))
except Exception:
    harness_exception("S17")

# ---------------------------------------- N: single code path guard ----------------------------------------
print("\n--- N: the shared functions may not read module-level fitted state ---", flush=True)
try:
    c, r = stop_test("N1_global_read_in_fit_predict", extra=[(
        "    labels_eval = lab.loc[mask_eval]\n", "    labels_eval = lab.loc[mask_eval]\n    _leak = len(df9)  # injected\n")])
    record("N1 a module-level data read injected into fit_predict -> the bytecode guard STOPS before Part A",
           not c["ok"] and "fit_predict reads module-level names outside the allowed set: ['df9']" in c["exc"] and
           c["out_files"] == ["ph1_analysis_plan.json"] and not c["fits"], short(c["exc"], 140))
except Exception:
    harness_exception("N1")


# ---------------------------------------- summary ----------------------------------------
print("\n" + "=" * 110)
n_fail = sum(1 for r_ in REPORT if r_[1] != "PASS")
for t, s_, dtl in REPORT:
    print("{}  {}".format(s_, t))
print("=" * 110)
print("{} tests: {} PASS, {} FAIL | {:.0f}s | logs in {}".format(len(REPORT), len(REPORT) - n_fail, n_fail,
                                                                  time.time() - T_ALL, LOGS))
