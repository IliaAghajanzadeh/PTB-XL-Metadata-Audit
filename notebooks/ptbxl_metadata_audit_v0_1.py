# =====================================================================================
# پروژه: PTB-XL Metadata Audit  —  ممیزیِ فقط-خواندنیِ یکپارچگی و متادیتا
# نسخه کد: v0.1
# تاریخ: 2026-09-09
# هدف: ثبتِ provenance، شکلِ داده، متادیتای ضبط، دموگرافیک، سلامتِ split،
#      اشتقاقِ برچسبِ ابرکلاس، و جدول‌های تقاطعیِ device/site در برابر fold و برچسب.
# متغیر مستقل: هیچ — این نسخه هیچ مدلی آموزش نمی‌دهد و هیچ عددِ عملکردی تولید نمی‌کند.
#
# ورودی‌های مجاز (read-only): ptbxl_database.csv | scp_statements.csv | RECORDS |
#                              ptbxl_v102_changelog.txt | ptbxl_v103_changelog.txt
# هیچ فایلِ سیگنالی از records100/ یا records500/ باز نمی‌شود. wfdb ایمپورت نمی‌شود.
#
# خروجی‌ها (همه در OUT_DIR):
#   1) v0.1_column_inventory.csv              — dtype + missingness + cardinality هر ستون
#   2) v0.1_labels.csv                        — ecg_id, patient_id, strat_fold + ۵ ستون باینری ابرکلاس
#   3) v0.1_crosstab_device_fold.csv          — device × strat_fold (شمارش)
#   4) v0.1_crosstab_site_fold.csv            — site × strat_fold (شمارش)
#   5) v0.1_prevalence_device_superclass.csv  — نرخ شیوع هر ابرکلاس درون هر device
#   6) v0.1_prevalence_site_superclass.csv    — نرخ شیوع هر ابرکلاس درون هر site
#   7) v0.1_manifest.json                     — provenance کامل + SHA-256 ورودی‌ها و خروجی‌ها
#
# ممنوعیت‌ها: آموزش مدل | AUROC/accuracy/هر متریکِ عملکردی | رسم نمودار |
#             نوشتن یا تغییر هر فایلی داخلِ پوشه‌ی دیتاست
#
# نکته درباره assert نشت: بررسیِ patient-disjoint بودنِ foldها انجام می‌شود، جزئیاتِ تخلف
# چاپ می‌شود، اما AssertionError عمداً در انتهای سلول پرتاب می‌شود تا خروجی‌های ممیزی
# (که برای تشخیصِ علت لازم‌اند) از دست نروند.
# =====================================================================================

import os, sys, json, ast, hashlib, platform
from datetime import datetime
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 220)

# ------------------------------- پیکربندی -------------------------------------------
DATA_ROOT = r"D:\ptbxl-metadata-audit\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT_DIR   = r"D:\ptbxl-metadata-audit\outputs\v0.1"
CODE_VERSION = "v0.1"
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# اعدادِ لنگرِ منتشرشده (برای مقایسه، نه برای اعمال) — منبع: مقاله‌ی اصلی PTB-XL
ANCHORS = {"n_records": 21799, "n_patients": 18869,
           "NORM": 9514, "MI": 5469, "STTC": 5235, "CD": 4898, "HYP": 2649}

os.makedirs(OUT_DIR, exist_ok=True)

# گاردِ ایمنی: خروجی هرگز نباید داخلِ پوشه‌ی دیتاست نوشته شود
_root_abs, _out_abs = os.path.abspath(DATA_ROOT), os.path.abspath(OUT_DIR)
assert not _out_abs.lower().startswith(_root_abs.lower()), "OUT_DIR داخلِ پوشه‌ی دیتاست است — متوقف شد."

RUN_STARTED = datetime.now().astimezone().isoformat()
OUTPUT_FILES = []          # نامِ فایل‌های تولیدشده
WARNINGS = []              # هشدارهای جمع‌شده برای خلاصه‌ی پایانی
LEAKAGE_FAIL = False


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


# =====================================================================================
# بخش ۱ — PROVENANCE
# =====================================================================================
hdr("SECTION 1 — PROVENANCE")

print("run started         :", RUN_STARTED)
print("python              :", sys.version.replace("\n", " "))
print("platform            :", platform.platform())
print("machine / processor :", platform.machine(), "|", platform.processor())
print("numpy               :", np.__version__)
print("pandas              :", pd.__version__)
print("dataset folder name :", os.path.basename(os.path.normpath(DATA_ROOT)))
print("dataset root        :", DATA_ROOT)
print("output dir          :", OUT_DIR)

DB_CSV      = os.path.join(DATA_ROOT, "ptbxl_database.csv")
SCP_CSV     = os.path.join(DATA_ROOT, "scp_statements.csv")
RECORDS_TXT = os.path.join(DATA_ROOT, "RECORDS")
CHANGELOGS  = ["ptbxl_v102_changelog.txt", "ptbxl_v103_changelog.txt"]

for p in (DB_CSV, SCP_CSV, RECORDS_TXT):
    assert os.path.isfile(p), "فایل پیدا نشد: " + p

print("\n--- SHA-256 فایل‌های ورودیِ اصلی ---")
INPUT_HASHES = {}
for p in (DB_CSV, SCP_CSV):
    h = sha256_of(p)
    INPUT_HASHES[os.path.basename(p)] = h
    print("{:26s} {}   ({:,} bytes)".format(os.path.basename(p), h, os.path.getsize(p)))

print("\n--- SHA-256 سایر ورودی‌های خوانده‌شده ---")
OTHER_HASHES = {}
for fn in [os.path.basename(RECORDS_TXT)] + CHANGELOGS:
    p = os.path.join(DATA_ROOT, fn)
    if os.path.isfile(p):
        h = sha256_of(p)
        OTHER_HASHES[fn] = h
        print("{:26s} {}   ({:,} bytes)".format(fn, h, os.path.getsize(p)))
    else:
        warn("یافت نشد: " + p)

for fn in CHANGELOGS:
    p = os.path.join(DATA_ROOT, fn)
    print("\n" + "-" * 100)
    print("--- محتوای کاملِ " + fn + " ---")
    print("-" * 100)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            print(f.read())
    else:
        print("[WARN] فایل موجود نیست.")


# =====================================================================================
# بخش ۲ — شکلِ داده
# =====================================================================================
hdr("SECTION 2 — DATA SHAPE")

df = pd.read_csv(DB_CSV, index_col="ecg_id")

print("shape (rows, cols)      :", df.shape)
print("unique ecg_id           :", df.index.nunique())
print("duplicated ecg_id       :", int(df.index.duplicated().sum()))
print("unique patient_id       :", df["patient_id"].nunique())
print("anchor n_records/n_pat  :", ANCHORS["n_records"], "/", ANCHORS["n_patients"])
if len(df) != ANCHORS["n_records"]:
    warn("تعدادِ رکورد ({}) با عددِ لنگرِ منتشرشده ({}) یکی نیست.".format(len(df), ANCHORS["n_records"]))
if df["patient_id"].nunique() != ANCHORS["n_patients"]:
    warn("تعدادِ بیمار ({}) با عددِ لنگر ({}) یکی نیست.".format(df["patient_id"].nunique(), ANCHORS["n_patients"]))

print("\n--- ستون‌ها، dtype، missingness، cardinality ---")
inventory = pd.DataFrame({
    "column":      list(df.columns),
    "dtype":       [str(t) for t in df.dtypes],
    "n_missing":   df.isna().sum().values,
    "pct_missing": np.round(df.isna().mean().values * 100, 3),
    "n_unique":    [int(df[c].nunique(dropna=True)) for c in df.columns],
})
print(inventory.to_string(index=False))
save_csv(inventory, "v0.1_column_inventory.csv", index=False)

# --- افزوده: بررسیِ سازگاریِ RECORDS با مسیرهای داخلِ CSV (هیچ فایلِ سیگنالی باز نمی‌شود) ---
print("\n--- بررسیِ سازگاریِ RECORDS (فقط خواندنِ فهرستِ مسیرها) ---")
with open(RECORDS_TXT, "r", encoding="utf-8", errors="replace") as f:
    records_lines = [ln.strip() for ln in f if ln.strip()]
rec_set = set(records_lines)
csv_paths = set(df["filename_lr"].astype(str)) | set(df["filename_hr"].astype(str))
print("خطوطِ RECORDS                :", len(records_lines), "| یکتا:", len(rec_set))
print("مسیرهای یکتا در CSV          :", len(csv_paths))
only_csv = sorted(csv_paths - rec_set)
only_rec = sorted(rec_set - csv_paths)
print("در CSV هست ولی در RECORDS نه :", len(only_csv), ("| نمونه: " + str(only_csv[:5])) if only_csv else "")
print("در RECORDS هست ولی در CSV نه :", len(only_rec), ("| نمونه: " + str(only_rec[:5])) if only_rec else "")
if only_csv or only_rec:
    warn("عدمِ تطابقِ RECORDS با مسیرهای CSV — ممکن است صرفاً تفاوتِ قالبِ نام باشد؛ نمونه‌ها را ببین.")


# =====================================================================================
# بخش ۳ — متادیتای ضبط
# =====================================================================================
hdr("SECTION 3 — RECORDING METADATA (device / site / nurse)")

for col in ["device", "site", "nurse"]:
    print("\n--- " + col + " ---")
    print("nunique (non-null):", int(df[col].nunique(dropna=True)),
          "| missing:", int(df[col].isna().sum()))
    vc = df[col].value_counts(dropna=False).to_frame("n")
    vc["pct"] = np.round(vc["n"] / len(df) * 100, 3)
    print(vc.to_string())

print("\n--- recording_date ---")
rd = pd.to_datetime(df["recording_date"], errors="coerce")
n_unparsable = int(rd.isna().sum() - df["recording_date"].isna().sum())
print("min:", rd.min(), "| max:", rd.max())
print("missing:", int(df["recording_date"].isna().sum()), "| unparsable:", n_unparsable)
print("\nتوزیعِ سالانه:")
print(rd.dt.year.value_counts(dropna=False).sort_index().to_frame("n").to_string())


# =====================================================================================
# بخش ۴ — دموگرافیک
# =====================================================================================
hdr("SECTION 4 — DEMOGRAPHICS")

print("--- sex ---")
vc = df["sex"].value_counts(dropna=False).to_frame("n")
vc["pct"] = np.round(vc["n"] / len(df) * 100, 3)
print(vc.to_string())

print("\n--- age ---")
age = df["age"]
print("dtype:", age.dtype, "| missing:", int(age.isna().sum()))
print("min :", age.min(), "| max:", age.max(), "| mean:", round(float(age.mean()), 3))
for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
    print("  p{:g} = {}".format(q * 100, age.quantile(q)))

print("\n۱۵ مقدارِ پرتکرارِ age (برای کشفِ sentinel):")
print(age.value_counts(dropna=False).head(15).to_frame("n").to_string())

print("\nشمارشِ مقادیرِ مشکوک / سنسورشده:")
checks = [("age > 89", (age > 89)), ("age > 90", (age > 90)),
          ("age >= 100", (age >= 100)), ("age >= 120 (غیرممکنِ فیزیولوژیک)", (age >= 120)),
          ("age == 300", (age == 300)), ("age <= 0", (age <= 0))]
for name, mask in checks:
    print("  {:35s} : {}".format(name, int(mask.sum())))
print("  {:35s} : {}".format("age اعشاری (غیرصحیح)", int((age.dropna() % 1 != 0).sum())))

sentinel_mask = (age >= 120)
if int(sentinel_mask.sum()) > 0:
    warn("مقدارِ sentinel در age پیدا شد ({} رکورد با age>=120). "
         "این‌ها سنین سانسورشده‌اند و باید قبل از هر مدل‌سازی صریحاً مدیریت شوند.".format(int(sentinel_mask.sum())))

print("\nتوزیعِ کاملِ سنینِ >= ۸۰ (مرزِ سانسور معمولاً اینجاست):")
print(age[age >= 80].value_counts().sort_index().to_frame("n").to_string())

print("\n--- height / weight ---")
for c in ["height", "weight"]:
    print("\n{}: missing {} ({:.2f}%)".format(c, int(df[c].isna().sum()), df[c].isna().mean() * 100))
    print(df[c].describe().to_string())


# =====================================================================================
# بخش ۵ — سلامتِ SPLIT
# =====================================================================================
hdr("SECTION 5 — SPLIT INTEGRITY")

print("--- strat_fold ---")
fold_tab = df["strat_fold"].value_counts(dropna=False).sort_index().to_frame("n_records")
fold_tab["pct"] = np.round(fold_tab["n_records"] / len(df) * 100, 3)
fold_tab["n_patients"] = df.groupby("strat_fold")["patient_id"].nunique()
print(fold_tab.to_string())

print("\n--- بررسیِ patient-disjoint بودنِ foldها ---")
folds_per_patient = df.groupby("patient_id")["strat_fold"].nunique()
violators = folds_per_patient[folds_per_patient > 1]
LEAKAGE_FAIL = len(violators) > 0
print("بیمارانی که در بیش از یک fold ظاهر می‌شوند:", len(violators))
if LEAKAGE_FAIL:
    warn("نشتِ سطح-بیمار در splitِ رسمی — این بلوکه‌کننده است.")
    print("\n۱۰ نمونه‌ی اولِ بیمارانِ متخلف و foldهایشان:")
    for pid in list(violators.index[:10]):
        sub = df[df["patient_id"] == pid]
        print("  patient_id={} | folds={} | n_records={}".format(
            pid, sorted(sub["strat_fold"].unique().tolist()), len(sub)))
else:
    print("PASS — هیچ بیماری بین دو fold مشترک نیست.")

print("\n--- توزیعِ تعدادِ رکورد به‌ازای هر بیمار ---")
rec_per_patient = df.groupby("patient_id").size()
print(rec_per_patient.value_counts().sort_index().to_frame("n_patients").to_string())
print("max records per patient:", int(rec_per_patient.max()),
      "| mean:", round(float(rec_per_patient.mean()), 4))

# --- افزوده: اعتبارسنجیِ انسانیِ برچسب × fold (فرضِ طرح: foldهای ۹ و ۱۰ اعتبارسنجی‌شده‌اند) ---
if "validated_by_human" in df.columns:
    print("\n--- افزوده: validated_by_human × strat_fold ---")
    ct_val = pd.crosstab(df["strat_fold"], df["validated_by_human"].fillna("__MISSING__"))
    print(ct_val.to_string())


# =====================================================================================
# بخش ۶ — اشتقاقِ برچسب
# =====================================================================================
hdr("SECTION 6 — LABEL DERIVATION")

scp = pd.read_csv(SCP_CSV, index_col=0)
print("scp_statements shape:", scp.shape)
print("ستون‌ها:", list(scp.columns))

diag = scp[scp["diagnostic"].fillna(0) == 1]
print("\nstatements با diagnostic == 1 :", len(diag), "از", len(scp))
print("\nتوزیعِ diagnostic_class در میانِ آن‌ها:")
print(diag["diagnostic_class"].value_counts(dropna=False).to_frame("n_statements").to_string())

code2class = {k: v for k, v in diag["diagnostic_class"].items() if isinstance(v, str)}

parsed = df["scp_codes"].apply(ast.literal_eval)
all_codes = set()
for d in parsed:
    all_codes.update(d.keys())
unknown_codes = sorted(all_codes - set(scp.index))
print("\nکدهای یکتا در scp_codes :", len(all_codes),
      "| کدهایی که در scp_statements نیستند:", len(unknown_codes))
if unknown_codes:
    warn("کدهای ناشناخته در scp_codes: " + str(unknown_codes[:20]))

# توزیعِ likelihood — یک تصمیمِ پیش‌ثبتی: قراردادِ استاندارد همه‌ی کلیدها را نگه می‌دارد
lik_counter = {}
for d in parsed:
    for v in d.values():
        lik_counter[v] = lik_counter.get(v, 0) + 1
print("\nتوزیعِ مقادیرِ likelihood در scp_codes (روی همه‌ی جفت‌های رکورد-کد):")
print(pd.Series(lik_counter).sort_index().to_frame("n").to_string())

extra_classes = sorted({v for v in code2class.values()} - set(SUPERCLASSES))
if extra_classes:
    warn("diagnostic_classهایی خارج از پنج ابرکلاس: " + str(extra_classes))


def to_classes(d, min_likelihood=None):
    out = set()
    for k, v in d.items():
        if k not in code2class:
            continue
        if min_likelihood is not None and not (float(v) > min_likelihood):
            continue
        out.add(code2class[k])
    return out


# قرارداد به‌کاررفته در v0.1_labels.csv: همه‌ی کلیدها بدون فیلترِ likelihood (قراردادِ بنچمارک)
LABEL_CONVENTION = "all keys of scp_codes mapped via diagnostic==1 (no likelihood threshold)"
rec_classes = parsed.apply(lambda d: to_classes(d, None))
lab = pd.DataFrame({c: rec_classes.apply(lambda s, c=c: int(c in s)) for c in SUPERCLASSES},
                   index=df.index)

print("\n--- تعدادِ رکورد در هر ابرکلاس (قرارداد: " + LABEL_CONVENTION + ") ---")
cmp_tab = pd.DataFrame({
    "derived": [int(lab[c].sum()) for c in SUPERCLASSES],
    "anchor_published": [ANCHORS[c] for c in SUPERCLASSES],
})
cmp_tab.index = SUPERCLASSES
cmp_tab["diff"] = cmp_tab["derived"] - cmp_tab["anchor_published"]
cmp_tab["prevalence_pct"] = np.round(cmp_tab["derived"] / len(df) * 100, 3)
print(cmp_tab.to_string())
if (cmp_tab["diff"] != 0).any():
    warn("اعدادِ مشتق‌شده با اعدادِ منتشرشده اختلاف دارند — علتِ محتمل: تغییرِ نسخه‌ی دیتاست "
         "یا تفاوتِ قرارداد. changelogهای بخش ۱ را در گزارش ذکر کن.")

n_labels = lab.sum(axis=1)
print("\nرکوردهای بدونِ هیچ ابرکلاس :", int((n_labels == 0).sum()),
      "({:.2f}%)".format((n_labels == 0).mean() * 100))
print("\nتوزیعِ تعدادِ برچسب به‌ازای هر رکورد:")
nl = n_labels.value_counts().sort_index().to_frame("n_records")
nl["pct"] = np.round(nl["n_records"] / len(df) * 100, 3)
print(nl.to_string())

# حساسیت به آستانه‌ی likelihood (فقط گزارش، روی خروجی اثر ندارد)
alt = parsed.apply(lambda d: to_classes(d, 0.0))
n_changed = int((rec_classes != alt).sum())
print("\nحساسیت: اگر فقط کدهای با likelihood>0 نگه داشته شوند،",
      n_changed, "رکورد ({:.2f}%) مجموعه‌ی برچسبش تغییر می‌کند.".format(n_changed / len(df) * 100))

labels_out = pd.concat([df[["patient_id", "strat_fold"]], lab], axis=1).reset_index()
save_csv(labels_out, "v0.1_labels.csv", index=False)


# =====================================================================================
# بخش ۷ — جدول‌های تقاطعی  (مهم‌ترین بخش)
# =====================================================================================
hdr("SECTION 7 — CROSSTABS (device / site  ×  fold و ابرکلاس)")


def cramers_v(ct):
    """Cramér's V خام و تصحیح‌شده‌ی Bergsma، بدونِ وابستگی به scipy."""
    obs = ct.values.astype(float)
    n = obs.sum()
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    exp = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum(np.where(exp > 0, (obs - exp) ** 2 / np.where(exp > 0, exp, 1.0), 0.0)))
    r, k = obs.shape
    dof = (r - 1) * (k - 1)
    v = float(np.sqrt(chi2 / (n * max(min(r, k) - 1, 1))))
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - dof / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = max(min(rc, kc) - 1, 1e-12)
    vc = float(np.sqrt(max(phi2c, 0.0) / denom))
    return {"n": int(n), "shape": (r, k), "chi2": chi2, "dof": dof,
            "cramers_v": v, "cramers_v_biascorrected": vc}


dfx = df.copy()
for c in ["device", "site", "nurse"]:
    dfx[c] = dfx[c].fillna("__MISSING__").astype(str)

CRAMER = {}

# ۷.۱ و ۷.۲ — device/site در برابر strat_fold
for col, fname, key in [("device", "v0.1_crosstab_device_fold.csv", "device_x_fold"),
                        ("site",   "v0.1_crosstab_site_fold.csv",   "site_x_fold")]:
    print("\n" + "-" * 100)
    print("--- " + col + " × strat_fold (شمارش) ---")
    ct = pd.crosstab(dfx[col], dfx["strat_fold"])
    ct_full = ct.copy()
    ct_full["TOTAL"] = ct.sum(axis=1)
    print(ct_full.to_string())
    print("\nسهمِ سطری (٪ هر " + col + " در هر fold):")
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(2).to_string())
    stats = cramers_v(ct)
    CRAMER[key] = stats
    print("\nCramér's V ({} × strat_fold): {:.4f}  | bias-corrected: {:.4f}  "
          "| chi2={:.2f}, dof={}, n={}, shape={}".format(
              col, stats["cramers_v"], stats["cramers_v_biascorrected"],
              stats["chi2"], stats["dof"], stats["n"], stats["shape"]))
    save_csv(ct_full, fname, index=True)

# ۷.۳ و ۷.۴ — نرخِ شیوعِ هر ابرکلاس درونِ هر device / site
overall_prev = lab.mean()
for col, fname in [("device", "v0.1_prevalence_device_superclass.csv"),
                   ("site",   "v0.1_prevalence_site_superclass.csv")]:
    print("\n" + "-" * 100)
    print("--- نرخِ شیوعِ ابرکلاس‌ها درونِ هر " + col + " ---")
    joined = dfx[[col]].join(lab)
    prev = joined.groupby(col)[SUPERCLASSES].mean()
    prev.insert(0, "n_records", joined.groupby(col).size())
    prev = prev.sort_values("n_records", ascending=False)
    disp = prev.copy()
    for c in SUPERCLASSES:
        disp[c] = (disp[c] * 100).round(2)
    overall_row = pd.DataFrame([[len(df)] + [round(overall_prev[c] * 100, 2) for c in SUPERCLASSES]],
                               index=["__OVERALL__"], columns=["n_records"] + SUPERCLASSES)
    disp = pd.concat([overall_row, disp])
    print("(اعداد بر حسب درصد؛ سطرِ __OVERALL__ خطِ مبنای کلِ دیتاست است)")
    print(disp.to_string())
    print("\nبیشینه‌ی فاصله از شیوعِ کلی، بر حسب واحدِ درصد:")
    spread = (prev[SUPERCLASSES].sub(overall_prev, axis=1).abs() * 100)
    print(spread.max().round(2).to_frame("max_abs_deviation_pp").to_string())
    save_csv(disp, fname, index=True)


# =====================================================================================
# بخش ۸ — MANIFEST
# =====================================================================================
hdr("SECTION 8 — MANIFEST")

manifest = {
    "code_version": CODE_VERSION,
    "run_started": RUN_STARTED,
    "run_finished": datetime.now().astimezone().isoformat(),
    "dataset_root": DATA_ROOT,
    "dataset_folder_name": os.path.basename(os.path.normpath(DATA_ROOT)),
    "output_dir": OUT_DIR,
    "input_files_sha256": INPUT_HASHES,
    "other_inputs_sha256": OTHER_HASHES,
    "library_versions": {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    },
    "dataset_summary": {
        "n_records": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "n_devices": int(df["device"].nunique(dropna=True)),
        "n_sites": int(df["site"].nunique(dropna=True)),
        "n_nurses": int(df["nurse"].nunique(dropna=True)),
        "superclass_counts": {c: int(lab[c].sum()) for c in SUPERCLASSES},
        "n_records_without_superclass": int((n_labels == 0).sum()),
    },
    "label_convention": LABEL_CONVENTION,
    "cramers_v": CRAMER,
    "checks": {
        "patient_disjoint_folds": "FAIL" if LEAKAGE_FAIL else "PASS",
        "n_patients_in_multiple_folds": int(len(violators)),
        "records_manifest_mismatch_csv_only": len(only_csv),
        "records_manifest_mismatch_records_only": len(only_rec),
    },
    "warnings": WARNINGS,
    "outputs": [],
}

for name in OUTPUT_FILES:
    p = os.path.join(OUT_DIR, name)
    manifest["outputs"].append({"file": name, "bytes": os.path.getsize(p), "sha256": sha256_of(p)})

MANIFEST_PATH = os.path.join(OUT_DIR, "v0.1_manifest.json")
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("[saved] " + MANIFEST_PATH)
print("SHA-256 v0.1_manifest.json:", sha256_of(MANIFEST_PATH))

print("\n--- محتوای manifest ---")
print(json.dumps(manifest, ensure_ascii=False, indent=2))


# =====================================================================================
# خلاصه‌ی پایانی (برای گزارشِ مرحله)
# =====================================================================================
hdr("SUMMARY — v0.1")
print("records / patients        :", len(df), "/", df["patient_id"].nunique())
print("device / site / nurse     :", df["device"].nunique(dropna=True), "/",
      df["site"].nunique(dropna=True), "/", df["nurse"].nunique(dropna=True))
print("age sentinel (>=120)      :", int((df['age'] >= 120).sum()))
print("بدونِ ابرکلاس              :", int((n_labels == 0).sum()))
print("چندبرچسبی (>=2)           :", int((n_labels >= 2).sum()))
print("patient-disjoint folds    :", "FAIL" if LEAKAGE_FAIL else "PASS")
print("Cramér's V device×fold    : {:.4f} (corrected {:.4f})".format(
    CRAMER["device_x_fold"]["cramers_v"], CRAMER["device_x_fold"]["cramers_v_biascorrected"]))
print("Cramér's V site×fold      : {:.4f} (corrected {:.4f})".format(
    CRAMER["site_x_fold"]["cramers_v"], CRAMER["site_x_fold"]["cramers_v_biascorrected"]))
print("فایل‌های تولیدشده          :", len(OUTPUT_FILES) + 1)
print("\nهشدارها ({}):".format(len(WARNINGS)))
for w in WARNINGS:
    print("  - " + w)

if LEAKAGE_FAIL:
    raise AssertionError(
        "نشتِ سطح-بیمار: {} بیمار در بیش از یک strat_fold ظاهر شده‌اند. "
        "خروجی‌های ممیزی نوشته شدند؛ قبل از هر مدل‌سازی این را حل کن.".format(len(violators)))

print("\nv0.1 با موفقیت تمام شد.")
