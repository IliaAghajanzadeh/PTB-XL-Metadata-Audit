# =====================================================================================
# Project: PTB-XL Metadata Audit  -  make_flowchart.py: the study-design flowchart (new Fig. 1 of the paper)
# Code version: flowchart-1.1 (FLOWCHART_VERSION; 1.1 = core-chat review: new wording of meta_head, h1h2, lr, final
#         and meta_f10; meta_f10 divided like h3e (two dividers at the same y); C7 hardened (arrow x from the box
#         geometry, not from the drawing table); C2 34 lines, C3 5 italic lines, C4/C5/C6/C11 for two dividers)
# Date: 2026-10-09
#
# Input : ONLY ROOT\outputs\verify\run_20261008-022605\paper_numbers.json (sha256 pinned in EXPECTED_PN_SHA256;
#         verify.all_checks_passed and verify.paper_final must be true). Four numbers of the figure (21,799 records,
#         18,869 patients, 370 checks, margin 0.01, 18 final models) are built from that file and the resulting lines
#         are asserted equal to the specified strings. Nothing else is read.
# Output: ROOT\outputs\figures\flowchart-1.0_<YYYYmmdd-HHMMSS>\ (created new; STOP if it exists; every file
#         exclusive-create): flowchart.pdf (vector, TrueType fonts), flowchart.png (600 dpi RGB),
#         flowchart_preview_A4.pdf, flowchart_layout.json, flowchart_manifest.json.
#         Self-checks C1-C11 are computed BEFORE any file is written; if one fails nothing is written. C12 (files on
#         disk == in-memory bytes, all recorded in the manifest) necessarily runs after writing.
# Run   : python make_flowchart.py --root D:\ptbxl-metadata-audit
#         or paste the whole file into ONE Jupyter cell (ROOT defaults to D:\ptbxl-metadata-audit).
#         Exit code 1 (script) or a FAIL banner (Jupyter) if any self-check fails.
# Layout: one IEEE column (85.8 mm), two lanes (left: metadata-only models; right: ECG networks), 8 rows, 11 boxes,
#         12 vertical downward arrows; all geometry in millimetres (data units = mm, y downward).
# Font  : Times New Roman 8 pt (fallback Liberation Serif -> TeX Gyre Termes -> DejaVu Serif, with a WARN).
# Provenance: blocks copied from make_figures.py (figures-1.1, sha256 ebc299bf...6e43) are marked
#         '# verbatim from make_figures.py <function>' or '# adapted from make_figures.py <function>: <change>'.
# Dependencies: matplotlib, numpy, Pillow, fontTools (shipped with matplotlib) and the standard library.
# =====================================================================================

import os, sys, io, re, json, math, hashlib, argparse, platform
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from matplotlib.text import Text
from matplotlib import font_manager
from PIL import Image
from fontTools.ttLib import TTFont

FLOWCHART_VERSION = "flowchart-1.1"
DEFAULT_ROOT = r"D:\ptbxl-metadata-audit"
EXPECTED_PN_SHA256 = "e540d16d65d9247669e8488bcc834ea854cdd5bf94f0181f3c9ceb6ff49e524c"
PN_RELPATH = ("outputs", "verify", "run_20261008-022605", "paper_numbers.json")
EXPECTED_VERSIONS = {"python": "3.10.20", "matplotlib": "3.10.9", "pillow": "12.3.0", "numpy": "1.26.4"}

# ------------------------------------- specification constants (prompt section 5) ----------------------------------
MM = 1.0 / 25.4                                   # inches per mm
DPI = 600
W = 85.8                                          # canvas width (one IEEE column)
M = 0.4                                           # outer margin
GAP = 3.6                                         # row bottom edge -> next row top edge
LANE_GAP = 4.2
LANE_W = (W - 2 * M - LANE_GAP) / 2.0             # 40.4
LANE_X = {"left": (M, M + LANE_W), "right": (M + LANE_W + LANE_GAP, W - M), "full": (M, W - M)}
X_LC = M + LANE_W / 2.0                           # Lc = 20.6
X_RC = M + LANE_W + LANE_GAP + LANE_W / 2.0       # Rc = 65.2
H_MAX = 125.0
FONT_SIZE = 8.0
PITCH = 1.15 * FONT_SIZE / 72.0 * 25.4            # 3.2456 mm, baseline to baseline
VPAD = 1.4
HEAD_L, HEAD_W = 1.2, 1.0
EDGE_LW, DIVIDER_LW, ARROW_LW = 0.6, 0.4, 0.6     # pt
INK, EDGE, GATE, WHITE = "#1a1a1a", "#333333", "#e9e9e9", "white"
FONT_CANDIDATES = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
N_ROWS = 8
# self-check thresholds (prompt section 7)
EXACT = 1e-9
PAD_H_MIN, CENTER_TOL = 2.0, 0.10
PAD_V_MIN, PAD_V_DIFF, DIVIDER_MIN = 1.0, 0.20, 1.0
CLEARANCE = 0.3
SHAFT_MIN = 2.0
INSIDE_MIN = 0.2
FRAME_MM = 0.15
MEDIABOX_TOL = 0.05
# A4 preview (prompt section 3)
A4_W, A4_H = 210.0, 297.0
PREVIEW_X, PREVIEW_TOP = 16.0, 30.0
GUIDES_X = (16.0, 101.8)
GUIDE_COLOR, GUIDE_LW = "#c8c8c8", 0.3

# ------------------------------------- content (prompt section 4), exactly as specified ----------------------------
# (box id, row, lane, gate?, sections); a section is a list of (line, italic); h3e has two sections
SPEC_BOXES = [
    ("data", 0, "full", False, [[("PTB-XL v1.0.3 (100 Hz): 21,799 ECGs from 18,869 patients", False),
                                 ("five diagnostic superclasses; patient-disjoint folds 1\u201310", False)]]),
    ("meta_head", 1, "left", False, [[("Metadata-only models", False), ("logistic regression and", False),
                                      ("gradient boosting", False), ("(blocks B1\u2013B5)", False)]]),
    ("ecg_head", 1, "right", False, [[("ECG networks", False), ("ResNet1d-Wang, Inception1d", False),
                                      ("with no metadata; + age, sex;", False), ("+ age, sex, acquisition", False)]]),
    ("h1h2", 2, "left", False, [[("H1, H2 evaluated on fold 9", False), ("(trained on folds 1\u20138)", False),
                                 ("confirmatory after the fact", True)]]),
    ("lr", 2, "right", False, [[("Learning rate tuned on fold 9", False), ("(trained on folds 1\u20138)", False),
                                ("with the + age, sex arm", False)]]),
    ("frp", 3, "right", True, [[("Final-run protocol sealed", False), ("and anchored in Bitcoin", False)]]),
    ("final", 4, "right", False, [[("18 final models (folds 1\u20139);", False),
                                   ("manifest anchored in Bitcoin", False)]]),
    ("open", 5, "full", True, [[("Fold 10 opened once; outputs hashed before any label was read;", False),
                                ("no ECG network trained afterwards", False)]]),
    ("meta_f10", 6, "left", False, [[("Metadata-only models", False), ("(trained on folds 1\u20139)", False),
                                     ("post hoc", True)],
                                    [("H2 contrast (B3 \u2212 B1)", False), ("post hoc", True)]]),
    ("h3e", 6, "right", False, [[("H3: + age, sex, acquisition", False), ("vs. + age, sex; margin \u00b10.01", False),
                                 ("pre-registered", True)],
                                [("E1\u2013E3", False), ("exploratory", True)]]),
    ("verify", 7, "full", False, [[("Read-only script recomputes every reported statistic", False),
                                   ("from the locked artifacts (370 checks)", False)]]),
]
STATUS_LINES = {"confirmatory after the fact", "post hoc", "pre-registered", "exploratory"}
# (arrow id, source box, target box, x)
SPEC_ARROWS = [("a1", "data", "meta_head", "Lc"), ("a2", "data", "ecg_head", "Rc"), ("a3", "meta_head", "h1h2", "Lc"),
               ("a4", "ecg_head", "lr", "Rc"), ("a5", "lr", "frp", "Rc"), ("a6", "frp", "final", "Rc"),
               ("a7", "final", "open", "Rc"), ("a8", "h1h2", "open", "Lc"), ("a9", "open", "meta_f10", "Lc"),
               ("a10", "open", "h3e", "Rc"), ("a11", "meta_f10", "verify", "Lc"), ("a12", "h3e", "verify", "Rc")]
X_OF = {"Lc": X_LC, "Rc": X_RC}


# verbatim from make_figures.py in_ipython
def in_ipython():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False


IN_IPYTHON = in_ipython()


# verbatim from make_figures.py hdr
def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# verbatim from make_figures.py sha256_bytes
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# verbatim from make_figures.py sha256_of
def sha256_of(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


# verbatim from make_figures.py write_once
def write_once(path, data):
    """Exclusive create; never overwrites. Returns the sha256 of the bytes on disk."""
    with open(path, "xb") as f:
        f.write(data)
    got = sha256_of(path)
    assert got == sha256_bytes(data), "write verification failed for " + path
    return got


# verbatim from make_figures.py px2mm
def px2mm(v):
    return v / DPI * 25.4


# verbatim from make_figures.py mm2px
def mm2px(v):
    return v * MM * DPI


CHECKS = []


# verbatim from make_figures.py check
def check(cid, description, ok, detail):
    CHECKS.append({"id": cid, "description": description, "status": "PASS" if ok else "FAIL", "detail": detail})


# =====================================================================================
# 0. Arguments, input, refusals (C1)
# =====================================================================================
hdr("make_flowchart {} | {}".format(FLOWCHART_VERSION, datetime.now().astimezone().isoformat()))
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--root", default=DEFAULT_ROOT)
ARGS = _ap.parse_known_args()[0]
ROOT = ARGS.root
PN_PATH = os.path.join(ROOT, *PN_RELPATH)
if not os.path.isfile(PN_PATH):
    raise RuntimeError("STOP: {} not found".format(PN_PATH))
with open(PN_PATH, "rb") as _f:
    PN_BYTES = _f.read()
PN_SHA256 = sha256_bytes(PN_BYTES)
PN = json.loads(PN_BYTES.decode("utf-8"))
print("input   : {}\nsha256  : {}".format(PN_PATH, PN_SHA256))
_refuse = []
if PN_SHA256 != EXPECTED_PN_SHA256:
    _refuse.append("sha256 {} != EXPECTED_PN_SHA256 {}".format(PN_SHA256, EXPECTED_PN_SHA256))
if PN.get("verify", {}).get("all_checks_passed") is not True:
    _refuse.append("verify.all_checks_passed is not true")
if PN.get("verify", {}).get("paper_final") is not True:
    _refuse.append("verify.paper_final is not true")
if _refuse:
    raise RuntimeError("REFUSED: " + "; ".join(_refuse))
print("sha256 matches the verified paper_numbers.json; all_checks_passed and paper_final are true")

# --- the four number substrings, built from the JSON ---
_ds, _ls = PN["dataset"], PN["locked_settings"]
N_RECORDS = "{:,}".format(_ds["n_records"])
N_PATIENTS = "{:,}".format(_ds["n_patients"])
N_CHECKS = "{:d}".format(PN["verify"]["n_checks"])
_sesoi = PN["h3"]["sesoi_h3"]
assert _sesoi == _ls["SESOI_H3"]["value"], "h3.sesoi_h3 != locked_settings.SESOI_H3.value"
SESOI = "{:g}".format(_sesoi)
assert float(SESOI) == _sesoi
N_MODELS = "{:d}".format(len(PN["ecg_fold10"]) * len(PN["ecg_fold10"]["resnet1d_wang"]) *
                         len(_ls["seeds_signal_models"]["value"]))
BUILT = {
    ("data", 0, 0): "PTB-XL v1.0.3 (100 Hz): {} ECGs from {} patients".format(N_RECORDS, N_PATIENTS),
    ("final", 0, 0): "{} final models (folds 1\u20139);".format(N_MODELS),
    ("h3e", 0, 1): "vs. + age, sex; margin \u00b1{}".format(SESOI),
    ("verify", 0, 1): "from the locked artifacts ({} checks)".format(N_CHECKS),
}
BOXES_CONTENT = []                                 # the content that is drawn: specification lines, JSON-built numbers
_c1 = []
for _bid, _row, _lane, _gate, _secs in SPEC_BOXES:
    _new = []
    for _si, _sec in enumerate(_secs):
        _ns = []
        for _li, (_txt, _it) in enumerate(_sec):
            _b = BUILT.get((_bid, _si, _li))
            if _b is not None:
                _c1.append({"box": _bid, "built": _b, "specified": _txt, "equal": _b == _txt})
                _txt = _b
            _ns.append((_txt, _it))
        _new.append(_ns)
    BOXES_CONTENT.append((_bid, _row, _lane, _gate, _new))
_numbers = {"n_records": N_RECORDS, "n_patients": N_PATIENTS, "n_checks": N_CHECKS, "sesoi_h3": SESOI,
            "n_final_models": N_MODELS}
check("C1", "input: sha256 pinned, all_checks_passed and paper_final true; the number substrings are built from "
      "paper_numbers.json and the resulting lines equal the specified strings", len(_c1) == 4 and
      all(x["equal"] for x in _c1) and _numbers == {"n_records": "21,799", "n_patients": "18,869", "n_checks": "370",
                                                    "sesoi_h3": "0.01", "n_final_models": "18"},
      {"sha256": PN_SHA256, "numbers": _numbers, "lines": _c1})

# --- versions (WARN only) ---
VERSIONS = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
            "pillow": __import__("PIL").__version__, "numpy": np.__version__, "os": platform.platform()}
WARNINGS = []
for _k, _e in EXPECTED_VERSIONS.items():
    if VERSIONS[_k] != _e:
        WARNINGS.append("{} {} != {} (renderer metrics may differ; the self-checks decide)".format(_k, VERSIONS[_k], _e))
print("versions: {}".format(VERSIONS))

# verbatim from make_figures.py font-resolution loop
FONT_NAME, FONT_PATH = None, None
for _name in FONT_CANDIDATES:
    try:
        _p = font_manager.findfont(font_manager.FontProperties(family=_name, weight="normal", style="normal"),
                                   fallback_to_default=False)
    except ValueError:
        continue
    if font_manager.FontProperties(fname=_p).get_name() == _name:
        FONT_NAME, FONT_PATH = _name, _p
        break
if FONT_NAME is None:
    raise RuntimeError("STOP: none of the fonts {} is installed".format(FONT_CANDIDATES))
print("font    : {} ({})".format(FONT_NAME, FONT_PATH))
if FONT_NAME != FONT_CANDIDATES[0]:
    WARNINGS.append("font fallback: {} is used instead of Times New Roman".format(FONT_NAME))
for _w in WARNINGS:
    print("[WARN] " + _w)
# the italic face of the same family (status lines)
ITALIC_PATH = font_manager.findfont(font_manager.FontProperties(family=FONT_NAME, weight="normal", style="italic"),
                                    fallback_to_default=False)
_itf = font_manager.get_font(ITALIC_PATH)
if _itf.family_name != FONT_NAME or "italic" not in _itf.style_name.lower():
    raise RuntimeError("STOP: no italic face of {} is installed (found {} {})".format(FONT_NAME, _itf.family_name,
                                                                                      _itf.style_name))
print("italic  : {} {} ({})".format(_itf.family_name, _itf.style_name, ITALIC_PATH))

# verbatim from make_figures.py RC
RC = {"font.family": [FONT_NAME], "font.size": FONT_SIZE, "font.weight": "normal", "font.style": "normal",
      "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False, "axes.unicode_minus": False,
      "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
      "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE, "legend.title_fontsize": FONT_SIZE,
      "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt", "savefig.facecolor": "white",
      "figure.facecolor": "white", "axes.facecolor": "white", "pdf.compression": 6, "svg.fonttype": "none"}


# =====================================================================================
# 1. Geometry (prompt section 5)
# =====================================================================================
def box_height(sections):
    n = sum(len(s) for s in sections)
    return n * PITCH + 2 * VPAD * len(sections)        # ordinary: n*PITCH + 2*VPAD; h3e: 5*PITCH + 4*VPAD


ROW_H = [max(box_height(secs) for _, row, _, _, secs in BOXES_CONTENT if row == r) for r in range(N_ROWS)]
ROW_TOP = []
_y = M
for r in range(N_ROWS):
    ROW_TOP.append(_y)
    _y += ROW_H[r] + (GAP if r < N_ROWS - 1 else 0.0)
H = M + sum(ROW_H) + (N_ROWS - 1) * GAP + M
GEOM = {}                                          # box id -> geometry
for _bid, _row, _lane, _gate, _secs in BOXES_CONTENT:
    x0, x1 = LANE_X[_lane]
    yt, yb = ROW_TOP[_row], ROW_TOP[_row] + ROW_H[_row]
    if len(_secs) == 1:
        sec_bounds = [(yt, yb)]
    else:                                          # h3e: section A = 3 lines, section B = 2 lines
        assert abs(box_height(_secs) - ROW_H[_row]) <= EXACT, "a divided box is not the tallest box of its row"
        yd = yt + len(_secs[0]) * PITCH + 2 * VPAD     # meta_f10 and h3e: 3 + 2 lines -> the same divider y
        sec_bounds = [(yt, yd), (yd, yb)]
    GEOM[_bid] = {"id": _bid, "row": _row, "lane": _lane, "gate": _gate, "x0": x0, "x1": x1, "y_top": yt,
                  "y_bottom": yb, "xc": (x0 + x1) / 2.0, "sections": _secs, "section_bounds": sec_bounds,
                  "divider_y": sec_bounds[0][1] if len(_secs) > 1 else None}
ARROWS = []
for _aid, _src, _dst, _xn in SPEC_ARROWS:
    ARROWS.append({"id": _aid, "source": _src, "target": _dst, "x_name": _xn, "x": X_OF[_xn],
                   "y_start": GEOM[_src]["y_bottom"], "y_tip": GEOM[_dst]["y_top"],
                   "y_head_base": GEOM[_dst]["y_top"] - HEAD_L})
print("H = {:.4f} mm (limit {} mm) | row heights {} | LANE_W {:.4f} | Lc {:.4f} | Rc {:.4f}".format(
    H, H_MAX, [round(h, 4) for h in ROW_H], LANE_W, X_LC, X_RC))
if H > H_MAX:
    raise RuntimeError("FAIL: H = {:.4f} mm > {} mm (fonts and gaps are not changed)".format(H, H_MAX))


# =====================================================================================
# 2. Drawing (the same function draws the figure and the A4 preview)
# =====================================================================================
# verbatim from make_figures.py new_figure
def new_figure(w_mm, h_mm):
    fig = Figure(figsize=(w_mm * MM, h_mm * MM), dpi=DPI)
    FigureCanvasAgg(fig)
    return fig


# verbatim from make_figures.py renderer_of
def renderer_of(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def mm_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                              # y grows downward
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def text_extent_mm(ax, t, r):
    """Rendered extent of a text in mm (x0, x1, y_top, y_bottom; y downward)."""
    bb = t.get_window_extent(r)
    (xa, ya), (xb, yb) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)


def draw_flowchart(ax, baselines=None):
    """Draws boxes, dividers, arrows and texts. baselines: {(box, section, line): y} from an earlier layout pass; if
    None, the text blocks are centred with the rendered extents (one measured translation per section)."""
    arts = {"boxes": {}, "dividers": {}, "shafts": {}, "heads": {}, "texts": []}
    for g in GEOM.values():
        rect = Rectangle((g["x0"], g["y_top"]), g["x1"] - g["x0"], g["y_bottom"] - g["y_top"],
                         facecolor=GATE if g["gate"] else WHITE, edgecolor=EDGE, linewidth=EDGE_LW,
                         joinstyle="miter", zorder=1, gid="box/" + g["id"])
        ax.add_patch(rect)
        arts["boxes"][g["id"]] = rect
        if g["divider_y"] is not None:
            ln = Line2D([g["x0"], g["x1"]], [g["divider_y"], g["divider_y"]], linewidth=DIVIDER_LW, color=EDGE,
                        solid_capstyle="butt", zorder=1.5, gid="divider/" + g["id"])
            ax.add_line(ln)
            arts["dividers"][g["id"]] = ln
    for a in ARROWS:
        x = a["x"]
        sh = Line2D([x, x], [a["y_start"], a["y_head_base"]], linewidth=ARROW_LW, color=EDGE, solid_capstyle="butt",
                    zorder=2, gid="arrow/{}/shaft".format(a["id"]))
        ax.add_line(sh)
        hd = Polygon([(x - HEAD_W / 2.0, a["y_head_base"]), (x + HEAD_W / 2.0, a["y_head_base"]), (x, a["y_tip"])],
                     closed=True, facecolor=EDGE, edgecolor="none", linewidth=0, zorder=2,
                     gid="arrow/{}/head".format(a["id"]))
        ax.add_patch(hd)
        arts["shafts"][a["id"]], arts["heads"][a["id"]] = sh, hd
    for g in GEOM.values():
        for si, (sec, (st, sb)) in enumerate(zip(g["sections"], g["section_bounds"])):
            n = len(sec)
            for li, (txt, italic) in enumerate(sec):
                if baselines is not None:
                    y = baselines[(g["id"], si, li)]
                else:                              # provisional: block of n pitches centred in the section
                    y = (st + sb) / 2.0 - n * PITCH / 2.0 + (li + 0.8) * PITCH
                t = ax.text(g["xc"], y, txt, ha="center", va="baseline", fontsize=FONT_SIZE, fontweight="normal",
                            fontstyle="italic" if italic else "normal", color=INK, zorder=3,
                            gid="text/{}/{}/{}".format(g["id"], si, li))
                arts["texts"].append({"box": g["id"], "section": si, "line": li, "text": txt, "italic": italic,
                                      "artist": t})
    if baselines is None:                          # one translation per section: centre the rendered extents
        r = renderer_of(ax.figure)
        for g in GEOM.values():
            for si, (st, sb) in enumerate(g["section_bounds"]):
                ts = [d for d in arts["texts"] if d["box"] == g["id"] and d["section"] == si]
                ext = [text_extent_mm(ax, d["artist"], r) for d in ts]
                shift = (st + sb) / 2.0 - (min(e[2] for e in ext) + max(e[3] for e in ext)) / 2.0
                for d in ts:
                    x, y = d["artist"].get_position()
                    d["artist"].set_position((x, y + shift))
    return arts


# verbatim from make_figures.py render_rgb
def render_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    assert (buf[:, :, 3] == 255).all(), "the rendered figure has transparent pixels"
    return Image.fromarray(buf[:, :, :3].copy(), mode="RGB")


# verbatim from make_figures.py png_bytes
def png_bytes(img, mode=None):
    b = io.BytesIO()
    (img.convert(mode) if mode else img).save(b, format="PNG", dpi=(DPI, DPI))
    return b.getvalue()


# adapted from make_figures.py pdf_bytes: Creator = FLOWCHART_VERSION
def pdf_bytes(fig):
    b = io.BytesIO()
    fig.savefig(b, format="pdf", dpi=DPI, facecolor="white", metadata={"Creator": FLOWCHART_VERSION,
                                                                       "CreationDate": None})
    return b.getvalue()


# =====================================================================================
# 3. Self-check helpers
# =====================================================================================
def rect_seg_dist(e, seg, half_lw):
    """Distance (mm) from a text extent e = (x0, x1, yt, yb) to an axis-parallel stroked segment
    seg = (xa, ya, xb, yb) with half stroke width half_lw (negative = overlap)."""
    xa, ya, xb, yb = seg
    sx0, sx1, sy0, sy1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
    dx = max(0.0, sx0 - e[1], e[0] - sx1)
    dy = max(0.0, sy0 - e[3], e[2] - sy1)
    return math.hypot(dx, dy) - half_lw


def rect_rect_dist(e, f):
    dx = max(0.0, f[0] - e[1], e[0] - f[1])
    dy = max(0.0, f[2] - e[3], e[2] - f[3])
    return math.hypot(dx, dy)


def pt2mm(v):
    return v / 72.0 * 25.4


# adapted from make_figures.py check_size: PDF font-embedding test kept; MediaBox and image-XObject tests added
def pdf_font_and_box(pdf_b, w_mm, h_mm):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf_b)
    wpd, hpd = (float(m.group(1)) / 72 * 25.4, float(m.group(2)) / 72 * 25.4) if m else (None, None)
    fonts_ok = b"/FontFile2" in pdf_b and b"/FontFile3" not in pdf_b and b"/Subtype /Type1" not in pdf_b \
        and b"/Subtype /Type3" not in pdf_b
    images = len(re.findall(rb"/Subtype\s*/Image", pdf_b))
    box_ok = m is not None and abs(wpd - w_mm) <= MEDIABOX_TOL and abs(hpd - h_mm) <= MEDIABOX_TOL
    return box_ok, fonts_ok, images, wpd, hpd


# adapted from make_figures.py FONT_CMAP / check_fonts: glyph coverage of the upright AND the italic face
CMAPS = {os.path.normcase(FONT_PATH): set(TTFont(FONT_PATH, fontNumber=0).getBestCmap().keys()),
         os.path.normcase(ITALIC_PATH): set(TTFont(ITALIC_PATH, fontNumber=0).getBestCmap().keys())}


# =====================================================================================
# 4. Main
# =====================================================================================
with matplotlib.rc_context(RC):
    hdr("Drawing")
    FIG = new_figure(W, H)
    AX = mm_axes(FIG, [0, 0, 1, 1])
    ARTS = draw_flowchart(AX)
    R = renderer_of(FIG)
    BASELINES = {(d["box"], d["section"], d["line"]): d["artist"].get_position()[1] for d in ARTS["texts"]}
    EXT = {(d["box"], d["section"], d["line"]): text_extent_mm(AX, d["artist"], R) for d in ARTS["texts"]}

    hdr("Self-checks C2-C11")
    # ---- C2 text inventory ----
    expected_inv = [(bid, txt) for bid, _, _, _, secs in BOXES_CONTENT for sec in secs for txt, _ in sec]
    drawn = [t for t in AX.texts if t.get_visible() and t.get_text() != ""]
    drawn_inv = [(t.get_gid().split("/")[1], t.get_text()) for t in drawn]
    others = [t.get_text() for t in FIG.texts if t.get_text()] + \
        [t.get_text() for t in (AX.title, AX._left_title, AX._right_title) if t.get_text()] + \
        ([AX.get_legend()] if AX.get_legend() else []) + list(FIG.legends)
    _axis_owned = {id(t) for t in AX.xaxis.findobj(Text) + AX.yaxis.findobj(Text)}
    stray = [t.get_text() for t in FIG.findobj(Text) if t.get_visible() and t.get_text() != ""
             and id(t) not in _axis_owned and t not in AX.texts]
    check("C2", "text inventory: the visible Text artists, in drawing order, equal the 34 specified lines (string, box, "
          "order); no other text, legend or title; the axis is off", drawn_inv == expected_inv and len(drawn_inv) == 34
          and not others and not stray and not AX.axison,
          {"n_drawn": len(drawn_inv), "n_expected": len(expected_inv), "axis_off": not AX.axison, "other_texts": others, "stray_texts": stray,
           "first_mismatch": next(([i, a, b] for i, (a, b) in enumerate(zip(drawn_inv, expected_inv)) if a != b), None)})

    # ---- C3 font ----
    bad3 = []
    for d in ARTS["texts"]:
        t, fp = d["artist"], d["artist"].get_fontproperties()
        path = os.path.normcase(font_manager.findfont(fp, fallback_to_default=False))
        want_italic = d["text"] in STATUS_LINES
        want_path = os.path.normcase(ITALIC_PATH if want_italic else FONT_PATH)
        missing = sorted({hex(ord(c)) for c in d["text"] if ord(c) not in CMAPS.get(path, set())})
        if fp.get_size_in_points() != FONT_SIZE or fp.get_weight() not in ("normal", 400) or \
                fp.get_style() != ("italic" if want_italic else "normal") or d["italic"] != want_italic or \
                path != want_path or missing or "$" in d["text"]:
            bad3.append({"text": d["text"], "size": fp.get_size_in_points(), "weight": fp.get_weight(),
                         "style": fp.get_style(), "font": path, "missing_glyphs": missing})
    n_italic = sum(d["artist"].get_fontproperties().get_style() == "italic" for d in ARTS["texts"])
    used = sorted({c for d in ARTS["texts"] for c in d["text"]})
    check("C3", "font: every text 8.0 pt, weight normal; exactly the lines whose text is a status string italic (5 "
          "instances: 'post hoc' twice; italic face of the same family), all others normal; glyph coverage complete "
          "(incl. U+2013, U+2212, U+00B1, ';')", not bad3 and n_italic == 5,
          {"font": FONT_NAME, "upright": FONT_PATH, "italic": ITALIC_PATH, "n_italic": n_italic, "violations": bad3,
           "n_distinct_characters": len(used)})

    # ---- C4 grid ----
    bad4 = []
    for g in GEOM.values():
        rect = ARTS["boxes"][g["id"]]
        lx0, lx1 = LANE_X[g["lane"]]
        if abs(g["x0"] - lx0) > EXACT or abs(g["x1"] - lx1) > EXACT:
            bad4.append((g["id"], "lane x"))
        if abs(rect.get_x() - g["x0"]) > EXACT or abs(rect.get_x() + rect.get_width() - g["x1"]) > EXACT or \
                abs(rect.get_y() - g["y_top"]) > EXACT or abs(rect.get_y() + rect.get_height() - g["y_bottom"]) > EXACT:
            bad4.append((g["id"], "artist != geometry"))
        if abs(g["y_top"] - ROW_TOP[g["row"]]) > EXACT or abs(g["y_bottom"] - ROW_TOP[g["row"]] - ROW_H[g["row"]]) > EXACT:
            bad4.append((g["id"], "row y"))
    full_ok = all(abs(GEOM[b]["x0"] - M) <= EXACT and abs(GEOM[b]["x1"] - (W - M)) <= EXACT
                  for b in GEOM if GEOM[b]["lane"] == "full")
    div_info = {}
    for bid, ln in ARTS["dividers"].items():
        g, xs_d, ys_d = GEOM[bid], list(ln.get_xdata()), list(ln.get_ydata())
        div_info[bid] = {"y": ys_d[0], "x": xs_d}
        if ys_d[0] != ys_d[1] or abs(ys_d[0] - g["divider_y"]) > EXACT or abs(xs_d[0] - g["x0"]) > EXACT or \
                abs(xs_d[1] - g["x1"]) > EXACT:
            bad4.append((bid, "divider not horizontal / not spanning its box x0..x1"))
    div_ys = [v["y"] for v in div_info.values()]
    if len(div_ys) != 2 or max(div_ys) - min(div_ys) > EXACT:
        bad4.append(("dividers", "not exactly two dividers at identical y"))
    gaps = [ROW_TOP[r + 1] - (ROW_TOP[r] + ROW_H[r]) for r in range(N_ROWS - 1)]
    gaps_ok = all(abs(gp - GAP) <= EXACT for gp in gaps)
    check("C4", "grid: left-lane boxes share x0/x1, right-lane boxes share x0/x1, full boxes span M..W-M (<= 1e-9 mm); "
          "boxes in a row share y_top / y_bottom; every row gap == GAP; both dividers at identical y (<= 1e-9) and "
          "spanning exactly their box x0..x1", not bad4 and full_ok and gaps_ok,
          {"lanes": {k: [round(v[0], 6), round(v[1], 6)] for k, v in LANE_X.items()}, "violations": bad4,
           "row_gaps_mm": [round(gp, 12) for gp in gaps], "dividers": div_info})

    # ---- C5 text fit ----
    bad5, min_h, max_hdiff, min_v, max_vdiff, min_div = [], 1e9, 0.0, 1e9, 0.0, 1e9
    for d in ARTS["texts"]:
        g, e = GEOM[d["box"]], EXT[(d["box"], d["section"], d["line"])]
        lp, rp = e[0] - g["x0"], g["x1"] - e[1]
        min_h, max_hdiff = min(min_h, lp, rp), max(max_hdiff, abs(lp - rp))
        if lp < PAD_H_MIN or rp < PAD_H_MIN or abs(lp - rp) > CENTER_TOL:
            bad5.append((d["text"], "horizontal", round(lp, 3), round(rp, 3)))
        if g["divider_y"] is not None:
            dd = min(abs(e[2] - g["divider_y"]), abs(e[3] - g["divider_y"]))
            min_div = min(min_div, dd)
            if dd < DIVIDER_MIN:
                bad5.append((d["text"], "divider", round(dd, 3)))
    sec_pads = []
    for g in GEOM.values():
        for si, (st, sb) in enumerate(g["section_bounds"]):
            ex = [EXT[(g["id"], si, li)] for li in range(len(g["sections"][si]))]
            tp, bp = min(e[2] for e in ex) - st, sb - max(e[3] for e in ex)
            sec_pads.append({"box": g["id"], "section": si, "top_mm": round(tp, 4), "bottom_mm": round(bp, 4)})
            min_v, max_vdiff = min(min_v, tp, bp), max(max_vdiff, abs(tp - bp))
            if tp < PAD_V_MIN or bp < PAD_V_MIN or abs(tp - bp) > PAD_V_DIFF:
                bad5.append((g["id"], si, "vertical", round(tp, 3), round(bp, 3)))
    check("C5", "text fit (rendered extents, mm): every line >= 2.0 mm from both box edges and centred (|left - right| "
          "<= 0.10); every box / section: top and bottom padding >= 1.0 and differing by <= 0.20; in both divided "
          "boxes (meta_f10, h3e) every line >= 1.0 from the divider", not bad5,
          {"min_horizontal_pad_mm": round(min_h, 4), "max_horizontal_asymmetry_mm": round(max_hdiff, 6),
           "min_vertical_pad_mm": round(min_v, 4), "max_vertical_pad_difference_mm": round(max_vdiff, 6),
           "min_distance_to_divider_mm": round(min_div, 4), "sections": sec_pads, "violations": bad5})

    # ---- C6 no collisions ----
    keys = list(EXT)
    text_overlaps = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = EXT[keys[i]], EXT[keys[j]]
            if min(a[1], b[1]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[2], b[2]) > 0:
                text_overlaps.append((keys[i], keys[j]))
    hl_box, hl_div, hl_arr = pt2mm(EDGE_LW) / 2, pt2mm(DIVIDER_LW) / 2, pt2mm(ARROW_LW) / 2
    segments = []
    for g in GEOM.values():
        segments += [("box {} top".format(g["id"]), (g["x0"], g["y_top"], g["x1"], g["y_top"]), hl_box),
                     ("box {} bottom".format(g["id"]), (g["x0"], g["y_bottom"], g["x1"], g["y_bottom"]), hl_box),
                     ("box {} left".format(g["id"]), (g["x0"], g["y_top"], g["x0"], g["y_bottom"]), hl_box),
                     ("box {} right".format(g["id"]), (g["x1"], g["y_top"], g["x1"], g["y_bottom"]), hl_box)]
        if g["divider_y"] is not None:
            segments.append(("divider " + g["id"], (g["x0"], g["divider_y"], g["x1"], g["divider_y"]), hl_div))
    for a in ARROWS:
        segments.append(("shaft " + a["id"], (a["x"], a["y_start"], a["x"], a["y_head_base"]), hl_arr))
    heads = [("head " + a["id"], (a["x"] - HEAD_W / 2, a["x"] + HEAD_W / 2, a["y_head_base"], a["y_tip"]))
             for a in ARROWS]
    min_clear, too_close = (1e9, None), []
    for k, e in EXT.items():
        for name, seg, hl in segments:
            dist = rect_seg_dist(e, seg, hl)
            if dist < min_clear[0]:
                min_clear = (dist, k, name)
            if dist < CLEARANCE:
                too_close.append((k, name, round(dist, 3)))
        for name, f in heads:
            dist = rect_rect_dist(e, f)
            if dist < min_clear[0]:
                min_clear = (dist, k, name)
            if dist < CLEARANCE:
                too_close.append((k, name, round(dist, 3)))
    check("C6", "no collisions: no two text extents overlap; no text extent within 0.3 mm of any box edge, either "
          "divider, any arrow shaft or head (stroke half-widths included)", not text_overlaps and not too_close,
          {"text_overlaps": text_overlaps, "too_close": too_close,
           "min_clearance_mm": round(min_clear[0], 4), "at": [list(min_clear[1]), min_clear[2]]})

    # ---- C7 arrows ----
    bad7 = []
    head_sizes = set()
    lane_centres = {k: (v[0] + v[1]) / 2.0 for k, v in LANE_X.items()}
    if abs(X_LC - lane_centres["left"]) > EXACT or abs(X_RC - lane_centres["right"]) > EXACT:
        bad7.append(("X_LC / X_RC", "!= lane centres from LANE_X"))
    required_x = {}
    for a in ARROWS:
        sh, hd = ARTS["shafts"][a["id"]], ARTS["heads"][a["id"]]
        xs, ys = list(sh.get_xdata()), list(sh.get_ydata())
        v = hd.get_xy()[:3]
        (bx0, by0), (bx1, by1), (tx, ty) = v
        head_sizes.add((round(bx1 - bx0, 12), round(ty - by0, 12)))
        lane_boxes = [GEOM[b] for b in (a["source"], a["target"]) if GEOM[b]["lane"] != "full"]
        req = [g["xc"] for g in lane_boxes]
        required_x[a["id"]] = [round(v, 9) for v in req]
        if not lane_boxes or xs[0] != xs[1] or any(abs(xs[0] - v) > EXACT for v in req) or \
                any(abs(xs[1] - v) > EXACT for v in req):
            bad7.append((a["id"], "shaft not vertical / x != centre of its lane box(es) from GEOM"))
        if any(abs(tx - v) > EXACT or abs((bx0 + bx1) / 2 - v) > EXACT for v in req):
            bad7.append((a["id"], "head x != centre of its lane box(es) from GEOM"))
        for b in (a["source"], a["target"]):
            g = GEOM[b]
            if not (xs[0] - g["x0"] >= HEAD_W / 2 + 0.5 and g["x1"] - xs[0] >= HEAD_W / 2 + 0.5):
                bad7.append((a["id"], "x not strictly inside box {} (>= HEAD_W/2 + 0.5 mm from its sides)".format(b)))
        if abs((bx0 + bx1) / 2 - xs[0]) > EXACT or abs(tx - xs[0]) > EXACT or abs(by0 - by1) > EXACT:
            bad7.append((a["id"], "head not symmetric about the shaft x"))
        if abs(ys[0] - GEOM[a["source"]]["y_bottom"]) > EXACT:
            bad7.append((a["id"], "shaft start != source y_bottom"))
        if abs(ty - GEOM[a["target"]]["y_top"]) > EXACT:
            bad7.append((a["id"], "tip != target y_top"))
        if abs(ys[1] - by0) > EXACT or not ys[0] < ys[1] < ty:
            bad7.append((a["id"], "shaft does not end at the head base / not downward"))
        if ys[1] - ys[0] < SHAFT_MIN:
            bad7.append((a["id"], "visible shaft {:.3f} mm < {}".format(ys[1] - ys[0], SHAFT_MIN)))
        for g in GEOM.values():                    # interior of any box (open interval), shaft and head widths
            for half, y0, y1 in ((hl_arr, ys[0], ys[1]), (HEAD_W / 2, by0, ty)):
                if xs[0] + half > g["x0"] and xs[0] - half < g["x1"] and \
                        min(y1, g["y_bottom"]) - max(y0, g["y_top"]) > EXACT:
                    bad7.append((a["id"], "enters box " + g["id"]))
    for i in range(len(ARROWS)):
        for j in range(i + 1, len(ARROWS)):
            a, b = ARROWS[i], ARROWS[j]
            if abs(a["x"] - b["x"]) < HEAD_W and min(a["y_tip"], b["y_tip"]) - max(a["y_start"], b["y_start"]) > 0:
                bad7.append((a["id"], b["id"], "overlap / cross"))
    shafts_len = {a["id"]: round(a["y_head_base"] - a["y_start"], 4) for a in ARROWS}
    check("C7", "arrows: exactly 12, vertical, shaft and head x == centre (x0 + x1)/2 of the lane box(es) among source "
          "and target (from GEOM, not from the drawing table); X_LC / X_RC == lane centres from LANE_X; x >= "
          "HEAD_W/2 + 0.5 mm inside both source and target boxes; shaft starts at the source bottom edge and "
          "the head tip lies on the target top edge (<= 1e-9), visible shaft >= 2.0 mm, no arrow enters any box "
          "interior, no two arrows overlap or cross, all heads identical",
          len(ARROWS) == 12 and len(ARTS["shafts"]) == 12 and not bad7 and len(head_sizes) == 1,
          {"violations": bad7, "lane_centres_mm": lane_centres, "required_x_mm": required_x,
           "drawn_x_mm": {a["id"]: list(ARTS["shafts"][a["id"]].get_xdata()) for a in ARROWS},
           "head_sizes_mm": sorted(head_sizes), "visible_shaft_mm": shafts_len,
           "min_visible_shaft_mm": min(shafts_len.values())})

    # ---- C8 canvas ----
    extents = []
    for g in GEOM.values():
        extents.append(("box " + g["id"], g["x0"] - hl_box, g["x1"] + hl_box, g["y_top"] - hl_box,
                        g["y_bottom"] + hl_box))
        if g["divider_y"] is not None:
            extents.append(("divider " + g["id"], g["x0"], g["x1"], g["divider_y"] - hl_div, g["divider_y"] + hl_div))
    for a in ARROWS:
        extents.append(("shaft " + a["id"], a["x"] - hl_arr, a["x"] + hl_arr, a["y_start"], a["y_head_base"]))
        extents.append(("head " + a["id"], a["x"] - HEAD_W / 2, a["x"] + HEAD_W / 2, a["y_head_base"], a["y_tip"]))
    for k, e in EXT.items():
        extents.append(("text " + "/".join(map(str, k)), e[0], e[1], e[2], e[3]))
    margins = [(min(x0, y0, W - x1, H - y1), name) for name, x0, x1, y0, y1 in extents]
    min_margin = min(margins)
    pos = AX.get_position()
    sx = AX.transData.transform([[1, 0]])[0][0] - AX.transData.transform([[0, 0]])[0][0]
    sy = AX.transData.transform([[0, 0]])[0][1] - AX.transData.transform([[0, 1]])[0][1]
    axes_ok = abs(pos.x0) < 1e-9 and abs(pos.y0) < 1e-9 and abs(pos.x1 - 1) < 1e-9 and abs(pos.y1 - 1) < 1e-9 and \
        abs(sx - sy) <= 1e-6 * sx
    check("C8", "canvas: W == 85.8, H <= 125.0; every artist extent (box strokes incl. half width) >= 0.2 mm inside the "
          "canvas; one axes [0, 0, 1, 1] with equal aspect (mm data units)",
          W == 85.8 and H <= H_MAX and min_margin[0] >= INSIDE_MIN and axes_ok,
          {"W_mm": W, "H_mm": round(H, 6), "min_inside_margin_mm": round(min_margin[0], 4), "at": min_margin[1],
           "axes_position": [pos.x0, pos.y0, pos.x1, pos.y1], "px_per_mm_x": sx, "px_per_mm_y": sy})

    # ---- render ----
    IMG = render_rgb(FIG)
    PNG = png_bytes(IMG)
    PDF = pdf_bytes(FIG)
    PDF2 = pdf_bytes(FIG)

    # ---- C9 raster ----
    im = Image.open(io.BytesIO(PNG))
    arr = np.asarray(im)
    fpx = int(math.ceil(mm2px(FRAME_MM)))
    frame = np.concatenate([arr[:fpx].reshape(-1, 3), arr[-fpx:].reshape(-1, 3), arr[:, :fpx].reshape(-1, 3),
                            arr[:, -fpx:].reshape(-1, 3)])
    dpi_png = im.info.get("dpi", (0, 0))
    ok9 = im.mode == "RGB" and abs(im.size[0] - mm2px(W)) <= 1 and abs(im.size[1] - mm2px(H)) <= 1 and \
        all(round(v) == DPI for v in dpi_png) and bool((frame == 255).all())
    check("C9", "raster: PNG at 600 dpi, RGB, pixel size == W x H mm (+/-1 px); the outer 0.15 mm frame is pure white",
          ok9, {"pixels": list(im.size), "expected_px": [round(mm2px(W), 3), round(mm2px(H), 3)],
                "dpi": [round(v, 3) for v in dpi_png], "mode": im.mode, "frame_px": fpx,
                "frame_all_white": bool((frame == 255).all())})

    # ---- C10 PDF ----
    box_ok, fonts_ok, n_images, wpd, hpd = pdf_font_and_box(PDF, W, H)
    check("C10", "PDF: MediaBox == W x H mm (+/-0.05); all fonts embedded as TrueType; no image XObjects; two "
          "in-memory renders byte-identical", box_ok and fonts_ok and n_images == 0 and PDF == PDF2,
          {"mediabox_mm": [None if wpd is None else round(wpd, 4), None if hpd is None else round(hpd, 4)],
           "truetype_embedded": fonts_ok, "image_xobjects": n_images, "deterministic": PDF == PDF2,
           "sha256": sha256_bytes(PDF)})

    # ---- C11 counts ----
    n_white = sum(1 for g in GEOM.values() if not g["gate"] and ARTS["boxes"][g["id"]].get_facecolor()[:3] == (1, 1, 1))
    n_grey = sum(1 for g in GEOM.values() if g["gate"] and
                 matplotlib.colors.to_hex(ARTS["boxes"][g["id"]].get_facecolor()) == GATE)
    n_div = sum(1 for ln in AX.get_lines() if (ln.get_gid() or "").startswith("divider/"))
    n_shafts = sum(1 for ln in AX.get_lines() if (ln.get_gid() or "").endswith("/shaft"))
    n_heads = sum(1 for p in AX.patches if (p.get_gid() or "").endswith("/head"))
    n_rects = sum(1 for p in AX.patches if (p.get_gid() or "").startswith("box/"))
    colours = set()
    for ln in AX.get_lines():
        colours.add(matplotlib.colors.to_hex(ln.get_color()))
    for p in AX.patches:
        colours.add(matplotlib.colors.to_hex(p.get_facecolor()))
        if p.get_linewidth() > 0:
            colours.add(matplotlib.colors.to_hex(p.get_edgecolor()))
    for t in AX.texts:
        colours.add(matplotlib.colors.to_hex(t.get_color()))
    check("C11", "counts: 11 boxes (9 white, 2 grey), 2 dividers, 12 arrows (12 shafts, 12 heads); no other artist; "
          "only the colours #1a1a1a, #333333, #e9e9e9 and white",
          n_rects == 11 and n_white == 9 and n_grey == 2 and n_div == 2 and n_shafts == 12 and n_heads == 12 and
          len(AX.get_lines()) == 14 and len(AX.patches) == 23 and colours <= {INK, EDGE, GATE, "#ffffff"},
          {"boxes": n_rects, "white": n_white, "grey": n_grey, "dividers": n_div, "shafts": n_shafts, "heads": n_heads,
           "lines": len(AX.get_lines()), "patches": len(AX.patches), "colours": sorted(colours)})

    # ---- A4 preview (vector; the same drawing at 1:1, same baselines) ----
    PREV = new_figure(A4_W, A4_H)
    PAX = mm_axes(PREV, [PREVIEW_X / A4_W, (A4_H - PREVIEW_TOP - H) / A4_H, W / A4_W, H / A4_H])
    draw_flowchart(PAX, BASELINES)
    for gx in GUIDES_X:
        PREV.add_artist(Line2D([gx / A4_W, gx / A4_W], [0, 1], transform=PREV.transFigure, color=GUIDE_COLOR,
                               linewidth=GUIDE_LW, solid_capstyle="butt"))
    PREV_PDF = pdf_bytes(PREV)
    PREV_PDF2 = pdf_bytes(PREV)
    _pb = PAX.get_position()
    PREVIEW_INFO = {"figure_left_mm": round(_pb.x0 * A4_W, 6), "figure_top_mm": round(A4_H - _pb.y1 * A4_H, 6),
                    "figure_size_mm": [round(_pb.width * A4_W, 6), round(_pb.height * A4_H, 6)],
                    "guides_x_mm": list(GUIDES_X), "deterministic": PREV_PDF == PREV_PDF2}
    print("A4 preview: {}".format(PREVIEW_INFO))

N_FAIL_PRE = sum(c["status"] == "FAIL" for c in CHECKS)
LAYOUT = {
    "document": "flowchart layout", "version": FLOWCHART_VERSION, "units": "mm, y downward from the top edge",
    "canvas": {"W": W, "H": H, "M": M, "GAP": GAP, "LANE_GAP": LANE_GAP, "LANE_W": LANE_W, "Lc": X_LC, "Rc": X_RC,
               "PITCH": PITCH, "VPAD": VPAD, "HEAD_L": HEAD_L, "HEAD_W": HEAD_W},
    "rows": [{"row": r, "y_top": ROW_TOP[r], "height": ROW_H[r]} for r in range(N_ROWS)],
    "boxes": [{"id": g["id"], "row": g["row"], "lane": g["lane"], "fill": GATE if g["gate"] else "white",
               "x0": g["x0"], "x1": g["x1"], "y_top": g["y_top"], "y_bottom": g["y_bottom"],
               "divider_y": g["divider_y"],
               "lines": [{"section": si, "text": txt, "italic": it, "baseline_y": BASELINES[(g["id"], si, li)],
                          "extent_mm": [round(v, 4) for v in EXT[(g["id"], si, li)]]}
                         for si, sec in enumerate(g["sections"]) for li, (txt, it) in enumerate(sec)]}
              for g in GEOM.values()],
    "arrows": [{"id": a["id"], "source": a["source"], "target": a["target"], "x": a["x"], "y_start": a["y_start"],
                "y_head_base": a["y_head_base"], "y_tip": a["y_tip"]} for a in ARROWS],
    "preview": PREVIEW_INFO, "self_checks": CHECKS,
}


def print_report():
    hdr("SELF-CHECK REPORT")
    for c in CHECKS:
        print("[{}] {} - {}\n       {}".format(c["status"], c["id"], c["description"],
                                               json.dumps(c["detail"], ensure_ascii=False)))


if N_FAIL_PRE:
    print_report()
    print("\n" + "#" * 100 + "\nFAIL: {} self-check(s) failed: {} - NOTHING WAS WRITTEN\n".format(
        N_FAIL_PRE, [c["id"] for c in CHECKS if c["status"] == "FAIL"]) + "#" * 100)
    if not IN_IPYTHON:
        sys.exit(1)
else:
    OUT_DIR = os.path.join(ROOT, "outputs", "figures", "{}_{}".format(FLOWCHART_VERSION,
                                                                    datetime.now().strftime("%Y%m%d-%H%M%S")))
    if os.path.exists(OUT_DIR):
        raise RuntimeError("STOP: {} already exists - an output folder is never overwritten".format(OUT_DIR))
    os.makedirs(OUT_DIR, exist_ok=False)
    FILES = {"flowchart.pdf": PDF, "flowchart.png": PNG, "flowchart_preview_A4.pdf": PREV_PDF,
             "flowchart_layout.json": json.dumps(LAYOUT, ensure_ascii=False, indent=2).encode("utf-8")}
    WRITTEN = {n: write_once(os.path.join(OUT_DIR, n), b) for n, b in FILES.items()}
    _script = globals().get("__file__")
    MANIFEST = {"document": "flowchart manifest", "version": FLOWCHART_VERSION,
                "created": datetime.now().astimezone().isoformat(), "versions": VERSIONS, "warnings": WARNINGS,
                "font": {"name": FONT_NAME, "upright": FONT_PATH, "italic": ITALIC_PATH,
                         "fallback": FONT_NAME != FONT_CANDIDATES[0]},
                "paper_numbers": {"path": PN_PATH, "sha256": PN_SHA256},
                "script": {"path": _script, "sha256": sha256_of(_script) if _script and os.path.isfile(_script)
                           else None, "note": None if _script else "run as a Jupyter cell (no script file)"},
                "self_checks_C1_to_C11": CHECKS,
                "outputs": [{"file": n, "bytes": len(FILES[n]), "sha256": WRITTEN[n]} for n in FILES]}
    MAN_BYTES = json.dumps(MANIFEST, ensure_ascii=False, indent=2).encode("utf-8")
    write_once(os.path.join(OUT_DIR, "flowchart_manifest.json"), MAN_BYTES)
    with open(os.path.join(OUT_DIR, "flowchart_manifest.json"), "r", encoding="utf-8") as _f:
        _man = json.load(_f)
    _listed = {o["file"]: o["sha256"] for o in _man["outputs"]}
    _on_disk = sorted(os.listdir(OUT_DIR))
    ok12 = _on_disk == sorted(list(FILES) + ["flowchart_manifest.json"]) and sorted(_listed) == sorted(FILES) and \
        all(sha256_of(os.path.join(OUT_DIR, n)) == sha256_bytes(FILES[n]) == _listed[n] for n in FILES) and \
        sha256_of(os.path.join(OUT_DIR, "flowchart_manifest.json")) == sha256_bytes(MAN_BYTES) and \
        _man["paper_numbers"]["sha256"] == PN_SHA256
    check("C12", "outputs: all files exist; sha256 on disk == in-memory sha256; all recorded in flowchart_manifest.json "
          "(the manifest itself verified against its in-memory bytes)", ok12,
          {"on_disk": _on_disk, "listed": sorted(_listed)})
    print_report()
    print("\noutput folder: {}".format(OUT_DIR))
    for n in sorted(WRITTEN):
        print("  {:28s} {}".format(n, WRITTEN[n]))
    print("  {:28s} {}".format("flowchart_manifest.json", sha256_bytes(MAN_BYTES)))
    print("H = {:.4f} mm | font {}".format(H, FONT_NAME))
    if WARNINGS:
        print("\nWARNINGS: {}".format(WARNINGS))
    N_FAIL = sum(c["status"] == "FAIL" for c in CHECKS)
    if N_FAIL == 0:
        print("\nALL SELF-CHECKS PASSED ({})".format(len(CHECKS)))
    else:
        print("\n" + "#" * 100 + "\nFAIL: {} of {} self-checks failed: {}\n".format(
            N_FAIL, len(CHECKS), [c["id"] for c in CHECKS if c["status"] == "FAIL"]) + "#" * 100)
        if not IN_IPYTHON:
            sys.exit(1)
