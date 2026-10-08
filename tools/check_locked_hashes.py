"""Check that the files in this repository are byte-identical to the hashed and timestamped originals.

Usage (from the repository root):  python tools/check_locked_hashes.py [repository_root]
Standard library only. Exit code 0 = all checks pass, 1 = at least one failure.

Run it twice: once on your local project before uploading, and once on a fresh clone / download of the
GitHub repository after uploading. Any line-ending conversion by git would show up here as a mismatch.
"""
import hashlib
import sys
from pathlib import Path

# Write-once documents: SHA-256 of the file bytes (paper_numbers.json, provenance.documents[].file_sha256).
LOCKED = {
    "pre_registration.json": "11dd446272654fc80d65b2fdd40259fa674bbb419a44fde84e589b6e297cde32",
    "dev_protocol.json": "babf101c1d69b322daef6907175fdc4c17b5a53755a68f13a998eefa0148dc2d",
    "v0.5_training_details.json": "6480c191aeaed9ff1e6b1da5caa0c1e46908d453d30106a32beb7cf83678be11",
    "lr_selection.json": "358d783b37bba6fcf61237a43817c1fd8594fe192f8ecdfca2c25b871a21415b",
    "final_run_protocol.json": "e1a5260e72d2e8b0d27dc709639468b7f9a48a7c8e2f8af2d663e105dbd329d4",
    "v0.8_manifest.json": "9725a0ee502784ffc267b1021ec3cb427966708205db9a09ce7e8c1fc8f30415",
    "v0.9_frozen_outputs.json": "133ca5fe9c9faf299cec5270483eb6ac2abf568a1c6fcc6bb5998d351528867f",
    "v0.9_results.json": "8996ac975f7091fd07d0127521d8413e14e957f878a930334fdf69983bbc506f",
    "v0.9_manifest.json": "710e3519bbb750fc427cf9bafe2f6075806610401d1f52f7682f1670417cdca6",
}
# Paper artifacts: at least one copy must match (older runs with the same file name may exist).
ARTIFACTS = {
    "paper_numbers.json": "e540d16d65d9247669e8488bcc834ea854cdd5bf94f0181f3c9ceb6ff49e524c",
    "make_figures.py": "ebc299bfae3a80d2726caeea6a281a36f25657b1d206ea91ec63ee448e296e43",
    "fig1_auroc_levels.pdf": "1d13b90c2c30d05c2961a2eb5d04f510346f0fcbe79ac89b5fe5dcb75bbd1aa6",
    "fig2_delta.pdf": "cc7c6f223231a25bbbfe0551723e433afa95b905a6faa3ec98fda2bb3c6cdc8c",
}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find(root, name):
    return sorted(p for p in root.rglob(name) if p.is_file() and not (set(p.parts) & SKIP_DIRS))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    failures = 0
    print("repository root:", root, "\n")
    print("Write-once documents (every copy must match; a .ots proof must sit next to at least one copy)")
    for name, want in LOCKED.items():
        copies = find(root, name)
        if not copies:
            print("  FAIL  {:<28} missing".format(name)); failures += 1; continue
        has_ots = False
        for p in copies:
            ok = sha256(p) == want
            ots = p.with_name(p.name + ".ots").is_file()
            has_ots = has_ots or ots
            print("  {}  {:<28} {}{}".format("OK  " if ok else "FAIL", name, p.relative_to(root),
                                              "  (+ .ots)" if ots else ""))
            failures += 0 if ok else 1
        if not has_ots:
            print("  FAIL  {:<28} no {}.ots found".format(name, name)); failures += 1
    print("\nPaper artifacts (at least one copy must match)")
    for name, want in ARTIFACTS.items():
        copies = find(root, name)
        matches = [p for p in copies if sha256(p) == want]
        for p in copies:
            print("  {}  {:<28} {}".format("OK  " if p in matches else "note", name, p.relative_to(root)))
        if not matches:
            print("  FAIL  {:<28} no copy with the recorded hash".format(name)); failures += 1
    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else "{} FAILURE(S)".format(failures)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
