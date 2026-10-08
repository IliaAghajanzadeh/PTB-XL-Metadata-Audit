# ph1 r1 - provenance list

Cell: `ph1_cell.py`, sha256 `1fc8f775233531342bf5b70af94c2c3e4c6534ea8a58c1cbb999279512950776`, 1700 lines; 537 of them are inside 53 marked blocks (30 verbatim, 23 adapted). Every block was checked mechanically against its source (all PASS; the harness repeats the check in P3, reading the sources from the notebooks).

Sources: v0.2.1 = `ptbxl_death_experiment_v0.2.1.ipynb - main code cell (cell 2 of 2)`; v0.3 = `ptbxl_v0.3_preregistration.ipynb - main code cell (cell 2 of 2)`; v0.9 = `v0.9_cell.py (sha256 9af1773d...)`. Line numbers are 1-based lines of that cell's source.

Undo order for an adapted block: delete lines tagged `# [ph1 added]`; remove `indent: N`; undo each `literal`; undo each `rename`.

| # | kind | ph1 lines | source | source lines | block | declared changes |
|---|---|---|---|---|---|---|
| 1 | verbatim | 85-97 | v0.2.1 | 59-71 | imports, display options | - |
| 2 | verbatim | 144-144 | v0.2.1 | 78-78 | superclasses | - |
| 3 | verbatim | 148-150 | v0.2.1 | 80-82 | seeds, bootstrap | - |
| 4 | verbatim | 153-154 | v0.2.1 | 84-85 | H1 threshold, learners | - |
| 5 | verbatim | 157-157 | v0.3 | 65-65 | H2 model seed | - |
| 6 | verbatim | 160-161 | v0.3 | 69-70 | learner roles | - |
| 7 | verbatim | 164-164 | v0.3 | 72-72 | compared blocks | - |
| 8 | verbatim | 168-173 | v0.3 | 74-79 | alpha, power, z values, SESOI | - |
| 9 | verbatim | 196-200 | v0.3 | 82-86 | input hashes | - |
| 10 | verbatim | 217-218 | v0.9 | 222-223 | fold-9 positives, folds 1-9 records | - |
| 11 | verbatim | 221-221 | v0.9 | 229-229 | fold-10 records | - |
| 12 | verbatim | 224-224 | v0.9 | 303-303 | fold-10 patients | - |
| 13 | verbatim | 227-229 | v0.9 | 311-313 | fold-10 label counts disclosed in FRP | - |
| 14 | verbatim | 259-262 | v0.9 | 358-361 | output paths outside the dataset folder | - |
| 15 | verbatim | 266-267 | v0.9 | 365-366 | run start, warnings | - |
| 16 | verbatim | 274-279 | v0.2.1 | 97-102 | sha256_of | - |
| 17 | verbatim | 284-351 | v0.2.1 | 113-180 | hdr, warn, note, _json_default, fast_auc, to_level_string, encode_categorical | - |
| 18 | verbatim | 356-368 | v0.3 | 201-213 | canonical JSON, self_sha256 procedure | - |
| 19 | verbatim | 373-380 | v0.9 | 455-462 | verify_self_sha256 | - |
| 20 | verbatim | 385-394 | v0.9 | 467-476 | plain, manifest_entry | - |
| 21 | verbatim | 399-409 | v0.9 | 499-509 | write_once_json | - |
| 22 | verbatim | 414-415 | v0.9 | 514-515 | sha256_bytes | - |
| 23 | verbatim | 420-427 | v0.9 | 543-550 | _find_nonfinite | - |
| 24 | verbatim | 432-433 | v0.9 | 555-556 | _partial | - |
| 25 | verbatim | 438-440 | v0.9 | 656-658 | csv_bytes | - |
| 26 | verbatim | 445-471 | v0.9 | 681-707 | exclusive_rename, write_once_bytes, write_once_json_atomic | - |
| 27 | adapted | 629-630 | v0.2.1 | 208-209 | train / eval masks | indent: 4 / literal: <= 8 => <= max_train_fold ;; == 9 => == eval_fold / renames: df9->frame |
| 28 | adapted | 643-654 | v0.2.1 | 232-243 | label derivation; scp is an argument | indent: 4 / renames: df9->frame |
| 29 | adapted | 659-693 | v0.2.1 | 291-325 | Section 1: age, sex, recording_year, level strings, encoder fit on the training rows | indent: 4 / renames: df9->frame |
| 30 | adapted | 702-732 | v0.2.1 | 335-365 | Section 1 feature frame, Section 2 feature blocks, design matrix shapes | indent: 4 |
| 31 | adapted | 746-746 | v0.2.1 | 384-384 | training labels | indent: 4 |
| 32 | adapted | 751-795 | v0.2.1 | 408-450 | Section 3 fit loop up to the training labels | indent: 4; 2 added line(s) |
| 33 | adapted | 799-812 | v0.2.1 | 452-465 | Section 3 fit + predict_proba per class | indent: 4 |
| 34 | adapted | 869-869 | v0.2.1 | 275-275 | chance AUPRC baseline = evaluation-fold prevalence | indent: 4 / literal: lab.loc[mask_eval] => labels_eval / renames: prev_fold9->prev_eval |
| 35 | adapted | 872-872 | v0.2.1 | 385-385 | evaluation labels | indent: 4 / literal: lab.loc[mask_eval] => labels_eval |
| 36 | adapted | 875-888 | v0.2.1 | 387-398 | patient-level bootstrap resamples, v0.2.1 construction | indent: 4 / literal: df9.loc[mask_eval, "patient_id"].to_numpy() => eval_patient_ids; 2 added line(s) |
| 37 | adapted | 894-894 | v0.2.1 | 403-403 | metric rows list | indent: 4 |
| 38 | adapted | 907-917 | v0.2.1 | 467-477 | per-class AUROC / AUPRC rows | indent: 4 / renames: prev_fold9->prev_eval |
| 39 | adapted | 920-946 | v0.2.1 | 483-509 | macro AUROC / AUPRC + patient-level bootstrap 95% CI | indent: 4 / renames: prev_fold9->prev_eval |
| 40 | adapted | 951-951 | v0.2.1 | 513-513 | metric rows frame | indent: 4 |
| 41 | adapted | 955-978 | v0.2.1 | 525-548 | Section 5 decision table | indent: 4 / renames: BLOCKS->BLOCK_NAMES |
| 42 | adapted | 983-992 | v0.2.1 | 553-562 | B3 >= 0.60 verdict | indent: 4 |
| 43 | adapted | 1003-1003 | v0.3 | 366-366 | point macro-AUROC per learner and block | indent: 4 |
| 44 | adapted | 1018-1037 | v0.3 | 496-515 | H2 statistics per learner | indent: 4 |
| 45 | adapted | 1048-1052 | v0.3 | 522-526 | MDE | indent: 4 / renames: Z_ALPHA_2->z_alpha_2, Z_POWER->z_power |
| 46 | adapted | 1055-1092 | v0.3 | 537-574 | difference test, TOST, achieved bound, learner disagreement | indent: 4 / renames: SESOI_H2->sesoi_h2 |
| 47 | adapted | 1096-1107 | v0.3 | 627-638 | results block per learner, pre_registration.json field names | indent: 4 |
| 48 | adapted | 1110-1114 | v0.3 | 595-599 | per-draw deltas frame | indent: 4 |
| 49 | verbatim | 1238-1243 | v0.9 | 1139-1144 | reference implementation exec | - |
| 50 | verbatim | 1366-1366 | v0.2.1 | 207-207 | working set folds 1-9 | - |
| 51 | adapted | 1474-1483 | v0.9 | 1485-1494 | fold-10 ids, patients | renames: fold10_meta->df10 |
| 52 | verbatim | 1486-1488 | v0.9 | 1531-1533 | fold-10 patient grouping and BOOT_IDX with REF | - |
| 53 | verbatim | 1491-1491 | v0.9 | 1536-1536 | fold-10 id array | - |

## Unmarked lines that repeat earlier code (new ph1 code, listed for completeness)

- `metrics_from_probs` loop headers `for block_name in BLOCK_NAMES / for learner in LEARNERS / for seed in SEEDS`, `per_class_auroc, per_class_auprc = [], []`, `for sc in SUPERCLASSES:` and `yev = y_eval_full[sc].to_numpy()` repeat v0.2.1 lines 408, 443-444, 447, 449 and 451 with one more indent level; the scores come from the probability frame (`scores_eval = P[...]`, `proba = scores_eval[sc]`) instead of a model.
- `LOCK_HASHES` (3 entries) = v0.9 lines 1537-1540 + closing brace, without the `perm_idx_sha256` entry (PERM_IDX is not computed in ph1).
- `EXPECTED_SELF_SHA256` / `EXPECTED_FILE_SHA256` hold 2 of the 5 entries of v0.9 lines 193-206 (pre_registration, final_run_protocol); `EXPECTED_REFERENCE_CODE_SHA256` = v0.9 line 212.
- The STEP 0 version / hash / snapshot code follows v0.9 STEP 0 in structure; the five-document loop is reduced to the two documents ph1 reads.

## Earlier code deliberately NOT copied

- v0.2.1 Section 0a (prevalence by fold), Section 0b chance-baseline CSV, Section 6 coefficients, Section 7 manifest, `save_csv` (answers Q2); only the fold prevalence vector used as `auprc_chance` is kept (v0.2.1 line 275).
- v0.2.1 line 451 (`yev = ...` inside the fit loop): no evaluation label is touched while fitting. The metric statements of the v0.2.1 fit loop (lines 467-477 and 483-509) are copied into `metrics_from_probs` (blocks 38-39); the Section 6 model capture (lines 479-481) and the timing print (line 511) are not copied.
- v0.2.1 lines 327-333 (print 'levels from folds 1-8'), 399-401 and 551-552 (prints that name 'fold 9') are replaced by neutral prints.
- v0.3 lines 368-415 (seed-42 refit of B1/B3) and 443-456 (BOOT_ROWS rebuild): replaced by the seed-42 fits of A2 and REF on `build_boot_idx(1942, 2000, 7)`; the equivalence is tested in A3 against pre_registration.json and v0.3_bootstrap_deltas.csv (answer Q4).
- v0.3 lines 417-438 and 486-494 (4-decimal sanity checks vs the v0.2.1 printout): superseded by the 1e-12 checks of Part A. v0.3 Part D (sealing pre_registration.json) and its manifest: not applicable.
- v0.3 `macro_on_rows` (per-class nan-skip): the per-draw deltas come from `REF.paired_bootstrap` (answer Q5); the two agree when no draw is NaN (checked in A3).
