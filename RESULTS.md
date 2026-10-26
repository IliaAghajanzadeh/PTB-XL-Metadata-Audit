# Results tables

Every value below is generated from `paper_numbers.json` (sha256 `e540d16d65d9247669e8488bcc834ea854cdd5bf94f0181f3c9ceb6ff49e524c`), the output of the read-only verification script (`run_20261008-022605`, 370 checks, 0 failed). Values are rounded to four decimals, as in the paper.

Macro-AUROC is the unweighted mean of the five per-class AUROCs (NORM, MI, STTC, CD, HYP). Intervals are percentile intervals from the paired, patient-level bootstrap (2,000 resamples, seed 7) unless stated otherwise. Status labels: **pre-registered** (H3), **confirmatory after the fact** (H1, H2 on fold 9), **post hoc** (all fold-10 metadata-only results), **exploratory** (E1–E3, closed list fixed before fold 10 was opened).

## Dataset (PTB-XL v1.0.3, 100 Hz)

| | records | patients |
|---|---|---|
| all | 21,799 | 18,869 |
| folds 1–8 (development training) | 17,418 | 15,023 |
| fold 9 (validation) | 2,183 | 1,942 |
| folds 1–9 (final training) | 19,601 | 16,965 |
| fold 10 (test) | 2,198 | 1,904 |

Positives per superclass: fold 9 — NORM 955, MI 540, STTC 528, CD 495, HYP 268; fold 10 — NORM 963, MI 550, STTC 521, CD 496, HYP 262. Records without any superclass statement: 411 (kept as all-negative).

## Metadata-only models: macro-AUROC [95% CI] (paper Table II, Fig. 2a)

Fold 9: trained on folds 1–8 (development run). Fold 10: trained on folds 1–9, **post hoc**. LR: logistic regression (deterministic). HGB: histogram gradient boosting, value and interval are means over seeds 42, 123, 2024. Note from the verification script: *ci95 = mean of the three per-seed marginal 95% percentile CIs (v0.2.1 convention); for LogisticRegression the three seeds are identical*.

| block | fold 9 LR | fold 9 HGB | fold 10 LR (post hoc) | fold 10 HGB (post hoc) |
|---|---|---|---|---|
| B0 intercept only | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] |
| B1 age, sex | 0.6890 [0.6743, 0.7042] | 0.6725 [0.6568, 0.6880] | 0.6846 [0.6687, 0.7007] | 0.6739 [0.6582, 0.6903] |
| B2 acquisition (device, site, nurse) | 0.6104 [0.5954, 0.6257] | 0.6102 [0.5948, 0.6259] | 0.6269 [0.6097, 0.6432] | 0.6200 [0.6031, 0.6364] |
| B3 = B1 + B2 | 0.6972 [0.6825, 0.7119] | 0.6937 [0.6790, 0.7084] | 0.7065 [0.6910, 0.7213] | 0.7057 [0.6907, 0.7203] |
| B4 recording year | 0.5094 [0.4902, 0.5296] | 0.4980 [0.4812, 0.5158] | 0.5257 [0.5068, 0.5444] | 0.5145 [0.4976, 0.5326] |
| B5 = B3 + year | 0.6969 [0.6823, 0.7117] | 0.6935 [0.6790, 0.7083] | 0.7061 [0.6906, 0.7210] | 0.7060 [0.6909, 0.7207] |

## H1 (confirmatory after the fact): metadata-only B3 reaches fold-9 macro-AUROC ≥ 0.6

| learner | fold 9 B3 | H1 |
|---|---|---|
| LR | 0.6972 | holds |
| HGB | 0.6937 | holds |

## H2: acquisition metadata on top of age and sex, metadata-only (Δ = B3 − B1, seed-42 refits; paper Table III)

Difference test: 95% CI excludes 0. Equivalence: two one-sided tests at ±0.02 (90% CI inside ±0.02); the ±0.02 margin was set after the fold-9 deltas were seen, so the achieved bound is also reported.

| fold | learner | B1 | B3 | Δ | SE | 95% CI | 90% CI | difference test | TOST ±0.02 | achieved bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 (confirmatory after the fact) | LR | 0.6890 | 0.6972 | +0.0082 | 0.0043 | [−0.0001, +0.0168] | [+0.0013, +0.0153] | no statistically significant difference between B1 and B3 | practically negligible | 0.0153 |
| 9 (confirmatory after the fact) | HGB | 0.6724 | 0.6923 | +0.0199 | 0.0049 | [+0.0106, +0.0298] | [+0.0120, +0.0282] | significant difference: B3 higher than B1 by +0.0199 (point delta) | inconclusive / not shown negligible | 0.0282 |
| 10 (post hoc) | LR | 0.6846 | 0.7065 | +0.0219 | 0.0044 | [+0.0132, +0.0304] | [+0.0149, +0.0289] | significant difference: B3 higher than B1 by +0.0219 (point delta) | inconclusive / not shown negligible | 0.0289 |
| 10 (post hoc) | HGB | 0.6732 | 0.7042 | +0.0309 | 0.0050 | [+0.0210, +0.0405] | [+0.0222, +0.0389] | significant difference: B3 higher than B1 by +0.0309 (point delta) | inconclusive / not shown negligible | 0.0389 |

## ECG networks on fold 10: macro-AUROC [95% CI] (three-seed ensembles; paper Fig. 2b)

Intervals are descriptive (marginal bootstrap). Published reference results (not directly comparable): ResNet1d-Wang 0.93, Inception1d 0.921.

| network | signal only | signal + age/sex | signal + age/sex + acquisition |
|---|---|---|---|
| ResNet1d-Wang (primary) | 0.9355 [0.9284, 0.9423] | 0.9362 [0.9292, 0.9428] | 0.9390 [0.9323, 0.9455] |
| Inception1d (robustness) | 0.9327 [0.9251, 0.9395] | 0.9340 [0.9272, 0.9402] | 0.9349 [0.9281, 0.9415] |

No bootstrap draw was excluded (all arms: n_valid = 2000, n_nan = 0).

## H3 (pre-registered): Δ = AUROC(signal + all) − AUROC(signal + age/sex) on fold 10 (paper Table III, Fig. 3)

Rule (a): 95% CI excludes 0. Rule (b): 90% CI inside ±0.01. Material: 95% CI entirely outside ±0.01.

| network | signal + age/sex | signal + all | Δ | 95% CI | 90% CI | rules holding | verdict |
|---|---|---|---|---|---|---|---|
| ResNet1d-Wang (primary) | 0.9362 | 0.9390 | +0.0028 | [+0.0010, +0.0048] | [+0.0013, +0.0044] | a, b | statistically detectable, below the materiality threshold |
| Inception1d (robustness) | 0.9340 | 0.9349 | +0.0009 | [−0.0016, +0.0033] | [−0.0013, +0.0030] | b | no material change |

### H3 per seed (secondary; not the H3 verdict)

| network | seed | Δ | 95% CI | 90% CI |
|---|---|---|---|---|
| resnet1d_wang | 42 | +0.0035 | [+0.0013, +0.0059] | [+0.0016, +0.0055] |
| resnet1d_wang | 123 | +0.0028 | [+0.0001, +0.0055] | [+0.0005, +0.0051] |
| resnet1d_wang | 2024 | +0.0017 | [−0.0007, +0.0043] | [−0.0003, +0.0039] |
| inception1d | 42 | +0.0021 | [−0.0015, +0.0057] | [−0.0010, +0.0052] |
| inception1d | 123 | +0.0023 | [−0.0011, +0.0056] | [−0.0005, +0.0051] |
| inception1d | 2024 | +0.0019 | [−0.0021, +0.0057] | [−0.0013, +0.0050] |

## Exploratory analyses (closed list fixed before fold 10 was opened)

### E1: Δ = AUROC(signal + age/sex) − AUROC(signal only)

| network | Δ | 95% CI | 90% CI |
|---|---|---|---|
| resnet1d_wang | +0.0007 | [−0.0008, +0.0023] | [−0.0006, +0.0021] |
| inception1d | +0.0013 | [−0.0009, +0.0036] | [−0.0006, +0.0032] |

### E2: per-class H3 deltas (uncorrected for multiplicity)

| network | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| resnet1d_wang | +0.0033 [+0.0014, +0.0051] | +0.0032 [+0.0004, +0.0061] | +0.0029 [+0.0006, +0.0054] | +0.0050 [+0.0004, +0.0102] | −0.0005 [−0.0063, +0.0052] |
| inception1d | +0.0004 [−0.0023, +0.0032] | +0.0047 [+0.0005, +0.0092] | −0.0006 [−0.0045, +0.0031] | +0.0020 [−0.0042, +0.0079] | −0.0022 [−0.0097, +0.0049] |

### E3: macro-AUROC drop when device/site/nurse are jointly permuted across test records (20 permutations, signal + all models)

| network | unpermuted | mean drop | min drop | max drop |
|---|---|---|---|---|
| resnet1d_wang | 0.9390 | 0.0053 | 0.0036 | 0.0069 |
| inception1d | 0.9349 | 0.0042 | 0.0028 | 0.0051 |

All 20 drops, resnet1d_wang: 0.0059, 0.0036, 0.0057, 0.0069, 0.0052, 0.0062, 0.0053, 0.0042, 0.0049, 0.0060, 0.0048, 0.0058, 0.0063, 0.0050, 0.0041, 0.0054, 0.0041, 0.0056, 0.0052, 0.0058

All 20 drops, inception1d: 0.0048, 0.0030, 0.0045, 0.0051, 0.0041, 0.0049, 0.0041, 0.0035, 0.0035, 0.0049, 0.0041, 0.0041, 0.0051, 0.0036, 0.0028, 0.0045, 0.0036, 0.0045, 0.0046, 0.0042
