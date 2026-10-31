# Result tables, units and provenance

The archived CSV files retain the reported numerical precision. Local absolute paths have been removed; source hashes and dataset/method IDs remain. Public summary coefficient cells are blank for DDAE references because DOCL's lambda/c do not apply to that method.

| Supplement | File(s) under `results/supplementary/` | Rows | Meaning |
|---|---|---:|---|
| S1 | `Table_S1_normal_only_seed_noise.csv` | 432 | Three principal methods, normal-only training; 24 datasets x 3 seeds x A/B |
| S2 | `Table_S2_mixed_seed_noise.csv` | 432 | Same three principal methods, transductive mixed training |
| S3 | `Table_S3_parameter_search_summary.csv` | 29 | All distinct lambda/c configurations; completed seeds are specified |
| S3 detail | `Table_S3_parameter_search_seed_noise.csv` | 2736 | Seed/bank detail for 14 three-seed and 15 single-seed configurations |
| S4 | `Table_S4_sampling_seed_noise.csv` | 576 | Four noise-pair sampling strategies |
| S5 | `Table_S5_component_seed_noise.csv` | 864 | Complete DOCL, DDAE reference and four component controls |
| S5 exceptions | `Table_S5_reverse_local_exceptions.csv` | 4 | Datasets with higher mean AP under reverse order |
| S6 | `Table_S6_normal_representation_diagnostics.csv` | 720 | Deduplicated complete-DOCL and control probe records |

## Metrics and repetitions

AP and ROC-AUC are reported as percentages in fields ending in `_percent`. A difference in AP is in percentage points (pp). Some all-baseline CSVs also retain original 0–1 metric columns; use their explicit `_percent` columns for a percentage table. The PR-AUC fields marked `trapezoidal` integrate the precision–recall curve and are not interchangeable with average precision.

Each training seed produces one model. A/B score that same model using independent fixed scoring-noise banks; they are not six independently trained models. Average A/B within each dataset/seed. For the macro AP, average the training seeds within each dataset and then weight datasets equally. For the reported training-seed SD, compute the macro AP separately for each of the three training seeds after bank averaging, then calculate sample SD with ddof=1. SD is not a confidence interval.

For parameter screening, only the listed seeds exist. A single-seed row has no training-seed SD; missing SD is not zero. Fourteen configurations have all three seeds; fifteen have seed 42 only. The final chosen pair is lambda=1.2,c=2.0, selected from completed three-seed comparisons.

## Source hashes

`source_score_sha256`/`score_sha256` identify the original saved score artifact. `source_receipt_sha256`/`receipt_sha256` identify its original execution receipt. These original artifacts are not bundled in this compact repository. Their hashes document lineage; the table-level checks do not claim to recompute AP from unavailable per-sample arrays.

`provenance/scientific_source_manifest.json` separately identifies byte-identical scientific source copies. `checksums.sha256` covers the supplied repository files. `provenance/archive_validation.json` distinguishes archived-record verification and synthetic implementation checks from new training or rescoring.

## External baselines

`results/baselines/baseline_comparison_complete.csv` retains all-method summaries, protocol disclosures and coverage. `per_dataset.csv` and `per_seed_AB.csv` contain finer results. `failure_and_incomplete_coverage.csv` preserves unsuccessful/unavailable cases. A result on fewer than 24 complete datasets is not a full-24 result; unavailable outcomes are not replaced with zero. Fixed additional baseline configurations are in `configs/additional_baseline_configurations.json`.

The compact executable package covers DOCL, matched DDAE and the local DDAE-C implementation, parameter settings, pairing rules and component controls. It does not bundle all third-party source trees or the full execution environments of all external baselines.

## Component interpretation

The four controls change order direction, remove the margin, replace it with a mean-matched fixed margin, or decouple Gaussian noise vectors. The fixed margin at c=2 is 0.2866060435771942. Matching its mean does not match the hinge activation pattern or gradients. All controls retain uniform time-step pair sampling. These controls must not be confused with the four alternative time-step pairing rules in S4.

The four controls have 288 genuinely new trained models at the selected coefficients. The complete-DOCL/DDAE references, pairing and mixed records can reuse audited existing runs; this is not described as an additional fresh retraining. S6 reuses common normal probe inputs across models; a probe-bank label is distinct from an A/B anomaly-scoring bank.
