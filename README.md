# DOCL: Diffusion-Ordered Contrastive Learning

Reproduction material for **Diffusion-Ordered Contrastive Learning for Tabular Anomaly Detection**.

This repository contains the final DOCL scientific implementation, experimental configurations, data preparation instructions, and supplementary results for the **24 selected datasets**. It is maintained separately from the development workspace. The manuscript is in preparation; this repository does not imply journal acceptance or a published article DOI.

## Contents

| Location | Contents |
|---|---|
| `src/experiments/docl.py` | Clean-anchor, shared-noise weak/strong views and cosine ordering objective |
| `src/core.py` | Frozen 100-epoch trainer, evaluation-noise replay, original R49 scorer |
| `src/sampling3.py` | Fixed, main-linked and curriculum time-step pairing controls |
| `src/ablation_adapter.py` | Reverse order, zero margin, mean-matched fixed margin, independent-noise controls |
| `src/baselines/native7.py` | Original local baseline adapter; the portable entry point exposes its DDAE-C implementation |
| `src/vendor/AnoDDAE/` | Required upstream AnoDDAE files and their MIT license |
| `configs/` | Final protocol, all 29 parameter configurations, additional baseline configurations |
| `data/` | Dataset/source manifest, array checksums, metadata, split audit; no raw third-party feature files |
| `results/supplementary/` | Tables S1–S6, including parameter, pairing, component and normal-probe records |
| `results/baselines/` | All-method baseline summaries, dataset/seed/bank records, and incomplete coverage |
| `scripts/` | Portable data preparation, one-model execution, summary reproduction and archive verification |
| `provenance/` | Scientific source hashes, historical environments and preparation validation |

## Main verified results

All values below use the same 24 selected datasets, training seeds 42/43/44, two scoring-noise banks A/B, 100 total epochs, and the original R49 score. The selected DOCL coefficients are **lambda = 1.2, c = 2.0**. DOCL's 20 reconstruction warm-up epochs are included in the total of 100.

| Setting | Method | Macro AP (%) | Training-seed SD (pp) |
|---|---|---:|---:|
| Normal-only | Local same-protocol DDAE + R49 | 58.289039 | 1.803176 |
| Normal-only | DOCL + R49 | 62.207715 | 1.121250 |
| Transductive mixed | Local same-protocol DDAE + R49 | 24.309408 | 2.388685 |
| Transductive mixed | DOCL + R49 | 22.638290 | 1.599138 |

DOCL improves normal-only macro AP by **3.918677 percentage points** over the matched local DDAE + R49 reproduction. In mixed training it decreases AP by **1.671118 percentage points**. Both directions are retained in the archive. These baseline values are local reproductions, not the DDAE paper's published numbers.

## Environment and installation

Use Python 3.12. Install dependencies into a separate environment:

```bash
python -m venv .venv
```

Activate it on Windows with `.venv\Scripts\activate`, or on Linux/macOS with `source .venv/bin/activate`, then run:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is a portable dependency specification. Receipt-declared historical versions are retained in `provenance/recorded_environments.json`; they are not replaced by the portable requirements. Reported experiments used PyTorch 2.5.1 with CUDA 12.4 on an NVIDIA RTX 4070 Laptop GPU. Device, numerical-library and software-version differences can change floating-point trajectories. Source equality and the same nominal seed alone do not establish a bitwise reproduction.

## Verify the supplied archive and tables

This command uses only the supplied files and does not train or score a model:

```bash
python scripts/verify_archive.py
python scripts/summarize_results.py
```

For pairing or parameter tables:

```bash
python scripts/summarize_results.py --table results/supplementary/Table_S4_sampling_seed_noise.csv
python scripts/summarize_results.py --table results/supplementary/Table_S3_parameter_search_seed_noise.csv --group lambda_value margin_coefficient
```

## Prepare the exact dataset inputs

The source URLs are pinned to upstream commits. The preparation script checks the original downloaded file hash and the resulting X/y array hashes, including dtype and row order. It does not regenerate downsampled data or select alternative source files.

```bash
python scripts/prepare_data.py --list
python scripts/prepare_data.py --download --dataset all
```

Raw third-party feature data is obtained from the original providers rather than redistributed in this repository. Read `docs/DATA.md` for the sources, conversion rules and rights information. Prepared inputs are written to `data/raw/`, which is excluded from version control.

## Run one model explicitly

The entry point prints a plan unless `--run` is supplied. For example:

```bash
python scripts/run_experiment.py --dataset old__wine --seed 42 --method DOCL
```

To actually train and score that model:

```bash
python scripts/run_experiment.py --dataset old__wine --seed 42 --method DOCL --device cuda --output outputs/wine_docl_s42 --run
```

Supported methods are `DDAE`, `DOCL`, `DDAE-C`, `fixed10_35`, `mainlinked12`, `gapcurriculum`, `reverse`, `zero_margin`, `fixed_margin`, and `independent_noise`. Use `--setting mixed` for the transductive full-X setting. Use `--lambda-value` and `--c` to reproduce a recorded parameter configuration. A new output directory is required for each model. There is no automatic retry, hyperparameter search, controller, scheduled task or background execution.

Repeat each requested method/configuration over the listed 24 dataset IDs and seeds 42, 43 and 44. Each trained model is scored with A/B; A/B are not additional independent training runs. The full normal-probe function is retained as `ablation_adapter.diagnostics`, which requires the matching complete-DOCL checkpoint. Its archived probe records are Table S6.

The DDAE-C implementation uses explicitly documented local choices, including alpha = 0.5; it is not an author-confirmed complete reproduction. Its original adapter also contains branches for other methods whose third-party sources are not bundled here. The portable entry point does not expose those branches. All-method comparison results, fixed additional configurations and coverage records are supplied; a complete standalone execution environment for every external baseline is outside this package's scope.

## Protocol and interpretation

1. Apply upstream `normalize_data` to the complete raw X while preserving its input dtype behavior; cast network inputs to FP32 afterward.
2. For normal-only training, use NumPy's legacy normal-row shuffle: half of the normal rows train, and the remaining normal rows plus all anomalies evaluate. Check original row IDs against `data/split_audit.csv`.
3. Retain 100 total epochs, the original MLP/Adam settings, the original batch-size rule based on total N, and evaluation-noise consumption every 10 epochs. DOCL adds its auxiliary objective only after epoch 20.
4. Preserve independent private random streams for auxiliary views and scoring. R49 sums unsquared per-sample L2 reconstruction residuals across t = 1,...,49, starting from a newly perturbed original query at every time step.
5. Average scoring banks and training seeds within each dataset, then average datasets equally. Report sample SD across the three seed macro means. AP and trapezoidal PR-AUC are distinct metrics.

Dataset and coefficient selection used labelled development results. The reported collection is not an independent blind validation benchmark. The 29-configuration search includes both single-seed screening and three-seed confirmations; see `configs/parameter_search.json` rather than interpreting every row as a three-seed result.

The component controls were actually rerun for 288 models at the final coefficients. Matched pre-existing full-DOCL/DDAE references, pairing runs and mixed runs were independently checked and reused where appropriate. Reused runs are not represented as newly trained models. The reverse constraint outperforms forward DOCL on four datasets; all four exceptions and the negative mixed-training result are preserved. Component AP differences are not additive causal shares of the total improvement. No equal-computation reconstruction-only control is claimed.

## Attribution and reuse

AnoDDAE upstream: [sattarov/AnoDDAE](https://github.com/sattarov/AnoDDAE), commit `bc1c5cc827cd8f6a6aa1ef5e75441bfd5b858ec0`. Its original MIT notice is retained. See `LICENSES/README.md` for license scope. Third-party datasets retain their source terms.

For an exact repository snapshot, record the commit SHA associated with the downloaded files. This GitHub repository is a source and result repository; no Zenodo DOI or permanent archival deposit is claimed.
