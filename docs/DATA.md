# Dataset access and input preparation

The paper uses 24 selected datasets. `data/datasets24.json` identifies every exact input by source repository, pinned commit, path, original file SHA256, array shape, dtype, row order and X/y array SHA256. `data/dataset_metadata.csv` supplies the paper's counts and anomaly fractions. The dataset IDs are preserved verbatim, including source-provided suffixes.

## Sources

| Source | Pinned commit | Selected inputs |
|---|---|---:|
| [ADBench](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965) | `3dac8221081e190f157d78e93bfa8867f90d0965` | 12 |
| [Outlier-detection](https://github.com/BELLoney/Outlier-detection/tree/209fbdba90ffbd3aa9b20de6944aaa543e012a7c) | `209fbdba90ffbd3aa9b20de6944aaa543e012a7c` | 12 |

The ADBench repository carries a [BSD-2-Clause license](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/LICENSE); users should also consult each original dataset's terms. No repository-wide LICENSE file was found at the pinned Outlier-detection commit. This package does not infer permission to relicense those datasets. It supplies their original download locations and exact preparation instructions instead of redistributing raw feature files.

## Conversion

- ADBench NPZ: retain X's original dtype, numerical values and row order; convert y to a one-dimensional int64 vector.
- Eleven Outlier-detection MAT files: read the published `trandata` matrix; retain all feature columns with their original dtype; use its final column as binary y. Do not redo the publisher's downsampling, category encoding or min-max transformation.
- Yeast1141 CSV: exclude the sequence-name identifier, retain `mcg,gvh,alm,mit,erl,pox,vac,nuc` in that order as float64, and label ERL as anomaly and CYT/NUC/MIT as normal. Unexpected classes cause an error instead of silently discarding rows.

After conversion, verify X/y array SHA256 values. An NPZ container hash can differ with ZIP metadata despite identical arrays; the manifest records the historical container hash as additional provenance, while the preparation script requires the pinned source file hash and exact array hashes.

The supplied upload package contains metadata and derived experimental tables. It does not contain raw feature matrices, pretrained models or per-sample score arrays. Reproduction commands obtain the pinned feature data and generate model/score artifacts locally.

## Preprocessing and splits

Apply original upstream full-X `normalize_data` before splitting; preserve the original dtype calculation and cast to FP32 afterward. A train-normal-only scaler is a different protocol and is not used here.

For each of seeds 42/43/44, shuffle normal-row IDs with `np.random.RandomState(seed).shuffle`. Use the first floor(n_normal/2) normal IDs for training. Evaluation IDs are the remaining shuffled normals followed by all anomaly IDs in original order. `data/split_audit.csv` records 72 split cases and the three array hashes for train IDs, test IDs and test labels.

Mixed training uses all original rows as unlabelled inputs and evaluates the same full X; it is transductive. Labels are used for reported metrics and development selection, not as training targets. Full-X standardization in normal-only experiments is also transductive preprocessing.
