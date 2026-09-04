# evapotranspiration-gap-filling

Deep learning (LSTM and Convolutional-Transformer) gap-filling of half-hourly
evapotranspiration (ET) data from weighing lysimeters and eddy covariance (EC)
towers, benchmarked against linear interpolation and Marginal Distribution
Sampling (MDS) across three gap-duration categories (isolated point, short
3-hour blocks, long 24-hour blocks).

Companion code for: *[Paper title, once finalized]*, submitted to *Journal of
Hydrology*.

## Overview

Models are trained and evaluated using a synthetic gap-masking framework: real
observed values are deliberately withheld (masked) so that final accuracy is
always checked against genuinely observed measurements, never against
previously-imputed values. Final generalization is additionally validated on
a fully independent holdout year (2023) never used in any stage of model
development.

## Repository structure

```
├── data_loading/
│   ├── lysimeter_loader.py       # Loads and normalizes yearly lysimeter CSVs
│   └── ec_loader.py              # Loads and normalizes yearly EC CSVs
├── preprocessing/
│   ├── gap_masking.py            # Isolated single-point synthetic masking
│   ├── block_gap_masking.py      # Contiguous block masking (short/long)
│   └── feature_engineering.py    # Weather gap-filling, temporal features, windowing
├── models/
│   ├── models.py                 # BiLSTMGapFiller, ConvTransformerGapFiller
│   ├── model_dataset.py          # ETWindowDataset, normalization
│   └── context_dropout.py        # Context-dropout training augmentation
├── training/
│   └── train.py                  # train_model, evaluate_model, train/val split
├── evaluation/
│   ├── baselines.py              # Linear interpolation and MDS baselines
│   └── uncertainty_and_significance.py  # Bootstrap CIs, paired significance tests
├── holdout_2023/
│   ├── mask_2023_holdout.py          # Applies masking to the independent holdout year
│   ├── build_2023_windows.py         # Builds model input windows for 2023
│   ├── evaluate_2023_holdout.py      # Runs trained models on 2023 (inference only)
│   └── baselines_2023_holdout.py     # Runs baselines on 2023
├── figures/                      # Scripts generating all paper figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Developed and tested with Python 3.12.7 and PyTorch 2.13.0+cu130 on an NVIDIA
RTX 4060 Laptop GPU. CPU-only execution is supported but substantially slower
for training.

## Reproducing the results

1. **Load and combine raw data**
   ```bash
   python data_loading/lysimeter_loader.py
   python data_loading/ec_loader.py
   ```
2. **Apply synthetic gap masking** (isolated point, short 3h, long 24h; run for
   both lysimeter and EC datasets)
   ```bash
   python preprocessing/gap_masking.py
   python preprocessing/block_gap_masking.py
   ```
3. **Build model input windows** (window half-size = 96 steps / 48 hours)
   ```bash
   python preprocessing/feature_engineering.py
   ```
4. **Train models** (LSTM and Conv-Transformer, with context-dropout
   augmentation)
   ```bash
   python training/train.py
   ```
5. **Evaluate on the held-out 2018–2022 test set and run baselines**
   ```bash
   python evaluation/baselines.py
   python evaluation/uncertainty_and_significance.py
   ```
6. **Validate on the independent 2023 holdout year**
   ```bash
   python holdout_2023/aggregate_1min_to_30min.py   # lysimeter-specific correction step
   python holdout_2023/mask_2023_holdout.py
   python holdout_2023/build_2023_windows.py
   python holdout_2023/evaluate_2023_holdout.py
   python holdout_2023/baselines_2023_holdout.py
   ```

## Data availability

The raw lysimeter and eddy covariance data are not included in this
repository. [Add your data availability statement here — e.g., available
from the corresponding author upon reasonable request, or archived at
a specific repository with a DOI.]

## Citation

If you use this code, please cite:

```
[Full citation, once the paper is published]
```

## License

MIT License — see [LICENSE](LICENSE) for details.
