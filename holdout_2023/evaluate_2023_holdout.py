import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_dataset import ETWindowDataset, compute_normalization_stats
from models import BiLSTMGapFiller, ConvTransformerGapFiller
from train import evaluate_model

MODEL_DIR = "block_models_w96_augmented"

# Each 2023 npz, paired with the ORIGINAL 2018-2022 npz used to fit the
# normalization statistics that the model was actually trained with (never
# recompute normalization from the 2023 data itself -- that would leak
# information about the test year into preprocessing).
COMBOS = [
    ("lysimeter_2023_point_windows.npz", "lysimeter_point_w96_windows.npz", "lysimeter_point_w96"),
    ("ec_2023_point_windows.npz", "ec_point_w96_windows.npz", "ec_point_w96"),
    ("lysimeter_2023_block_short3h_windows.npz", "lysimeter_block_short3h_w96_windows.npz", "lysimeter_block_short3h_w96"),
    ("ec_2023_block_short3h_windows.npz", "ec_block_short3h_w96_windows.npz", "ec_block_short3h_w96"),
    ("lysimeter_2023_block_long24h_windows.npz", "lysimeter_block_long24h_w96_windows.npz", "lysimeter_block_long24h_w96"),
    ("ec_2023_block_long24h_windows.npz", "ec_block_long24h_w96_windows.npz", "ec_block_long24h_w96"),
]
MODEL_TYPES = ["lstm", "conv_transformer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

results = []

for npz_2023, npz_original, ckpt_label in COMBOS:
    print(f"\n{'='*20} {ckpt_label} (2023 holdout) {'='*20}")

    # Normalization stats from the ORIGINAL training data, not 2023
    feature_mean, feature_std = compute_normalization_stats(npz_original, subset="train")

    # For the 2023 file, use subset="all" then filter to sample_type=="eval"
    # ourselves, since 2023 has no "train" split of its own -- every point
    # here is either a synthetic gap (our test target) or a real gap.
    raw = np.load(npz_2023, allow_pickle=True)
    sample_type = raw["sample_type"]
    n_eval = (sample_type == "eval").sum()
    print(f"  2023 holdout eval points available: {n_eval}")

    eval_ds = ETWindowDataset(npz_2023, subset="eval", feature_mean=feature_mean, feature_std=feature_std)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=0)
    n_features = eval_ds.n_features

    for model_type in MODEL_TYPES:
        ckpt_path = os.path.join(MODEL_DIR, f"{model_type}_{ckpt_label}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found: {ckpt_path} -- skipping")
            continue

        if model_type == "lstm":
            model = BiLSTMGapFiller(n_features=n_features, hidden_dim=64, num_layers=2, dropout=0.2)
        else:
            model = ConvTransformerGapFiller(n_features=n_features, conv_channels=64, d_model=64,
                                               n_heads=4, n_transformer_layers=2, dropout=0.2)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        metrics = evaluate_model(model, eval_loader, device)
        print(f"  {model_type}: MAE={metrics['MAE']:.5f}, RMSE={metrics['RMSE']:.5f}, "
              f"NSE={metrics['NSE']:.4f} (n={metrics['n_samples']})")

        results.append({"label": ckpt_label, "model": model_type, **metrics})

results_df = pd.DataFrame(results)
results_df.to_csv("holdout_2023_dl_results.csv", index=False)
print(f"\n\n{'='*25} 2023 HOLD-OUT RESULTS SUMMARY {'='*25}")
print(results_df.to_string(index=False))
print("\nSaved holdout_2023_dl_results.csv")
print("\nThese models were trained EXCLUSIVELY on 2018-2022 data. 2023 was never used "
      "in training, validation, or hyperparameter selection at any point.")
