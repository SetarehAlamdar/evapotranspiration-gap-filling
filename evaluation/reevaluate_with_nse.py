import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_dataset import ETWindowDataset, compute_normalization_stats
from models import BiLSTMGapFiller, ConvTransformerGapFiller
from train import evaluate_model

MODEL_DIR = "block_models_w96_augmented"
BATCH_SIZE = 128

# All 12 final-configuration combinations (window=96, context-dropout augmented)
COMBOS = [
    ("lysimeter_point_w96_windows.npz", "lysimeter_point_w96"),
    ("ec_point_w96_windows.npz", "ec_point_w96"),
    ("lysimeter_block_short3h_w96_windows.npz", "lysimeter_block_short3h_w96"),
    ("ec_block_short3h_w96_windows.npz", "ec_block_short3h_w96"),
    ("lysimeter_block_long24h_w96_windows.npz", "lysimeter_block_long24h_w96"),
    ("ec_block_long24h_w96_windows.npz", "ec_block_long24h_w96"),
]
MODEL_TYPES = ["lstm", "conv_transformer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []

for npz_path, label in COMBOS:
    print(f"\n{'='*20} {label} {'='*20}")

    feature_mean, feature_std = compute_normalization_stats(npz_path, subset="train")
    eval_ds = ETWindowDataset(npz_path, subset="eval",
                               feature_mean=feature_mean, feature_std=feature_std)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_features = eval_ds.n_features

    for model_type in MODEL_TYPES:
        ckpt_path = os.path.join(MODEL_DIR, f"{model_type}_{label}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found: {ckpt_path} -- skipping")
            continue

        if model_type == "lstm":
            model = BiLSTMGapFiller(n_features=n_features, hidden_dim=64, num_layers=2, dropout=0.2)
        else:
            model = ConvTransformerGapFiller(
                n_features=n_features, conv_channels=64, d_model=64,
                n_heads=4, n_transformer_layers=2, dropout=0.2,
            )

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        metrics = evaluate_model(model, eval_loader, device)
        print(f"  {model_type}: MAE={metrics['MAE']:.5f}, RMSE={metrics['RMSE']:.5f}, "
              f"NRMSE={metrics['NRMSE_pct']:.2f}%, NSE={metrics['NSE']:.4f}")

        results.append({"dataset_block": label, "model": model_type, **metrics})

results_df = pd.DataFrame(results)
out_path = os.path.join(MODEL_DIR, "final_dl_metrics_with_nse.csv")
results_df.to_csv(out_path, index=False)

print(f"\n\n{'='*20} FINAL METRICS (with NSE) {'='*20}")
print(results_df.to_string(index=False))
print(f"\nSaved {out_path}")
