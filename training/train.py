"""
train.py

Training loop shared by both BiLSTMGapFiller and ConvTransformerGapFiller,
plus an evaluation function computing MAE/RMSE/NRMSE.

Key design points:

1. The "eval" subset (synthetic gaps) is NEVER touched during training or
   early stopping -- it's reserved exclusively for the final reported
   metrics. Instead, we carve out an internal validation split from the
   "train" pool itself, used only to decide when to stop training.

2. Early stopping: training halts once validation loss hasn't improved for
   `patience` consecutive epochs, and we keep the best-performing model
   weights seen so far (not just whatever the last epoch happened to be).

3. Works identically for both model architectures -- pass in whichever
   model instance you want trained.

Usage:
    python train.py
(edit the CONFIG section below to point at lysimeter_windows.npz or
ec_windows.npz, and choose which model to train)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model_dataset import ETWindowDataset, compute_normalization_stats
from models import BiLSTMGapFiller, ConvTransformerGapFiller
from context_dropout import ContextDropoutWrapper


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    verbose: bool = True,
):
    """
    Trains `model` using MSE loss and Adam, with early stopping based on
    validation loss. Returns the model loaded with the BEST validation
    weights seen (not necessarily the final epoch's weights), plus a
    history dict of per-epoch train/val loss for plotting later.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val improvement for {patience} epochs). "
                          f"Best val_loss={best_val_loss:.5f}")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, loader, device):
    """
    Computes MAE, RMSE, NRMSE, and NSE (Nash-Sutcliffe Efficiency) over
    every batch in `loader`. Call this ONLY with the "eval" subset for
    final reported results.

    NSE = 1 - (sum of squared residuals) / (sum of squared deviations from
    the observed mean). NSE=1 is a perfect match; NSE=0 is equivalent to
    always predicting the observed mean; negative values are worse than
    that baseline.
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    mae = np.mean(np.abs(all_preds - all_true))
    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    nrmse = 100 * rmse / np.mean(all_true)
    ss_res = np.sum((all_true - all_preds) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    nse = 1 - ss_res / ss_tot

    return {"MAE": mae, "RMSE": rmse, "NRMSE_pct": nrmse, "NSE": nse, "n_samples": len(all_true)}


def make_train_val_split(train_dataset, val_fraction: float = 0.15, seed: int = 42):
    """Randomly splits the 'train' subset into an actual-training slice and
    an internal validation slice used only for early stopping."""
    n_val = int(round(len(train_dataset) * val_fraction))
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)
    return train_subset, val_subset


if __name__ == "__main__":
    # ------------------------- CONFIG -------------------------
    NPZ_PATH = "lysimeter_windows.npz"     # or "ec_windows.npz"
    MODEL_TYPE = "lstm"                     # "lstm" or "conv_transformer"
    BATCH_SIZE = 128
    EPOCHS = 50
    PATIENCE = 7
    LEARNING_RATE = 1e-3
    # ------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Computing normalization statistics from the training subset...")
    feature_mean, feature_std = compute_normalization_stats(NPZ_PATH, subset="train")

    full_train_ds = ETWindowDataset(NPZ_PATH, subset="train",
                                     feature_mean=feature_mean, feature_std=feature_std)
    eval_ds = ETWindowDataset(NPZ_PATH, subset="eval",
                               feature_mean=feature_mean, feature_std=feature_std)

    train_subset, val_subset = make_train_val_split(full_train_ds, val_fraction=0.15)
    print(f"Train: {len(train_subset)} | Internal val (early stopping only): {len(val_subset)} "
          f"| Held-out eval (final metrics only): {len(eval_ds)}")

    # Context-dropout augmentation applied ONLY to the training subset.
    # val_subset and eval_ds are deliberately left unaugmented, since they
    # need to reflect real, unaltered data for honest early-stopping
    # decisions and final reported metrics.
    train_subset = ContextDropoutWrapper(train_subset, dropout_prob=0.3, max_dropout_fraction=0.5)
    print("Applied context-dropout augmentation to training subset "
          "(dropout_prob=0.3, max_dropout_fraction=0.5).")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    n_features = full_train_ds.n_features

    if MODEL_TYPE == "lstm":
        model = BiLSTMGapFiller(n_features=n_features, hidden_dim=64, num_layers=2, dropout=0.2)
    elif MODEL_TYPE == "conv_transformer":
        model = ConvTransformerGapFiller(
            n_features=n_features, conv_channels=64, d_model=64,
            n_heads=4, n_transformer_layers=2, dropout=0.2,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    print(f"\nTraining {MODEL_TYPE}...")
    model, history = train_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
    )

    print("\nEvaluating on held-out synthetic gaps (final reported metrics)...")
    metrics = evaluate_model(model, eval_loader, device)
    print(f"MAE:   {metrics['MAE']:.5f}")
    print(f"RMSE:  {metrics['RMSE']:.5f}")
    print(f"NRMSE: {metrics['NRMSE_pct']:.2f}%")
    print(f"(n = {metrics['n_samples']} held-out samples)")

    torch.save(model.state_dict(), f"{MODEL_TYPE}_{NPZ_PATH.replace('.npz', '')}_best.pt")
    print(f"\nSaved model weights to {MODEL_TYPE}_{NPZ_PATH.replace('.npz', '')}_best.pt")
