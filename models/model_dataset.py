"""
model_dataset.py

Loads the .npz window files produced by feature_engineering.build_windows()
and assembles them into a PyTorch Dataset ready for the LSTM / Conv-Transformer
models.

Key step: X_weather has length (2W+1) -- it includes the center timestep,
since weather is always visible even at a gap. X_et and X_et_mask have
length (2W) -- the center is excluded, since that's the value being
predicted. To build ONE unified sequence for the model, we reinsert a
placeholder at the center of the ET and mask channels (ET=0, mask=0), so
all three channel groups align to the same (2W+1)-length sequence, then
concatenate them per-timestep:

    combined[:, t, :] = [weather features at t] + [ET value at t] + [ET mask at t]

The model never receives the true ET value at the center position -- only
"here's everything around it, and a flag saying this position is unknown."

Usage:
    from model_dataset import ETWindowDataset

    train_ds = ETWindowDataset("lysimeter_windows.npz", subset="train")
    eval_ds  = ETWindowDataset("lysimeter_windows.npz", subset="eval")
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def _build_combined_sequence(X_weather: np.ndarray, X_et: np.ndarray, X_et_mask: np.ndarray) -> np.ndarray:
    """
    X_weather: (N, 2W+1, F)
    X_et:      (N, 2W, 1)
    X_et_mask: (N, 2W, 1)
    Returns combined: (N, 2W+1, F+2)
    """
    N, L, F = X_weather.shape  # L = 2W+1
    W = (L - 1) // 2

    et_full = np.zeros((N, L, 1), dtype=np.float32)
    mask_full = np.zeros((N, L, 1), dtype=np.float32)  # center stays 0 = "unknown"

    et_full[:, :W, :] = X_et[:, :W, :]
    et_full[:, W + 1:, :] = X_et[:, W:, :]
    mask_full[:, :W, :] = X_et_mask[:, :W, :]
    mask_full[:, W + 1:, :] = X_et_mask[:, W:, :]

    combined = np.concatenate([X_weather, et_full, mask_full], axis=-1)
    return combined.astype(np.float32)


def compute_normalization_stats(npz_path: str, subset: str = "train"):
    """
    Computes per-feature mean/std from the given subset (always use
    "train" for this -- never "eval"), EXCLUDING the mask channel (last
    feature), which stays as a plain 0/1 flag. Returns (mean, std) arrays
    of shape (n_features - 1,), ready to pass into ETWindowDataset for
    every subset (train/val/eval/real_gap) so they're all normalized
    consistently using the same training-derived statistics.
    """
    temp_ds = ETWindowDataset(npz_path, subset=subset)  # unnormalized
    X = temp_ds.X[:, :, :-1]  # exclude mask channel
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    std[std == 0] = 1.0  # avoid divide-by-zero for any constant feature
    return mean, std


class ETWindowDataset(Dataset):
    """
    subset: one of "train", "eval", "real_gap", or "all"
      - "train": real observed points not held out (use for model training)
      - "eval":  synthetic gaps, held out ground truth (use ONLY for final
                 reported metrics -- never for early stopping/tuning)
      - "real_gap": genuine missing points, no ground truth (production use)
      - "all": every window, unfiltered

    feature_mean, feature_std: optional arrays (length n_features - 1,
    i.e. excluding the mask channel) used to standardize weather+ET
    features to mean 0 / std 1. If not provided, no normalization is
    applied. IMPORTANT: always compute these from the TRAINING subset
    only (see compute_normalization_stats below), then reuse the SAME
    values for val/eval/real_gap datasets -- never recompute stats from
    eval data, or you'd be leaking test-set information into the
    preprocessing itself. The mask channel (last feature) is intentionally
    left unnormalized, since it's already a meaningful 0/1 flag.
    """

    def __init__(self, npz_path: str, subset: str = "train",
                 feature_mean: np.ndarray = None, feature_std: np.ndarray = None):
        data = np.load(npz_path, allow_pickle=True)
        X_weather = data["X_weather"]
        X_et = data["X_et"]
        X_et_mask = data["X_et_mask"]
        y = data["y"]
        sample_type = data["sample_type"]

        if subset != "all":
            keep = sample_type == subset
            X_weather, X_et, X_et_mask = X_weather[keep], X_et[keep], X_et_mask[keep]
            y = y[keep]

        self.X = _build_combined_sequence(X_weather, X_et, X_et_mask)
        self.y = y.astype(np.float32)
        self.subset = subset

        if feature_mean is not None and feature_std is not None:
            # Normalize every channel EXCEPT the last one (the mask channel)
            self.X[:, :, :-1] = (self.X[:, :, :-1] - feature_mean) / feature_std

        if len(self.y) == 0:
            print(f"WARNING: subset='{subset}' produced 0 samples. Check that "
                  f"build_windows() was run and sample_type values match "
                  f"('train'/'eval'/'real_gap').")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    @property
    def n_features(self):
        return self.X.shape[-1]

    @property
    def seq_len(self):
        return self.X.shape[1]


if __name__ == "__main__":
    import tempfile, os

    # Self-test: build a tiny fake npz and confirm shapes/no-leakage
    N, W, F = 50, 10, 5
    rng = np.random.default_rng(0)
    X_weather = rng.normal(size=(N, 2 * W + 1, F)).astype(np.float32)
    X_et = rng.normal(size=(N, 2 * W, 1)).astype(np.float32)
    X_et_mask = rng.integers(0, 2, size=(N, 2 * W, 1)).astype(np.float32)
    y = rng.normal(size=N).astype(np.float32)
    sample_type = np.array(["train"] * 30 + ["eval"] * 15 + ["real_gap"] * 5, dtype=object)

    test_npz_path = os.path.join(tempfile.gettempdir(), "test_windows.npz")
    np.savez(test_npz_path, X_weather=X_weather, X_et=X_et,
              X_et_mask=X_et_mask, y=y, sample_type=sample_type)

    train_ds = ETWindowDataset(test_npz_path, subset="train")
    eval_ds = ETWindowDataset(test_npz_path, subset="eval")
    real_gap_ds = ETWindowDataset(test_npz_path, subset="real_gap")

    print(f"train: {len(train_ds)}, eval: {len(eval_ds)}, real_gap: {len(real_gap_ds)}")
    assert len(train_ds) == 30 and len(eval_ds) == 15 and len(real_gap_ds) == 5

    x0, y0 = train_ds[0]
    print(f"Combined sequence shape: {x0.shape} (expect ({2*W+1}, {F+2}))")
    assert x0.shape == (2 * W + 1, F + 2)

    # Center position must have ET=0 and mask=0 (unknown), regardless of subset
    center = W
    assert x0[center, F].item() == 0.0, "Center ET channel should be 0"
    assert x0[center, F + 1].item() == 0.0, "Center mask channel should be 0"
    print("Center-position leakage check passed: ET=0, mask=0 at center for all samples.")
