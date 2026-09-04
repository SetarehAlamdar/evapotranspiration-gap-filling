"""
context_dropout.py

Training-time data augmentation: randomly blanks out an ADDITIONAL
contiguous chunk of the ET/mask context (on top of whatever the window
already has masked), for a random subset of training examples.

Why this helps: most training windows have plenty of real ET history
nearby (long gaps are relatively rare in the data), so the model rarely
practices operating with sparse local context during training. But at
evaluation time on long block gaps, points near the center of the gap
have almost no real ET context left. This creates a train/eval
distribution mismatch -- the model is being tested in a regime it barely
saw during training.

Context dropout closes this gap for free (no architecture change, no
larger sequences, no extra compute per step): during training, we
sometimes simulate an even sparser context than the window naturally
provides, forcing the model to learn to rely more on weather variables
alone when local ET history isn't available.

IMPORTANT: only ever wrap the TRAINING subset with this. Validation
(early stopping) and the held-out evaluation set must never be augmented,
or you'd be measuring performance on artificially altered data rather
than the real held-out gaps.

Usage:
    from context_dropout import ContextDropoutWrapper

    train_subset, val_subset = make_train_val_split(full_train_ds, val_fraction=0.15)
    train_subset = ContextDropoutWrapper(train_subset)   # augmented
    # val_subset stays exactly as-is (unaugmented)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ContextDropoutWrapper(Dataset):
    """
    Wraps any dataset that returns (X, y) pairs where X is a
    (seq_len, n_features) tensor whose LAST TWO feature columns are the
    ET value channel and the ET mask channel (exactly the layout produced
    by model_dataset._build_combined_sequence). For a random subset of
    examples (controlled by dropout_prob), zeroes out an additional random
    contiguous block of the ET+mask channels.

    dropout_prob: fraction of training examples that get this augmentation
                  applied at all (others pass through unchanged).
    max_dropout_fraction: the additional dropped block's length is drawn
                  uniformly between 10% and this fraction of the sequence
                  length, so we see a range of severities, not just one
                  fixed size.
    """

    def __init__(self, base_dataset, dropout_prob: float = 0.3,
                 max_dropout_fraction: float = 0.5, seed: int = None):
        self.base = base_dataset
        self.dropout_prob = dropout_prob
        self.max_dropout_fraction = max_dropout_fraction
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, y = self.base[idx]

        if self.rng.random() < self.dropout_prob:
            X = X.clone()
            seq_len = X.shape[0]

            frac = self.rng.uniform(0.1, self.max_dropout_fraction)
            block_len = max(1, min(seq_len, int(round(frac * seq_len))))
            max_start = seq_len - block_len
            start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
            end = start + block_len

            X[start:end, -2] = 0.0  # ET value channel
            X[start:end, -1] = 0.0  # ET mask channel (0 = "treat as unobserved")

        return X, y


if __name__ == "__main__":
    # Self-test using a tiny fake base dataset (no torch training needed,
    # just checking the augmentation logic itself)
    class FakeDataset(Dataset):
        def __init__(self, n=20, seq_len=49, n_features=13):
            self.n = n
            self.seq_len = seq_len
            self.n_features = n_features

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            X = torch.ones(self.seq_len, self.n_features)  # all 1s, easy to check zeroing
            y = torch.tensor(0.5)
            return X, y

    base = FakeDataset(n=200, seq_len=49, n_features=13)
    wrapped = ContextDropoutWrapper(base, dropout_prob=1.0, max_dropout_fraction=0.5, seed=0)

    n_examples_with_dropout = 0
    for i in range(len(wrapped)):
        X, y = wrapped[i]
        et_mask_col = X[:, -1]
        n_zeroed = (et_mask_col == 0).sum().item()
        if n_zeroed > 0:
            n_examples_with_dropout += 1
            assert n_zeroed <= int(0.5 * 49) + 1, "Dropout block exceeded max_dropout_fraction!"
        # Confirm weather features (all other columns) were NEVER touched
        assert (X[:, :-2] == 1.0).all(), "Augmentation incorrectly modified weather features!"

    print(f"{n_examples_with_dropout} / {len(wrapped)} examples had dropout applied "
          f"(dropout_prob=1.0, so should be all of them)")
    assert n_examples_with_dropout == len(wrapped)

    # Now check dropout_prob=0.3 gives roughly 30% (not exactly, since random)
    wrapped_partial = ContextDropoutWrapper(base, dropout_prob=0.3, seed=1)
    n_with = sum(1 for i in range(len(wrapped_partial))
                 if (wrapped_partial[i][0][:, -1] == 0).any())
    pct = 100 * n_with / len(wrapped_partial)
    print(f"With dropout_prob=0.3: {pct:.1f}% of examples augmented (expect ~30%)")

    print("\nAll self-tests passed.")
