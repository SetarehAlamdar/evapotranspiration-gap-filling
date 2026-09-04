"""
models.py

Two architectures for ET gap-filling, both consuming the unified
(2W+1)-length sequence produced by model_dataset.ETWindowDataset (weather
channels + ET channel + mask channel, with the center position always
carrying ET=0, mask=0):

1. BiLSTMGapFiller
   A bidirectional LSTM reads the full sequence. Because it's
   bidirectional, the hidden state at the center timestep has already
   "seen" both the steps before AND after it (forward pass carries context
   up to and including the center; backward pass carries context from the
   end back to the center) -- exactly the bidirectional-context design we
   agreed on for gap-filling (as opposed to forecasting).

2. ConvTransformerGapFiller
   A 1D convolution first embeds local temporal patterns (e.g. short-term
   trends around each point), then a standard Transformer encoder
   (bidirectional/non-causal self-attention -- every position can attend
   to every other position, including both before and after the center)
   refines these embeddings using positional encoding. The center
   position's final representation is used for prediction.

Both models output a single scalar (predicted ET at the center position)
from a small MLP head applied to the center timestep's learned
representation.
"""

import math
import torch
import torch.nn as nn


class BiLSTMGapFiller(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        seq_len = x.shape[1]
        center = seq_len // 2
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_dim*2)
        center_repr = lstm_out[:, center, :]    # (batch, hidden_dim*2)
        out = self.head(center_repr).squeeze(-1)  # (batch,)
        return out


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to the conv-embedded sequence
    so the Transformer knows each position's location relative to the center gap."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class ConvTransformerGapFiller(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 500,
    ):
        super().__init__()
        assert conv_kernel_size % 2 == 1, "conv_kernel_size should be odd to preserve sequence length with 'same' padding"

        # 1D conv over the time axis: embeds local temporal patterns.
        # Conv1d expects (batch, channels, seq_len), so we transpose in forward().
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )
        self.conv_activation = nn.ReLU()

        self.input_proj = nn.Linear(conv_channels, d_model) if conv_channels != d_model else nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._last_attn_weights = None  # populated on request, for interpretability later

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        seq_len = x.shape[1]
        center = seq_len // 2

        conv_in = x.transpose(1, 2)                  # (batch, n_features, seq_len)
        conv_out = self.conv_activation(self.conv(conv_in))  # (batch, conv_channels, seq_len)
        conv_out = conv_out.transpose(1, 2)           # (batch, seq_len, conv_channels)

        embedded = self.input_proj(conv_out)          # (batch, seq_len, d_model)
        embedded = self.pos_encoding(embedded)

        # Standard TransformerEncoder is already non-causal (no mask applied),
        # so every position attends to every other position -- bidirectional
        # context by construction, matching the gap-filling design goal.
        encoded = self.transformer(embedded)           # (batch, seq_len, d_model)

        center_repr = encoded[:, center, :]
        out = self.head(center_repr).squeeze(-1)
        return out


if __name__ == "__main__":
    # Self-test with random data (no real data needed to check shapes/forward pass)
    torch = __import__("torch")
    batch, seq_len, n_features = 8, 21, 7  # matches (2*10+1, 5 weather + ET + mask) example

    x = torch.randn(batch, seq_len, n_features)

    lstm_model = BiLSTMGapFiller(n_features=n_features, hidden_dim=32, num_layers=2)
    out_lstm = lstm_model(x)
    print(f"BiLSTMGapFiller output shape: {out_lstm.shape} (expect ({batch},))")
    assert out_lstm.shape == (batch,)

    ct_model = ConvTransformerGapFiller(
        n_features=n_features, conv_channels=32, d_model=32, n_heads=4, n_transformer_layers=2,
    )
    out_ct = ct_model(x)
    print(f"ConvTransformerGapFiller output shape: {out_ct.shape} (expect ({batch},))")
    assert out_ct.shape == (batch,)

    print("Both models produce correctly-shaped output for a forward pass.")
