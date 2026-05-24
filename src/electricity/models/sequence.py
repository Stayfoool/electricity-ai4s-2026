from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency for TFT-like experiments.
    torch = None
    nn = None


@dataclass(frozen=True)
class DailySequenceFrame:
    dates: list[pd.Timestamp]
    times: list[np.ndarray]
    features: np.ndarray
    targets: np.ndarray
    base_pred: np.ndarray
    residual: np.ndarray


def make_daily_sequence_frame(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    base_pred_col: str,
    feature_cols: list[str],
    residual_col: str = "residual",
) -> DailySequenceFrame:
    work = df[[time_col, target_col, base_pred_col, residual_col, *feature_cols]].copy()
    work["date"] = work[time_col].dt.normalize()
    work["slot"] = work[time_col].dt.hour * 4 + work[time_col].dt.minute // 15

    dates: list[pd.Timestamp] = []
    times: list[np.ndarray] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    base_preds: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for date, group in work.groupby("date", sort=True):
        if len(group) != 96:
            continue
        group = group.sort_values("slot")
        if not np.array_equal(group["slot"].to_numpy(), np.arange(96)):
            continue
        feature_values = group[feature_cols].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all():
            continue
        dates.append(pd.Timestamp(date))
        times.append(group[time_col].to_numpy())
        features.append(feature_values)
        targets.append(group[target_col].to_numpy(dtype=float))
        base_preds.append(group[base_pred_col].to_numpy(dtype=float))
        residuals.append(group[residual_col].to_numpy(dtype=float))

    if not dates:
        return DailySequenceFrame(
            dates=[],
            times=[],
            features=np.empty((0, 96, len(feature_cols)), dtype=float),
            targets=np.empty((0, 96), dtype=float),
            base_pred=np.empty((0, 96), dtype=float),
            residual=np.empty((0, 96), dtype=float),
        )

    return DailySequenceFrame(
        dates=dates,
        times=times,
        features=np.stack(features),
        targets=np.stack(targets),
        base_pred=np.stack(base_preds),
        residual=np.stack(residuals),
    )


def flatten_daily_features(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"expected 3D daily features, got shape={features.shape}")
    return features.reshape(features.shape[0], features.shape[1] * features.shape[2])


def temporal_context_samples(
    features: np.ndarray,
    *,
    offsets: tuple[int, ...] = (-8, -4, -2, -1, 0, 1, 2, 4, 8),
) -> np.ndarray:
    """Build slot-level local temporal context features.

    The competition provides the full next-day feature curve, so using nearby slots
    inside the same day is allowed. Edge slots use nearest-value padding.
    """

    if features.ndim != 3:
        raise ValueError(f"expected 3D daily features, got shape={features.shape}")
    if not offsets:
        raise ValueError("offsets must not be empty")
    max_offset = max(abs(int(offset)) for offset in offsets)
    padded = np.pad(features, ((0, 0), (max_offset, max_offset), (0, 0)), mode="edge")
    pieces = []
    slots = features.shape[1]
    for offset in offsets:
        start = max_offset + int(offset)
        pieces.append(padded[:, start : start + slots, :])
    context = np.concatenate(pieces, axis=2)
    return context.reshape(features.shape[0] * features.shape[1], context.shape[2])


def build_mlp_residual_model(
    *,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    alpha: float = 0.01,
    learning_rate_init: float = 0.001,
    max_iter: int = 1500,
    random_state: int = 2026,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_temporal_ridge_residual_model(
    *,
    alpha: float = 100.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


if nn is not None:

    class SmallGRUPriceModel(nn.Module):
        """Small daily curve model that directly predicts 96 electricity prices."""

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int = 32,
            num_layers: int = 1,
            dropout: float = 0.2,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            if input_size <= 0:
                raise ValueError("input_size must be positive")
            if hidden_size <= 0:
                raise ValueError("hidden_size must be positive")
            if num_layers <= 0:
                raise ValueError("num_layers must be positive")

            self.input_norm = nn.LayerNorm(input_size)
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_size = hidden_size * (2 if bidirectional else 1)
            self.head = nn.Sequential(
                nn.LayerNorm(output_size),
                nn.Dropout(dropout),
                nn.Linear(output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            projected = self.input_proj(x)
            temporal, _ = self.gru(projected)
            return self.head(temporal).squeeze(-1)


    class _TCNBlock(nn.Module):
        def __init__(
            self,
            *,
            channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd for same-length TCN blocks")
            padding = dilation * (kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: batch, time, channels
            y = self.net(x.transpose(1, 2)).transpose(1, 2)
            return self.norm(x + y)


    class SmallTCNPriceModel(nn.Module):
        """Small non-causal TCN for direct daily 96-point price curves."""

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int = 32,
            levels: int = 3,
            kernel_size: int = 5,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if input_size <= 0:
                raise ValueError("input_size must be positive")
            if levels <= 0:
                raise ValueError("levels must be positive")
            self.input_norm = nn.LayerNorm(input_size)
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.Sequential(
                *[
                    _TCNBlock(
                        channels=hidden_size,
                        kernel_size=kernel_size,
                        dilation=2**idx,
                        dropout=dropout,
                    )
                    for idx in range(levels)
                ]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            x = self.input_proj(x)
            x = self.blocks(x)
            return self.head(x).squeeze(-1)


    class SmallTFTLikeResidual(nn.Module):
        """Small gated-attention residual model for daily 96-point curves.

        This is intentionally much smaller than a full TFT. It keeps only the
        useful competition-safe ideas: variable projection, recurrent temporal
        context, gated residual path, and low-head self-attention.
        """

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int = 16,
            num_heads: int = 1,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            if input_size <= 0:
                raise ValueError("input_size must be positive")
            self.input_norm = nn.LayerNorm(input_size)
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
            )
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid(),
            )
            self.dropout = nn.Dropout(dropout)
            self.out = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            projected = torch.relu(self.input_proj(x))
            temporal, _ = self.gru(projected)
            attended, _ = self.attn(temporal, temporal, temporal, need_weights=False)
            gate = self.gate(torch.cat([temporal, attended], dim=-1))
            fused = gate * attended + (1.0 - gate) * temporal
            residual = self.out(self.dropout(fused)).squeeze(-1)
            return residual


def torch_available() -> bool:
    return torch is not None and nn is not None
