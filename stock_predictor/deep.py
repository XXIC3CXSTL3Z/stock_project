from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _prepare_sequences(
    train_df: pd.DataFrame, feature_cols: Iterable[str], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences across all tickers."""
    feature_cols = list(feature_cols)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, grp in train_df.groupby("ticker"):
        grp = grp.sort_values("date")
        features = grp[feature_cols].to_numpy(dtype=np.float32)
        targets = grp["target_next_return"].to_numpy(dtype=np.float32)
        if len(grp) <= seq_len:
            continue
        for idx in range(seq_len, len(grp)):
            X_list.append(features[idx - seq_len : idx])
            y_list.append(targets[idx])

    if not X_list:
        raise ValueError("Not enough rows per ticker to build sequences for deep models.")

    return np.stack(X_list), np.array(y_list)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        encoded = self.encoder(z)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def _normalize_sequences(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features for stability."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    return X_norm, mean, std


def train_sequence_model(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    seq_len: int = 10,
    arch: str = "lstm",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """Train an LSTM/Transformer on sliding windows of engineered features."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, y = _prepare_sequences(train_df, feature_cols, seq_len)
    X, mean, std = _normalize_sequences(X)

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=True)

    input_size = len(feature_cols)
    if arch == "lstm":
        model: nn.Module = LSTMRegressor(input_size=input_size)
    elif arch == "transformer":
        model = TransformerRegressor(input_size=input_size)
    else:
        raise ValueError("arch must be 'lstm' or 'transformer'")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            opt.step()

    return model, mean, std


@torch.no_grad()
def predict_sequence_model(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int = 10,
    device: Optional[str] = None,
    method: str = "deep",
) -> pd.DataFrame:
    """
    Use a trained sequence model to predict next returns for each ticker.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = list(feature_cols)
    model.eval()
    model.to(device)

    rows = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < seq_len:
            continue
        seq = grp[feature_cols].to_numpy(dtype=np.float32)[-seq_len:]
        seq_norm = (seq - mean) / std
        tensor_x = torch.tensor(seq_norm, dtype=torch.float32, device=device).unsqueeze(0)
        pred = float(model(tensor_x).cpu().item())
        rows.append(
            {
                "ticker": ticker,
                "predicted_return": pred,
                "samples": len(grp),
                "method": method,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    total_positive = result[result["predicted_return"] > 0]["predicted_return"].sum()
    if total_positive > 0:
        result["recommended_weight"] = (
            result["predicted_return"].clip(lower=0) / total_positive
        )
    else:
        result["recommended_weight"] = 0.0
    return result
