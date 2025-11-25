from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:  # optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # noqa: BLE001
    SummaryWriter = None

from tqdm.auto import tqdm


def _prepare_sequences(
    train_df: pd.DataFrame, feature_cols: Iterable[str], seq_len: int, target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences across all tickers."""
    feature_cols = list(feature_cols)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, grp in train_df.groupby("ticker"):
        grp = grp.sort_values("date")
        features = grp[feature_cols].to_numpy(dtype=np.float32)
        targets = grp[target_col].to_numpy(dtype=np.float32)
        if len(grp) <= seq_len:
            continue
        for idx in range(seq_len, len(grp)):
            X_list.append(features[idx - seq_len : idx])
            y_list.append(targets[idx])

    if not X_list:
        raise ValueError("Not enough rows per ticker to build sequences for deep models.")

    return np.stack(X_list), np.array(y_list)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.act(out[:, -1, :])
        return self.head(out).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        attn_out, weights = self.attn(z, z, z, need_weights=True, average_attn_weights=False)
        self.attn_weights = weights.detach()
        z = self.dropout(attn_out) + z
        z = self.feedforward(z) + z
        z = self.norm(z)
        return self.head(z[:, -1, :]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    """1D CNN feature extractor feeding an LSTM head."""

    def __init__(self, input_size: int, hidden_size: int = 32, cnn_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.permute(0, 2, 1)  # (batch, features, seq)
        z = torch.relu(self.conv(z))
        z = z.permute(0, 2, 1)  # back to (batch, seq, channels)
        out, _ = self.lstm(z)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return self.head(out).squeeze(-1)


class CrossAssetAttention(nn.Module):
    """Multi-head attention with a global token for multi-ticker fusion."""

    def __init__(self, input_size: int, hidden_size: int = 48, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.proj = nn.Linear(input_size, hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.head = nn.Linear(hidden_size, 1)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        cls_token = self.cls.expand(bsz, -1, -1)
        z = self.proj(x)
        z = torch.cat([cls_token, z], dim=1)
        attn_out, weights = self.attn(z, z, z, need_weights=True, average_attn_weights=False)
        self.attn_weights = weights.detach()
        z = attn_out + z
        z = self.ff(z) + z
        pooled = z[:, 0, :]
        return self.head(pooled).squeeze(-1)


def _normalize_sequences(
    X: np.ndarray, rolling_norm_window: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features for stability."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)

    def _norm(seq: np.ndarray) -> np.ndarray:
        if rolling_norm_window:
            local = seq[-rolling_norm_window:]
            mu = local.mean(axis=0)
            sigma = local.std(axis=0)
            sigma = np.where(sigma < 1e-6, 1e-6, sigma)
            return (seq - mu) / sigma
        return (seq - mean) / std

    X_norm = np.stack([_norm(seq) for seq in X])
    return X_norm, mean, std


def _assert_normalization(mean: np.ndarray, std: np.ndarray, feature_cols: List[str]) -> None:
    if len(mean) != len(feature_cols) or len(std) != len(feature_cols):
        raise ValueError(
            f"Normalization mismatch: expected {len(feature_cols)} features, "
            f"got mean/std shapes {mean.shape} / {std.shape}."
        )


def save_checkpoint(
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: List[str],
    checkpoint_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "feature_cols": feature_cols,
        "metadata": metadata or {},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, arch: str, input_size: int) -> Tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if arch == "lstm":
        model: nn.Module = LSTMRegressor(input_size=input_size)
    elif arch == "transformer":
        model = TransformerRegressor(input_size=input_size)
    elif arch == "cnn_lstm":
        model = CNNLSTMRegressor(input_size=input_size)
    elif arch == "fusion":
        model = CrossAssetAttention(input_size=input_size)
    else:
        raise ValueError(f"Unsupported arch {arch}")
    model.load_state_dict(payload["state_dict"])
    return model, payload


def train_sequence_model(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    seq_len: int = 10,
    arch: str = "lstm",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    target_col: str = "target_return_1d",
    patience: int = 5,
    val_split: float = 0.15,
    checkpoint_path: Optional[Path] = None,
    log_progress: bool = True,
    early_stopping: bool = True,
    rolling_norm_window: Optional[int] = None,
    tensorboard_logdir: Optional[Path] = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray, List[dict]]:
    """Train an LSTM/Transformer/CNN-LSTM on sliding windows with logging + checkpoints."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_df = train_df.dropna(subset=list(feature_cols) + [target_col]).copy()

    X, y = _prepare_sequences(train_df, feature_cols, seq_len, target_col=target_col)
    X, mean, std = _normalize_sequences(X, rolling_norm_window=rolling_norm_window)
    _assert_normalization(mean, std, list(feature_cols))

    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)

    if val_split > 0 and len(dataset) > 2:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        if train_size < 1:
            train_ds, val_ds = dataset, None
        else:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    input_size = len(feature_cols)
    if arch == "lstm":
        model: nn.Module = LSTMRegressor(input_size=input_size)
    elif arch == "transformer":
        model = TransformerRegressor(input_size=input_size)
    elif arch == "cnn_lstm":
        model = CNNLSTMRegressor(input_size=input_size)
    elif arch == "fusion":
        model = CrossAssetAttention(input_size=input_size)
    else:
        raise ValueError("arch must be 'lstm', 'transformer', 'cnn_lstm', or 'fusion'")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: List[dict] = []
    best_state = None
    best_val = float("inf")
    patience_ctr = 0
    writer = None
    if tensorboard_logdir and SummaryWriter:
        tensorboard_logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_logdir))

    epoch_iter = range(epochs)
    if log_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)

    for epoch in epoch_iter:
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        val_loss = None
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    preds = model(batch_x)
                    loss = loss_fn(preds, batch_y)
                    val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else None

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar("loss/val", val_loss, epoch)

        if val_loss is not None and val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if checkpoint_path:
            save_checkpoint(
                model,
                mean,
                std,
                list(feature_cols),
                Path(checkpoint_path),
                metadata={"arch": arch, "seq_len": seq_len, "rolling_norm_window": rolling_norm_window},
            )
            history_path = Path(checkpoint_path).with_suffix(".history.csv")
            pd.DataFrame(history).to_csv(history_path, index=False)

        if early_stopping and val_loader and patience_ctr >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    if writer:
        writer.close()

    return model, mean, std, history


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
    horizon_label: str = "1d",
    normalization_check: bool = True,
    rolling_norm_window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Use a trained sequence model to predict next returns for each ticker.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = list(feature_cols)
    if normalization_check:
        _assert_normalization(mean, std, feature_cols)

    model.eval()
    model.to(device)

    rows = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < seq_len:
            continue
        seq = grp[feature_cols].to_numpy(dtype=np.float32)[-seq_len:]
        if rolling_norm_window:
            local = seq[-rolling_norm_window:]
            mu = local.mean(axis=0)
            sigma = local.std(axis=0)
            sigma = np.where(sigma < 1e-6, 1e-6, sigma)
            seq_norm = (seq - mu) / sigma
        else:
            seq_norm = (seq - mean) / std
        tensor_x = torch.tensor(seq_norm, dtype=torch.float32, device=device).unsqueeze(0)
        pred = float(model(tensor_x).cpu().item())
        rows.append(
            {
                "ticker": ticker,
                "predicted_return": pred,
                "samples": len(grp),
                "method": method,
                "horizon": horizon_label,
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


def visualize_attention(
    attn_weights: Optional[torch.Tensor], save_path: Path, title: str = "Attention Weights"
) -> Optional[Path]:
    """Save a simple heatmap of the most recent attention weights."""
    if attn_weights is None:
        return None
    weights = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()  # average over heads
    plt.figure(figsize=(4, 3))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Weight")
    plt.xlabel("Source timestep")
    plt.ylabel("Target timestep")
    plt.title(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def walk_forward_sequence_backtest(
    features_df: pd.DataFrame,
    feature_cols: Iterable[str],
    seq_len: int = 10,
    arch: str = "lstm",
    epochs: int = 10,
    patience: int = 3,
    horizon_label: str = "1d",
) -> pd.DataFrame:
    """
    Walk-forward backtest for deep models (slow but illustrative).
    Trains on history up to each decision date, predicts, and records realized returns.
    """
    dates = sorted(features_df["date"].unique())
    results: List[dict] = []
    for idx in range(seq_len, len(dates) - 1):
        date = dates[idx]
        train_slice = features_df[features_df["date"] < date]
        latest = features_df[features_df["date"] == date]
        target_col = f"target_return_{horizon_label}"
        if target_col not in features_df.columns:
            target_col = f"target_return_{horizon_label.replace('d','')}d"
        if train_slice.empty or latest.empty:
            continue
        model, mean, std, _ = train_sequence_model(
            train_df=train_slice,
            feature_cols=feature_cols,
            seq_len=seq_len,
            arch=arch,
            epochs=epochs,
            target_col=target_col,
            patience=patience,
            early_stopping=True,
            log_progress=False,
        )
        preds = predict_sequence_model(
            model, latest, feature_cols, mean, std, seq_len=seq_len, method=f"deep_{arch}", horizon_label=horizon_label
        )
        for _, row in preds.iterrows():
            realized = latest[latest["ticker"] == row["ticker"]]
            actual = realized[target_col].iloc[0] if not realized.empty else np.nan
            results.append(
                {
                    "date": date,
                    "ticker": row["ticker"],
                    "predicted_return": row["predicted_return"],
                    "actual_return": actual,
                    "horizon": horizon_label,
                }
            )
    return pd.DataFrame(results)
