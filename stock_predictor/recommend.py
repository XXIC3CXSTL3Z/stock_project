from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .data import load_price_data
from .deep import load_checkpoint, predict_sequence_model, train_sequence_model, visualize_attention
from .features import engineer_features
from .fetch import fetch_latest_prices
from .model import select_top_features, train_and_predict, tune_hyperparameters
from .portfolio import black_litterman_weights, markowitz_weights, risk_parity_weights


def generate_recommendations(
    data_path: Optional[Path] = None,
    tickers: Optional[Iterable[str]] = None,
    model_type: str = "random_forest",
    top_n: Optional[int] = 5,
    period: str = "6mo",
    auto_feature_k: Optional[int] = 8,
    cv_folds: int = 0,
    use_markowitz: bool = True,
    risk_aversion: float = 1.0,
    horizons: Sequence[int] = (1, 5, 10),
    sharpe_ranking: bool = True,
    shrink_cov: bool = True,
    weighting: str = "markowitz",
    checkpoint_dir: Optional[Path] = None,
    load_checkpoints: bool = False,
    hyper_grid: Optional[Sequence[str]] = None,
    crypto: bool = False,
) -> pd.DataFrame:
    """End-to-end orchestration to produce ranked investment ideas."""
    if tickers:
        prices = fetch_latest_prices(tickers, period=period, crypto=crypto)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    train_df, latest_df, feature_cols = engineer_features(prices, horizons=horizons)

    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(
            train_df, feature_cols, top_k=auto_feature_k, target_col="target_return_1d"
        )

    if train_df.empty or latest_df.empty:
        raise ValueError("Not enough rows to build features and targets. Add more history.")

    horizon_frames = []
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        preds_h = train_and_predict(
            train_df=train_df,
            latest_df=latest_df,
            feature_cols=feature_cols,
            model_type=model_type,
            cv_folds=cv_folds,
            target_col=target_col,
            horizon_label=f"{horizon}d",
            sharpe_ranking=sharpe_ranking,
            checkpoint_dir=checkpoint_dir,
            load_checkpoints=load_checkpoints,
        )

        preds_h["markowitz_weight"] = 0.0
        if use_markowitz and not preds_h.empty:
            if weighting == "risk_parity":
                weights = risk_parity_weights(prices)
            elif weighting == "black_litterman":
                weights = black_litterman_weights(
                    preds_h, prices, risk_aversion=risk_aversion, shrink=shrink_cov
                )
            else:
                weights = markowitz_weights(
                    preds_h, prices, risk_aversion=risk_aversion, shrink=shrink_cov
                )
            if not weights.empty:
                preds_h["markowitz_weight"] = preds_h["ticker"].map(weights).fillna(0.0)

        if top_n is not None:
            preds_h = preds_h.head(top_n)
        _normalize_column(preds_h, "recommended_weight")
        _normalize_column(preds_h, "markowitz_weight")
        horizon_frames.append(preds_h)

    if not horizon_frames:
        return pd.DataFrame()

    predictions = pd.concat(horizon_frames, ignore_index=True)

    if hyper_grid:
        tuning = tune_hyperparameters(
            train_df=train_df,
            feature_cols=feature_cols,
            model_grid=hyper_grid,
            target_col="target_return_1d",
            cv_folds=cv_folds or 3,
        )
        predictions.attrs["tuning_results"] = tuning

    return predictions


def generate_deep_recommendations(
    data_path: Optional[Path] = None,
    tickers: Optional[Iterable[str]] = None,
    period: str = "6mo",
    arch: str = "lstm",
    seq_len: int = 12,
    epochs: int = 25,
    auto_feature_k: Optional[int] = 10,
    use_markowitz: bool = True,
    risk_aversion: float = 1.0,
    top_n: Optional[int] = 5,
    horizons: Sequence[int] = (1, 5, 10),
    shrink_cov: bool = True,
    weighting: str = "markowitz",
    checkpoint_path: Optional[Path] = None,
    load_checkpoint: bool = False,
    patience: int = 5,
    log_progress: bool = True,
    crypto: bool = False,
    attention_dir: Optional[Path] = Path("artifacts"),
) -> pd.DataFrame:
    """Train an LSTM/Transformer on sequences of engineered features and rank tickers."""
    if tickers:
        prices = fetch_latest_prices(tickers, period=period, crypto=crypto)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    train_df, _, feature_cols = engineer_features(prices, horizons=horizons)
    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(
            train_df, feature_cols, top_k=auto_feature_k, target_col="target_return_1d"
        )

    horizon_frames = []
    latest_dates = train_df.groupby("ticker")["date"].max()
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        model = None
        mean = None
        std = None
        history = []
        target_std = float(train_df[target_col].std()) if len(train_df) > 1 else 0.0
        if load_checkpoint and checkpoint_path and checkpoint_path.exists():
            model, payload = load_checkpoint(checkpoint_path, arch=arch, input_size=len(feature_cols))
            mean = payload["mean"]
            std = payload["std"]
        else:
            model, mean, std, history = train_sequence_model(
                train_df=train_df,
                feature_cols=feature_cols,
                seq_len=seq_len,
                arch=arch,
                epochs=epochs,
                target_col=target_col,
                patience=patience,
                checkpoint_path=checkpoint_path,
                log_progress=log_progress,
            )
        preds_h = predict_sequence_model(
            model=model,
            df=train_df,
            feature_cols=feature_cols,
            mean=mean,
            std=std,
            seq_len=seq_len,
            method=f"deep_{arch}",
            horizon_label=f"{horizon}d",
        )

        preds_h["latest_date"] = preds_h["ticker"].map(latest_dates)
        if target_std > 0:
            preds_h["sharpe_score"] = preds_h["predicted_return"] / (target_std + 1e-6)
            preds_h = preds_h.sort_values("sharpe_score", ascending=False).reset_index(drop=True)
            preds_h["rank"] = preds_h.index + 1
        preds_h["markowitz_weight"] = 0.0
        if use_markowitz and not preds_h.empty:
            if weighting == "risk_parity":
                weights = risk_parity_weights(prices)
            elif weighting == "black_litterman":
                weights = black_litterman_weights(
                    preds_h, prices, risk_aversion=risk_aversion, shrink=shrink_cov
                )
            else:
                weights = markowitz_weights(
                    preds_h, prices, risk_aversion=risk_aversion, shrink=shrink_cov
                )
            if not weights.empty:
                preds_h["markowitz_weight"] = preds_h["ticker"].map(weights).fillna(0.0)

        if top_n is not None:
            preds_h = preds_h.head(top_n)
        _normalize_column(preds_h, "recommended_weight")
        _normalize_column(preds_h, "markowitz_weight")
        preds_h.attrs["history"] = history
        preds_h.attrs["model"] = model
        preds_h.attrs["mean"] = mean
        preds_h.attrs["std"] = std

        if attention_dir and hasattr(model, "attn_weights"):
            path = Path(attention_dir) / f"attention_{arch}_{horizon}d.png"
            saved = visualize_attention(getattr(model, "attn_weights"), path)
            if saved:
                preds_h.attrs["attention_path"] = saved

        horizon_frames.append(preds_h)

    if not horizon_frames:
        return pd.DataFrame()

    result = pd.concat(horizon_frames, ignore_index=True)
    # bubble up last attention path for dashboards
    for frame in horizon_frames[::-1]:
        if "attention_path" in frame.attrs:
            result.attrs["attention_path"] = frame.attrs["attention_path"]
            break
    return result


def _normalize_column(df: pd.DataFrame, column: str) -> None:
    """Normalize a weight column to sum to 1 if it has positive values."""
    if column not in df.columns:
        return
    positive_sum = df[column].clip(lower=0).sum()
    if positive_sum > 0:
        df[column] = df[column].clip(lower=0) / positive_sum


def format_recommendations(predictions: pd.DataFrame) -> str:
    """Render a lightweight text table."""
    if predictions.empty:
        return "No recommendations available."

    lines = [
        "Rank  Ticker  Horizon  Predicted Return  Weight  Markowitz Wt  Method        Samples  CV RMSE  Sharpe  Latest Date",
        "-" * 135,
    ]
    for _, row in predictions.iterrows():
        cv_display = f"{row['cv_rmse']:.4f}" if pd.notna(row.get("cv_rmse")) else "-"
        sharpe_display = f"{row['sharpe_score']:.3f}" if pd.notna(row.get("sharpe_score")) else "-"
        latest_display = (
            pd.to_datetime(row["latest_date"]).date()
            if "latest_date" in row and pd.notna(row.get("latest_date"))
            else "-"
        )
        method_display = row.get("method", "-")
        lines.append(
            f"{int(row['rank']):>4}  "
            f"{row['ticker']:<6}  "
            f"{row.get('horizon','-'):>7}  "
            f"{row['predicted_return']*100:>15.2f}%  "
            f"{row['recommended_weight']*100:>6.2f}%  "
            f"{row['markowitz_weight']*100:>12.2f}%  "
            f"{method_display:<12}  "
            f"{int(row['samples']):>7}  "
            f"{cv_display:>7}  "
            f"{sharpe_display:>6}  "
            f"{latest_display}"
        )

    best = predictions.iloc[0]
    lines.append("\nBest portfolio choice:")
    lines.append(
        f"- {best['ticker']} | predicted {best['predicted_return']*100:.2f}% | "
        f"Markowitz weight {best['markowitz_weight']*100:.2f}%"
    )

    return "\n".join(lines)
