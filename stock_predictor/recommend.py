from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .data import load_price_data
from .deep import predict_sequence_model, train_sequence_model
from .features import engineer_features
from .fetch import fetch_latest_prices
from .model import select_top_features, train_and_predict
from .portfolio import markowitz_weights


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
) -> pd.DataFrame:
    """End-to-end orchestration to produce ranked investment ideas."""
    if tickers:
        prices = fetch_latest_prices(tickers, period=period)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    train_df, latest_df, feature_cols = engineer_features(prices)

    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(train_df, feature_cols, top_k=auto_feature_k)

    if train_df.empty or latest_df.empty:
        raise ValueError("Not enough rows to build features and targets. Add more history.")

    predictions = train_and_predict(
        train_df=train_df,
        latest_df=latest_df,
        feature_cols=feature_cols,
        model_type=model_type,
        cv_folds=cv_folds,
    )

    predictions["markowitz_weight"] = 0.0
    if use_markowitz and not predictions.empty:
        weights = markowitz_weights(predictions, prices, risk_aversion=risk_aversion)
        if not weights.empty:
            predictions["markowitz_weight"] = predictions["ticker"].map(weights).fillna(0.0)

    if top_n is not None:
        predictions = predictions.head(top_n)
    _normalize_column(predictions, "recommended_weight")
    _normalize_column(predictions, "markowitz_weight")

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
) -> pd.DataFrame:
    """Train an LSTM/Transformer on sequences of engineered features and rank tickers."""
    if tickers:
        prices = fetch_latest_prices(tickers, period=period)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    train_df, _, feature_cols = engineer_features(prices)
    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(train_df, feature_cols, top_k=auto_feature_k)

    model, mean, std = train_sequence_model(
        train_df=train_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        arch=arch,
        epochs=epochs,
    )
    predictions = predict_sequence_model(
        model=model,
        df=train_df,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        method=f"deep_{arch}",
    )

    latest_dates = train_df.groupby("ticker")["date"].max()
    predictions["latest_date"] = predictions["ticker"].map(latest_dates)

    predictions["markowitz_weight"] = 0.0
    if use_markowitz and not predictions.empty:
        weights = markowitz_weights(predictions, prices, risk_aversion=risk_aversion)
        if not weights.empty:
            predictions["markowitz_weight"] = predictions["ticker"].map(weights).fillna(0.0)

    if top_n is not None:
        predictions = predictions.head(top_n)
    _normalize_column(predictions, "recommended_weight")
    _normalize_column(predictions, "markowitz_weight")

    return predictions


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
        "Rank  Ticker  Predicted Return  Weight  Markowitz Wt  Method        Samples  CV RMSE  Latest Date",
        "-" * 115,
    ]
    for _, row in predictions.iterrows():
        cv_display = f"{row['cv_rmse']:.4f}" if pd.notna(row.get("cv_rmse")) else "-"
        latest_display = (
            pd.to_datetime(row["latest_date"]).date()
            if "latest_date" in row and pd.notna(row.get("latest_date"))
            else "-"
        )
        method_display = row.get("method", "-")
        lines.append(
            f"{int(row['rank']):>4}  "
            f"{row['ticker']:<6}  "
            f"{row['predicted_return']*100:>15.2f}%  "
            f"{row['recommended_weight']*100:>6.2f}%  "
            f"{row['markowitz_weight']*100:>12.2f}%  "
            f"{method_display:<12}  "
            f"{int(row['samples']):>7}  "
            f"{cv_display:>7}  "
            f"{latest_display}"
        )

    best = predictions.iloc[0]
    lines.append("\nBest portfolio choice:")
    lines.append(
        f"- {best['ticker']} | predicted {best['predicted_return']*100:.2f}% | "
        f"Markowitz weight {best['markowitz_weight']*100:.2f}%"
    )

    return "\n".join(lines)
