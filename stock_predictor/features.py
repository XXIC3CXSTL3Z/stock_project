from typing import List, Tuple

import pandas as pd


FeatureResult = Tuple[pd.DataFrame, pd.DataFrame, List[str]]


def _volatility_regime(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Classify high/low volatility regimes based on rolling std vs its median.

    Returns a binary indicator where 1 = high volatility.
    """
    rolling_std = series.rolling(window).std()
    median_std = rolling_std.expanding().median()
    return (rolling_std > median_std).astype(float)


def engineer_features(
    prices: pd.DataFrame,
    short_window: int = 5,
    mid_window: int = 10,
    long_window: int = 20,
) -> FeatureResult:
    """
    Build model-ready features and training labels.

    Returns (train_df, latest_df, feature_columns).
    - train_df contains historical rows with a known next-day return target.
    - latest_df contains the most recent feature row per ticker for inference.
    """
    df = prices.copy()

    def _per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        group["return_1d"] = group["close"].pct_change(1)
        group["return_5d"] = group["close"].pct_change(short_window)
        group["return_10d"] = group["close"].pct_change(mid_window)
        group["momentum_5d"] = (group["close"] - group["close"].shift(short_window)) / group[
            "close"
        ].shift(short_window)
        group["momentum_10d"] = (group["close"] - group["close"].shift(mid_window)) / group[
            "close"
        ].shift(mid_window)
        group["volatility_5d"] = group["return_1d"].rolling(short_window).std()
        group["volatility_10d"] = group["return_1d"].rolling(mid_window).std()
        group["sma_ratio"] = group["close"] / group["close"].rolling(long_window).mean()
        group["ema_short"] = group["close"].ewm(span=short_window, adjust=False).mean()
        group["ema_long"] = group["close"].ewm(span=long_window, adjust=False).mean()
        group["ema_ratio"] = group["ema_short"] / group["ema_long"] - 1
        group["vol_regime"] = _volatility_regime(group["close"], window=long_window)
        group["target_next_return"] = group["close"].pct_change().shift(-1)
        return group

    df = df.groupby("ticker", group_keys=False).apply(_per_ticker)

    feature_cols: List[str] = [
        "return_1d",
        "return_5d",
        "return_10d",
        "momentum_5d",
        "momentum_10d",
        "volatility_5d",
        "volatility_10d",
        "sma_ratio",
        "ema_ratio",
        "vol_regime",
    ]

    model_columns = ["ticker", "date"] + feature_cols + ["target_next_return"]
    df = df[model_columns]

    train_df = df.dropna(subset=feature_cols + ["target_next_return"]).reset_index(drop=True)
    latest_df = df.sort_values("date").groupby("ticker").tail(1)
    latest_df = latest_df.dropna(subset=feature_cols).reset_index(drop=True)

    return train_df, latest_df, feature_cols
