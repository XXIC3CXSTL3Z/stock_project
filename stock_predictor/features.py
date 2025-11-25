from typing import Iterable, List, Sequence, Tuple

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


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma + 1e-9)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
        }
    )


def engineer_features(
    prices: pd.DataFrame,
    short_window: int = 5,
    mid_window: int = 10,
    long_window: int = 20,
    horizons: Sequence[int] = (1, 5, 10),
) -> FeatureResult:
    """
    Build model-ready features and training labels.

    Returns (train_df, latest_df, feature_columns).
    - train_df contains historical rows with a known next-day return target.
    - latest_df contains the most recent feature row per ticker for inference.
    """
    df = prices.copy()
    horizons = list(horizons)

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
        rsi = _rsi(group["close"])
        group["rsi_14"] = rsi
        macd_df = _macd(group["close"])
        group = pd.concat([group, macd_df], axis=1)
        bb_df = _bollinger_bands(group["close"], window=mid_window)
        group = pd.concat([group, bb_df], axis=1)

        # Multi-horizon targets: pct change over horizon, shifted to align with prediction time.
        for horizon in horizons:
            group[f"target_return_{horizon}d"] = group["close"].pct_change(horizon).shift(-horizon)

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
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
    ]

    target_cols = [f"target_return_{h}d" for h in horizons]
    model_columns = ["ticker", "date"] + feature_cols + target_cols
    df = df[model_columns]

    train_df = df.dropna(subset=feature_cols).reset_index(drop=True)
    latest_df = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest_df = latest_df.dropna(subset=feature_cols).reset_index(drop=True)

    return train_df, latest_df, feature_cols
