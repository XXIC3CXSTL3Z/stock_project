from typing import Iterable, List, Optional, Sequence, Tuple

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


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    std = std.replace(0, pd.NA)
    return (series - mean) / (std + 1e-6)


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = volume.copy()
    obv[:] = 0.0
    sign = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (volume.fillna(0) * sign).cumsum()


def _regime_flags(close: pd.Series, vol: pd.Series, long_window: int) -> pd.DataFrame:
    trend = close / close.rolling(long_window).mean()
    bull_bear = (trend > 1.0).astype(float)
    vol_crush = (vol < vol.rolling(long_window).median()).astype(float)
    return pd.DataFrame({"regime_bull": bull_bear, "regime_vol_crush": vol_crush})


def _beta_and_corr(
    asset_returns: pd.Series, benchmark_returns: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    cols = {}
    for col in benchmark_returns.columns:
        bench = benchmark_returns[col]
        cov = asset_returns.rolling(window).cov(bench)
        var = bench.rolling(window).std(ddof=0) ** 2
        cols[f"beta_{col.lower()}"] = cov / (var + 1e-9)
        cols[f"corr_{col.lower()}"] = asset_returns.rolling(window).corr(bench)
    return pd.DataFrame(cols)


def engineer_features(
    prices: pd.DataFrame,
    short_window: int = 5,
    mid_window: int = 10,
    long_window: int = 20,
    horizons: Sequence[int] = (1, 5, 10),
    benchmark_returns: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    rolling_norm_window: Optional[int] = None,
    include_volume: bool = True,
    crypto: bool = False,
) -> FeatureResult:
    """
    Build model-ready features and training labels.

    Returns (train_df, latest_df, feature_columns).
    - train_df contains historical rows with a known next-day return target.
    - latest_df contains the most recent feature row per ticker for inference.
    - rolling_norm_window standardizes features per ticker over a trailing window.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    horizons = list(horizons)

    if macro_df is not None and not macro_df.empty:
        macro_df = macro_df.copy()
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        df = df.merge(macro_df, on="date", how="left")
        df = df.sort_values(["ticker", "date"]).ffill()

    bench_rets = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_rets = benchmark_returns.copy()
        bench_rets.index = pd.to_datetime(bench_rets.index)

    def _per_ticker(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").reset_index(drop=True)
        group["return_1d"] = group["close"].pct_change(1, fill_method=None)
        group["return_5d"] = group["close"].pct_change(short_window, fill_method=None)
        group["return_10d"] = group["close"].pct_change(mid_window, fill_method=None)
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

        if include_volume and "volume" in group.columns:
            group["volume"] = group["volume"].ffill()
            group["volume_z"] = _rolling_zscore(group["volume"], long_window)
            group["obv"] = _on_balance_volume(group["close"], group["volume"])
            group["obv_slope"] = group["obv"].diff(short_window)
        else:
            group["volume_z"] = 0.0
            group["obv"] = 0.0
            group["obv_slope"] = 0.0

        if bench_rets is not None:
            aligned = bench_rets.reindex(group["date"]).ffill()
            asset_returns = group["return_1d"].copy()
            asset_returns.index = group["date"]
            beta_df = _beta_and_corr(asset_returns, aligned, window=long_window)
            beta_df.index = group.index
            group = pd.concat([group, beta_df], axis=1)

        if crypto:
            group["weekend_flag"] = (group["date"].dt.dayofweek >= 5).astype(float)
            group["overnight_return"] = group["close"].pct_change(fill_method=None).where(group["weekend_flag"] == 0)
        else:
            group["weekend_flag"] = 0.0
            group["overnight_return"] = group["return_1d"]

        regime_df = _regime_flags(group["close"], group["volatility_5d"], long_window=long_window)
        group = pd.concat([group, regime_df], axis=1)

        # Multi-horizon targets: pct change over horizon, shifted to align with prediction time.
        for horizon in horizons:
            group[f"target_return_{horizon}d"] = group["close"].pct_change(horizon, fill_method=None).shift(-horizon)

        return group

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
        "volume_z",
        "obv",
        "obv_slope",
        "weekend_flag",
        "overnight_return",
        "regime_bull",
        "regime_vol_crush",
    ]

    if bench_rets is not None:
        for col in bench_rets.columns:
            feature_cols.append(f"beta_{col.lower()}")
            feature_cols.append(f"corr_{col.lower()}")

    if macro_df is not None and not macro_df.empty:
        macro_cols = [c for c in macro_df.columns if c != "date"]
        feature_cols.extend(macro_cols)

    target_cols = [f"target_return_{h}d" for h in horizons]
    model_columns = ["ticker", "date"] + feature_cols + target_cols

    grouped_parts = [_per_ticker(grp) for _, grp in df.groupby("ticker")]
    if grouped_parts:
        df = pd.concat(grouped_parts, ignore_index=True)
    else:
        df = pd.DataFrame(columns=model_columns)

    df = df[model_columns]

    df[feature_cols] = df[feature_cols].fillna(0.0)

    train_df = df.dropna(subset=feature_cols).reset_index(drop=True)
    if rolling_norm_window:
        normalized_parts = []
        for _, grp in train_df.groupby("ticker"):
            grp = grp.sort_values("date").copy()
            for col in feature_cols:
                rolling = grp[col].rolling(rolling_norm_window)
                grp[col] = (grp[col] - rolling.mean()) / (rolling.std() + 1e-6)
            normalized_parts.append(grp)
        train_df = pd.concat(normalized_parts, ignore_index=True)

    latest_df = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest_df = latest_df.dropna(subset=feature_cols).reset_index(drop=True)

    return train_df, latest_df, feature_cols
