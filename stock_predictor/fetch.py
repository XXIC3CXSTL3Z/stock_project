import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import yfinance as yf


def _cache_path(cache_dir: Optional[Path], key: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return cache_dir / f"{sanitized}.pkl"


def _load_cache(cache_dir: Optional[Path], key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, key)
    if path and path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:
            return None
    return None


def _save_cache(cache_dir: Optional[Path], key: str, df: pd.DataFrame) -> None:
    path = _cache_path(cache_dir, key)
    if path:
        try:
            df.to_pickle(path)
        except Exception:
            pass


def _download_with_retry(**kwargs) -> pd.DataFrame:
    retries = kwargs.pop("retries", 3)
    backoff = kwargs.pop("backoff", 1.5)
    last_exc = None
    for attempt in range(retries):
        try:
            return yf.download(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def _align_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing business days per ticker."""
    aligned = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        all_days = pd.bdate_range(grp["date"].min(), grp["date"].max(), freq="B")
        grp = grp.set_index("date").reindex(all_days).ffill().reset_index()
        grp = grp.rename(columns={"index": "date"})
        grp["ticker"] = ticker
        aligned.append(grp)
    return pd.concat(aligned, ignore_index=True)


def fetch_latest_prices(
    tickers: Iterable[str],
    period: str = "6mo",
    interval: str = "1d",
    crypto: bool = False,
    cache_dir: Optional[Path] = None,
    align_business: bool = True,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Download recent OHLC data and return a normalized price DataFrame.

    Parameters:
        tickers: iterable of ticker symbols.
        period: lookback window supported by yfinance (e.g., 1mo, 3mo, 6mo, 1y).
        interval: bar interval (1d recommended for this demo).
        crypto: when True, auto-append -USD to bare crypto tickers for yfinance.
        cache_dir: optional directory for caching downloads (pickle).
        align_business: align per-ticker history to business days with ffill (disabled automatically for crypto).
        retries/backoff: retry controls for yfinance download.
    """
    symbols: List[str] = []
    for t in tickers:
        symbol = t.strip().upper()
        if not symbol:
            continue
        if crypto and "-" not in symbol:
            symbol = f"{symbol}-USD"
        symbols.append(symbol)

    if not symbols:
        raise ValueError("No tickers provided for live fetch.")

    align_prices = align_business and not crypto
    cache_key = f"{'_'.join(symbols)}_{period}_{interval}_{'bdays' if align_prices else 'raw'}"
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None and not cached.empty:
        return cached

    data = _download_with_retry(
        tickers=symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
        retries=retries,
        backoff=backoff,
    )

    rows = []
    is_multi = isinstance(data.columns, pd.MultiIndex)

    for symbol in symbols:
        if is_multi:
            if symbol not in data:
                continue
            close_series = data[symbol]["Close"].dropna()
            volume_series = data[symbol].get("Volume")
        else:
            if "Close" not in data:
                continue
            close_series = data["Close"].dropna()
            volume_series = data.get("Volume")

        vol_df = volume_series.dropna() if volume_series is not None else None

        for ts, price in close_series.items():
            row = {"date": ts, "ticker": symbol, "close": float(price)}
            if vol_df is not None and ts in vol_df:
                try:
                    row["volume"] = float(vol_df.loc[ts])
                except Exception:
                    pass
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Fetched dataset is empty; check ticker symbols or network access.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if align_prices:
        df = _align_business_days(df)
    _save_cache(cache_dir, cache_key, df)
    return df


def fetch_benchmark_prices(
    benchmarks: Sequence[str] = ("SPY", "QQQ"),
    period: str = "2y",
    interval: str = "1d",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch benchmark ETF prices for comparisons."""
    return fetch_latest_prices(
        benchmarks,
        period=period,
        interval=interval,
        cache_dir=cache_dir,
        align_business=True,
    )


def fetch_macro_indicators(
    symbols: Sequence[str] = ("^VIX", "^IRX", "^TNX"),
    period: str = "2y",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Pull macro series such as VIX and rates.
    Returns a dataframe with columns date, feature.
    """
    rows: List[dict] = []
    for sym in symbols:
        try:
            df = fetch_latest_prices([sym], period=period, cache_dir=cache_dir, align_business=True)
        except Exception:
            continue
        df = df[["date", "close"]].rename(columns={"close": f"macro_{sym.strip('^').lower()}"})
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    merged = rows[0]
    for extra in rows[1:]:
        merged = merged.merge(extra, on="date", how="outer")
    merged = merged.sort_values("date").ffill().dropna(how="all")
    return merged
