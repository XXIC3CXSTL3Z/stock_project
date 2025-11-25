from typing import Iterable, List

import pandas as pd
import yfinance as yf


def fetch_latest_prices(
    tickers: Iterable[str],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download recent OHLC data and return a normalized price DataFrame.

    Parameters:
        tickers: iterable of ticker symbols.
        period: lookback window supported by yfinance (e.g., 1mo, 3mo, 6mo, 1y).
        interval: bar interval (1d recommended for this demo).
    """
    symbols: List[str] = [t.strip().upper() for t in tickers if t.strip()]
    if not symbols:
        raise ValueError("No tickers provided for live fetch.")

    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    rows = []
    is_multi = isinstance(data.columns, pd.MultiIndex)

    for symbol in symbols:
        if is_multi:
            if symbol not in data:
                continue
            close_series = data[symbol]["Close"].dropna()
        else:
            if "Close" not in data:
                continue
            close_series = data["Close"].dropna()

        for ts, price in close_series.items():
            rows.append({"date": ts, "ticker": symbol, "close": float(price)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Fetched dataset is empty; check ticker symbols or network access.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df
