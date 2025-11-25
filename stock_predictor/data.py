from pathlib import Path
from typing import Union

import pandas as pd


def load_price_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load end-of-day price data.

    Expected CSV columns: date, ticker, close.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find data file at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"date", "ticker", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df
