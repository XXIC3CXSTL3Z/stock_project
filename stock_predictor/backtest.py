from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features import engineer_features
from .model import train_and_predict


def walk_forward_validation(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    window: int = 60,
    model_type: str = "random_forest",
    min_samples: int = 16,
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Walk-forward validation across tickers and horizons.

    For each ticker, iteratively trains on data up to time t and evaluates on t.
    """
    full_train, _, engineered_features = engineer_features(prices, horizons=horizons)
    feature_cols = list(feature_cols) if feature_cols else engineered_features

    all_rows: List[dict] = []
    for ticker, grp in full_train.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for idx in range(window, len(grp)):
            train_slice = grp.iloc[:idx]
            test_row = grp.iloc[[idx]]
            for horizon in horizons:
                target_col = f"target_return_{horizon}d"
                if pd.isna(test_row[target_col].iloc[0]):
                    continue
                preds = train_and_predict(
                    train_df=train_slice,
                    latest_df=test_row,
                    feature_cols=feature_cols,
                    model_type=model_type,
                    min_samples=min_samples,
                    target_col=target_col,
                    horizon_label=f"{horizon}d",
                    sharpe_ranking=False,
                    cv_folds=0,
                )
                if preds.empty:
                    continue
                pred_val = preds.loc[preds["ticker"] == ticker, "predicted_return"].iloc[0]
                all_rows.append(
                    {
                        "ticker": ticker,
                        "date": test_row["date"].iloc[0],
                        "horizon": horizon,
                        "predicted_return": pred_val,
                        "actual_return": test_row[target_col].iloc[0],
                    }
                )

    return pd.DataFrame(all_rows)


def generate_backtest_report(walk_forward_df: pd.DataFrame) -> Dict[str, float]:
    """Compute simple backtest diagnostics."""
    if walk_forward_df.empty:
        return {}

    metrics: Dict[str, float] = {}
    walk_forward_df = walk_forward_df.copy()
    walk_forward_df["error"] = walk_forward_df["predicted_return"] - walk_forward_df["actual_return"]

    metrics["rmse"] = float(np.sqrt(np.mean(np.square(walk_forward_df["error"]))))
    metrics["mae"] = float(np.mean(np.abs(walk_forward_df["error"])))
    metrics["hit_rate"] = float(
        np.mean(
            np.sign(walk_forward_df["predicted_return"]) == np.sign(walk_forward_df["actual_return"])
        )
    )
    metrics["avg_pred_return"] = float(np.mean(walk_forward_df["predicted_return"]))
    metrics["avg_actual_return"] = float(np.mean(walk_forward_df["actual_return"]))

    ret_series = walk_forward_df.sort_values("date")["actual_return"].fillna(0.0)
    if ret_series.std() > 0:
        metrics["sharpe"] = float(
            (ret_series.mean() / (ret_series.std() + 1e-6)) * np.sqrt(252)
        )
    else:
        metrics["sharpe"] = 0.0

    metrics["cum_return"] = float((1 + ret_series).prod() - 1)
    return metrics
