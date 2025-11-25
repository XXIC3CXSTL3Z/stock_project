from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make_regressor(model_type: str):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=2, n_jobs=-1
        )
    if model_type == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_type == "gbrt":
        return GradientBoostingRegressor(random_state=42)
    raise ValueError(f"Unsupported model type: {model_type}")


def select_top_features(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    top_k: int = 8,
) -> List[str]:
    """Select top features using a RandomForest importance heuristic."""
    feature_cols = list(feature_cols)
    if len(feature_cols) <= top_k:
        return feature_cols

    model = RandomForestRegressor(
        n_estimators=300, random_state=7, min_samples_leaf=2, n_jobs=-1
    )
    model.fit(train_df[feature_cols], train_df["target_next_return"])
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def _cross_val_rmse(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_type: str,
    cv_folds: int,
) -> Optional[float]:
    """Compute time-series CV RMSE for one ticker."""
    if cv_folds < 2 or len(df) <= cv_folds + 2:
        return None

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    reg = _make_regressor(model_type)
    scores = []

    X_all = df[feature_cols]
    y_all = df["target_next_return"]

    for train_idx, test_idx in tscv.split(df):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        scores.append(rmse)

    return float(np.mean(scores)) if scores else None


def train_and_predict(
    train_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    feature_cols: Iterable[str],
    model_type: str = "random_forest",
    min_samples: int = 16,
    cv_folds: int = 0,
) -> pd.DataFrame:
    """
    Fit per-ticker regressors and predict next-day returns.

    Falls back to a simple mean-return baseline when the dataset is too small.
    """
    feature_cols = list(feature_cols)
    rows: List[dict] = []

    for ticker, ticker_train in train_df.groupby("ticker"):
        ticker_latest = latest_df[latest_df["ticker"] == ticker]
        if ticker_latest.empty:
            continue

        X = ticker_train[feature_cols]
        y = ticker_train["target_next_return"]
        samples = len(ticker_train)

        cv_rmse = _cross_val_rmse(ticker_train, feature_cols, model_type, cv_folds)

        try:
            if samples >= max(min_samples, len(feature_cols) + 2):
                model = _make_regressor(model_type)
                model.fit(X, y)
                predicted_return = float(model.predict(ticker_latest[feature_cols])[0])
                method_used = model_type
            else:
                predicted_return = float(y.mean())
                method_used = "mean_baseline"
        except Exception:
            predicted_return = float(y.mean())
            method_used = "mean_baseline"

        rows.append(
            {
                "ticker": ticker,
                "predicted_return": predicted_return,
                "samples": samples,
                "method": method_used,
                "latest_date": ticker_latest["date"].iloc[0],
                "volatility": float(
                    ticker_train["volatility_5d"].mean(skipna=True)
                ),  # proxy for risk
                "cv_rmse": cv_rmse,
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
