from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
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
    target_col: str = "target_return_1d",
) -> List[str]:
    """Select top features using a RandomForest importance heuristic."""
    train_df = train_df.dropna(subset=list(feature_cols) + [target_col])
    feature_cols = list(feature_cols)
    if len(feature_cols) <= top_k:
        return feature_cols

    model = RandomForestRegressor(
        n_estimators=300, random_state=7, min_samples_leaf=2, n_jobs=-1
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def _cross_val_rmse(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_type: str,
    cv_folds: int,
    target_col: str,
) -> Optional[float]:
    """Compute time-series CV RMSE for one ticker."""
    df = df.dropna(subset=list(feature_cols) + [target_col])
    if cv_folds < 2 or len(df) <= cv_folds + 2:
        return None

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    reg = _make_regressor(model_type)
    scores = []

    X_all = df[feature_cols]
    y_all = df[target_col]

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
    target_col: str = "target_return_1d",
    horizon_label: str = "1d",
    sharpe_ranking: bool = True,
    checkpoint_dir: Optional[Path] = None,
    load_checkpoints: bool = False,
) -> pd.DataFrame:
    """
    Fit per-ticker regressors and predict next-day returns.

    Falls back to a simple mean-return baseline when the dataset is too small.
    """
    feature_cols = list(feature_cols)
    rows: List[dict] = []
    checkpoints: Dict[str, Path] = {}
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for ticker, ticker_train in train_df.groupby("ticker"):
        ticker_latest = latest_df[latest_df["ticker"] == ticker]
        if ticker_latest.empty:
            continue
        ticker_train = ticker_train.dropna(subset=feature_cols + [target_col])
        if ticker_train.empty:
            continue

        X = ticker_train[feature_cols]
        y = ticker_train[target_col]
        samples = len(ticker_train)

        cv_rmse = _cross_val_rmse(
            ticker_train, feature_cols, model_type, cv_folds=cv_folds, target_col=target_col
        )

        checkpoint_path = None
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"{ticker}_{horizon_label}_{model_type}.joblib"
            checkpoints[ticker] = checkpoint_path

        model = None
        if load_checkpoints and checkpoint_path and checkpoint_path.exists():
            try:
                model = joblib.load(checkpoint_path)
                method_used = f"{model_type}_checkpoint"
            except Exception:
                model = None

        try:
            if samples >= max(min_samples, len(feature_cols) + 2):
                model = model or _make_regressor(model_type)
                model.fit(X, y)
                predicted_return = float(model.predict(ticker_latest[feature_cols])[0])
                method_used = model_type
            else:
                predicted_return = float(y.mean())
                method_used = "mean_baseline"

            if checkpoint_path and model is not None:
                joblib.dump(model, checkpoint_path)
        except Exception:
            predicted_return = float(y.mean())
            method_used = "mean_baseline"

        target_std = float(y.std()) if len(y) > 1 else 0.0
        sharpe_score = (
            predicted_return / (target_std + 1e-6) if sharpe_ranking and target_std > 0 else np.nan
        )

        rows.append(
            {
                "ticker": ticker,
                "predicted_return": predicted_return,
                "horizon": horizon_label,
                "samples": samples,
                "method": method_used,
                "latest_date": ticker_latest["date"].iloc[0],
                "volatility": float(
                    ticker_train["volatility_5d"].mean(skipna=True)
                ),  # proxy for risk
                "cv_rmse": cv_rmse,
                "sharpe_score": sharpe_score,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if sharpe_ranking and "sharpe_score" in result:
        result = result.sort_values("sharpe_score", ascending=False, na_position="last")
    else:
        result = result.sort_values("predicted_return", ascending=False)

    result = result.reset_index(drop=True)
    result["rank"] = result.index + 1

    total_positive = result[result["predicted_return"] > 0]["predicted_return"].sum()
    if total_positive > 0:
        result["recommended_weight"] = (
            result["predicted_return"].clip(lower=0) / total_positive
        )
    else:
        result["recommended_weight"] = 0.0

    return result


def tune_hyperparameters(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    model_grid: Sequence[str],
    target_col: str = "target_return_1d",
    cv_folds: int = 3,
) -> pd.DataFrame:
    """Simple hyperparameter / model-type sweep with walk-forward CV RMSE."""
    results = []
    for model_type in model_grid:
        rmses = []
        for ticker, ticker_train in train_df.groupby("ticker"):
            cv_rmse = _cross_val_rmse(
                ticker_train, feature_cols, model_type=model_type, cv_folds=cv_folds, target_col=target_col
            )
            if cv_rmse is not None:
                rmses.append(cv_rmse)
        avg_rmse = float(np.mean(rmses)) if rmses else None
        results.append({"model_type": model_type, "cv_rmse": avg_rmse})

    return pd.DataFrame(results).sort_values("cv_rmse", na_position="last")
