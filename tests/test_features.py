import numpy as np
import pandas as pd

from stock_predictor.features import engineer_features
from stock_predictor.model import train_and_predict


def test_engineer_features_adds_enhancements():
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    df = pd.DataFrame(
        {
            "date": dates.tolist() + dates.tolist(),
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": list(100 + (pd.Series(range(len(dates))) * 0.1)) + list(
                50 + (pd.Series(range(len(dates))) * 0.05)
            ),
            "volume": [1_000_000 + i * 1000 for i in range(len(dates))] * 2,
        }
    )
    train_df, latest_df, feature_cols = engineer_features(df, horizons=[1, 5])
    assert {"obv", "volume_z", "regime_bull", "regime_vol_crush"}.issubset(set(feature_cols))
    assert not train_df.empty
    assert not latest_df.empty


def test_engineer_features_clips_infinite_values():
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    close = pd.Series([100, 0, 102, 103, 104, 0, 106, 107, 108, 109, 110, 111], index=dates)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "close": close,
            "volume": [10_000] * len(dates),
        }
    )

    train_df, latest_df, feature_cols = engineer_features(df, horizons=[1, 3])

    assert np.isfinite(train_df[feature_cols].to_numpy()).all()
    assert np.isfinite(latest_df[feature_cols].to_numpy()).all()

    preds = train_and_predict(
        train_df=train_df,
        latest_df=latest_df,
        feature_cols=feature_cols,
        model_type="random_forest",
        cv_folds=0,
        target_col="target_return_1d",
        horizon_label="1d",
    )

    assert not preds.empty
    assert preds["predicted_return"].notna().all()
