import pandas as pd

from stock_predictor.features import engineer_features


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
