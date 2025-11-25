import pandas as pd

from stock_predictor import fetch


def test_crypto_fetch_keeps_weekends(monkeypatch):
    dates = pd.date_range("2024-01-05", periods=4, freq="D")  # Fri-Mon includes weekend
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
    volume = pd.Series([10_000, 11_000, 12_000, 13_000], index=dates)
    fake_df = pd.DataFrame({"Close": close, "Volume": volume})

    def fake_download(**kwargs):
        return fake_df

    monkeypatch.setattr(fetch.yf, "download", fake_download)

    result = fetch.fetch_latest_prices(["BTC"], period="7d", crypto=True, cache_dir=None, align_business=True)

    # Weekend (Sat/Sun) rows should remain because business-day alignment is disabled for crypto.
    assert len(result) == 4
    assert set(result["date"].dt.dayofweek) == {0, 4, 5, 6}
