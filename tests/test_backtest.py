import pandas as pd

from stock_predictor.backtest import simulate_portfolio_nav, walk_forward_validation


def test_simulate_portfolio_nav_runs():
    dates = pd.date_range("2023-01-02", periods=80, freq="B")
    prices = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "close": list(100 + (pd.Series(range(len(dates))) * 0.1)) + list(
                50 + (pd.Series(range(len(dates))) * 0.08)
            ),
        }
    )
    wf = walk_forward_validation(prices=prices, horizons=[1], window=20, model_type="linear")
    nav_df, metrics = simulate_portfolio_nav(wf, prices, weighting="risk_parity", rolling_cov_window=20)
    assert not nav_df.empty
    assert "final_nav" in metrics
