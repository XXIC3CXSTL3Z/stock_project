import numpy as np
import pandas as pd


def compute_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot price history into wide matrix of log returns."""
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    log_returns = np.log(wide).diff().dropna(how="all")
    return log_returns.dropna(axis=1, how="all")


def markowitz_weights(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    risk_aversion: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    Compute heuristic Markowitz-style weights using predicted returns and historical covariance.
    Negative weights are clipped to `min_weight` and renormalized.
    """
    returns = compute_return_matrix(prices)
    cov = returns.cov()
    mu = predictions.set_index("ticker")["predicted_return"]

    tickers = [t for t in cov.columns if t in mu.index]
    if not tickers:
        return pd.Series(dtype=float)

    cov = cov.loc[tickers, tickers]
    mu = mu.loc[tickers]

    try:
        inv_cov = np.linalg.pinv(cov.values)
        raw = inv_cov.dot(mu.values) / risk_aversion
        weights = pd.Series(raw, index=tickers)
    except Exception:
        # Fallback: proportional to predicted returns
        weights = mu.copy()

    weights = weights.clip(lower=min_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = pd.Series(1.0 / len(weights), index=weights.index)
    return weights
