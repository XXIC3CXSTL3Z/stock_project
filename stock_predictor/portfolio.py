import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


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
    shrink: bool = True,
) -> pd.Series:
    """
    Compute heuristic Markowitz-style weights using predicted returns and historical covariance.
    Negative weights are clipped to `min_weight` and renormalized.
    """
    returns = compute_return_matrix(prices)
    if shrink:
        lw = LedoitWolf().fit(returns.fillna(0.0))
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
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


def risk_parity_weights(prices: pd.DataFrame) -> pd.Series:
    """
    Risk-parity allocation based on inverse volatility.
    """
    returns = compute_return_matrix(prices)
    vol = returns.std()
    inv_vol = 1.0 / (vol + 1e-6)
    weights = inv_vol / inv_vol.sum()
    return weights


def black_litterman_weights(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    tau: float = 0.05,
    risk_aversion: float = 2.0,
    min_weight: float = 0.0,
    shrink: bool = True,
) -> pd.Series:
    """
    Simplified Black-Litterman blending equal-weighted prior with model views.
    """
    returns = compute_return_matrix(prices)
    if shrink:
        lw = LedoitWolf().fit(returns.fillna(0.0))
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
        cov = returns.cov()

    mu = predictions.set_index("ticker")["predicted_return"]
    tickers = [t for t in cov.columns if t in mu.index]
    if not tickers:
        return pd.Series(dtype=float)

    cov = cov.loc[tickers, tickers]
    mu = mu.loc[tickers]

    prior = pd.Series(1.0 / len(tickers), index=tickers)
    try:
        inv_cov = np.linalg.pinv(cov.values)
        bl_mean = prior + tau * inv_cov.dot(mu.values)
        weights = pd.Series(bl_mean, index=tickers)
    except Exception:
        weights = prior + tau * mu

    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()
    return weights
