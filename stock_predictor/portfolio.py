from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def compute_return_matrix(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Pivot price history into wide matrix of returns."""
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if log:
        rets = np.log(wide).diff()
    else:
        rets = wide.pct_change()
    rets = rets.dropna(how="all")
    return rets.dropna(axis=1, how="all")


def _covariance(returns: pd.DataFrame, shrink: bool = True, window: Optional[int] = None) -> pd.DataFrame:
    if window:
        returns = returns.tail(window)
    returns = returns.fillna(0.0)
    if shrink:
        lw = LedoitWolf().fit(returns)
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
        cov = returns.cov()
    return cov


def markowitz_weights(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    risk_aversion: float = 1.0,
    min_weight: float = 0.0,
    shrink: bool = True,
    cov_window: Optional[int] = None,
) -> pd.Series:
    """
    Compute heuristic Markowitz-style weights using predicted returns and historical covariance.
    Negative weights are clipped to `min_weight` and renormalized.
    """
    returns = compute_return_matrix(prices)
    cov = _covariance(returns, shrink=shrink, window=cov_window)

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
    vol = returns.std().replace(0, np.nan)
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
    view_confidence: float = 1.0,
    cov_window: Optional[int] = None,
) -> pd.Series:
    """
    Simplified Black-Litterman blending equal-weighted prior with model views.
    view_confidence scales the contribution of model views.
    """
    returns = compute_return_matrix(prices)
    cov = _covariance(returns, shrink=shrink, window=cov_window)

    mu = predictions.set_index("ticker")["predicted_return"]
    tickers = [t for t in cov.columns if t in mu.index]
    if not tickers:
        return pd.Series(dtype=float)

    cov = cov.loc[tickers, tickers]
    mu = mu.loc[tickers]

    prior = pd.Series(1.0 / len(tickers), index=tickers)
    try:
        inv_cov = np.linalg.pinv(cov.values)
        bl_mean = prior + tau * view_confidence * inv_cov.dot(mu.values)
        weights = pd.Series(bl_mean, index=tickers)
    except Exception:
        weights = prior + tau * mu

    weights = weights.clip(lower=min_weight)
    weights = weights / weights.sum()
    return weights


def hrp_weights(prices: pd.DataFrame, shrink: bool = True, cov_window: Optional[int] = None) -> pd.Series:
    """
    Hierarchical Risk Parity allocation using correlation clustering.
    """
    returns = compute_return_matrix(prices)
    cov = _covariance(returns, shrink=shrink, window=cov_window)
    corr = cov.corr().fillna(0.0)
    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
    link = linkage(squareform(dist.values), "single")
    sort_ix = dendrogram(link, no_plot=True)["leaves"]
    ordered = cov.index[sort_ix]

    def _alloc(cov_mat: pd.DataFrame, assets: list[str]) -> pd.Series:
        if len(assets) == 1:
            return pd.Series(1.0, index=assets)
        split = len(assets) // 2
        left, right = assets[:split], assets[split:]
        w_left = _alloc(cov_mat, left)
        w_right = _alloc(cov_mat, right)
        var_left = float(w_left.values.T @ cov_mat.loc[left, left].values @ w_left.values)
        var_right = float(w_right.values.T @ cov_mat.loc[right, right].values @ w_right.values)
        alpha = 1 - var_left / (var_left + var_right + 1e-9)
        w_left *= alpha
        w_right *= 1 - alpha
        return pd.concat([w_left, w_right])

    weights = _alloc(cov.loc[ordered, ordered], list(ordered))
    weights = weights / weights.sum()
    return weights


def volatility_target_weights(
    base_weights: pd.Series,
    prices: pd.DataFrame,
    target_vol: float = 0.1,
    cov_window: int = 60,
    shrink: bool = True,
) -> pd.Series:
    """Scale weights to hit a target annualized volatility."""
    returns = compute_return_matrix(prices)
    cov = _covariance(returns, shrink=shrink, window=cov_window)
    aligned = cov.loc[base_weights.index, base_weights.index]
    port_var = float(base_weights.values.T @ aligned.values @ base_weights.values)
    port_vol = np.sqrt(port_var * 252)
    if port_vol > 0:
        scale = target_vol / port_vol
        base_weights = base_weights * scale
    if base_weights.abs().sum() > 0:
        base_weights = base_weights / base_weights.abs().sum()
    return base_weights


def kelly_position_sizing(
    predictions: pd.DataFrame, prices: pd.DataFrame, shrink: bool = True, cov_window: Optional[int] = None
) -> pd.Series:
    """Kelly sizing using predicted return vector and covariance."""
    returns = compute_return_matrix(prices)
    cov = _covariance(returns, shrink=shrink, window=cov_window)
    mu = predictions.set_index("ticker")["predicted_return"]
    tickers = [t for t in cov.columns if t in mu.index]
    if not tickers:
        return pd.Series(dtype=float)
    cov = cov.loc[tickers, tickers]
    mu = mu.loc[tickers]
    try:
        inv = np.linalg.pinv(cov.values)
        w = inv.dot(mu.values)
        weights = pd.Series(w, index=tickers)
        weights = weights.clip(lower=0)
        weights = weights / (weights.abs().sum() + 1e-9)
        return weights
    except Exception:
        return mu / (mu.abs().sum() + 1e-9)
