from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .deep import walk_forward_sequence_backtest
from .features import engineer_features
from .model import train_and_predict
from .portfolio import (
    black_litterman_weights,
    hrp_weights,
    kelly_position_sizing,
    markowitz_weights,
    risk_parity_weights,
    volatility_target_weights,
)


def walk_forward_validation(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    window: int = 60,
    model_type: str = "random_forest",
    min_samples: int = 16,
    feature_cols: Optional[Iterable[str]] = None,
    model_params: Optional[Dict] = None,
    regime_aware: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward validation across tickers and horizons.

    For each ticker, iteratively trains on data up to time t and evaluates on t.
    """
    full_train, _, engineered_features = engineer_features(prices, horizons=horizons)
    feature_cols = list(feature_cols) if feature_cols else engineered_features

    def _process_ticker(ticker: str, grp: pd.DataFrame) -> List[dict]:
        rows: List[dict] = []
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
                    model_params=model_params,
                    regime_aware=regime_aware,
                )
                if preds.empty:
                    continue
                pred_val = preds.loc[preds["ticker"] == ticker, "predicted_return"].iloc[0]
                rows.append(
                    {
                        "ticker": ticker,
                        "date": test_row["date"].iloc[0],
                        "horizon": horizon,
                        "predicted_return": pred_val,
                        "actual_return": test_row[target_col].iloc[0],
                    }
                )
        return rows

    grouped = list(full_train.groupby("ticker"))
    if n_jobs and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(_process_ticker)(t, g) for t, g in grouped)
        all_rows = [item for sublist in results for item in sublist]
    else:
        all_rows = []
        for ticker, grp in grouped:
            all_rows.extend(_process_ticker(ticker, grp))

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


def _blend_horizons(wf_df: pd.DataFrame, horizon_weights: Dict[int, float]) -> pd.DataFrame:
    weights = pd.Series(horizon_weights)
    weights = weights / weights.sum()
    merged: List[dict] = []
    for (date, ticker), grp in wf_df.groupby(["date", "ticker"]):
        expected = 0.0
        actual = 0.0
        for h, w in weights.items():
            row = grp[grp["horizon"] == h]
            if row.empty:
                continue
            expected += float(row["predicted_return"].iloc[0]) * float(w)
            actual += float(row["actual_return"].iloc[0]) * float(w)
        merged.append({"date": date, "ticker": ticker, "predicted_return": expected, "actual_return": actual})
    return pd.DataFrame(merged)


def _drawdown(nav: pd.Series) -> Tuple[float, pd.Series]:
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    return float(dd.min()), dd


def simulate_portfolio_nav(
    wf_df: pd.DataFrame,
    prices: pd.DataFrame,
    weighting: str = "markowitz",
    risk_aversion: float = 1.0,
    shrink_cov: bool = True,
    transaction_cost: float = 0.0005,
    slippage: float = 0.0005,
    turnover_limit: Optional[float] = None,
    leverage: float = 1.0,
    vol_target: Optional[float] = None,
    rolling_cov_window: int = 60,
    benchmark_prices: Optional[pd.DataFrame] = None,
    horizon_weights: Optional[Dict[int, float]] = None,
    use_kelly: bool = False,
    use_hrp: bool = False,
    plots_dir: Path = Path("artifacts"),
    rl_policy: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Portfolio-level NAV simulation with turnover, costs, slippage, leverage, and benchmark comparison.
    """
    if wf_df.empty:
        return pd.DataFrame(), {}

    wf_df = wf_df.copy()
    if horizon_weights:
        wf_df = _blend_horizons(wf_df, horizon_weights=horizon_weights)
        wf_df["horizon"] = "blend"
        horizon_pick = "blend"
    else:
        horizon_values = [h for h in wf_df["horizon"].unique() if pd.notna(h)]
        horizon_pick = sorted(horizon_values)[0] if horizon_values else None
        if horizon_pick is not None:
            wf_df = wf_df[wf_df["horizon"] == horizon_pick]

    nav_rows: List[dict] = []
    prev_weights = pd.Series(dtype=float)
    nav = 1.0
    benchmark_nav = {}
    bench_returns = None
    last_reward = 0.0
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_wide = benchmark_prices.pivot(index="date", columns="ticker", values="close").sort_index()
        bench_returns = bench_wide.pct_change().fillna(0)
        benchmark_nav = {b: 1.0 for b in bench_returns.columns}

    for date in sorted(wf_df["date"].unique()):
        preds = wf_df[wf_df["date"] == date]
        if preds.empty:
            continue
        history_prices = prices[prices["date"] <= date]
        if weighting == "risk_parity":
            weights = risk_parity_weights(history_prices)
        elif weighting == "black_litterman":
            weights = black_litterman_weights(
                preds, history_prices, risk_aversion=risk_aversion, shrink=shrink_cov, cov_window=rolling_cov_window
            )
        elif use_hrp:
            weights = hrp_weights(history_prices, shrink=shrink_cov, cov_window=rolling_cov_window)
        else:
            weights = markowitz_weights(
                preds,
                history_prices,
                risk_aversion=risk_aversion,
                shrink=shrink_cov,
                cov_window=rolling_cov_window,
            )

        if use_kelly:
            kelly = kelly_position_sizing(preds, history_prices, shrink=shrink_cov, cov_window=rolling_cov_window)
            if not kelly.empty:
                weights = (weights.add(kelly, fill_value=0) / 2).fillna(0)

        weights = weights.clip(lower=0)
        if prev_weights.empty:
            prev_weights = pd.Series(0.0, index=weights.index)
        turnover = float(np.abs(weights - prev_weights.reindex(weights.index).fillna(0)).sum())
        if turnover_limit and turnover > turnover_limit:
            blend = turnover_limit / (turnover + 1e-6)
            weights = prev_weights.reindex(weights.index).fillna(0) * (1 - blend) + weights * blend
            turnover = turnover_limit

        gross = float(weights.abs().sum())
        if gross > leverage and gross > 0:
            weights = weights * (leverage / gross)

        if vol_target:
            weights = volatility_target_weights(
                weights, history_prices, target_vol=vol_target, cov_window=rolling_cov_window, shrink=shrink_cov
            )

        if rl_policy:
            weights = weights * (1 + max(-0.2, last_reward))
            weights = weights.clip(lower=0)
            if weights.sum() > 0:
                weights = weights / weights.sum()

        realized = preds.set_index("ticker")["actual_return"]
        realized = realized.reindex(weights.index).fillna(0.0)
        cost = turnover * (transaction_cost + slippage)
        port_ret = float((weights * realized).sum() * leverage - cost)
        nav *= 1 + port_ret
        last_reward = port_ret

        bench_snapshot = {}
        if benchmark_nav:
            if date in bench_returns.index:
                bench_rets = bench_returns.loc[date]
                for b in benchmark_nav:
                    benchmark_nav[b] *= 1 + float(bench_rets.get(b, 0))
            bench_snapshot = benchmark_nav.copy()

        nav_rows.append(
            {
                "date": date,
                "nav": nav,
                "turnover": turnover,
                "leverage": float(weights.abs().sum()),
                "portfolio_return": port_ret,
                **{f"weight_{k}": v for k, v in weights.items()},
                **{f"bench_{k}": v for k, v in bench_snapshot.items()},
            }
        )
        prev_weights = weights

    nav_df = pd.DataFrame(nav_rows).sort_values("date")
    metrics: Dict[str, float] = {}
    if not nav_df.empty:
        nav_series = nav_df.set_index("date")["nav"]
        dd, dd_series = _drawdown(nav_series)
        metrics["max_drawdown"] = float(dd)
        metrics["final_nav"] = float(nav_series.iloc[-1])
        metrics["cumulative_return"] = float(nav_series.iloc[-1] - 1)
        metrics["avg_turnover"] = float(nav_df["turnover"].mean())
        metrics["volatility"] = float(nav_df["portfolio_return"].std() * np.sqrt(252))
        if len(nav_df) > 20:
            rolling = nav_df["portfolio_return"].rolling(20)
            nav_df["rolling_sharpe"] = rolling.mean() / (rolling.std() + 1e-9) * np.sqrt(252)
            metrics["rolling_sharpe"] = float(nav_df["rolling_sharpe"].dropna().mean())
        nav_df["drawdown"] = dd_series.values
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not nav_df.empty:
        _plot_nav(nav_df, plots_dir=plots_dir)
    return nav_df, metrics


def _plot_nav(nav_df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(nav_df["date"], nav_df["nav"], label="Portfolio NAV")
    bench_cols = [c for c in nav_df.columns if c.startswith("bench_")]
    for col in bench_cols:
        plt.plot(nav_df["date"], nav_df[col], label=col.replace("bench_", "Benchmark "))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "nav_curve.png")
    plt.close()
    if "drawdown" in nav_df:
        plt.figure(figsize=(8, 3))
        plt.plot(nav_df["date"], nav_df["drawdown"], color="red")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(plots_dir / "drawdown.png")
        plt.close()
    if "rolling_sharpe" in nav_df:
        plt.figure(figsize=(8, 3))
        plt.plot(nav_df["date"], nav_df["rolling_sharpe"], color="purple")
        plt.xlabel("Date")
        plt.ylabel("Rolling Sharpe (20d)")
        plt.tight_layout()
        plt.savefig(plots_dir / "rolling_sharpe.png")
        plt.close()
