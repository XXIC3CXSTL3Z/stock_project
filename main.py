import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional

from stock_predictor.backtest import (
    generate_backtest_report,
    simulate_portfolio_nav,
    walk_forward_validation,
)
from stock_predictor.config import apply_env_overrides, load_config, merge_config
from stock_predictor.recommend import (
    format_recommendations,
    generate_deep_recommendations,
    generate_recommendations,
)


def _add_common_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Ticker symbols to fetch live data for (uses yfinance).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="sample_data/sample_prices.csv",
        help="Path to CSV with columns: date,ticker,close (used when --symbols is absent).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="6mo",
        help="Lookback period for live fetch (e.g., 1mo, 3mo, 6mo, 1y).",
    )
    parser.add_argument(
        "--crypto",
        action="store_true",
        help="Interpret tickers as crypto symbols (auto-append -USD).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional YAML/JSON config file; values in the file override CLI flags.",
    )


def _parse_horizon_weights(raw: Optional[list[str]]) -> Optional[Dict[int, float]]:
    if not raw:
        return None
    weights: Dict[int, float] = {}
    for item in raw:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            weights[int(key)] = float(value)
        except ValueError:
            continue
    return weights or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool to rank stocks by predicted returns and suggest portfolio weights.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="Classical models + allocations")
    _add_common_sources(rank)
    rank.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear", "ridge", "gbrt"],
        help="Model type for return prediction.",
    )
    rank.add_argument("--top", type=int, default=5, help="Number of tickers to display.")
    rank.add_argument(
        "--auto-top-k",
        type=int,
        default=8,
        help="Auto feature selection: keep top-k features by importance.",
    )
    rank.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Time-series cross-validation folds per ticker (0 disables).",
    )
    rank.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Higher values shrink Markowitz weights toward uniform.",
    )
    rank.add_argument(
        "--no-markowitz",
        action="store_true",
        help="Disable Markowitz allocation and only show score-based weights.",
    )
    rank.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Prediction horizons in days.",
    )
    rank.add_argument(
        "--weighting",
        type=str,
        default="markowitz",
        choices=["markowitz", "risk_parity", "black_litterman"],
        help="Allocation method.",
    )
    rank.add_argument(
        "--no-shrink-cov",
        action="store_true",
        help="Disable Ledoit-Wolf covariance shrinkage.",
    )
    rank.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to save/load sklearn checkpoints.",
    )
    rank.add_argument(
        "--load-checkpoints",
        action="store_true",
        help="Load existing checkpoints when available.",
    )
    rank.add_argument(
        "--hyper-grid",
        type=str,
        nargs="*",
        help="Optional list of model types to sweep for hyperparameter tuning.",
    )
    rank.add_argument(
        "--vol-target",
        type=float,
        help="Target annualized volatility for allocation scaling.",
    )
    rank.add_argument(
        "--rolling-cov-window",
        type=int,
        default=60,
        help="Rolling window for covariance estimation.",
    )
    rank.add_argument(
        "--hrp",
        action="store_true",
        help="Use hierarchical risk parity allocation.",
    )
    rank.add_argument(
        "--kelly",
        action="store_true",
        help="Blend Kelly sizing into allocations.",
    )
    rank.add_argument(
        "--view-confidence",
        type=float,
        default=1.0,
        help="Black-Litterman view confidence scaling.",
    )
    rank.add_argument(
        "--regime-aware",
        action="store_true",
        help="Switch model choice based on volatility regime.",
    )
    rank.add_argument(
        "--bayes-trials",
        type=int,
        default=0,
        help="Optuna/Bayesian tuning trials.",
    )
    rank.add_argument(
        "--bayes-model",
        type=str,
        default="random_forest",
        help="Model to tune with Optuna when --bayes-trials>0.",
    )
    rank.add_argument(
        "--horizon-weights",
        type=str,
        nargs="*",
        help="Blend horizons with weights like 1=0.5 5=0.3 10=0.2.",
    )
    rank.add_argument(
        "--ensemble-deep",
        action="store_true",
        help="Blend classical model with deep model outputs.",
    )

    deep = subparsers.add_parser("deep", help="LSTM/Transformer sequence models")
    _add_common_sources(deep)
    deep.add_argument(
        "--arch",
        type=str,
        default="lstm",
        choices=["lstm", "transformer"],
        help="Deep architecture to use.",
    )
    deep.add_argument("--seq-len", type=int, default=12, help="Sequence length for training.")
    deep.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    deep.add_argument("--top", type=int, default=5, help="Number of tickers to display.")
    deep.add_argument(
        "--auto-top-k",
        type=int,
        default=10,
        help="Auto feature selection: keep top-k features by importance.",
    )
    deep.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Higher values shrink Markowitz weights toward uniform.",
    )
    deep.add_argument(
        "--no-markowitz",
        action="store_true",
        help="Disable Markowitz allocation and only show score-based weights.",
    )
    deep.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Prediction horizons in days.",
    )
    deep.add_argument(
        "--weighting",
        type=str,
        default="markowitz",
        choices=["markowitz", "risk_parity", "black_litterman"],
        help="Allocation method.",
    )
    deep.add_argument(
        "--no-shrink-cov",
        action="store_true",
        help="Disable Ledoit-Wolf covariance shrinkage.",
    )
    deep.add_argument(
        "--checkpoint",
        type=str,
        help="Path to save/load deep-model checkpoints.",
    )
    deep.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs).",
    )
    deep.add_argument(
        "--device",
        type=str,
        help="Force device selection (cpu or cuda).",
    )
    deep.add_argument(
        "--tensorboard-logdir",
        type=str,
        help="Optional TensorBoard log directory.",
    )
    deep.add_argument(
        "--rolling-norm-window",
        type=int,
        help="Rolling window for on-the-fly feature normalization.",
    )
    deep.add_argument(
        "--vol-target",
        type=float,
        help="Target annualized volatility for allocation scaling.",
    )
    deep.add_argument(
        "--rolling-cov-window",
        type=int,
        default=60,
        help="Rolling window for covariance estimation.",
    )
    deep.add_argument(
        "--hrp",
        action="store_true",
        help="Use hierarchical risk parity allocation.",
    )
    deep.add_argument(
        "--kelly",
        action="store_true",
        help="Blend Kelly sizing into allocations.",
    )
    deep.add_argument(
        "--view-confidence",
        type=float,
        default=1.0,
        help="Black-Litterman view confidence scaling.",
    )
    deep.add_argument(
        "--horizon-weights",
        type=str,
        nargs="*",
        help="Blend horizons with weights like 1=0.5 5=0.3 10=0.2.",
    )

    backtest = subparsers.add_parser("backtest", help="Walk-forward validation + report")
    _add_common_sources(backtest)
    backtest.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear", "ridge", "gbrt"],
        help="Model for walk-forward validation.",
    )
    backtest.add_argument(
        "--window",
        type=int,
        default=60,
        help="Minimum history window before scoring.",
    )
    backtest.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Prediction horizons in days.",
    )
    backtest.add_argument(
        "--weighting",
        type=str,
        default="markowitz",
        choices=["markowitz", "risk_parity", "black_litterman"],
        help="Allocation method.",
    )
    backtest.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0005,
        help="Per-unit transaction cost applied on turnover.",
    )
    backtest.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage penalty on turnover.",
    )
    backtest.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Maximum gross leverage.",
    )
    backtest.add_argument(
        "--vol-target",
        type=float,
        help="Target annualized volatility for allocation scaling.",
    )
    backtest.add_argument(
        "--rolling-cov-window",
        type=int,
        default=60,
        help="Rolling window for covariance estimation.",
    )
    backtest.add_argument(
        "--hrp",
        action="store_true",
        help="Use hierarchical risk parity allocation.",
    )
    backtest.add_argument(
        "--kelly",
        action="store_true",
        help="Blend Kelly sizing into allocations.",
    )
    backtest.add_argument(
        "--horizon-weights",
        type=str,
        nargs="*",
        help="Blend horizons with weights like 1=0.5 5=0.3 10=0.2.",
    )
    backtest.add_argument(
        "--rl-policy",
        action="store_true",
        help="Enable a lightweight reinforcement learning policy overlay.",
    )

    live = subparsers.add_parser("live", help="Live signal generation + alerts")
    _add_common_sources(live)
    live.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear", "ridge", "gbrt"],
        help="Model type for return prediction.",
    )
    live.add_argument("--top", type=int, default=5, help="Number of tickers to display.")
    live.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Prediction horizons in days.",
    )
    live.add_argument(
        "--weighting",
        type=str,
        default="markowitz",
        choices=["markowitz", "risk_parity", "black_litterman"],
        help="Allocation method.",
    )
    live.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Polling interval (seconds) when loop mode is enabled.",
    )
    live.add_argument(
        "--loop",
        action="store_true",
        help="Continuously refresh signals instead of running once.",
    )
    live.add_argument(
        "--horizon-weights",
        type=str,
        nargs="*",
        help="Blend horizons with weights like 1=0.5 5=0.3 10=0.2.",
    )
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard UI")
    return parser


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "config", None):
        return argparse.Namespace(**apply_env_overrides(vars(args)))
    cfg = load_config(Path(args.config))
    merged: Dict[str, Optional[str]] = merge_config(vars(args), cfg)
    merged = apply_env_overrides(merged)
    return argparse.Namespace(**merged)


def run_rank(args: argparse.Namespace) -> Optional[int]:
    args = _apply_config(args)
    horizon_weights = _parse_horizon_weights(getattr(args, "horizon_weights", None))
    predictions = generate_recommendations(
        data_path=Path(args.data) if not args.symbols else None,
        tickers=args.symbols,
        model_type=args.model,
        top_n=args.top,
        period=args.period,
        auto_feature_k=args.auto_top_k,
        cv_folds=args.cv_folds,
        use_markowitz=not args.no_markowitz,
        risk_aversion=args.risk_aversion,
        horizons=args.horizons,
        weighting=args.weighting,
        shrink_cov=not args.no_shrink_cov,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        load_checkpoints=args.load_checkpoints,
        hyper_grid=args.hyper_grid,
        crypto=args.crypto,
        vol_target=args.vol_target,
        rolling_cov_window=args.rolling_cov_window,
        use_hrp=args.hrp,
        use_kelly=args.kelly,
        view_confidence=args.view_confidence,
        regime_aware=args.regime_aware,
        bayes_trials=args.bayes_trials,
        bayes_model=args.bayes_model,
        horizon_weights=horizon_weights,
        ensemble_with_deep=args.ensemble_deep,
    )
    print(format_recommendations(predictions))
    tuning = predictions.attrs.get("tuning_results")
    if tuning is not None:
        print("\nHyperparameter sweep:")
        print(tuning)
    return 0


def run_deep(args: argparse.Namespace) -> Optional[int]:
    args = _apply_config(args)
    horizon_weights = _parse_horizon_weights(getattr(args, "horizon_weights", None))
    predictions = generate_deep_recommendations(
        data_path=Path(args.data) if not args.symbols else None,
        tickers=args.symbols,
        period=args.period,
        arch=args.arch,
        seq_len=args.seq_len,
        epochs=args.epochs,
        auto_feature_k=args.auto_top_k,
        use_markowitz=not args.no_markowitz,
        risk_aversion=args.risk_aversion,
        top_n=args.top,
        horizons=args.horizons,
        weighting=args.weighting,
        shrink_cov=not args.no_shrink_cov,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        load_checkpoint=bool(args.checkpoint),
        patience=args.patience,
        crypto=args.crypto,
        vol_target=args.vol_target,
        rolling_cov_window=args.rolling_cov_window,
        use_hrp=args.hrp,
        use_kelly=args.kelly,
        view_confidence=args.view_confidence,
        device=args.device,
        rolling_norm_window=args.rolling_norm_window,
        tensorboard_logdir=Path(args.tensorboard_logdir) if args.tensorboard_logdir else None,
        horizon_weights=horizon_weights,
    )
    print(format_recommendations(predictions))
    return 0


def run_backtest(args: argparse.Namespace) -> Optional[int]:
    args = _apply_config(args)
    horizon_weights = _parse_horizon_weights(getattr(args, "horizon_weights", None))
    if args.symbols:
        from stock_predictor.fetch import fetch_latest_prices

        prices = fetch_latest_prices(args.symbols, period=args.period, crypto=args.crypto, cache_dir=Path("artifacts/cache"))
    else:
        from stock_predictor.data import load_price_data

        prices = load_price_data(Path(args.data))

    wf = walk_forward_validation(
        prices=prices,
        horizons=args.horizons,
        window=args.window,
        model_type=args.model,
    )
    report = generate_backtest_report(wf)
    print("Walk-forward results:")
    print(wf.tail())
    print("\nPrediction error report:")
    for k, v in report.items():
        print(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

    from stock_predictor.fetch import fetch_benchmark_prices

    benchmarks = fetch_benchmark_prices(cache_dir=Path("artifacts/cache"))
    nav_df, nav_metrics = simulate_portfolio_nav(
        wf_df=wf,
        prices=prices,
        weighting=args.weighting,
        risk_aversion=1.0,
        shrink_cov=True,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        leverage=args.leverage,
        vol_target=args.vol_target,
        rolling_cov_window=args.rolling_cov_window,
        benchmark_prices=benchmarks,
        horizon_weights=horizon_weights,
        use_hrp=args.hrp,
        use_kelly=args.kelly,
        plots_dir=Path("artifacts"),
        rl_policy=args.rl_policy,
    )
    if not nav_df.empty:
        print("\nPortfolio NAV:")
        print(nav_df[["date", "nav", "turnover", "leverage"]].tail())
    if nav_metrics:
        print("\nPortfolio metrics:")
        for k, v in nav_metrics.items():
            print(f"- {k}: {v:.4f}")
    return 0


def run_dashboard(_: argparse.Namespace) -> Optional[int]:
    # Launch Streamlit programmatically for convenience.
    subprocess.run(["streamlit", "run", "stock_predictor/dashboard.py"], check=False)
    return 0


def run_live(args: argparse.Namespace) -> Optional[int]:
    args = _apply_config(args)
    horizon_weights = _parse_horizon_weights(getattr(args, "horizon_weights", None))
    from stock_predictor.live import live_signal_loop, live_signal_once

    live_kwargs = dict(
        model_type=args.model,
        top_n=args.top,
        horizons=args.horizons,
        weighting=args.weighting,
        horizon_weights=horizon_weights,
        crypto=args.crypto,
    )
    if args.loop:
        live_signal_loop(tickers=args.symbols, period=args.period, **live_kwargs)
    else:
        preds = live_signal_once(tickers=args.symbols, period=args.period, **live_kwargs)
        print(format_recommendations(preds))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "deep":
            return run_deep(args) or 0
        if args.command == "backtest":
            return run_backtest(args) or 0
        if args.command == "dashboard":
            return run_dashboard(args) or 0
        if args.command == "live":
            return run_live(args) or 0
        return run_rank(args) or 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
