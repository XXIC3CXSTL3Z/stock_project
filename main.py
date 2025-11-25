import argparse
from pathlib import Path
from typing import Optional

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool to rank stocks by predicted returns and suggest portfolio weights.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="Classical models + Markowitz allocation")
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

    return parser


def run_rank(args: argparse.Namespace) -> Optional[int]:
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
    )
    print(format_recommendations(predictions))
    return 0


def run_deep(args: argparse.Namespace) -> Optional[int]:
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
    )
    print(format_recommendations(predictions))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "deep":
            return run_deep(args) or 0
        return run_rank(args) or 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
