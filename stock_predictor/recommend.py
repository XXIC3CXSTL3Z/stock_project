from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from .data import load_price_data
from .deep import (
    load_checkpoint,
    predict_sequence_model,
    train_sequence_model,
    visualize_attention,
)
from .features import engineer_features
from .fetch import fetch_benchmark_prices, fetch_latest_prices, fetch_macro_indicators
from .model import select_top_features, train_and_predict, tune_hyperparameters
from .portfolio import (
    black_litterman_weights,
    hrp_weights,
    kelly_position_sizing,
    markowitz_weights,
    risk_parity_weights,
    volatility_target_weights,
)


def _apply_allocation(
    preds: pd.DataFrame,
    prices: pd.DataFrame,
    weighting: str,
    risk_aversion: float,
    shrink_cov: bool,
    vol_target: Optional[float],
    cov_window: int,
    use_hrp: bool,
    use_kelly: bool,
    view_confidence: float = 1.0,
) -> pd.DataFrame:
    preds = preds.copy()
    preds["markowitz_weight"] = 0.0
    if preds.empty:
        return preds

    if use_hrp:
        weights = hrp_weights(prices, shrink=shrink_cov, cov_window=cov_window)
    elif weighting == "risk_parity":
        weights = risk_parity_weights(prices)
    elif weighting == "black_litterman":
        weights = black_litterman_weights(
            preds, prices, risk_aversion=risk_aversion, shrink=shrink_cov, view_confidence=view_confidence, cov_window=cov_window
        )
    else:
        weights = markowitz_weights(
            preds, prices, risk_aversion=risk_aversion, shrink=shrink_cov, cov_window=cov_window
        )
    if use_kelly:
        kelly = kelly_position_sizing(preds, prices, shrink=shrink_cov, cov_window=cov_window)
        if not kelly.empty:
            weights = (weights.add(kelly, fill_value=0) / 2).fillna(0)
    if not weights.empty:
        if vol_target:
            weights = volatility_target_weights(
                weights, prices, target_vol=vol_target, cov_window=cov_window, shrink=shrink_cov
            )
        preds["markowitz_weight"] = preds["ticker"].map(weights).fillna(0.0)
    return preds


def _blend_horizon_frames(frames: list[pd.DataFrame], horizon_weights: Dict[int, float]) -> pd.DataFrame:
    weights = pd.Series(horizon_weights)
    weights = weights / weights.sum()
    merged = []
    stacked = pd.concat(frames, ignore_index=True)
    for ticker, grp in stacked.groupby("ticker"):
        pred = 0.0
        weight = 0.0
        for h, w in weights.items():
            sub = grp[grp["horizon"] == f"{h}d"]
            if sub.empty:
                continue
            pred += float(sub["predicted_return"].iloc[0]) * float(w)
            weight += float(sub["recommended_weight"].iloc[0]) * float(w)
        merged.append(
            {
                "ticker": ticker,
                "predicted_return": pred,
                "recommended_weight": weight,
                "markowitz_weight": weight,
                "horizon": "blend",
                "method": grp["method"].iloc[0],
                "samples": grp["samples"].max(),
                "latest_date": grp["latest_date"].max(),
            }
        )
    result = pd.DataFrame(merged).sort_values("predicted_return", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def _ensemble_predictions(base: pd.DataFrame, deep_preds: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """Blend classical and deep predictions."""
    blended = base.merge(
        deep_preds[["ticker", "horizon", "predicted_return"]].rename(
            columns={"predicted_return": "deep_pred"}
        ),
        on=["ticker", "horizon"],
        how="left",
    )
    blended["predicted_return"] = (1 - weight) * blended["predicted_return"] + weight * blended["deep_pred"].fillna(
        blended["predicted_return"]
    )
    blended.drop(columns=["deep_pred"], inplace=True)
    blended = blended.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    blended["rank"] = blended.index + 1
    total_positive = blended[blended["predicted_return"] > 0]["predicted_return"].sum()
    if total_positive > 0:
        blended["recommended_weight"] = blended["predicted_return"].clip(lower=0) / total_positive
    return blended


def generate_recommendations(
    data_path: Optional[Path] = None,
    tickers: Optional[Iterable[str]] = None,
    model_type: str = "random_forest",
    top_n: Optional[int] = 5,
    period: str = "6mo",
    auto_feature_k: Optional[int] = 8,
    cv_folds: int = 0,
    use_markowitz: bool = True,
    risk_aversion: float = 1.0,
    horizons: Sequence[int] = (1, 5, 10),
    sharpe_ranking: bool = True,
    shrink_cov: bool = True,
    weighting: str = "markowitz",
    checkpoint_dir: Optional[Path] = None,
    load_checkpoints: bool = False,
    hyper_grid: Optional[Sequence[str]] = None,
    crypto: bool = False,
    benchmark: bool = True,
    horizon_weights: Optional[Dict[int, float]] = None,
    vol_target: Optional[float] = None,
    rolling_cov_window: int = 60,
    use_hrp: bool = False,
    use_kelly: bool = False,
    view_confidence: float = 1.0,
    regime_aware: bool = False,
    bayes_trials: int = 0,
    bayes_model: str = "random_forest",
    ensemble_with_deep: bool = False,
    deep_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """End-to-end orchestration to produce ranked investment ideas."""
    if tickers:
        prices = fetch_latest_prices(tickers, period=period, crypto=crypto)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    benchmark_prices = None
    macro_df = None
    if benchmark:
        try:
            benchmark_prices = fetch_benchmark_prices(cache_dir=Path("artifacts/cache"))
        except Exception:
            benchmark_prices = None
        try:
            macro_df = fetch_macro_indicators(cache_dir=Path("artifacts/cache"))
        except Exception:
            macro_df = None
    bench_returns = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_returns = (
            benchmark_prices.pivot(index="date", columns="ticker", values="close").sort_index().pct_change()
        )

    train_df, latest_df, feature_cols = engineer_features(
        prices,
        horizons=horizons,
        benchmark_returns=bench_returns,
        macro_df=macro_df,
        crypto=crypto,
    )

    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(
            train_df, feature_cols, top_k=auto_feature_k, target_col="target_return_1d"
        )

    if train_df.empty or latest_df.empty:
        raise ValueError("Not enough rows to build features and targets. Add more history.")

    horizon_frames = []
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        preds_h = train_and_predict(
            train_df=train_df,
            latest_df=latest_df,
            feature_cols=feature_cols,
            model_type=model_type,
            cv_folds=cv_folds,
            target_col=target_col,
            horizon_label=f"{horizon}d",
            sharpe_ranking=sharpe_ranking,
            checkpoint_dir=checkpoint_dir,
            load_checkpoints=load_checkpoints,
            regime_aware=regime_aware,
        )

        if use_markowitz and not preds_h.empty:
            preds_h = _apply_allocation(
                preds_h,
                prices=prices,
                weighting=weighting,
                risk_aversion=risk_aversion,
                shrink_cov=shrink_cov,
                vol_target=vol_target,
                cov_window=rolling_cov_window,
                use_hrp=use_hrp,
                use_kelly=use_kelly,
                view_confidence=view_confidence,
            )
        else:
            preds_h["markowitz_weight"] = 0.0
        else:
            preds_h["markowitz_weight"] = 0.0

        if top_n is not None:
            preds_h = preds_h.head(top_n)
        _normalize_column(preds_h, "recommended_weight")
        _normalize_column(preds_h, "markowitz_weight")
        horizon_frames.append(preds_h)

    if not horizon_frames:
        return pd.DataFrame()

    predictions = pd.concat(horizon_frames, ignore_index=True)
    if horizon_weights:
        blended = _blend_horizon_frames(horizon_frames, horizon_weights=horizon_weights)
        predictions = pd.concat([predictions, blended], ignore_index=True)

    if hyper_grid:
        tuning = tune_hyperparameters(
            train_df=train_df,
            feature_cols=feature_cols,
            model_grid=hyper_grid,
            target_col="target_return_1d",
            cv_folds=cv_folds or 3,
            bayes_trials=bayes_trials,
            bayes_model=bayes_model,
        )
        predictions.attrs["tuning_results"] = tuning

    if ensemble_with_deep:
        deep_kwargs = deep_params or {}
        deep_preds = generate_deep_recommendations(
            data_path=data_path,
            tickers=tickers,
            period=period,
            top_n=top_n,
            horizons=horizons,
            weighting=weighting,
            shrink_cov=shrink_cov,
            crypto=crypto,
            attention_dir=Path("artifacts"),
            **deep_kwargs,
        )
        predictions = _ensemble_predictions(predictions, deep_preds, weight=0.5)

    return predictions


def generate_deep_recommendations(
    data_path: Optional[Path] = None,
    tickers: Optional[Iterable[str]] = None,
    period: str = "6mo",
    arch: str = "lstm",
    seq_len: int = 12,
    epochs: int = 25,
    auto_feature_k: Optional[int] = 10,
    use_markowitz: bool = True,
    risk_aversion: float = 1.0,
    top_n: Optional[int] = 5,
    horizons: Sequence[int] = (1, 5, 10),
    shrink_cov: bool = True,
    weighting: str = "markowitz",
    checkpoint_path: Optional[Path] = None,
    load_checkpoint: bool = False,
    patience: int = 5,
    log_progress: bool = True,
    crypto: bool = False,
    attention_dir: Optional[Path] = Path("artifacts"),
    prices_df: Optional[pd.DataFrame] = None,
    vol_target: Optional[float] = None,
    rolling_cov_window: int = 60,
    use_hrp: bool = False,
    use_kelly: bool = False,
    view_confidence: float = 1.0,
    device: Optional[str] = None,
    rolling_norm_window: Optional[int] = None,
    tensorboard_logdir: Optional[Path] = None,
    horizon_weights: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Train an LSTM/Transformer on sequences of engineered features and rank tickers."""
    if prices_df is not None:
        prices = prices_df
    elif tickers:
        prices = fetch_latest_prices(tickers, period=period, crypto=crypto)
    elif data_path:
        prices = load_price_data(data_path)
    else:
        raise ValueError("Provide either a data CSV path or a list of tickers to fetch.")

    benchmark_prices = None
    macro_df = None
    if not crypto:
        try:
            benchmark_prices = fetch_benchmark_prices(cache_dir=Path("artifacts/cache"))
        except Exception:
            benchmark_prices = None
        try:
            macro_df = fetch_macro_indicators(cache_dir=Path("artifacts/cache"))
        except Exception:
            macro_df = None
    bench_returns = None
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_returns = (
            benchmark_prices.pivot(index="date", columns="ticker", values="close").sort_index().pct_change()
        )

    train_df, _, feature_cols = engineer_features(
        prices, horizons=horizons, benchmark_returns=bench_returns, macro_df=macro_df, crypto=crypto
    )
    if auto_feature_k and len(feature_cols) > auto_feature_k:
        feature_cols = select_top_features(
            train_df, feature_cols, top_k=auto_feature_k, target_col="target_return_1d"
        )

    horizon_frames = []
    latest_dates = train_df.groupby("ticker")["date"].max()
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        model = None
        mean = None
        std = None
        history = []
        target_std = float(train_df[target_col].std()) if len(train_df) > 1 else 0.0
        if load_checkpoint and checkpoint_path and checkpoint_path.exists():
            model, payload = load_checkpoint(checkpoint_path, arch=arch, input_size=len(feature_cols))
            mean = payload["mean"]
            std = payload["std"]
        else:
            model, mean, std, history = train_sequence_model(
                train_df=train_df,
                feature_cols=feature_cols,
                seq_len=seq_len,
                arch=arch,
                epochs=epochs,
                target_col=target_col,
                patience=patience,
                checkpoint_path=checkpoint_path,
                log_progress=log_progress,
                device=device,
                rolling_norm_window=rolling_norm_window,
                tensorboard_logdir=tensorboard_logdir,
            )
        preds_h = predict_sequence_model(
            model=model,
            df=train_df,
            feature_cols=feature_cols,
            mean=mean,
            std=std,
            seq_len=seq_len,
            method=f"deep_{arch}",
            horizon_label=f"{horizon}d",
            rolling_norm_window=rolling_norm_window,
        )

        preds_h["latest_date"] = preds_h["ticker"].map(latest_dates)
        if target_std > 0:
            preds_h["sharpe_score"] = preds_h["predicted_return"] / (target_std + 1e-6)
            preds_h = preds_h.sort_values("sharpe_score", ascending=False).reset_index(drop=True)
            preds_h["rank"] = preds_h.index + 1
        if use_markowitz and not preds_h.empty:
            preds_h = _apply_allocation(
                preds_h,
                prices=prices,
                weighting=weighting,
                risk_aversion=risk_aversion,
                shrink_cov=shrink_cov,
                vol_target=vol_target,
                cov_window=rolling_cov_window,
                use_hrp=use_hrp,
                use_kelly=use_kelly,
                view_confidence=view_confidence,
            )

        if top_n is not None:
            preds_h = preds_h.head(top_n)
        _normalize_column(preds_h, "recommended_weight")
        _normalize_column(preds_h, "markowitz_weight")
        preds_h.attrs["history"] = history
        preds_h.attrs["model"] = model
        preds_h.attrs["mean"] = mean
        preds_h.attrs["std"] = std

        if attention_dir and hasattr(model, "attn_weights"):
            path = Path(attention_dir) / f"attention_{arch}_{horizon}d.png"
            saved = visualize_attention(getattr(model, "attn_weights"), path)
            if saved:
                preds_h.attrs["attention_path"] = saved

        horizon_frames.append(preds_h)

    if not horizon_frames:
        return pd.DataFrame()

    if horizon_weights:
        blended = _blend_horizon_frames(horizon_frames, horizon_weights=horizon_weights)
        horizon_frames.append(blended)

    result = pd.concat(horizon_frames, ignore_index=True)
    for frame in horizon_frames[::-1]:
        if "history" in frame.attrs:
            result.attrs["history"] = frame.attrs["history"]
            break
    # bubble up last attention path for dashboards
    for frame in horizon_frames[::-1]:
        if "attention_path" in frame.attrs:
            result.attrs["attention_path"] = frame.attrs["attention_path"]
            break
    return result


def _normalize_column(df: pd.DataFrame, column: str) -> None:
    """Normalize a weight column to sum to 1 if it has positive values."""
    if column not in df.columns:
        return
    positive_sum = df[column].clip(lower=0).sum()
    if positive_sum > 0:
        df[column] = df[column].clip(lower=0) / positive_sum


def format_recommendations(predictions: pd.DataFrame) -> str:
    """Render a lightweight text table."""
    if predictions.empty:
        return "No recommendations available."

    lines = [
        "Rank  Ticker  Horizon  Predicted Return  Weight  Markowitz Wt  Method        Samples  CV RMSE  Sharpe  Latest Date",
        "-" * 135,
    ]
    for _, row in predictions.iterrows():
        cv_display = f"{row['cv_rmse']:.4f}" if pd.notna(row.get("cv_rmse")) else "-"
        sharpe_display = f"{row['sharpe_score']:.3f}" if pd.notna(row.get("sharpe_score")) else "-"
        latest_display = (
            pd.to_datetime(row["latest_date"]).date()
            if "latest_date" in row and pd.notna(row.get("latest_date"))
            else "-"
        )
        method_display = row.get("method", "-")
        lines.append(
            f"{int(row['rank']):>4}  "
            f"{row['ticker']:<6}  "
            f"{row.get('horizon','-'):>7}  "
            f"{row['predicted_return']*100:>15.2f}%  "
            f"{row['recommended_weight']*100:>6.2f}%  "
            f"{row['markowitz_weight']*100:>12.2f}%  "
            f"{method_display:<12}  "
            f"{int(row['samples']):>7}  "
            f"{cv_display:>7}  "
            f"{sharpe_display:>6}  "
            f"{latest_display}"
        )

    best = predictions.iloc[0]
    lines.append("\nBest portfolio choice:")
    lines.append(
        f"- {best['ticker']} | predicted {best['predicted_return']*100:.2f}% | "
        f"Markowitz weight {best['markowitz_weight']*100:.2f}%"
    )

    return "\n".join(lines)
