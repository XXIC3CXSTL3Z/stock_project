from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from .backtest import simulate_portfolio_nav, walk_forward_validation
from .fetch import fetch_latest_prices
from .recommend import (
    format_recommendations,
    generate_deep_recommendations,
    generate_recommendations,
)
from .deep import visualize_attention


def _parse_tickers(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def run_dashboard() -> None:
    st.set_page_config(page_title="Forecast Dashboard", layout="wide")
    st.title("Multi-Horizon Forecast Dashboard")
    st.caption("Classical + deep models with walk-forward features and portfolio overlays.")

    with st.sidebar:
        st.header("Data")
        tickers_text = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,BTC")
        crypto = st.checkbox("Crypto tickers", value=False)
        period = st.selectbox("Lookback", options=["3mo", "6mo", "1y"], index=1)
        top_n = st.slider("Top N", min_value=3, max_value=20, value=5)
        horizons: Sequence[int] = st.multiselect("Horizons (days)", options=[1, 5, 10], default=[1, 5, 10])
        weighting = st.selectbox("Weighting", options=["markowitz", "risk_parity", "black_litterman"])
        shrink_cov = st.checkbox("Ledoit-Wolf shrinkage", value=True)

        st.header("Model")
        mode = st.selectbox("Mode", options=["Classical", "Deep LSTM", "Deep Transformer"])
        model_type = st.selectbox(
            "Classical model",
            options=["random_forest", "ridge", "linear", "gbrt"],
            index=0,
            disabled=mode != "Classical",
        )
        epochs = st.slider("Epochs (deep)", min_value=5, max_value=60, value=20, disabled="Deep" not in mode)
        seq_len = st.slider(
            "Sequence length", min_value=5, max_value=30, value=12, disabled="Deep" not in mode
        )

    if st.button("Run forecast"):
        tickers = _parse_tickers(tickers_text)
        with st.spinner("Running pipeline..."):
            prices = fetch_latest_prices(tickers, period=period, crypto=crypto, cache_dir=Path("artifacts/cache"))
            if mode == "Classical":
                preds = generate_recommendations(
                    tickers=tickers,
                    period=period,
                    top_n=top_n,
                    model_type=model_type,
                    horizons=horizons,
                    weighting=weighting,
                    shrink_cov=shrink_cov,
                    crypto=crypto,
                )
            else:
                arch = "lstm" if "LSTM" in mode else "transformer"
                preds = generate_deep_recommendations(
                    tickers=tickers,
                    period=period,
                    top_n=top_n,
                    arch=arch,
                    seq_len=seq_len,
                    epochs=epochs,
                    horizons=horizons,
                    weighting=weighting,
                    shrink_cov=shrink_cov,
                    crypto=crypto,
                    attention_dir=Path("artifacts"),
                )

        st.subheader("Recommendations")
        st.dataframe(preds)
        st.text(format_recommendations(preds))

        price_wide = prices.pivot(index="date", columns="ticker", values="close")
        st.subheader("Price chart")
        st.line_chart(price_wide)

        heatmap = preds.pivot_table(index="ticker", columns="horizon", values="predicted_return")
        st.subheader("Predicted return heatmap")
        st.dataframe(heatmap)

        if "markowitz_weight" in preds:
            st.subheader("Allocation (Markowitz/Risk Parity/Black-Litterman)")
            alloc = preds[["ticker", "horizon", "markowitz_weight"]]
            st.bar_chart(alloc, x="ticker", y="markowitz_weight", color="horizon")

        if st.checkbox("Run quick backtest"):
            wf = walk_forward_validation(prices=prices, horizons=horizons, window=40, model_type="random_forest")
            nav_df, nav_metrics = simulate_portfolio_nav(
                wf, prices, weighting=weighting, benchmark_prices=None, plots_dir=Path("artifacts")
            )
            if not nav_df.empty:
                st.subheader("NAV curve")
                st.line_chart(nav_df.set_index("date")[["nav"]])
            st.write("Backtest metrics", nav_metrics)

        history = preds.attrs.get("history")
        if history:
            hist_df = pd.DataFrame(history)
            st.subheader("Training curves")
            st.line_chart(hist_df.set_index("epoch")[["train_loss", "val_loss"]])

        attention_path = preds.attrs.get("attention_path")
        if attention_path:
            st.subheader("Attention heatmap (Transformer)")
            st.image(str(attention_path))


if __name__ == "__main__":
    run_dashboard()
