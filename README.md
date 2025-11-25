# Stock Prediction CLI

Python CLI that fetches prices (or reads a CSV), engineers richer technical factors, trains classical or deep models, and suggests portfolio weights using Markowitz / risk-parity / Black-Litterman allocations. Ships with a small sample dataset so you can run it out of the box.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Classical models + Markowitz:
```bash
python main.py rank --symbols AAPL MSFT GOOG --period 6mo --model random_forest --cv-folds 3 --auto-top-k 8 --top 5 --horizons 1 5 10 --weighting markowitz
```

Deep models (LSTM/Transformer):
```bash
python main.py deep --symbols AAPL MSFT GOOG --arch transformer --seq-len 12 --epochs 20 --top 5 --patience 5 --horizons 1 5 10
```

Walk-forward backtest:
```bash
python main.py backtest --symbols AAPL MSFT GOOG --window 80 --horizons 1 5 10
```

Streamlit dashboard:
```bash
python -m streamlit run stock_predictor/dashboard.py
```

Offline with the sample CSV:
```bash
python main.py rank --data sample_data/sample_prices.csv --top 3
```

Live/alerts mode:
```bash
python main.py live --symbols AAPL MSFT --period 3mo --model random_forest --loop
```

Key flags:
- `--symbols`: tickers to fetch live via yfinance (overrides `--data`).
- `--data`: CSV path with `date,ticker,close` columns.
- `--period`: lookback window for live fetch (1mo, 3mo, 6mo, 1y, etc.).
- `--crypto`: treat tickers as crypto symbols (auto-append `-USD`).
- `rank` command: `--model` (random_forest, linear, ridge, gbrt), `--cv-folds`, `--auto-top-k`, `--risk-aversion`, `--horizons`, `--weighting` (markowitz, risk_parity, black_litterman), `--no-shrink-cov`, `--checkpoint-dir`, `--hyper-grid` for tuning.
- `deep` command: `--arch` (lstm, transformer), `--seq-len`, `--epochs`, `--patience` (early stopping), `--auto-top-k`, `--risk-aversion`, `--horizons`, `--weighting`, `--no-shrink-cov`, `--checkpoint`.
- `--top`: number of tickers to display.
- `--config`: load YAML/JSON config (values in the file override CLI flags).
- `backtest` command: run walk-forward validation and print a Sharpe/return report.
- Advanced allocation flags: `--vol-target`, `--rolling-cov-window`, `--hrp`, `--kelly`, `--horizon-weights`.
- Bayesian tuning: `--bayes-trials` and `--bayes-model`.
- Deep extras: `--tensorboard-logdir`, `--rolling-norm-window`, `--device`, `--horizon-weights`.

Example output (truncated):
```
Rank  Ticker  Horizon  Predicted Return  Weight  Markowitz Wt  Method        Samples  CV RMSE  Sharpe  Latest Date
---------------------------------------------------------------------------------------------------------------------------------------
   1  MSFT       1d            0.45%   48.18%        50.12%  random_forest        15   0.012   0.522  2024-01-30
   2  AAPL       5d            0.39%   41.10%        32.88%  random_forest        15   0.014   0.417  2024-01-30

Best portfolio choice:
- MSFT | predicted 0.45% | Markowitz weight 50.12%

Note: This demo is for educational purposes only and is not financial advice.
```

## Data format

CSV must include:
```
date,ticker,close
2024-01-02,AAPL,185.64
...
```
Dates should be ISO-8601, tickers are case-insensitive, and prices should be numeric. More history per ticker yields better signals; at least 15 rows per ticker is recommended.

## How it works

- Feature engineering: multi-horizon returns, RSI/MACD/Bollinger, EMA ratios, short/medium volatility, volatility-regime flag, moving-average ratio.
- Auto feature selection: RandomForest importances keep the strongest factors.
- Validation: time-series CV, walk-forward validation, backtest report with Sharpe/cumulative return.
- Models: classical (RandomForest, Linear, Ridge, GBRT) and deep (LSTM or Transformer with GELU, dropout, layer norm, early stopping, checkpoints, loss logging).
- Allocation: Markowitz (Ledoit–Wolf shrinkage), risk-parity, and Black–Litterman options.
- Dashboard: Streamlit UI with multi-horizon forecasts and attention heatmaps for Transformer runs.
- Config: load YAML/JSON configs to drive the CLI; optional model checkpoint save/load.

## Notes

- Predictions are illustrative only; do not use for real trading decisions.
- Replace the sample CSV with your own data for more realistic outputs, or fetch live prices with `--symbols`.
- Deep models and yfinance fetches need network access and may take longer to run.
