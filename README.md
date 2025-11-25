# Stock Prediction CLI

Python CLI that fetches prices (or reads a CSV), engineers richer technical factors, trains classical or deep models, and suggests portfolio weights using a Markowitz-style allocation. Ships with a small sample dataset so you can run it out of the box.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Classical models + Markowitz:
```bash
python main.py rank --symbols AAPL MSFT GOOG --period 6mo --model random_forest --cv-folds 3 --auto-top-k 8 --top 5
```

Deep models (LSTM/Transformer):
```bash
python main.py deep --symbols AAPL MSFT GOOG --arch transformer --seq-len 12 --epochs 20 --top 5
```

Offline with the sample CSV:
```bash
python main.py rank --data sample_data/sample_prices.csv --top 3
```

Key flags:
- `--symbols`: tickers to fetch live via yfinance (overrides `--data`).
- `--data`: CSV path with `date,ticker,close` columns.
- `--period`: lookback window for live fetch (1mo, 3mo, 6mo, 1y, etc.).
- `rank` command: `--model` (random_forest, linear, ridge, gbrt), `--cv-folds`, `--auto-top-k`, `--risk-aversion`, `--no-markowitz`.
- `deep` command: `--arch` (lstm, transformer), `--seq-len`, `--epochs`, `--auto-top-k`, `--risk-aversion`, `--no-markowitz`.
- `--top`: number of tickers to display.

Example output (truncated):
```
Rank  Ticker  Predicted Return  Weight  Markowitz Wt  Method        Samples  CV RMSE  Latest Date
---------------------------------------------------------------------------------------------------
   1  MSFT              0.45%   48.18%        50.12%  random_forest        15   0.012  2024-01-30
   2  AAPL              0.39%   41.10%        32.88%  random_forest        15   0.014  2024-01-30

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

- Feature engineering: multi-horizon returns and momentum, EMA ratios, short/medium volatility, volatility-regime flag, moving-average ratio.
- Auto feature selection: RandomForest importances keep the strongest factors.
- Cross-validation: optional time-series CV RMSE per ticker.
- Models: classical (RandomForest, Linear, Ridge, GBRT) and deep (LSTM or Transformer sequence models).
- Allocation: Markowitz-style weights from predicted returns and historical covariance, alongside score-based weights.

## Notes

- Predictions are illustrative only; do not use for real trading decisions.
- Replace the sample CSV with your own data for more realistic outputs, or fetch live prices with `--symbols`.
- Deep models and yfinance fetches need network access and may take longer to run.
