# Pairs_Trading
This project implements a simple pairs-trading workflow. 
It downloads historical Adjusted Close prices, tests for cointegration between stock pairs, 
computes price ratios and z-scores, generates trading signals, 
runs a basic backtest and saves diagnostic plots (heatmap, ratio charts, z-score, signals).
---

## Description
This project implements a simple **pairs trading** strategy:
- Downloads historical price data (Adjusted Close) for a list of tickers.
- Tests for cointegration between pairs of stocks.
- Computes price ratios and z-scores, generates buy/sell signals.
- Runs a basic backtest and saves plots.


## Run order
1. First run **`get_data.py`** to download and save CSV files into the `DATA/` folder:

```bash
python CODE/get_data.py
```

2. Then run the main analysis script **`pairs_trading.py`** which loads the CSVs and runs the analysis/backtest:

```bash
python CODE/pairs_trading.py
```

Plots are saved to the `PLOTS/` folder and CSV files to `DATA/`.

---

## Requirements

```bash
pip install -r requirements.txt
```

---
