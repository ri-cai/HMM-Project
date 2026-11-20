# HMM–Random Forest Regime Trading Strategy

A trading strategy that tries to detect market regimes (risk-on vs risk-off) using a Hidden Markov Model, then uses a Random Forest to predict the regime two days ahead. Backtested on VUAA (Vanguard S&P 500 UCITS ETF).

## How it works

Markets tend to switch between two broad states: one trending up with lower volatility, and one where drawdowns and choppiness dominate. The pipeline:

1. Engineer technical features from daily price data (returns, volatility, momentum, RSI, relative volume, etc.)
2. Compress the HMM inputs to 3 components with PCA
3. Fit a 2-state Gaussian HMM on a rolling 504-day window (refit monthly, filtered daily in between)
4. Label the states **Risk-On** / **Risk-Off**: the state with lower standardised volatility and drawdown depth is Risk-On
5. Train a Random Forest (walk-forward, retrained every 126 days) to predict the regime at $t+2$ from the HMM output plus technical indicators
6. Go long when the model predicts Risk-On, sit in cash when it predicts Risk-Off

The signal from the close of day $t$ is filled at the close of $t+1$. Transaction costs of 0.1% per unit of turnover are applied.

## Features used

Daily returns, log returns, 10-day rolling volatility, 4-day momentum, 10/50/99-day moving averages and their crossover, a normalised MA slope, price z-score, drawdown from the 252-day high, RSI, relative volume, and rolling kurtosis/skewness. The HMM sees a PCA-compressed subset; the Random Forest sees a wider set including the HMM's Risk-On probability and its daily change.

## What it outputs

- Classification report and confusion matrix for the out-of-sample RF predictions, compared against a persistence baseline (regime at $t+2$ = regime at $t$)
- Latest regime forecast with probability
- Strategy vs buy-and-hold: Sharpe, CAGR, volatility, max drawdown, Calmar ratio, hit rate, time invested
- Permutation importance of the labelling features (pre-out-of-sample data only)
- Plots: regime overlay on price, cumulative strategy vs benchmark, feature importance, confusion matrix

## Setup

Python 3.9+.

```bash
pip install -r requirements.txt
```

Download daily VUAA history as CSV (e.g. Investing.com) with `Date`, `Price` (or `Close`) and `Vol.` (or `Volume`) columns, put it in the project folder, and set the path at the top of the script:

```python
DATA_PATH = "VUAA Historical Data.csv"
```

Then run:

```bash
python hmm_regime_strategy.py
```

## Things to keep in mind

- This is a backtest, not a live system. There's no execution or broker integration.
- Every feature uses data up to day $t$ only, the HMM is fit on trailing windows, and the RF trains only on rows whose target was observable at training time. Still, review the pipeline before reading too much into the results.
- HMM fits that fail to converge on a window are dropped.
- The 0.1% transaction cost is a rough estimate. Real spreads and slippage will vary.
