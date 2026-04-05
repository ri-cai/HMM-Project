# HMM-Project
Hidden Markov Model, with machine learning to forecast market regimes
HMM–Random Forest Regime Trading Strategy
A trading strategy that tries to detect market regimes (think: "risk-on" vs "risk-off" periods) using a Hidden Markov Model, then uses a Random Forest to predict what regime tomorrow will be. Backtested on VUAA (Vanguard S&P 500 UCITS ETF).
How it works
The idea is that markets tend to switch between two broad states — one where things are generally trending up with lower volatility, and one where drawdowns and choppiness dominate. The pipeline goes:

Engineer a bunch of technical features from daily price data (returns, volatility, momentum, RSI, relative volume, etc.)
Compress the key ones down to 3 components with PCA
Fit a Gaussian HMM on a rolling 504-day window to classify each day into one of two hidden states
Label those states as "Inflationary" or "Deflationary" using KMeans on per-state averages of volatility, drawdown, momentum, and volume
Train a Random Forest (retrained every ~126 days) to predict tomorrow's regime using the HMM output + technical indicators
Go long when the model predicts inflationary, sit in cash when it predicts deflationary

Transaction costs of 0.1% are applied on every position switch.
Features used
The model pulls from daily returns, log returns, 10-day rolling volatility, 4-day momentum, several moving averages (10/50/99-day) and their crossovers, a normalised MA slope, price z-score, drawdown from highs, RSI, relative volume, and rolling kurtosis/skewness. The HMM sees a PCA-compressed subset of these; the Random Forest sees a wider set including the HMM's state probabilities and their deltas.
What it outputs

Classification report and confusion matrix for the RF predictions
Predicted next regime with confidence score
Performance comparison (strategy vs buy-and-hold) including Sharpe, CAGR, max drawdown, Calmar ratio, hit rate
Plots: regime overlay on price, cumulative strategy vs benchmark, RF feature importances

Setup
You'll need Python 3.9+ and these packages:
numpy
pandas
scikit-learn
hmmlearn
matplotlib
joblib
Install with:
bashpip install -r requirements.txt
You also need a CSV of daily VUAA data with Date, Price (or Close), and Vol. (or volume) columns. You can grab one from Investing.com or Kaggle. Drop it in the project folder and update the path at the top of the script:
pythonX = "VUAA Historical Data.csv"
Then just run:
bashpython hmm_regime_strategy.py
Things to keep in mind

This is a backtest, not a live system. There's no execution or broker integration.
Rolling windows and expanding training splits are used throughout to avoid look-ahead bias, but it's worth reviewing the pipeline yourself before reading too much into the results.
HMM fitting doesn't always converge — when it fails on a window, that result is just dropped.
The 0.1% transaction cost is a rough estimate. Real spreads and slippage will vary.
