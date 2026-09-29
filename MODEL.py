import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "VUAA Historical Data.csv"

HMM_WINDOW = 504           # trailing window (trading days) used to fit each HMM
HMM_REFIT_EVERY = 21       # refit the HMM monthly, filter daily in between
HMM_STATES = 2
HMM_RESTARTS = 10          # EM restarts; the fit with the highest log-likelihood is kept
PCA_COMPONENTS = 3

RF_BURN_IN = 252           # rows of HMM output required before the first RF fit
RF_RETRAIN_EVERY = 126

FILL_DELAY = 1             # signal from close t is filled at close t + FILL_DELAY
HORIZON = FILL_DELAY + 1   # so the position first earns the return of day t + HORIZON
TRANSACTION_COST = 0.001   # per unit of turnover

FI_FORWARD_DAYS = 20       # horizon for the labelling-feature importance check
SEED = 42

HMM_FEATURES = ["log_r", "vlt", "ma_s", "drawdown", "priceZ", "rsi", "mmt", "Rel_Volume"]
LABEL_FEATURES = ["vlt", "drawdown"]
LABEL_CANDIDATES = ["vlt", "drawdown", "Rel_Volume", "mmt", "rsi", "priceZ", "ma_s", "kurt", "skew"]
RF_FEATURES = ["p_risk_on", "p_risk_on_delta", "ma_c", "mmt", "rtn", "rsi", "ma_s",
               "drawdown", "priceZ", "vlt", "Rel_Volume", "kurt", "skew"]
REGIME_NAMES = {0: "Risk-Off", 1: "Risk-On"}


# ---------------------------------------------------------------------------
# Data loading and feature engineering (every feature uses data up to day t only)
# ---------------------------------------------------------------------------
def to_number(x):
    if isinstance(x, str):
        x = x.strip().replace(",", "")
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(x[-1:], 1.0)
        if mult != 1.0:
            x = x[:-1]
        try:
            return float(x) * mult
        except ValueError:
            return np.nan
    return x


def load_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Price" not in df.columns:
        for c in ["Adj Close", "Close", "close"]:
            if c in df.columns:
                df["Price"] = df[c]
                break
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = df["date"]
    if "Vol." not in df.columns:
        for c in ["Volume", "volume"]:
            if c in df.columns:
                df["Vol."] = df[c]
                break

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Price"] = df["Price"].apply(to_number).astype(float)
    df["Volume"] = df["Vol."].apply(to_number).astype(float)
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def wilder_rsi(price, window=14):
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(df):
    p = df["Price"]
    df["rtn"] = p.pct_change()
    df["log_r"] = np.log(p).diff()
    df["mmt"] = p.pct_change(4)
    df["vlt"] = df["log_r"].rolling(10).std()
    df["ma10"] = p.rolling(10).mean()
    df["ma50"] = p.rolling(50).mean()
    df["ma99"] = p.rolling(99).mean()
    df["ma_c"] = (df["ma10"] > df["ma99"]).astype(int)
    df["ma_s"] = (df["ma50"] - df["ma50"].shift(10)) / p
    df["priceZ"] = (p - df["ma50"]) / p.rolling(50).std()
    df["drawdown"] = p / p.rolling(252, min_periods=1).max() - 1
    df["rsi"] = wilder_rsi(p)
    df["Rel_Volume"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["kurt"] = df["rtn"].rolling(20).kurt()
    df["skew"] = df["rtn"].rolling(20).skew()
    df = df.replace([np.inf, -np.inf], np.nan)
    needed = sorted(set(HMM_FEATURES + RF_FEATURES + LABEL_CANDIDATES) - {"p_risk_on", "p_risk_on_delta"})
    return df.dropna(subset=needed).reset_index(drop=True)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Rolling HMM
# ---------------------------------------------------------------------------
import logging
import warnings
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore", module="hmmlearn")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def fit_hmm(X):
    best, best_ll = None, -np.inf
    for r in range(HMM_RESTARTS):
        model = GaussianHMM(n_components=HMM_STATES, covariance_type="diag", n_iter=500,
                            tol=1e-4, random_state=SEED + r)
        try:
            model.fit(X)
        except Exception:
            continue
        if not model.monitor_.converged:
            continue
        ll = model.score(X)
        if ll > best_ll:
            best, best_ll = model, ll
    return best


def risk_on_state(model, X, label_data):
    # The state with the lowest standardised volatility and drawdown depth is Risk-On
    states = model.predict(X)
    if len(np.unique(states)) < HMM_STATES:
        return None
    std = label_data.std(axis=0)
    std[std == 0] = 1.0
    z = (label_data - label_data.mean(axis=0)) / std
    scores = [-z[states == s].mean(axis=0).sum() for s in range(HMM_STATES)]
    return int(np.argmax(scores))


def run_hmm_block(start, end, hmm_raw, label_raw):
    # Fit on the window ending at `start` (inclusive), then filter each day up to `end`
    fit_slice = slice(start - HMM_WINDOW + 1, start + 1)
    scaler = StandardScaler().fit(hmm_raw[fit_slice])
    pca = PCA(n_components=PCA_COMPONENTS).fit(scaler.transform(hmm_raw[fit_slice]))
    X_fit = pca.transform(scaler.transform(hmm_raw[fit_slice]))

    model = fit_hmm(X_fit)
    if model is None:
        return []
    risk_on = risk_on_state(model, X_fit, label_raw[fit_slice])
    if risk_on is None:
        return []

    out = []
    for i in range(start, end):
        X_i = pca.transform(scaler.transform(hmm_raw[i - HMM_WINDOW + 1: i + 1]))
        # The last row of the forward-backward posterior is the filtered probability at day i
        out.append((i, model.predict_proba(X_i)[-1, risk_on]))
    return out


from joblib import Parallel, delayed


def add_hmm_regimes(df):
    hmm_raw = df[HMM_FEATURES].values
    label_raw = np.column_stack([df["vlt"].values, df["drawdown"].abs().values])
    n = len(df)
    starts = range(HMM_WINDOW - 1, n, HMM_REFIT_EVERY)
    blocks = Parallel(n_jobs=-1)(
        delayed(run_hmm_block)(s, min(s + HMM_REFIT_EVERY, n), hmm_raw, label_raw) for s in starts
    )

    p = np.full(n, np.nan)
    for block in blocks:
        for i, prob in block:
            p[i] = prob
    df["p_risk_on"] = p
    df["p_risk_on_delta"] = df["p_risk_on"].diff()
    df["regime"] = np.where(np.isnan(p), np.nan, (p >= 0.5).astype(float))
    df["target"] = df["regime"].shift(-HORIZON)
    return df


# ---------------------------------------------------------------------------
# Walk-forward Random Forest
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier


def make_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=20,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def prob_class_one(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def walk_forward_rf(df):
    rows = df.index[df[RF_FEATURES + ["regime"]].notna().all(axis=1)]
    df["p_pred"] = np.nan
    model = None
    for k in range(RF_BURN_IN, len(rows), RF_RETRAIN_EVERY):
        t = rows[k]
        # Only train on rows whose target was already observable at the close of day t
        train_rows = rows[:k]
        train_rows = train_rows[train_rows + HORIZON <= t]
        train = df.loc[train_rows].dropna(subset=["target"])
        if train["target"].nunique() < 2:
            continue
        model = make_rf().fit(train[RF_FEATURES], train["target"].astype(int))
        test_rows = rows[k: k + RF_RETRAIN_EVERY]
        df.loc[test_rows, "p_pred"] = prob_class_one(model, df.loc[test_rows, RF_FEATURES])
    df["pred"] = np.where(df["p_pred"].isna(), np.nan, (df["p_pred"] >= 0.5).astype(float))
    return df, model, rows


# ---------------------------------------------------------------------------
# Labelling-feature check: permutation importance for forward returns,
# using only data from before the first out-of-sample prediction
# ---------------------------------------------------------------------------
from sklearn.inspection import permutation_importance


def labelling_feature_importance(df, oos_start):
    fwd = df["Price"].shift(-FI_FORWARD_DAYS) / df["Price"] - 1
    pre = df.loc[: oos_start - 1 - FI_FORWARD_DAYS].copy()
    pre["fwd_up"] = (fwd.loc[pre.index] > 0).astype(int)

    split = int(len(pre) * 0.7)
    train = pre.iloc[: split - FI_FORWARD_DAYS]
    test = pre.iloc[split:]
    if len(train) < 100 or len(test) < 50 or train["fwd_up"].nunique() < 2:
        return None

    model = make_rf().fit(train[LABEL_CANDIDATES], train["fwd_up"])
    result = permutation_importance(model, test[LABEL_CANDIDATES], test["fwd_up"],
                                    scoring="balanced_accuracy", n_repeats=30,
                                    random_state=SEED, n_jobs=-1)
    return pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                        index=LABEL_CANDIDATES).sort_values("mean")


# ---------------------------------------------------------------------------
# Backtest and performance
# ---------------------------------------------------------------------------
def backtest(df):
    first = df["pred"].first_valid_index()
    signal = df["pred"].copy()
    signal.loc[first:] = signal.loc[first:].ffill()
    df["position"] = signal.shift(HORIZON)

    bt = df.dropna(subset=["position", "rtn"]).copy()
    turnover = bt["position"].diff().abs()
    turnover.iloc[0] = bt["position"].iloc[0]
    bt["strategy"] = bt["position"] * bt["rtn"] - TRANSACTION_COST * turnover
    bt["benchmark"] = bt["rtn"]
    return bt


def performance(returns, position=None):
    n = len(returns)
    equity = (1 + returns).cumprod()
    ann_ret = equity.iloc[-1] ** (252 / n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    max_dd = (equity / equity.cummax() - 1).min()
    invested = returns if position is None else returns[position > 0]
    return {
        "Total Return (%)": (equity.iloc[-1] - 1) * 100,
        "CAGR (%)": ann_ret * 100,
        "Annualised Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": returns.mean() / returns.std() * np.sqrt(252),
        "Max Drawdown (%)": max_dd * 100,
        "Calmar Ratio": ann_ret / abs(max_dd) if max_dd != 0 else np.nan,
        "Hit Rate When Invested (%)": (invested > 0).mean() * 100 if len(invested) else np.nan,
        "Time Invested (%)": 100.0 if position is None else (position > 0).mean() * 100,
    }


from sklearn.metrics import classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = build_features(load_prices(DATA_PATH))
    df = add_hmm_regimes(df)
    df, rf_model, rf_rows = walk_forward_rf(df)

    oos = df.dropna(subset=["pred", "target"])
    if oos.empty:
        raise RuntimeError("Not enough data for an out-of-sample period; use a longer price history.")
    y_true = oos["target"].astype(int)
    y_pred = oos["pred"].astype(int)
    y_persist = oos["regime"].astype(int)

    print(f"Out-of-sample period: {oos['Date'].iloc[0].date()} to {oos['Date'].iloc[-1].date()} "
          f"({len(oos)} days)\n")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=[REGIME_NAMES[0], REGIME_NAMES[1]], zero_division=0))
    print(f"{'':<34}{'Accuracy':>10}{'Balanced':>10}")
    print(f"{'Random Forest':<34}{(y_true == y_pred).mean():>10.2%}"
          f"{balanced_accuracy_score(y_true, y_pred):>10.2%}")
    print(f"{f'Persistence (regime t+{HORIZON} = t)':<34}{(y_true == y_persist).mean():>10.2%}"
          f"{balanced_accuracy_score(y_true, y_persist):>10.2%}")

    last = df.loc[rf_rows[-1]]
    p_last = last["p_pred"]
    if not np.isnan(p_last):
        label = REGIME_NAMES[int(p_last >= 0.5)]
        conf = max(p_last, 1 - p_last)
        print(f"\nRegime forecast for {HORIZON} trading days after {last['Date'].date()}: "
              f"{label.upper()} (probability {conf:.2%})")

    bt = backtest(df)
    print(f"\nBacktest: {bt['Date'].iloc[0].date()} to {bt['Date'].iloc[-1].date()}, "
          f"{TRANSACTION_COST:.2%} cost per unit turnover, fill at close t+{FILL_DELAY}")
    results = pd.DataFrame({
        "Strategy": performance(bt["strategy"], bt["position"]),
        "Buy & Hold": performance(bt["benchmark"]),
    })
    print(results.round(2).to_string())

    fi = labelling_feature_importance(df, oos.index[0])
    if fi is not None:
        print(f"\nPermutation importance for {FI_FORWARD_DAYS}-day forward return direction "
              f"(pre-out-of-sample data only, balanced accuracy drop):")
        print(fi.sort_values("mean", ascending=False).round(4).to_string())

    # Plots
    colors = {0: "red", 1: "green"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    hist = df.dropna(subset=["regime"])
    axes[0].plot(hist["Date"], hist["Price"], color="black", linewidth=1, label="Price")
    for r, c in colors.items():
        pts = hist[hist["regime"] == r]
        axes[0].scatter(pts["Date"], pts["Price"], color=c, s=8, label=REGIME_NAMES[r])
    axes[0].set_title("Filtered HMM Regimes")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")

    axes[1].plot(bt["Date"], (1 + bt["benchmark"]).cumprod(), color="gray", linestyle="--", label="Buy & Hold")
    axes[1].plot(bt["Date"], (1 + bt["strategy"]).cumprod(), color="blue", label="Strategy")
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_title("Out-of-Sample Strategy vs Buy & Hold (net of costs)")
    axes[1].set_ylabel("Growth of 1")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if fi is not None:
        ax1.barh(fi.index, fi["mean"], xerr=fi["std"], color="black")
        ax1.set_title(f"Permutation Importance ({FI_FORWARD_DAYS}d forward return, pre-OOS)")
        ax1.set_xlabel("Drop in balanced accuracy")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=[REGIME_NAMES[0], REGIME_NAMES[1]]).plot(
        ax=ax2, cmap=plt.cm.Blues, values_format="d")
    ax2.set_title("Out-of-Sample Confusion Matrix")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
