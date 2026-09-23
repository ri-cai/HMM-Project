import numpy as np
import pandas as pd

# Set data frame using csv from Investing.com/kaggle, then cleaned it by resetting index and dropping missing values
X = r"C:\Users\Hyper\Downloads\VS\Data\VUAA Historical Data.csv"
df = pd.read_csv(X)
df = df.reset_index(drop=True)

# Ensure all relevent column names are in place
if "Price" not in df.columns:
    if "Close" in df.columns:
        df["Price"] = df["Close"]
    elif "close" in df.columns:
        df["Price"] = df["close"]
if "Date" not in df.columns and "date" in df.columns:
    df["Date"] = df["date"]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
if "Vol." not in df.columns:
    df["Vol."] = df["volume"]

# LOOK-AHEAD FIX 1: enforce chronological order before any rolling, shift or cummax feature is built.
# Investing.com exports newest-first by default; left unsorted, every feature below runs backwards in time.
df = df.sort_values("Date").reset_index(drop=True)

# Features engineering for HMM and RF
df["rtn"] = df["Price"].pct_change(fill_method=None)
df["mmt"] = df["Price"] - df["Price"].shift(4)
df["ma10"] = df["Price"].rolling(window=10).mean()
df["ma99"] = df["Price"].rolling(window=99).mean()
df["ma_c"] = (df["ma10"] > df["ma99"]).astype(int)
df["log_r"] = np.log(df["Price"] / df["Price"].shift(1))
df["vlt"] = df["log_r"].rolling(window=10).std()
df['ma50'] = df['Price'].rolling(window=50).mean()
df['ma50_s'] = df['ma50'] - df['ma50'].shift(10)
df['ma_s'] = df['ma50_s'] / df['Price']
df["priceZ"] = (df["Price"] - df["Price"].rolling(50).mean()) / df["Price"].rolling(50).std()
df["drawdown"] = (df["Price"] / df["Price"].cummax()) - 1
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = np.where(loss == 0, np.inf, gain / loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['rsi'] = compute_rsi(df['Price'])
def convert_volume_str(volume_str):
    try:
        if isinstance(volume_str, str):
            volume_str = volume_str.strip()
            if volume_str.endswith('K'):
                return float(volume_str[:-1]) * 1_000
            elif volume_str.endswith('M'):
                return float(volume_str[:-1]) * 1_000_000
            elif volume_str.endswith('B'):
                return float(volume_str[:-1]) * 1_000_000_000
            else:
                return float(volume_str)  # no suffix
        return volume_str  # already numeric
    except Exception:
        return np.nan
df["Volume"] = df["Vol."].apply(convert_volume_str)
df["Volume_Rolling_Avg"] = df["Volume"].rolling(window=20).mean()
df["Rel_Volume"] = df["Volume"] / df["Volume_Rolling_Avg"]
df["pv_ratio"] = df["Price"] / df["Volume"]
df["kurt"] = df["Price"].rolling(window=10).kurt()
df["skew"] = df["Price"].rolling(window=10).skew()

# Define HMM features then clean all the data by removing Nan
from sklearn.preprocessing import StandardScaler
hmm_feature = ["rtn", "log_r", "vlt", "ma_s", "drawdown", "priceZ", "rsi", "mmt", "Rel_Volume"]
all_features = ["log_r", "rtn", "mmt", "ma_c", "log_r", "vlt", "ma_s", "priceZ", "drawdown", "rsi", "Volume", "Rel_Volume", "pv_ratio", "kurt", "skew"]
df = df.dropna(subset=all_features) 

# LOOK-AHEAD FIX 2: the scaler and PCA were previously fitted on the whole series, so every window's
# features were normalised using future data. They are now refitted inside each rolling window instead.
from sklearn.decomposition import PCA
hmm_raw = df[hmm_feature].values
n_components = 2

# Setting up the rolling HMM fitting
import random
random.seed(42)  # FIX 10: fixed seed so every run is reproducible and old vs new can be compared
random_seeds = random.sample(range(1, 100001), 15)
hmm_window = 504

# Begins the fitting in the initial window (Unsupervised Learning)
from hmmlearn.hmm import GaussianHMM
def compute_hmm_for_index(i):
    # Standardises HMM features to remove weighting bias, then compresses into principal components,
    # fitted on this window's past data only (imported locally, as StandardScaler is re-imported later in this function)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    raw_window = hmm_raw[i - hmm_window: i]
    scaler_w = StandardScaler().fit(raw_window)
    pca_w = PCA(n_components=3).fit(scaler_w.transform(raw_window))
    X_window = pca_w.transform(scaler_w.transform(raw_window))
    best_score = float('-inf')
    best_seed = None

    # Cross-validation (only on windows to ensure temporal order)
    n_folds = 4
    window_size = len(X_window)
    fold_size = window_size // (n_folds + 1)

    for seed in random_seeds:
        fold_scores = []

        for fold in range(n_folds):
            train_start = 0
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = val_start + fold_size

            if val_end > window_size:
                break

            X_train = X_window[train_start:train_end]
            X_val = X_window[val_start:val_end]

            try:
                model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=seed)
                model.fit(X_train)
                val_score = model.score(X_val)
                fold_scores.append(val_score)
            except Exception:
                continue

        if fold_scores:
            avg_val_score = np.mean(fold_scores)
            if avg_val_score > best_score:
                best_score = avg_val_score
                best_seed = seed

    if best_seed is None:
        return None

    # FIX 5: refit the chosen seed on the full window. Previously the model kept was whichever fold model
    # was fitted last: trained on only the first 80% of the window, or left over from another seed if a fit failed
    try:
        best_model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=best_seed)
        best_model.fit(X_window)
    except Exception:
        return None
    
    # Viterbi algo
    window_states = best_model.predict(X_window)
    unique_states = np.unique(window_states)
    if len(unique_states) < 2:
        return None

    # Pull features for labeling
    drawdown_window = df["drawdown"].iloc[i - hmm_window: i].values
    vlt_window = df["vlt"].iloc[i - hmm_window: i].values
    Rel_Volume_window = df["Rel_Volume"].iloc[i - hmm_window: i].values
    mmt_window = df["mmt"].iloc[i - hmm_window: i].values
    

    window_df = pd.DataFrame({
        "state": window_states,
        "drawdown": drawdown_window,
        "vlt": vlt_window,
        "Rel_Volume": Rel_Volume_window,
        "mmt": mmt_window
    })

    # Compute per-state averages
    state_metrics = []
    for state in np.unique(window_states):
        state_data = window_df[window_df["state"] == state]
        mean_vlt = state_data["vlt"].mean()
        mean_drawdown = state_data["drawdown"].mean()
        mean_Rel_Volume = state_data["Rel_Volume"].mean()
        mean_mmt = state_data["mmt"].mean()
        state_metrics.append([mean_vlt, abs(mean_drawdown), mean_Rel_Volume, mean_mmt])

    # FIX 7: the clustering step grouped two state vectors into two clusters, which always puts each state in its own
    # cluster, so it never changed the labels. Replaced with the identical direct comparison. This also removes a
    # crash when the HMM decodes only one state in a window: that day is now skipped (see the check above)
    inflation_scores = [m[0] - m[1] for m in state_metrics]
    inflationary_state = unique_states[int(np.argmax(inflation_scores))]
    regime_names = {
        state: "Inflationary" if state == inflationary_state else "Deflationary"
        for state in unique_states
    }

    # Determine current state
    full_state_path = best_model.predict(X_window)
    state_today = pd.Series(full_state_path[-5:]).mode()[0]
    x_today = pca_w.transform(scaler_w.transform(hmm_raw[i].reshape(1, -1)))
    regime_label_today = regime_names.get(state_today, "Unknown")

    # FIX 6: state probability from the forward pass over the window plus today. Previously it was computed on
    # today's observation alone, which ignores the transition matrix and all history
    state_prob_today = best_model.predict_proba(np.vstack([X_window, x_today]))[-1]

    # FIX 8: HMM state numbers are arbitrary and can swap between refits, so a raw state number meant different
    # things on different days. Features are now relative to the Inflationary state: H_S = 1 if today's state is
    # Inflationary, state_0_prob = probability of the Inflationary state, state_1_prob = of the Deflationary state
    infl_idx = int(inflationary_state)
    result_entry = {
        'original_index': df.index[i],
        'H_S': int(state_today == inflationary_state),
        'Regime_Label': regime_label_today,
        'Chosen_Seed': best_seed
    }
    result_entry['state_0_prob'] = state_prob_today[infl_idx]
    result_entry['state_1_prob'] = state_prob_today[1 - infl_idx]

    return result_entry

# Runs the rolling in parallel for every window and cleans out any empty sets using the model from the initial window
from joblib import Parallel, delayed
hmm_results_parallel = Parallel(n_jobs=2, backend='threading')(delayed(compute_hmm_for_index)(i) for i in range(hmm_window, len(df)))
hmm_results = [res for res in hmm_results_parallel if res is not None]

# Converts the list into df, then merges and aligns the data into original dataset
hmm_results_df = pd.DataFrame(hmm_results)
hmm_results_df.set_index('original_index', inplace=True)

# Slices df to begin after initial window, then joins the results to df
df_rolling = df.iloc[hmm_window:].copy()
df_rolling = df_rolling.join(hmm_results_df)

# Computes the change in state probabilities, then adding the values to df and cleans
for comp in range(n_components):
    prob_col = f'state_{comp}_prob'
    delta_col = f'{prob_col}_delta'
    df_rolling[delta_col] = df_rolling[prob_col].diff()
delta_cols = [f'state_{i}_prob_delta' for i in range(n_components)]
df_rolling = df_rolling.dropna(subset=delta_cols)

# Shifts data to set up a target variable (tomorrows regime)
df_rolling["Next_Regime"] = df_rolling["Regime_Label"].shift(-1)
df_rolling = df_rolling.dropna(subset=["Next_Regime"])

# Defines features for RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
rf_features = ["H_S", "ma_c", "mmt", "rtn", "rsi", "ma_s", "drawdown", "priceZ", "vlt", "Rel_Volume"] + [f"state_{i}_prob_delta" for i in range(n_components)]
rf_features_to_scale = [feat for feat in rf_features if feat != "H_S"]

# Seeds to try for each retrain
rf_seeds = random.sample(range(1, 100001), 15)
# LOOK-AHEAD FIX 3: fixed burn-in instead of 20% of the total sample length. A fraction of the full length
# means the first prediction date and every retrain date depended on how much future data existed.
rf_burn_in = 252
start_idx = rf_burn_in
rolling_predictions = []
rolling_true = df_rolling["Next_Regime"].iloc[start_idx:len(df_rolling) - 1].tolist()

# Sets initial model as none so it can identify a best model in fitting
rfmodel = None
last_trained_day = None
best_rf_seed = None
best_rf_score = None

for current_day in range(start_idx, len(df_rolling) - 1):
    if (last_trained_day is None) or (current_day - last_trained_day >= 126):
        train_x = df_rolling[rf_features].iloc[:current_day].copy()
        train_y = df_rolling["Next_Regime"].iloc[:current_day]

        valid_idx = train_x[rf_features_to_scale].dropna().index.intersection(train_y.dropna().index)
        train_x = train_x.loc[valid_idx]
        train_y = train_y.loc[valid_idx]

        if len(train_x) == 0:
            rolling_predictions.append(None)
            continue

        scaler = StandardScaler()
        train_x[rf_features_to_scale] = scaler.fit_transform(train_x[rf_features_to_scale])

        # FIX 9: seed chosen on a chronological validation split (the most recent 20% of the training data), then
        # refitted on all of it. Previously it was chosen on training accuracy, which rewards memorising the data
        n_val = max(1, int(len(train_x) * 0.2))
        fit_x, val_x = train_x.iloc[:-n_val], train_x.iloc[-n_val:]
        fit_y, val_y = train_y.iloc[:-n_val], train_y.iloc[-n_val:]

        best_score = -float("inf")
        best_seed_local = None

        for seed in rf_seeds:
            try:
                rf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=10, min_samples_leaf=5, min_samples_split=15, n_jobs=2)
                rf.fit(fit_x, fit_y)
                score = rf.score(val_x, val_y)

                if score > best_score:
                    best_score = score
                    best_seed_local = seed

            except Exception:
                continue

        if best_seed_local is None:
            rfmodel = None
        else:
            rfmodel = RandomForestClassifier(random_state=best_seed_local, n_estimators=200, max_depth=10, min_samples_leaf=5, min_samples_split=15, n_jobs=2)
            rfmodel.fit(train_x, train_y)
        best_rf_seed = best_seed_local
        best_rf_score = best_score
        last_trained_day = current_day

    test_x = df_rolling[rf_features].iloc[current_day:current_day + 1].copy()

    if test_x[rf_features_to_scale].isna().any().any() or rfmodel is None:
        rolling_predictions.append(None)
        continue

    test_x[rf_features_to_scale] = scaler.transform(test_x[rf_features_to_scale])
    pred = rfmodel.predict(test_x)[0]
    rolling_predictions.append(pred)

# Final classification report
from sklearn.metrics import classification_report
filtered_pairs = [(t, p) for t, p in zip(rolling_true, rolling_predictions) if p is not None]
if filtered_pairs:
    filtered_true, filtered_predictions = zip(*filtered_pairs)
    print(classification_report(filtered_true, filtered_predictions))

# Naive benchmark for the accuracy above: predict that tomorrow's regime equals today's. The regime label is very
# persistent, so this baseline is high, and model accuracy only means something relative to it
persist_labels = df_rolling["Regime_Label"].iloc[start_idx:len(df_rolling) - 1].tolist()
persist_pairs = [(t, q) for t, q, p in zip(rolling_true, persist_labels, rolling_predictions) if p is not None]
if persist_pairs:
    rf_acc = np.mean([t == p for t, p in filtered_pairs])
    persist_acc = np.mean([t == q for t, q in persist_pairs])
    print(f"RF accuracy: {rf_acc:.2%} | Persistence baseline (tomorrow = today): {persist_acc:.2%}")

# Final best RF info
print(f"Seed used by the final selected RF model: {best_rf_seed} | Validation Accuracy: {best_rf_score:.2%}")

# Adds rolling predictions to df
df_rolling.loc[df_rolling.index[start_idx:-1], "Rolling_Predicted_Regime"] = rolling_predictions

# Cleans and aligns data ready for predictions
latest_features = df_rolling[rf_features].iloc[-1:].copy()
latest_features[rf_features_to_scale] = scaler.transform(latest_features[rf_features_to_scale])

# Predicts regime, computes confidence probability and prints seeds used 
probs = rfmodel.predict_proba(latest_features)[0]
predicted_index = np.argmax(probs)
latest_seed = int(df_rolling["Chosen_Seed"].iloc[-1])
print(f"Seed used by the final selected HMM model: {latest_seed}")
next_regime = rfmodel.classes_[predicted_index]
confidence = probs[predicted_index]
print(f"Predicted next regime: {next_regime.upper()} with confidence: {confidence:.2%}")

# Sets positions based off regime prediction
df_rolling["Position"] = df_rolling["Rolling_Predicted_Regime"].map({
    "Inflationary": 1,
    "Deflationary": 0,
})

# Defines costs
transaction_cost = 0.001

# LOOK-AHEAD FIX 4: a signal computed from day t's close is filled at day t+1's close, so it first earns day t+2's return.
# Previously it earned day t+1's return, which assumed filling at the same close the signal was computed from.
df_rolling["Held_Position"] = df_rolling["Position"].shift(2)

# Calculate position changes from previous day
df_rolling["Prev_Position"] = df_rolling["Held_Position"].shift(1).fillna(0)

# Costs implimented, charged on the day the held position actually changes
df_rolling["Transaction_Cost"] = np.where(
    ((df_rolling["Prev_Position"] == 1) & (df_rolling["Held_Position"] == 0)) |
    ((df_rolling["Prev_Position"] == 0) & (df_rolling["Held_Position"] == 1)),
    transaction_cost,
    0.0
)

# Compute Strategy Return
df_rolling["Strategy_Return"] = (
    df_rolling["Held_Position"] * df_rolling["rtn"]
    - df_rolling["Transaction_Cost"]
)

# Compute cumulative strategy
df_rolling["Cumulative_Strategy"] = (1 + df_rolling["Strategy_Return"]).cumprod()

# Compute cumulative benchmark (buy and hold)
df_rolling["Cumulative_Benchmark"] = (1 + df_rolling["rtn"]).cumprod()

# Find the first valid strategy return (HMM start)
strategy_start_idx = df_rolling["Strategy_Return"].first_valid_index()

# Reset benchmark to start from the same point
if strategy_start_idx is not None:
    df_rolling.loc[:strategy_start_idx, "Cumulative_Benchmark"] = np.nan
    df_rolling["Cumulative_Benchmark"] /= df_rolling["Cumulative_Benchmark"].dropna().iloc[0]

# Scale strategy as well
df_rolling["Cumulative_Strategy"] /= df_rolling["Cumulative_Strategy"].dropna().iloc[0]


def evaluate_strategy(df, strategy_col="Strategy_Return", benchmark_col="rtn", risk_free_rate=0.00):
    def compute_metrics(returns, cumulative_col):
        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
        final_cumulative = df[cumulative_col].iloc[-1]
        cagr = final_cumulative ** (365.25 / days) - 1

        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        hit_rate = (returns > 0).mean()

        return {
            "Total Return (%)": cumulative_return * 100,
            "Annualized Return (%)": annualized_return * 100,
            "Annualized Volatility (%)": annualized_volatility * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown * 100,
            "CAGR (%)": cagr * 100,
            "Calmar Ratio": calmar_ratio,
            "Hit Rate (%)": hit_rate * 100
        }

    df = df.dropna(subset=[strategy_col, benchmark_col])
    strategy_returns = df[strategy_col]
    benchmark_returns = df[benchmark_col]

    strategy_metrics = compute_metrics(strategy_returns, "Cumulative_Strategy")
    benchmark_metrics = compute_metrics(benchmark_returns, "Cumulative_Benchmark")

    return {
        "Strategy": strategy_metrics,
        "Buy_and_Hold": benchmark_metrics
    }

# Run evaluation
performance = evaluate_strategy(df_rolling)

# Display nicely
print("\n Strategy Performance Metrics:")
for section_name, metrics_dict in performance.items():
    print(f"\n{section_name} Metrics:")
    for metric_name, value in metrics_dict.items():
        print(f"  {metric_name:<30}: {value:.2f}")

# Data Visualization
import matplotlib.pyplot as plt

# Regime and Backtest plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# Regime Overlay (actual regimes)
axes[0].plot(df_rolling["Date"], df_rolling["Price"], color="black", linewidth=1, label="Price")
colors = {"Deflationary": "red", "Inflationary": "green"}
for regime in colors:
    regime_data = df_rolling[df_rolling["Regime_Label"] == regime]
    axes[0].scatter(regime_data["Date"], regime_data["Price"], color=colors[regime], label=regime.capitalize(), s=20)

axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].set_title("Regime Overlay")

# Strategy vs Benchmark
axes[1].plot(df_rolling["Date"], df_rolling["Cumulative_Benchmark"], label="Benchmark", color="gray", linestyle="--")
axes[1].plot(df_rolling["Date"], df_rolling["Cumulative_Strategy"], label="Strategy", color="blue")
axes[1].axhline(y=1, color="black", linewidth=2)
axes[1].set_ylabel("Cumulative Return")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].set_title("Strategy vs Benchmark")
axes[1].grid(True)
axes[1].xaxis.set_ticks([])

# Filter rows with predicted regimes
predicted_regimes = df_rolling.dropna(subset=["Rolling_Predicted_Regime"])

# Plot predicted regime markers on benchmark 
for regime, color in colors.items():
    regime_points = predicted_regimes[predicted_regimes["Rolling_Predicted_Regime"] == regime]
    axes[1].scatter(
        regime_points["Date"],
        regime_points["Cumulative_Benchmark"],
        color=color,
        label=f"Predicted {regime}",
        s=30,
        alpha=0.6
    )

# Adjust legend to include predicted regimes
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, loc="upper left")

# Feature Importances + Confusion Matrix plot
fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# RF Feature Importances
importances = rfmodel.feature_importances_
feature_names = train_x.columns
ax1.barh(feature_names, importances, color='black')
ax1.set_title("RF Feature Importances")
ax1.set_xlabel("Importance")
ax1.set_ylabel("Feature")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
if filtered_pairs:
    cm = confusion_matrix(filtered_true, filtered_predictions, labels=["Deflationary", "Inflationary"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Deflationary", "Inflationary"])
    disp.plot(ax=ax2, cmap=plt.cm.Blues, values_format='d')
    ax2.set_title("Confusion Matrix")

plt.tight_layout()
plt.show()
