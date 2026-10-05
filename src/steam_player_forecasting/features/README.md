# Features & Time-Series Preprocessing Module

The `steam_player_forecasting.features` package provides modular, leakage-free preprocessing, transformations, scalers, and statistical diagnostic utilities designed specifically for time-series forecasting.

---

## Architecture & Principles

### Strict Anti-Leakage Protocol
In temporal forecasting, data leakage occurs whenever information from the future (validation or test windows) contaminates feature creation, scaling, or transformations.

1. **Chronological Splitting First:** The dataset is split chronologically into Train, Validation, and Test partitions *before* calculating any distributional statistics.
2. **Train-Only Fitting:** Scalers (`TimeSeriesScaler`) and variance-stabilizing transforms (`BoxCoxTransformer`, `LogTransformer`, `DifferencingTransformer`) fit parameters ($\mu, \sigma, \min, \max, \lambda$, offsets, initial boundary conditions) **strictly on the training window**.
3. **Out-of-Sample Transformation & Inversion:** Validation and test subsets are transformed using the fitted training parameters, and can be inverted back to their original physical level units without numerical degradation.

---

## Module Overview

| File | Primary Classes / Functions | Description |
| :--- | :--- | :--- |
| `split.py` | `train_val_test_split()`, `split_by_game()`, `expanding_window_cv()`, `TimeSeriesSplitResult` | Chronological dataset partitioning without future lookahead |
| `transforms.py` | `LogTransformer`, `BoxCoxTransformer`, `DifferencingTransformer`, `difference_series()`, `invert_difference_series()` | Variance stabilization and differencing with stateful inversion |
| `scalers.py` | `TimeSeriesScaler` (`MinMaxScaler`, `StandardScaler`, `RobustScaler`) | Train-only fitted feature scalers with 1D/2D and Series/DataFrame support |
| `diagnostics.py` | `check_stationarity()`, `compute_acf_pacf()`, `decompose_time_series()`, `compute_rolling_stats()` | STL decomposition, ADF/KPSS unit-root tests, and autocorrelation |

---

## Usage Examples

### 1. Chronological Train / Validation / Test Splitting

```python
from steam_player_forecasting.data import load_processed_data
from steam_player_forecasting.features import train_val_test_split

df = load_processed_data(game_name="Counter-Strike: Global Offensive")

# Hold out last 12 months for test, preceding 12 months for validation
res = train_val_test_split(df, val_months=12, test_months=12, date_col="Month_Year")

print(f"Train size: {len(res.train)} months")
print(f"Validation size: {len(res.val)} months")
print(f"Test size: {len(res.test)} months")
print("Split dates:", res.split_dates)
```

### 2. Leakage-Free Feature Scaling

```python
from steam_player_forecasting.features import TimeSeriesScaler

# Instantiate scaler
scaler = TimeSeriesScaler(scaler_type="minmax")

# Fit STRICTLY on training target
scaler.fit(res.train["Avg_players"])

# Transform train, val, and test partitions
train_scaled = scaler.transform(res.train["Avg_players"])
val_scaled = scaler.transform(res.val["Avg_players"])
test_scaled = scaler.transform(res.test["Avg_players"])

# Exact inverse transformation of model forecasts
y_pred_original = scaler.inverse_transform(test_scaled)
```

### 3. Box-Cox Power Transformation

```python
from steam_player_forecasting.features import BoxCoxTransformer

transformer = BoxCoxTransformer()
# Lambda is estimated by maximum likelihood strictly on train
train_bc = transformer.fit_transform(res.train["Avg_players"])
print(f"Estimated lambda: {transformer.lambda_:.4f}")

# Transform validation or test
val_bc = transformer.transform(res.val["Avg_players"])

# Invert back to level scale
val_level = transformer.inverse_transform(val_bc)
```

### 4. Stationarity Testing & Decomposition

```python
from steam_player_forecasting.features import check_stationarity, decompose_time_series

# Joint ADF and KPSS tests
diag = check_stationarity(res.train["Avg_players"])
print("Stationarity conclusion:", diag["conclusion"])
print(f"ADF p-value: {diag['adf']['p_value']:.4f}")
print(f"KPSS p-value: {diag['kpss']['p_value']:.4f}")

# STL Decomposition (Trend, Seasonal, Residual)
decomp = decompose_time_series(res.train.set_index("Month_Year")["Avg_players"], period=12, method="stl")
```

---

## Unit Testing

Run the feature and transformation test suite:

```bash
pytest tests/test_features.py -v
```
