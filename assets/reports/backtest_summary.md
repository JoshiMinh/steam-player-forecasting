| Rank | Model | Paradigm | Mean_MASE | CI_Low | CI_High | Std_MASE | Median_MASE | Raw_Mean_MASE | Diverged_Units | Fold0_Mean_MASE | Mean_Seed_Std | Mean_Rank | Units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Holt-Winters | Exponential Smoothing | 0.546 | 0.471 | 0.618 | 0.195 | 0.550 | 0.546 | 0 | 0.526 |  | 1.593 | 27 |
| 2 | SARIMA | Econometric | 0.801 | 0.560 | 1.185 | 0.884 | 0.605 | 144431566.854 | 1 | 0.672 |  | 3.259 | 27 |
| 3 | Ridge | Linear ML | 0.857 | 0.602 | 1.243 | 0.898 | 0.704 | 1.217 | 1 | 0.562 |  | 3.667 | 27 |
| 4 | GRU | Recurrent DL | 0.860 | 0.746 | 0.974 | 0.307 | 0.844 | 0.860 | 0 | 0.735 | 0.095 | 5.741 | 27 |
| 5 | Random Forest | Tree Ensemble ML | 0.918 | 0.831 | 1.006 | 0.235 | 0.882 | 0.918 | 0 | 0.832 |  | 6.407 | 27 |
| 6 | LSTM | Recurrent DL | 0.922 | 0.824 | 1.025 | 0.273 | 0.898 | 0.922 | 0 | 0.834 | 0.077 | 6.148 | 27 |
| 7 | XGBoost | Gradient Boosted ML | 0.939 | 0.851 | 1.030 | 0.239 | 0.931 | 0.939 | 0 | 0.923 |  | 6.889 | 27 |
| 8 | RNN | Recurrent DL Baseline | 0.949 | 0.859 | 1.037 | 0.239 | 0.971 | 0.949 | 0 | 0.913 | 0.093 | 6.963 | 27 |
| 9 | Seasonal Naive | Seasonal Baseline | 1.036 | 0.974 | 1.101 | 0.173 | 1.051 | 1.036 | 0 | 1.119 |  | 8.185 | 27 |
| 10 | ARIMA(1,1,1) | Econometric Baseline | 1.116 | 0.979 | 1.295 | 0.427 | 1.024 | 1.116 | 0 | 1.294 |  | 8.481 | 27 |
| 11 | Naive | Naive Baseline | 1.242 | 1.010 | 1.521 | 0.694 | 1.002 | 1.242 | 0 | 1.357 |  | 8.667 | 27 |
