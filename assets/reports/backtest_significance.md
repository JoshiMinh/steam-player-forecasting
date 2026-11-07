| Model_A | Model_B | Units | Mean_MASE_Diff | CI_Low | CI_High | Boot_P | Bootstrap_Significant | DM_Games_Tested | DM_Games_A_Better | DM_Games_Significant | DM_Median_P | Involves_Tuned_Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Holt-Winters | SARIMA | 27 | -0.111 | -0.175 | -0.052 | 0.001 | True | 7 | 7 | 0 | 0.148 | True |
| Holt-Winters | Ridge | 27 | -0.159 | -0.251 | -0.082 | 0.000 | True | 7 | 6 | 1 | 0.098 | False |
| Holt-Winters | GRU | 27 | -0.314 | -0.388 | -0.248 | 0.000 | True | 7 | 7 | 6 | 0.002 | False |
| Holt-Winters | Random Forest | 27 | -0.371 | -0.444 | -0.302 | 0.000 | True | 7 | 7 | 7 | 0.004 | False |
| Holt-Winters | LSTM | 27 | -0.376 | -0.448 | -0.310 | 0.000 | True | 7 | 7 | 5 | 0.001 | False |
| Holt-Winters | XGBoost | 27 | -0.392 | -0.473 | -0.317 | 0.000 | True | 7 | 7 | 7 | 0.009 | True |
| Holt-Winters | RNN | 27 | -0.403 | -0.463 | -0.346 | 0.000 | True | 7 | 7 | 7 | 0.001 | False |
| Holt-Winters | Seasonal Naive | 27 | -0.489 | -0.569 | -0.409 | 0.000 | True | 7 | 7 | 6 | 0.000 | False |
| Holt-Winters | ARIMA(1,1,1) | 27 | -0.570 | -0.719 | -0.456 | 0.000 | True | 7 | 7 | 6 | 0.000 | False |
| Holt-Winters | Naive | 27 | -0.696 | -0.934 | -0.502 | 0.000 | True | 7 | 7 | 7 | 0.007 | False |
| GRU | LSTM | 27 | -0.062 | -0.110 | -0.014 | 0.012 | True | 7 | 5 | 3 | 0.511 | False |
| GRU | SARIMA | 27 | 0.203 | 0.111 | 0.303 | 0.000 | True | 7 | 0 | 2 | 0.208 | True |
| LSTM | SARIMA | 27 | 0.265 | 0.178 | 0.361 | 0.000 | True | 7 | 0 | 4 | 0.019 | True |
| XGBoost | SARIMA | 27 | 0.282 | 0.203 | 0.366 | 0.000 | True | 7 | 0 | 4 | 0.008 | True |
| Ridge | SARIMA | 27 | 0.048 | -0.049 | 0.153 | 0.374 | False | 7 | 4 | 0 | 0.575 | True |
