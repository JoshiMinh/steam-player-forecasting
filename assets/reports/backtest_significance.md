| Model_A | Model_B | Units | Mean_MASE_Diff | CI_Low | CI_High | Boot_P | Bootstrap_Significant | DM_Games_Tested | DM_Games_A_Better | DM_Games_Significant | DM_Median_P | Involves_Tuned_Model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Holt-Winters | SARIMA | 27 | -0.255 | -0.595 | -0.055 | 0.001 | True | 7 | 7 | 0 | 0.251 | True |
| Holt-Winters | Ridge | 27 | -0.311 | -0.668 | -0.093 | 0.000 | True | 7 | 6 | 0 | 0.158 | False |
| Holt-Winters | GRU | 27 | -0.314 | -0.388 | -0.248 | 0.000 | True | 7 | 7 | 6 | 0.002 | False |
| Holt-Winters | Random Forest | 27 | -0.371 | -0.444 | -0.302 | 0.000 | True | 7 | 7 | 7 | 0.004 | False |
| Holt-Winters | LSTM | 27 | -0.376 | -0.448 | -0.310 | 0.000 | True | 7 | 7 | 5 | 0.001 | False |
| Holt-Winters | XGBoost | 27 | -0.392 | -0.473 | -0.317 | 0.000 | True | 7 | 7 | 7 | 0.009 | True |
| Holt-Winters | RNN | 27 | -0.403 | -0.463 | -0.346 | 0.000 | True | 7 | 7 | 7 | 0.001 | False |
| Holt-Winters | Seasonal Naive | 27 | -0.489 | -0.569 | -0.409 | 0.000 | True | 7 | 7 | 6 | 0.000 | False |
| Holt-Winters | ARIMA(1,1,1) | 27 | -0.570 | -0.719 | -0.456 | 0.000 | True | 7 | 7 | 6 | 0.000 | False |
| Holt-Winters | Naive | 27 | -0.696 | -0.934 | -0.502 | 0.000 | True | 7 | 7 | 7 | 0.007 | False |
| GRU | LSTM | 27 | -0.062 | -0.110 | -0.014 | 0.012 | True | 7 | 5 | 3 | 0.511 | False |
| GRU | SARIMA | 27 | 0.060 | -0.302 | 0.289 | 0.598 | False | 7 | 1 | 2 | 0.208 | True |
| LSTM | SARIMA | 27 | 0.121 | -0.228 | 0.344 | 0.441 | False | 7 | 1 | 3 | 0.231 | True |
| XGBoost | SARIMA | 27 | 0.138 | -0.204 | 0.350 | 0.395 | False | 7 | 1 | 4 | 0.008 | True |
| Ridge | SARIMA | 27 | 0.056 | -0.387 | 0.491 | 0.764 | False | 7 | 3 | 0 | 0.327 | True |
