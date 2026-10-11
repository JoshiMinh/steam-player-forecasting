# Validation vs Holdout Test Benchmark Comparison

Comparison of error metrics between Validation (2019-09 to 2020-08) and Test (2020-09 to 2021-08).

| Game | Model | Paradigm | MAE_Val | MAE_Test | MAPE_Val | MAPE_Test | MASE_Val | MASE_Test | MAE_Delta_Pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Counter-Strike: Global Offensive | Ridge | Linear ML | 10586.72 | 13289.18 | 1.82 | 2.03 | 0.195 | 0.249 | 25.5 |
| Counter-Strike: Global Offensive | Holt-Winters | Exponential Smoothing | 11017.69 | 14482.81 | 1.89 | 2.21 | 0.203 | 0.271 | 31.5 |
| Counter-Strike: Global Offensive | GRU | Recurrent DL | 23691.6 | 24360.55 | 4.01 | 3.67 | 0.437 | 0.456 | 2.8 |
| Counter-Strike: Global Offensive | RNN | Recurrent DL Baseline | 21885.32 | 25056.04 | 3.63 | 3.78 | 0.404 | 0.469 | 14.5 |
| Counter-Strike: Global Offensive | LSTM | Recurrent DL | 29835.28 | 31094.75 | 4.94 | 4.68 | 0.551 | 0.582 | 4.2 |
| Counter-Strike: Global Offensive | SARIMA | Econometric | 7968.89 | 31473.44 | 1.37 | 4.85 | 0.147 | 0.589 | 295.0 |
| Counter-Strike: Global Offensive | XGBoost | Gradient Boosted ML | 32862.5 | 36494.15 | 5.45 | 5.56 | 0.606 | 0.683 | 11.1 |
| Counter-Strike: Global Offensive | Random Forest | Tree Ensemble ML | 31035.45 | 41250.25 | 5.15 | 6.28 | 0.573 | 0.772 | 32.9 |
| Counter-Strike: Global Offensive | ARIMA(1,1,1) | Econometric Baseline | 34706.96 | 53074.28 | 5.76 | 8.07 | 0.641 | 0.993 | 52.9 |
| Counter-Strike: Global Offensive | Naive | Naive Baseline | 31958.62 | 53528.18 | 5.3 | 8.15 | 0.59 | 1.002 | 67.5 |
| Counter-Strike: Global Offensive | Seasonal Naive | Seasonal Baseline | 48822.78 | 57002.04 | 8.32 | 8.85 | 0.901 | 1.067 | 16.8 |
| Dota 2 | Ridge | Linear ML | 32877.2 | 19200.79 | 7.96 | 4.34 | 1.172 | 0.664 | -41.6 |
| Dota 2 | Holt-Winters | Exponential Smoothing | 13924.28 | 19894.46 | 3.32 | 4.42 | 0.496 | 0.688 | 42.9 |
| Dota 2 | SARIMA | Econometric | 13004.39 | 23463.86 | 3.14 | 5.31 | 0.463 | 0.811 | 80.4 |
| Dota 2 | Random Forest | Tree Ensemble ML | 33490.1 | 23442.29 | 7.82 | 5.14 | 1.193 | 0.811 | -30.0 |
| Dota 2 | XGBoost | Gradient Boosted ML | 31112.35 | 25401.84 | 7.41 | 5.63 | 1.109 | 0.878 | -18.4 |
| Dota 2 | LSTM | Recurrent DL | 36388.96 | 26687.03 | 8.36 | 5.92 | 1.297 | 0.923 | -26.7 |
| Dota 2 | RNN | Recurrent DL Baseline | 37929.65 | 28420.21 | 9.09 | 6.28 | 1.352 | 0.983 | -25.1 |
| Dota 2 | Seasonal Naive | Seasonal Baseline | 34171.44 | 30714.52 | 8.25 | 6.87 | 1.218 | 1.062 | -10.1 |
| Dota 2 | GRU | Recurrent DL | 41033.72 | 35996.43 | 9.76 | 8.16 | 1.462 | 1.245 | -12.3 |
| Dota 2 | ARIMA(1,1,1) | Econometric Baseline | 40041.18 | 37268.3 | 9.22 | 8.08 | 1.427 | 1.289 | -6.9 |
| Dota 2 | Naive | Naive Baseline | 42990.3 | 51195.49 | 9.91 | 11.19 | 1.532 | 1.77 | 19.1 |
| Grand Theft Auto V | SARIMA | Econometric | 6223.54 | 5565.8 | 5.51 | 4.5 | 0.477 | 0.444 | -10.6 |
| Grand Theft Auto V | Holt-Winters | Exponential Smoothing | 5181.69 | 5781.39 | 4.87 | 4.89 | 0.397 | 0.461 | 11.6 |
| Grand Theft Auto V | Ridge | Linear ML | 7073.77 | 6097.83 | 6.91 | 5.2 | 0.542 | 0.486 | -13.8 |
| Grand Theft Auto V | Random Forest | Tree Ensemble ML | 11294.43 | 6964.79 | 9.98 | 5.6 | 0.866 | 0.556 | -38.3 |
| Grand Theft Auto V | LSTM | Recurrent DL | 11575.18 | 7295.66 | 10.22 | 5.98 | 0.887 | 0.582 | -37.0 |
| Grand Theft Auto V | RNN | Recurrent DL Baseline | 14850.72 | 8006.35 | 13.17 | 6.55 | 1.139 | 0.639 | -46.1 |
| Grand Theft Auto V | GRU | Recurrent DL | 12800.46 | 8269.64 | 11.22 | 6.85 | 0.981 | 0.66 | -35.4 |
| Grand Theft Auto V | XGBoost | Gradient Boosted ML | 11892.76 | 9469.62 | 10.49 | 7.61 | 0.912 | 0.755 | -20.4 |
| Grand Theft Auto V | Naive | Naive Baseline | 10772.79 | 9677.18 | 9.55 | 8.58 | 0.826 | 0.772 | -10.2 |
| Grand Theft Auto V | ARIMA(1,1,1) | Econometric Baseline | 11930.79 | 10878.22 | 10.53 | 9.02 | 0.915 | 0.868 | -8.8 |
| Grand Theft Auto V | Seasonal Naive | Seasonal Baseline | 10796.86 | 11339.65 | 9.65 | 9.53 | 0.828 | 0.905 | 5.0 |
| Rust | Holt-Winters | Exponential Smoothing | 3313.65 | 3725.29 | 4.4 | 4.32 | 0.326 | 0.377 | 12.4 |
| Rust | SARIMA | Econometric | 4008.22 | 3888.24 | 5.35 | 4.46 | 0.394 | 0.394 | -3.0 |
| Rust | Naive | Naive Baseline | 6666.16 | 4560.03 | 8.24 | 5.02 | 0.656 | 0.462 | -31.6 |
| Rust | Ridge | Linear ML | 4549.14 | 4943.53 | 6.01 | 5.6 | 0.448 | 0.501 | 8.7 |
| Rust | GRU | Recurrent DL | 5049.29 | 7030.84 | 6.33 | 7.74 | 0.497 | 0.712 | 39.2 |
| Rust | Random Forest | Tree Ensemble ML | 5096.55 | 7302.51 | 6.4 | 8.05 | 0.501 | 0.74 | 43.3 |
| Rust | ARIMA(1,1,1) | Econometric Baseline | 6855.24 | 7910.9 | 8.46 | 8.7 | 0.674 | 0.801 | 15.4 |
| Rust | LSTM | Recurrent DL | 5631.23 | 8557.04 | 6.85 | 9.48 | 0.554 | 0.867 | 52.0 |
| Rust | RNN | Recurrent DL Baseline | 4723.01 | 9092.99 | 5.9 | 10.17 | 0.465 | 0.921 | 92.5 |
| Rust | XGBoost | Gradient Boosted ML | 5238.96 | 10062.14 | 6.59 | 11.11 | 0.515 | 1.019 | 92.1 |
| Rust | Seasonal Naive | Seasonal Baseline | 8476.61 | 11205.82 | 10.9 | 12.67 | 0.834 | 1.135 | 32.2 |
| Team Fortress 2 | Holt-Winters | Exponential Smoothing | 3240.82 | 3410.62 | 6.79 | 5.96 | 0.769 | 0.796 | 5.2 |
| Team Fortress 2 | LSTM | Recurrent DL | 6521.0 | 3929.47 | 13.0 | 6.82 | 1.547 | 0.917 | -39.7 |
| Team Fortress 2 | RNN | Recurrent DL Baseline | 6061.84 | 4126.19 | 12.02 | 7.33 | 1.438 | 0.963 | -31.9 |
| Team Fortress 2 | Ridge | Linear ML | 3096.93 | 4435.55 | 6.57 | 7.77 | 0.735 | 1.035 | 43.2 |
| Team Fortress 2 | SARIMA | Econometric | 3561.85 | 4440.17 | 7.61 | 7.91 | 0.845 | 1.037 | 24.7 |
| Team Fortress 2 | GRU | Recurrent DL | 7338.68 | 4532.56 | 14.24 | 8.06 | 1.741 | 1.058 | -38.2 |
| Team Fortress 2 | Random Forest | Tree Ensemble ML | 4216.45 | 5191.53 | 8.37 | 8.95 | 1.0 | 1.212 | 23.1 |
| Team Fortress 2 | XGBoost | Gradient Boosted ML | 3628.22 | 5284.54 | 7.49 | 9.09 | 0.861 | 1.234 | 45.7 |
| Team Fortress 2 | Seasonal Naive | Seasonal Baseline | 4702.17 | 5491.39 | 9.48 | 9.89 | 1.115 | 1.282 | 16.8 |
| Team Fortress 2 | ARIMA(1,1,1) | Econometric Baseline | 5636.17 | 12486.85 | 10.74 | 21.31 | 1.337 | 2.915 | 121.5 |
| Team Fortress 2 | Naive | Naive Baseline | 6151.83 | 15929.39 | 11.39 | 27.45 | 1.459 | 3.719 | 158.9 |
| Tom Clancy's Rainbow Six Siege | Ridge | Linear ML | 7156.66 | 5149.47 | 8.49 | 5.09 | 0.565 | 0.388 | -28.0 |
| Tom Clancy's Rainbow Six Siege | Holt-Winters | Exponential Smoothing | 6961.82 | 6425.97 | 8.37 | 6.42 | 0.55 | 0.485 | -7.7 |
| Tom Clancy's Rainbow Six Siege | GRU | Recurrent DL | 8973.16 | 7866.74 | 10.77 | 7.71 | 0.709 | 0.593 | -12.3 |
| Tom Clancy's Rainbow Six Siege | LSTM | Recurrent DL | 9894.25 | 8681.09 | 11.52 | 8.48 | 0.781 | 0.655 | -12.3 |
| Tom Clancy's Rainbow Six Siege | SARIMA | Econometric | 7666.51 | 9206.76 | 9.52 | 9.41 | 0.605 | 0.694 | 20.1 |
| Tom Clancy's Rainbow Six Siege | Naive | Naive Baseline | 19434.16 | 10543.99 | 22.53 | 10.32 | 1.535 | 0.795 | -45.7 |
| Tom Clancy's Rainbow Six Siege | Random Forest | Tree Ensemble ML | 10322.2 | 11293.63 | 12.24 | 11.11 | 0.815 | 0.852 | 9.4 |
| Tom Clancy's Rainbow Six Siege | XGBoost | Gradient Boosted ML | 10509.47 | 12569.17 | 12.3 | 12.39 | 0.83 | 0.948 | 19.6 |
| Tom Clancy's Rainbow Six Siege | ARIMA(1,1,1) | Econometric Baseline | 15368.31 | 12922.95 | 17.74 | 12.78 | 1.214 | 0.975 | -15.9 |
| Tom Clancy's Rainbow Six Siege | RNN | Recurrent DL Baseline | 10924.65 | 13875.33 | 12.72 | 13.72 | 0.863 | 1.046 | 27.0 |
| Tom Clancy's Rainbow Six Siege | Seasonal Naive | Seasonal Baseline | 14901.62 | 15132.11 | 17.54 | 15.27 | 1.177 | 1.141 | 1.5 |
| Warframe | Holt-Winters | Exponential Smoothing | 2832.09 | 2625.59 | 6.06 | 4.82 | 0.645 | 0.602 | -7.3 |
| Warframe | Ridge | Linear ML | 3089.22 | 2647.3 | 6.59 | 4.89 | 0.704 | 0.607 | -14.3 |
| Warframe | GRU | Recurrent DL | 4348.87 | 3071.43 | 8.88 | 5.52 | 0.991 | 0.705 | -29.4 |
| Warframe | SARIMA | Econometric | 2050.94 | 3201.62 | 4.34 | 5.8 | 0.467 | 0.734 | 56.1 |
| Warframe | Random Forest | Tree Ensemble ML | 4431.46 | 3844.42 | 8.89 | 7.03 | 1.01 | 0.882 | -13.2 |
| Warframe | XGBoost | Gradient Boosted ML | 4496.11 | 4114.85 | 9.08 | 7.42 | 1.025 | 0.944 | -8.5 |
| Warframe | RNN | Recurrent DL Baseline | 3956.21 | 4250.01 | 7.75 | 7.72 | 0.902 | 0.975 | 7.4 |
| Warframe | Naive | Naive Baseline | 4326.94 | 4270.17 | 9.0 | 7.74 | 0.986 | 0.98 | -1.3 |
| Warframe | LSTM | Recurrent DL | 3996.54 | 4745.06 | 7.92 | 8.57 | 0.911 | 1.089 | 18.7 |
| Warframe | ARIMA(1,1,1) | Econometric Baseline | 4928.3 | 5304.91 | 9.65 | 9.6 | 1.123 | 1.217 | 7.6 |
| Warframe | Seasonal Naive | Seasonal Baseline | 4199.4 | 5415.18 | 8.56 | 10.04 | 0.957 | 1.242 | 29.0 |
