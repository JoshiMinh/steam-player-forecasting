# SARIMA Hyperparameter Selection & In-Sample Diagnostics

Optimal specifications selected on 12-month validation horizon (2019-09 to 2020-08) prior to unblinded testing.

| Game | Train_Obs | Optimal_Order | Optimal_Seasonal_Order | Specification | AIC | BIC | Val_MAE | Val_RMSE | Val_MAPE | Val_sMAPE | Val_MASE | LB_pValue | Residuals_White_Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Counter-Strike: Global Offensive | 86 | (1, 0, 0) | (1, 1, 0, 12) | SARIMA(1, 0, 0)x(1, 1, 0, 12) | 1378.59 | 1384.93 | 7968.89 | 8997.64 | 1.37 | 1.37 | 0.147 | 0.8914 | True |
| Dota 2 | 86 | (0, 1, 0) | (0, 1, 1, 12) | SARIMA(0, 1, 0)x(0, 1, 1, 12) | 1420.4 | 1424.59 | 13004.39 | 14951.1 | 3.14 | 3.16 | 0.463 | 0.5747 | True |
| Grand Theft Auto V | 53 | (1, 0, 1) | (1, 1, 1, 12) | SARIMA(1, 0, 1)x(1, 1, 1, 12) | 585.24 | 591.72 | 6223.54 | 8941.08 | 5.51 | 5.8 | 0.477 | 0.0 | False |
| Rust | 69 | (2, 1, 1) | (1, 0, 1, 12) | SARIMA(2, 1, 1)x(1, 0, 1, 12) | 1104.3 | 1116.24 | 3537.24 | 4691.71 | 4.67 | 4.54 | 0.348 | 0.749 | True |
| Team Fortress 2 | 86 | (1, 1, 2) | (0, 1, 1, 12) | SARIMA(1, 1, 2)x(0, 1, 1, 12) | 1151.69 | 1161.99 | 3561.85 | 5784.58 | 7.61 | 6.9 | 0.845 | 0.3534 | True |
| Tom Clancy's Rainbow Six Siege | 45 | (0, 1, 1) | (0, 1, 1, 12) | SARIMA(0, 1, 1)x(0, 1, 1, 12) | 379.21 | 381.88 | 7666.51 | 8940.34 | 9.52 | 9.52 | 0.605 | 0.5716 | True |
| Warframe | 78 | (1, 1, 2) | (1, 0, 1, 12) | SARIMA(1, 1, 2)x(1, 0, 1, 12) | 1221.61 | 1234.37 | 2050.94 | 2918.23 | 4.34 | 4.26 | 0.467 | 0.2953 | True |
