# Econometric Residual Diagnostics for Optimal SARIMA Models

Ljung-Box test for white noise serial independence and Jarque-Bera normality tests.

| Game | Specification | Residual_Mean | Residual_Std | Durbin_Watson | Ljung_Box_Stat | Ljung_Box_pValue | Jarque_Bera_Stat | Jarque_Bera_pValue | Is_Normal | Is_White_Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Counter-Strike: Global Offensive | SARIMA(1, 0, 0)x(1, 1, 0, 12) | 4359.7 | 33344.67 | 1.829 | 6.455 | 0.8914 | 367.68 | 0.0 | False | True |
| Dota 2 | SARIMA(0, 1, 0)x(0, 1, 1, 12) | -616.35 | 43052.25 | 2.14 | 10.471 | 0.5747 | 206.418 | 0.0 | False | True |
| Grand Theft Auto V | SARIMA(1, 0, 1)x(1, 1, 1, 12) | 9843.08 | 18324.83 | 0.48 | 71.451 | 0.0 | 3.177 | 0.2042 | True | False |
| Rust | SARIMA(2, 1, 1)x(1, 0, 1, 12) | 595.27 | 6874.11 | 1.483 | 8.45 | 0.749 | 26.589 | 0.0 | False | True |
| Team Fortress 2 | SARIMA(1, 1, 2)x(0, 1, 1, 12) | -1631.97 | 9367.13 | 1.175 | 13.218 | 0.3534 | 860.138 | 0.0 | False | True |
| Tom Clancy's Rainbow Six Siege | SARIMA(0, 1, 1)x(0, 1, 1, 12) | -1040.7 | 9277.74 | 1.632 | 10.506 | 0.5716 | 10.957 | 0.0042 | False | True |
| Warframe | SARIMA(1, 1, 2)x(1, 0, 1, 12) | -253.72 | 6376.46 | 1.368 | 14.085 | 0.2953 | 121.193 | 0.0 | False | True |
