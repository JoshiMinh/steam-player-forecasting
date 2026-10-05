"""Hyperparameter optimization, AIC/BIC model selection, and grid search for SARIMA."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from steam_player_forecasting.evaluation.metrics import calculate_metrics
from steam_player_forecasting.models.diagnostics import evaluate_residuals
from steam_player_forecasting.models.statistical import SARIMAForecaster


def select_sarima_order(
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray | None = None,
    candidate_orders: list[tuple[int, int, int]] | None = None,
    candidate_seasonal_orders: list[tuple[int, int, int, int]] | None = None,
    criterion: Literal["val_mae", "val_rmse", "val_mape", "val_smape", "val_mase", "aic", "bic"] = "val_mae",
    s: int = 12,
    maxiter: int = 50,
    verbose: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], pd.DataFrame, SARIMAForecaster]:
    """Systematically search and select optimal SARIMA order using AIC/BIC and validation error.

    Evaluates candidate specifications on:
      1. Information criteria: AIC, BIC (penalized log-likelihood on training set).
      2. Residual diagnostics: Ljung-Box test for Gaussian white noise residuals.
      3. Out-of-sample multi-step accuracy: MAE, RMSE, MAPE, sMAPE, MASE on validation set.

    Args:
        y_train: Training time series.
        y_val: Validation time series for out-of-sample evaluation. If None, criterion defaults to 'aic'.
        candidate_orders: List of (p, d, q) orders to test.
        candidate_seasonal_orders: List of (P, D, Q, s) seasonal orders to test.
        criterion: Target selection metric (e.g. 'val_mae', 'val_rmse', 'aic', 'bic').
        s: Seasonal cycle length (default 12).
        maxiter: Maximum iterations for MLE optimization.
        verbose: Whether to print optimization progress.

    Returns:
        Tuple of (best_order, best_seasonal_order, results_dataframe, best_fitted_model).
    """
    if y_val is None and criterion.startswith("val_"):
        criterion = "aic"

    if candidate_orders is None:
        candidate_orders = [
            (1, 1, 1),
            (1, 1, 0),
            (0, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (0, 1, 2),
            (2, 1, 0),
            (1, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
        ]

    if candidate_seasonal_orders is None:
        candidate_seasonal_orders = [
            (1, 1, 1, s),
            (0, 1, 1, s),
            (1, 0, 1, s),
            (1, 1, 0, s),
            (0, 0, 0, s),
        ]

    train_arr = np.asarray(y_train, dtype=float).ravel()
    val_arr = np.asarray(y_val, dtype=float).ravel() if y_val is not None else None
    val_steps = len(val_arr) if val_arr is not None else 0

    records: list[dict[str, Any]] = []
    models_dict: dict[tuple, SARIMAForecaster] = {}

    for ord_tuple in candidate_orders:
        for sord_tuple in candidate_seasonal_orders:
            spec_key = (ord_tuple, sord_tuple)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAForecaster(
                        order=ord_tuple,
                        seasonal_order=sord_tuple,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        maxiter=maxiter,
                    )
                    model.fit(y_train)

                aic_val = model.aic_ if model.aic_ is not None else np.nan
                bic_val = model.bic_ if model.bic_ is not None else np.nan

                # Residual white-noise check
                try:
                    res_diag = evaluate_residuals(model.resid_, lags=min(12, len(train_arr) // 2 - 1))
                    lb_pvalue = res_diag["ljung_box"]["min_p_value"]
                    is_white_noise = res_diag["is_white_noise"]
                except Exception:
                    lb_pvalue = np.nan
                    is_white_noise = False

                row: dict[str, Any] = {
                    "order": ord_tuple,
                    "seasonal_order": sord_tuple,
                    "order_str": f"SARIMA{ord_tuple}x{sord_tuple}",
                    "aic": aic_val,
                    "bic": bic_val,
                    "lb_pvalue": lb_pvalue,
                    "is_white_noise": is_white_noise,
                }

                # Validation forecast evaluation
                if val_arr is not None and val_steps > 0:
                    fc = model.predict(steps=val_steps)
                    fc_arr = np.asarray(fc, dtype=float).ravel()
                    metrics = calculate_metrics(
                        y_true=val_arr,
                        y_pred=fc_arr,
                        y_train=train_arr,
                        seasonal_period=s,
                    )
                    row["val_mae"] = metrics["MAE"]
                    row["val_rmse"] = metrics["RMSE"]
                    row["val_mape"] = metrics["MAPE"]
                    row["val_smape"] = metrics["sMAPE"]
                    row["val_mase"] = metrics.get("MASE", np.nan)

                records.append(row)
                models_dict[spec_key] = model

                if verbose:
                    val_err_str = f", Val MAE: {row.get('val_mae', np.nan):.1f}" if val_arr is not None else ""
                    print(f"Evaluated {row['order_str']} -> AIC: {aic_val:.1f}{val_err_str}")

            except Exception as e:
                if verbose:
                    print(f"Skipped {spec_key}: {e}")
                continue

    if not records:
        raise RuntimeError("No candidate SARIMA specifications converged successfully.")

    df_results = pd.DataFrame(records)

    # Sort results according to target criterion
    if criterion in df_results.columns:
        df_results = df_results.sort_values(by=criterion, ascending=True).reset_index(drop=True)
    else:
        df_results = df_results.sort_values(by="aic", ascending=True).reset_index(drop=True)

    best_row = df_results.iloc[0]
    best_order: tuple[int, int, int] = best_row["order"]
    best_seasonal_order: tuple[int, int, int, int] = best_row["seasonal_order"]
    best_model = models_dict[(best_order, best_seasonal_order)]

    return best_order, best_seasonal_order, df_results, best_model


# Alias
grid_search_sarima = select_sarima_order
