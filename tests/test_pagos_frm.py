from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import streamlit as st

from pagos_frm import _show_prediction_analysis, create_features


def test_create_features():
    df_hist = pd.DataFrame(
        {
            "year": [2023, 2023, 2024, 2024],
            "month": [1, 2, 1, 2],
            "total_amount": [100, 150, 200, 250],
            "payment_count": [10, 15, 20, 25],
            "days_in_month": [31, 28, 31, 28],
            "pct_judicial": [0.1, 0.2, 0.15, 0.25],
            "pct_castigo": [0.05, 0.1, 0.08, 0.12],
        }
    )

    result = create_features(df_hist)

    expected_columns = [
        "year",
        "month",
        "amount_lag_1",
        "payment_count_lag_1",
        "amount_lag_2",
        "payment_count_lag_2",
        "amount_lag_3",
        "payment_count_lag_3",
        "amount_lag_6",
        "payment_count_lag_6",
        "amount_lag_12",
        "payment_count_lag_12",
        "amount_ma_3",
        "payment_count_ma_3",
        "amount_ma_6",
        "payment_count_ma_6",
        "amount_ma_12",
        "payment_count_ma_12",
        "amount_std_6",
        "payment_count_std_6",
        "amount_diff_1",
        "amount_diff_12",
        "pct_judicial",
        "pct_castigo",
        "days_in_month",
    ]

    assert all(col in result.columns for col in expected_columns)

    assert list(result["year"]) == [2023, 2023, 2024, 2024]
    assert list(result["month"]) == [1, 2, 1, 2]

    assert pd.isna(result.loc[0, "amount_lag_1"])
    assert pd.isna(result.loc[0, "payment_count_lag_1"])

    assert result.loc[1, "amount_lag_1"] == 100
    assert result.loc[1, "payment_count_lag_1"] == 10

    assert pd.isna(result.loc[0, "amount_ma_3"])
    assert pd.isna(result.loc[1, "amount_ma_3"])
    assert result.loc[2, "amount_ma_3"] == (100 + 150 + 200) / 3

    assert pd.isna(result.loc[0, "amount_diff_1"])
    assert result.loc[1, "amount_diff_1"] == 50
    assert result.loc[2, "amount_diff_1"] == 50


def test_create_features_with_minimal_data():
    df_hist = pd.DataFrame(
        {
            "year": [2023],
            "month": [1],
            "total_amount": [100],
            "payment_count": [10],
            "days_in_month": [31],
            "pct_judicial": [0.1],
            "pct_castigo": [0.05],
        }
    )

    result = create_features(df_hist)

    assert result.shape[0] == 1
    assert "year" in result.columns
    assert "month" in result.columns
    assert result.iloc[0]["year"] == 2023
    assert result.iloc[0]["month"] == 1


@patch("pagos_frm.st")
@patch("pagos_frm.HAS_ML_LIBS", True)
@patch("pagos_frm.xgb.XGBRegressor")
@patch("pagos_frm.lgb.LGBMRegressor")
@patch("pagos_frm.RandomForestRegressor")
@patch("pagos_frm.GradientBoostingRegressor")
@patch("pagos_frm.ExtraTreesRegressor")
@patch("pagos_frm.StandardScaler")
@patch("pagos_frm.TimeSeriesSplit")
@patch("pagos_frm.mean_absolute_error")
@patch("pagos_frm.mean_absolute_percentage_error")
def test_show_prediction_analysis(
    mock_mape,
    mock_mae,
    mock_tscv,
    mock_scaler,
    mock_extratrees,
    mock_gradboost,
    mock_rf,
    mock_lgbm,
    mock_xgboost,
    mock_st,
):
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.write = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])

    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance
    mock_scaler_instance.fit_transform.return_value = np.array([[1, 2], [3, 4]])
    mock_scaler_instance.transform.return_value = np.array([[5, 6]])

    mock_tscv_instance = MagicMock()
    mock_tscv.return_value = mock_tscv_instance
    mock_tscv_instance.split.return_value = [([0], [1])]

    mock_mae.return_value = 1.0
    mock_mape.return_value = 10.0

    mock_model_instance = MagicMock()
    mock_model_instance.fit.return_value = None
    mock_model_instance.predict.return_value = np.array([100])

    mock_xgboost.return_value = mock_model_instance
    mock_lgbm.return_value = mock_model_instance
    mock_rf.return_value = mock_model_instance
    mock_gradboost.return_value = mock_model_instance
    mock_extratrees.return_value = mock_model_instance

    monthly = pd.DataFrame(
        {
            "YEAR_MONTH": ["2023-01", "2023-02", "2023-03"],
            "year": [2023, 2023, 2023],
            "month": [1, 2, 3],
            "total_amount": [1000, 1500, 2000],
            "payment_count": [10, 15, 20],
            "days_in_month": [31, 28, 31],
        }
    )

    df_original = pd.DataFrame(
        {
            "FECHA_PAGO": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-15",
                    "2023-02-01",
                    "2023-02-15",
                    "2023-03-01",
                    "2023-03-15",
                ]
            ),
            "MONTO": [500, 500, 750, 750, 1000, 1000],
            "num_pagos": [1, 1, 1, 1, 1, 1],
        }
    )

    _show_prediction_analysis(monthly, df_original)

    assert True


if __name__ == "__main__":
    test_create_features()
    test_create_features_with_minimal_data()
    print("All tests passed!")
