import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from pagos_frm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pagos_frm import crear_features, _show_prediction_analysis
import streamlit as st


def test_crear_features():
    """Test the crear_features function."""
    # Create a sample DataFrame with the expected columns
    df_hist = pd.DataFrame({
        'AÑO': [2023, 2023, 2024, 2024],
        'MES_NUM': [1, 2, 1, 2],
        'monto_total': [100, 150, 200, 250],
        'num_pagos': [10, 15, 20, 25],
        'dias_mes': [31, 28, 31, 28],
        'pct_judicial': [0.1, 0.2, 0.15, 0.25],
        'pct_castigo': [0.05, 0.1, 0.08, 0.12]
    })
    
    # Call the function
    result = crear_features(df_hist)
    
    # Check that the result has the expected columns
    expected_columns = ['año', 'mes', 'monto_lag_1', 'num_pagos_lag_1', 
                       'monto_lag_2', 'num_pagos_lag_2', 'monto_lag_3', 'num_pagos_lag_3',
                       'monto_lag_6', 'num_pagos_lag_6', 'monto_lag_12', 'num_pagos_lag_12',
                       'monto_ma_3', 'num_pagos_ma_3', 'monto_ma_6', 'num_pagos_ma_6',
                       'monto_ma_12', 'num_pagos_ma_12', 'monto_std_6', 'num_pagos_std_6',
                       'monto_diff_1', 'monto_diff_12', 'pct_judicial', 'pct_castigo', 'dias_mes']
    
    assert all(col in result.columns for col in expected_columns)
    
    # Check that the año and mes columns are correct
    assert list(result['año']) == [2023, 2023, 2024, 2024]
    assert list(result['mes']) == [1, 2, 1, 2]
    
    # Check that lag features are computed correctly (first row should have NaN for lags)
    assert pd.isna(result.loc[0, 'monto_lag_1'])
    assert pd.isna(result.loc[0, 'num_pagos_lag_1'])
    
    # Second row should have the first row's values for lag 1
    assert result.loc[1, 'monto_lag_1'] == 100
    assert result.loc[1, 'num_pagos_lag_1'] == 10
    
    # Check moving averages
    # For monto_ma_3, first two rows should have NaN, third row should be mean of first three
    assert pd.isna(result.loc[0, 'monto_ma_3'])
    assert pd.isna(result.loc[1, 'monto_ma_3'])
    assert result.loc[2, 'monto_ma_3'] == (100 + 150 + 200) / 3  # 150.0
    
    # Check differences - first element is NaN because there's no previous value
    assert pd.isna(result.loc[0, 'monto_diff_1'])  # First row, no previous
    assert result.loc[1, 'monto_diff_1'] == 50  # 150 - 100
    assert result.loc[2, 'monto_diff_1'] == 50  # 200 - 150


def test_crear_features_with_minimal_data():
    """Test crear_features with minimal data."""
    # Create a minimal DataFrame
    df_hist = pd.DataFrame({
        'AÑO': [2023],
        'MES_NUM': [1],
        'monto_total': [100],
        'num_pagos': [10],
        'dias_mes': [31],
        'pct_judicial': [0.1],
        'pct_castigo': [0.05]
    })
    
    # Call the function
    result = crear_features(df_hist)
    
    # Check that we get a DataFrame with the expected shape
    assert result.shape[0] == 1  # One row
    assert 'año' in result.columns
    assert 'mes' in result.columns
    assert result.iloc[0]['año'] == 2023
    assert result.iloc[0]['mes'] == 1


@patch('pagos_frm.st')
@patch('pagos_frm.HAS_ML_LIBS', True)
@patch('pagos_frm.xgb.XGBRegressor')
@patch('pagos_frm.lgb.LGBMRegressor')
@patch('pagos_frm.RandomForestRegressor')
@patch('pagos_frm.GradientBoostingRegressor')
@patch('pagos_frm.ExtraTreesRegressor')
@patch('pagos_frm.StandardScaler')
@patch('pagos_frm.TimeSeriesSplit')
@patch('pagos_frm.mean_absolute_error')
@patch('pagos_frm.mean_absolute_percentage_error')
def test_show_prediction_analysis(mock_mape, mock_mae, mock_tscv, mock_scaler,
                                 mock_extratrees, mock_gradboost, mock_rf,
                                 mock_lgbm, mock_xgboost, mock_st):
    """Test the _show_prediction_analysis function with mocked dependencies."""
    # Setup mocks
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.write = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
    
    # Mock the scaler
    mock_scaler_instance = MagicMock()
    mock_scaler.return_value = mock_scaler_instance
    mock_scaler_instance.fit_transform.return_value = np.array([[1, 2], [3, 4]])
    mock_scaler_instance.transform.return_value = np.array([[5, 6]])
    
    # Mock TimeSeriesSplit
    mock_tscv_instance = MagicMock()
    mock_tscv.return_value = mock_tscv_instance
    mock_tscv_instance.split.return_value = [([0], [1])]  # One split
    
     # Mock metrics
    mock_mae.return_value = 1.0
    mock_mape.return_value = 10.0  # 10%
     
    # Mock models
    mock_model_instance = MagicMock()
    mock_model_instance.fit.return_value = None
    mock_model_instance.predict.return_value = np.array([100])
    
    mock_xgboost.return_value = mock_model_instance
    mock_lgbm.return_value = mock_model_instance
    mock_rf.return_value = mock_model_instance
    mock_gradboost.return_value = mock_model_instance
    mock_extratrees.return_value = mock_model_instance
        
    # Create sample data
    monthly = pd.DataFrame({
        'AÑO_MES': ['2023-01', '2023-02', '2023-03'],
        'año': [2023, 2023, 2023],
        'mes': [1, 2, 3],
        'monto_total': [1000, 1500, 2000],
        'num_pagos': [10, 15, 20],
        'dias_mes': [31, 28, 31]
    })
    
    df_original = pd.DataFrame({
        'FECHA_PAGO': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15', '2023-03-01', '2023-03-15']),
        'MONTO': [500, 500, 750, 750, 1000, 1000],
        'num_pagos': [1, 1, 1, 1, 1, 1]  # This will be aggregated
    })
    
    # Call the function
    _show_prediction_analysis(monthly, df_original)
    
    # Verify that we didn't get an error (the function completed)
    # Since we're mocking everything, we mainly want to ensure no exceptions were raised
    assert True  # If we got here without exception, the test passes


if __name__ == "__main__":
    test_crear_features()
    test_crear_features_with_minimal_data()
    print("All tests passed!")