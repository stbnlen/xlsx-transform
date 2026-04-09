import sys
import os
import pandas as pd
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from q_banco
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from q_banco import show_q_banco_view
import streamlit as st


@patch('q_banco.st')
def test_show_q_banco_view_with_missing_columns(mock_st):
    """Test show_q_banco_view when required columns are missing."""
    # Setup
    mock_st.file_uploader.return_value = None  # No file uploaded
    
    # Call the function
    show_q_banco_view()
    
    # Verify that file_uploader was called
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader"
    )


@patch('q_banco.st')
@patch('q_banco.pd.read_excel')
def test_show_q_banco_view_with_data(mock_read_excel, mock_st):
    """Test show_q_banco_view with valid data."""
    # Setup mock file uploader to return a mock file
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file
    
    # Setup mock pandas read_excel to return a test DataFrame
    test_df = pd.DataFrame({
        'rut': ['12345678-9', '98765432-1'],
        'dv': ['9', '1'],
        'n_operacion_principal': ['OP001', 'OP002'],
        'origen_core': ['Core1', 'Core2'],
        'nombre_completo_cliente': ['Cliente 1', 'Cliente 2'],
        'SUCURSAL': ['Sucursal A', 'Sucursal B'],
        'CARTERA': ['Cartera X', 'Cartera Y'],
        'ESTADO CRM': ['Activo', 'Inactivo'],
        'ESTADO JUDICIAL': ['No Judicial', 'Judicial'],
        'saldo_capital': [100000, 200000],
        '% DESCUENTO': [10, 15],
        'comuna_particular': ['Comuna 1', 'Comuna 2']
    })
    mock_read_excel.return_value = test_df
    
    # Call the function
    show_q_banco_view()
    
    # Verify that the functions were called appropriately
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader"
    )
    mock_read_excel.assert_called_once_with(mock_uploaded_file)
    mock_st.subheader.assert_any_call("Original Data Preview:")
    mock_st.dataframe.assert_any_call(test_df)
    mock_st.write.assert_any_call(f"Original shape: {test_df.shape}")
    
    # Verify download button was created
    mock_st.download_button.assert_called_once()


@patch('q_banco.st')
@patch('q_banco.pd.read_excel')
def test_show_q_banco_view_with_missing_columns_error(mock_read_excel, mock_st):
    """Test show_q_banco_view when some required columns are missing."""
    # Setup mock file uploader to return a mock file
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file
    
    # Setup mock pandas read_excel to return a DataFrame missing some columns
    test_df = pd.DataFrame({
        'rut': ['12345678-9', '98765432-1'],
        'dv': ['9', '1'],
        # Missing several required columns
        'nombre_completo_cliente': ['Cliente 1', 'Cliente 2'],
    })
    mock_read_excel.return_value = test_df
    
    # Call the function
    show_q_banco_view()
    
    # Verify error message was shown
    mock_st.error.assert_called()
    # Check that the error message contains missing columns
    error_call_args = mock_st.error.call_args[0][0]
    assert "Missing columns" in error_call_args
    # Should mention some of the missing columns
    assert 'n_operacion_principal' in error_call_args or 'origen_core' in error_call_args