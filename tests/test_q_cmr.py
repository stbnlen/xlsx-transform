import sys
import os
import pandas as pd
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from q_cmr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from q_cmr import show_q_cmr_view
import streamlit as st


@patch('q_cmr.st')
def test_show_q_cmr_view_with_no_file(mock_st):
    """Test show_q_cmr_view when no file is uploaded."""
    # Setup
    mock_st.file_uploader.return_value = None  # No file uploaded
    
    # Call the function
    show_q_cmr_view()
    
    # Verify that file_uploader was called
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_cmr_uploader"
    )


@patch('q_cmr.st')
@patch('q_cmr.pd.read_excel')
def test_show_q_cmr_view_with_data(mock_read_excel, mock_st):
    """Test show_q_cmr_view with valid data."""
    # Setup mock file uploader to return a mock file
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file
    
    # Setup mock pandas read_excel to return a test DataFrame
    test_df = pd.DataFrame({
        'rut': ['12345678-9', '98765432-1'],
        'n_operacion_principal': ['OP001', 'OP002'],
        'dv': ['9', '1'],
        'nombre_completo_cliente': ['Cliente 1', 'Cliente 2'],
        'CARTERA': ['Cartera X', 'Cartera Y'],
        'CATEGORIA': ['Categoria A', 'Categoria B'],
        'SUCURSAL': ['Sucursal A', 'Sucursal B'],
        'EJECUTIVA ASIGNADA': ['Ejecutivo 1', 'Ejecutivo 2'],
        'ESTADO JUDICIAL': ['No Judicial', 'Judicial'],
        'DESCUENTO CAMPAÑA': [5, 10],
        'SALDO_DEUDA': [50000, 75000],
        'ESTADO INICIAL': ['Normal', 'Vencido'],
        'TRAMO': ['Tramo 1', 'Tramo 2'],
        'estado_cuenta': ['Al día', 'Vencido']
    })
    mock_read_excel.return_value = test_df
    
    # Call the function
    show_q_cmr_view()
    
    # Verify that the functions were called appropriately
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_cmr_uploader"
    )
    mock_read_excel.assert_called_once_with(mock_uploaded_file)
    mock_st.subheader.assert_any_call("Original Data Preview:")
    mock_st.dataframe.assert_any_call(test_df)
    mock_st.write.assert_any_call(f"Original shape: {test_df.shape}")
    
    # Verify download button was created
    mock_st.download_button.assert_called_once()


@patch('q_cmr.st')
@patch('q_cmr.pd.read_excel')
def test_show_q_cmr_view_with_missing_columns_error(mock_read_excel, mock_st):
    """Test show_q_cmr_view when some required columns are missing."""
    # Setup mock file uploader to return a mock file
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file
    
    # Setup mock pandas read_excel to return a DataFrame missing some columns
    test_df = pd.DataFrame({
        'rut': ['12345678-9', '98765432-1'],
        'dv': ['9', '1'],
        'nombre_completo_cliente': ['Cliente 1', 'Cliente 2'],
        # Missing several required columns
    })
    mock_read_excel.return_value = test_df
    
    # Call the function
    show_q_cmr_view()
    
    # Verify error message was shown
    mock_st.error.assert_called()
    # Check that the error message contains missing columns
    error_call_args = mock_st.error.call_args[0][0]
    assert "Missing columns" in error_call_args
    # Should mention some of the missing columns
    assert 'n_operacion_principal' in error_call_args or 'CARTERA' in error_call_args