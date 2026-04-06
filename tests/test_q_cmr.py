import sys
import os
import pandas as pd
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from q_cmr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from q_cmr import show_q_cmr_view
from utils import (
    normalize_column_name,
    find_matching_column,
    validate_required_columns,
)
import streamlit as st


def test_normalize_column_name():
    """Test the normalize_column_name function from utils."""
    # Test basic normalization
    assert normalize_column_name("RUT") == "rut"
    assert normalize_column_name("nombre_completo") == "nombrecompleto"
    assert normalize_column_name("SALDO_CAPITAL") == "saldocapital"
    
    # Test with multiple underscores
    assert normalize_column_name("SALDO__CAPITAL___") == "saldocapital"
    
    # Test with numbers and special characters (only underscores removed)
    assert normalize_column_name("COLUMN_123_TEST!") == "column123test!"
    
    # Test with non-string input
    assert normalize_column_name(123) == "123"
    assert normalize_column_name(None) == "none"


def test_find_matching_column():
    """Test the find_matching_column function from utils."""
    # Test exact match
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'rut') == 'rut'
    
    # Test case insensitive match
    columns = pd.Index(['RUT', 'Nombre', 'Edad'])
    assert find_matching_column(columns, 'rut') == 'RUT'
    
    # Test underscore insensitive match
    columns = pd.Index(['nombre_completo', 'edad'])
    assert find_matching_column(columns, 'nombrecompleto') == 'nombre_completo'
    
    # Test combined case and underscore insensitive (underscores removed)
    columns = pd.Index(['NOMBRE_COMPLETO_CLIENTE', 'edad'])
    assert find_matching_column(columns, 'nombrecompletocliente') == 'NOMBRE_COMPLETO_CLIENTE'
    
    # Test not found
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'apellido') is None


def test_validate_required_columns():
    """Test the validate_required_columns function from utils."""
    # Test all columns present
    columns = pd.Index(['rut', 'dv', 'nombre_completo', 'saldo_capital'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert missing == []
    assert mapping == {
        'rut': 'rut',
        'dv': 'dv',
        'nombre_completo': 'nombre_completo',
        'saldo_capital': 'saldo_capital'
    }
    
    # Test some columns missing
    columns = pd.Index(['rut', 'nombre_completo'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert set(missing) == {'dv', 'saldo_capital'}
    assert mapping == {
        'rut': 'rut',
        'nombre_completo': 'nombre_completo'
    }
    
    # Test case and underscore insensitive matching
    columns = pd.Index(['RUT', 'DV', 'NOMBRE_COMPLETO', 'SALDO_CAPITAL'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert missing == []
    assert mapping == {
        'rut': 'RUT',
        'dv': 'DV',
        'nombre_completo': 'NOMBRE_COMPLETO',
        'saldo_capital': 'SALDO_CAPITAL'
    }


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