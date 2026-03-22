import sys
import os
import pandas as pd
import io
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from pagos_bci
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pagos_bci import show_pagos_bci_view
import streamlit as st


@patch('pagos_bci.st')
def test_show_pagos_bci_view_with_no_file(mock_st):
    """Test show_pagos_bci_view when no file is uploaded."""
    # Setup
    mock_st.file_uploader.return_value = None  # No file uploaded
    
    # Call the function
    show_pagos_bci_view()
    
    # Verify that file_uploader was called
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader"
    )
    
    # Verify header and message when no file
    mock_st.header.assert_called_once_with("PAGOS BCI")
    mock_st.write.assert_called_once_with(
        "En construcción - Por favor sube un archivo Excel para ver la vista previa"
    )


@patch('pagos_bci.st')
@patch('pagos_bci.pd.read_excel')
def test_show_pagos_bci_view_with_data(mock_read_excel, mock_st):
    """Test show_pagos_bci_view with valid data."""
    # Setup mock file uploader to return a mock file
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file
    
    # Setup mock pandas read_excel to return a test DataFrame
    test_df = pd.DataFrame({
        'col1': ['value1', 'value2'],
        'col2': ['value3', 'value4'],
        'col3': [100, 200]
    })
    mock_read_excel.return_value = test_df
    
    # Call the function
    show_pagos_bci_view()
    
    # Verify that the functions were called appropriately
    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader"
    )
    mock_read_excel.assert_called_once_with(mock_uploaded_file)
    mock_st.header.assert_called_once_with("PAGOS BCI")
    mock_st.subheader.assert_called_once_with("Data Preview (First 5 rows):")
    
    # Check that dataframe was called with the head of our test dataframe
    # We can't directly compare DataFrames in assert_called_with due to ambiguity error
    # So we check that it was called once and verify the argument separately
    mock_st.dataframe.assert_called_once()
    called_args = mock_st.dataframe.call_args[0]
    called_df = called_args[0]  # First positional argument
    pd.testing.assert_frame_equal(called_df, test_df.head())
    
    mock_st.write.assert_any_call(f"Shape: {test_df.shape}")
    mock_st.write.assert_any_call(f"Columns: {list(test_df.columns)}")