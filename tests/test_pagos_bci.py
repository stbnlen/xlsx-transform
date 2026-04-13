from unittest.mock import MagicMock, patch

import pandas as pd

from pagos_bci import show_pagos_bci_view


@patch("pagos_bci.st")
def test_show_pagos_bci_view_with_no_file(mock_st):
    mock_st.file_uploader.return_value = None

    show_pagos_bci_view()

    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader"
    )

    mock_st.header.assert_called_once_with("PAGOS BCI")
    mock_st.write.assert_called_once_with(
        "Under construction - Please upload an Excel file to preview"
    )


@patch("pagos_bci.st")
@patch("pagos_bci.pd.read_excel")
def test_show_pagos_bci_view_with_data(mock_read_excel, mock_st):
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file

    test_df = pd.DataFrame(
        {"col1": ["value1", "value2"], "col2": ["value3", "value4"], "col3": [100, 200]}
    )
    mock_read_excel.return_value = test_df

    show_pagos_bci_view()

    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader"
    )
    mock_read_excel.assert_called_once_with(mock_uploaded_file)
    mock_st.header.assert_called_once_with("PAGOS BCI")
    mock_st.subheader.assert_called_once_with("Data Preview (First 5 rows):")

    mock_st.dataframe.assert_called_once()
    called_args = mock_st.dataframe.call_args[0]
    called_df = called_args[0]
    pd.testing.assert_frame_equal(called_df, test_df.head())

    mock_st.write.assert_any_call(f"Shape: {test_df.shape}")
    mock_st.write.assert_any_call(f"Columns: {list(test_df.columns)}")
