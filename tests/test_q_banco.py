from unittest.mock import MagicMock, patch

import pandas as pd

from q_banco import show_q_banco_view


@patch("q_banco.st")
def test_show_q_banco_view_with_missing_columns(mock_st):
    mock_st.file_uploader.return_value = None

    show_q_banco_view()

    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader"
    )


@patch("q_banco.st")
@patch("q_banco.pd.read_excel")
def test_show_q_banco_view_with_data(mock_read_excel, mock_st):
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file

    test_df = pd.DataFrame(
        {
            "rut": ["12345678-9", "98765432-1"],
            "dv": ["9", "1"],
            "n_operacion_principal": ["OP001", "OP002"],
            "origen_core": ["Core1", "Core2"],
            "nombre_completo_cliente": ["Cliente 1", "Cliente 2"],
            "SUCURSAL": ["Sucursal A", "Sucursal B"],
            "CARTERA": ["Cartera X", "Cartera Y"],
            "ESTADO CRM": ["Activo", "Inactivo"],
            "ESTADO JUDICIAL": ["No Judicial", "Judicial"],
            "saldo_capital": [100000, 200000],
            "% DESCUENTO": [10, 15],
            "comuna_particular": ["Comuna 1", "Comuna 2"],
        }
    )
    mock_read_excel.return_value = test_df

    show_q_banco_view()

    mock_st.file_uploader.assert_called_once_with(
        "Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader"
    )
    mock_read_excel.assert_called_once_with(mock_uploaded_file)
    mock_st.subheader.assert_any_call("Original Data Preview:")
    mock_st.dataframe.assert_any_call(test_df)
    mock_st.write.assert_any_call(f"Original shape: {test_df.shape}")

    mock_st.download_button.assert_called_once()


@patch("q_banco.st")
@patch("q_banco.pd.read_excel")
def test_show_q_banco_view_with_missing_columns_error(mock_read_excel, mock_st):
    mock_uploaded_file = MagicMock()
    mock_st.file_uploader.return_value = mock_uploaded_file

    test_df = pd.DataFrame(
        {
            "rut": ["12345678-9", "98765432-1"],
            "dv": ["9", "1"],
            "nombre_completo_cliente": ["Cliente 1", "Cliente 2"],
        }
    )
    mock_read_excel.return_value = test_df

    show_q_banco_view()

    mock_st.error.assert_called()
    error_call_args = mock_st.error.call_args[0][0]
    assert "Missing columns" in error_call_args
    assert (
        "n_operacion_principal" in error_call_args or "origen_core" in error_call_args
    )
