"""Tests for the Flujo COP tab schema in pages/asig.py."""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.asig import (  # noqa: E402
    COP_STOCK_COLUMNS,
    process_flujo_cop_data,
    process_flujo_cop_file,
    read_cop_stock_file,
)
from utils import validate_required_columns  # noqa: E402


def test_cop_stock_columns_schema():
    """The expected COP stock columns are declared."""
    assert COP_STOCK_COLUMNS == [
        "RUT COM",
        "DV",
        "AISGNACION",
        "Demandado",
        "SALDO DEUDOR",
        "RUT COMPLETO",
        "ESTADO CRM",
        "Flujo/Stock",
        "FF",
    ]


def test_cop_stock_validation_detects_missing_columns():
    """Missing stock columns are reported."""
    df = pd.DataFrame(columns=["RUT COM", "DV", "Demandado"])

    missing, mapping = validate_required_columns(df.columns, COP_STOCK_COLUMNS)

    assert set(missing) == {
        "AISGNACION",
        "SALDO DEUDOR",
        "RUT COMPLETO",
        "ESTADO CRM",
        "Flujo/Stock",
        "FF",
    }
    assert mapping == {"RUT COM": "RUT COM", "DV": "DV", "Demandado": "Demandado"}


def test_cop_stock_validation_matches_case_insensitive():
    """Stock column matching is case-insensitive and trims whitespace."""
    df = pd.DataFrame(
        columns=[
            "rut com",
            " DV ",
            "aisgnacion",
            "DEMANDADO",
            "saldo deudor",
            "rut completo",
            "estado crm",
            "flujo/stock",
            "ff",
        ]
    )

    missing, mapping = validate_required_columns(df.columns, COP_STOCK_COLUMNS)

    assert missing == []
    assert mapping["RUT COM"] == "rut com"
    assert mapping["FF"] == "ff"


def _build_stock_workbook(sheet_names: list[str], data_sheet: str) -> io.BytesIO:
    """Build an in-memory workbook with the COP stock data on data_sheet."""
    stock_df = pd.DataFrame(
        {
            "RUT COM": ["19513991", "22345678"],
            "DV": ["1", "9"],
            "AISGNACION": ["A1", "A2"],
            "Demandado": ["Juan Pérez", "Ana Ruiz"],
            "SALDO DEUDOR": [1500000, 0],
            "RUT COMPLETO": ["19513991-1", "22345678-9"],
            "ESTADO CRM": ["Activo", "Inactivo"],
            "Flujo/Stock": ["Stock", "Stock"],
            "FF": ["2026-01-15", "2026-02-20"],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name in sheet_names:
            if name == data_sheet:
                stock_df.to_excel(writer, sheet_name=name, index=False)
            else:
                pd.DataFrame({"placeholder": [1]}).to_excel(
                    writer, sheet_name=name, index=False
                )
    buffer.seek(0)
    return buffer


def test_read_cop_stock_file_reads_hoja2():
    """The stock reader returns the data from the 'Hoja2' sheet."""
    buffer = _build_stock_workbook(["Hoja1", "Hoja2"], data_sheet="Hoja2")

    result = read_cop_stock_file(buffer)

    assert list(result.columns) == COP_STOCK_COLUMNS
    assert len(result) == 2
    assert result["RUT COM"].astype(str).tolist() == ["19513991", "22345678"]


def test_read_cop_stock_file_missing_hoja2_raises():
    """A clear error is raised when the 'Hoja2' sheet is absent."""
    buffer = _build_stock_workbook(["Hoja1"], data_sheet="Hoja1")

    with pytest.raises(ValueError, match="Hoja2"):
        read_cop_stock_file(buffer)


def create_sample_cop_flujo() -> pd.DataFrame:
    """Create a flujo file with repeated RUTs."""
    return pd.DataFrame(
        {
            "RUT DEUDOR": ["11111111", "22222222", "11111111"],
            "DV": ["1", "2", "1"],
            "FECHA RECEPCIÓN FACTURA": [
                "2026-08-01",
                "2026-08-02",
                "2026-08-01",
            ],
            "NOMBRE DEUDOR": ["Pedro Silva", "Maria Ruiz", "Pedro Silva"],
            "SALDO DEUDOR": [100, 200, 50],
            "ESTADO CRM": ["Contactado", "Sin contacto", "Contactado"],
        }
    )


def create_sample_cop_stock() -> pd.DataFrame:
    """Create a stock dataframe with the COP layout."""
    return pd.DataFrame(
        {
            "RUT COM": ["19513991", "22345678"],
            "DV": ["1", "9"],
            "AISGNACION": ["A1", "A2"],
            "Demandado": ["Juan Pérez", "Ana Ruiz"],
            "SALDO DEUDOR": [1500000, 0],
            "RUT COMPLETO": ["19513991-1", "22345678-9"],
            "ESTADO CRM": ["Activo", "Inactivo"],
            "Flujo/Stock": ["Stock", "Stock"],
            "FF": ["2026-01-15", "2026-02-20"],
        }
    )


def test_process_flujo_cop_file_groups_repeated_ruts():
    """Repeated flujo RUTs are collapsed into one row with summed balance."""
    result = process_flujo_cop_file(create_sample_cop_flujo())

    assert len(result) == 2
    row = result[result["RUT COM"] == "11111111"].iloc[0]
    assert row["SALDO DEUDOR"] == 150
    assert row["DV"] == "1"
    assert row["AISGNACION"] == "2026-08-01"
    assert row["Demandado"] == "Pedro Silva"
    assert row["ESTADO CRM"] == "Contactado"


def test_process_flujo_cop_file_column_layout():
    """Flujo rows are mapped to the stock column layout."""
    result = process_flujo_cop_file(create_sample_cop_flujo())

    assert list(result.columns) == COP_STOCK_COLUMNS
    row = result[result["RUT COM"] == "11111111"].iloc[0]
    assert row["RUT COMPLETO"] == "11111111-1"
    assert row["Flujo/Stock"] == "FLUJO"
    assert row["AISGNACION"] == "2026-08-01"
    assert row["FF"] == ""


def test_process_flujo_cop_file_missing_fecha_recepcion():
    """Flujo files without FECHA RECEPCIÓN FACTURA get empty AISGNACION."""
    df_flujo = create_sample_cop_flujo().drop(columns=["FECHA RECEPCIÓN FACTURA"])

    result = process_flujo_cop_file(df_flujo)

    assert (result["AISGNACION"] == "").all()


def test_process_flujo_cop_data_appends_new_ruts():
    """Grouped flujo records are appended to the stock."""
    df_stock = create_sample_cop_stock()

    combined_df, accepted, discarded = process_flujo_cop_data(
        df_stock, create_sample_cop_flujo()
    )

    assert accepted == 2
    assert discarded == 0
    assert len(combined_df) == 4
    assert list(combined_df.columns) == COP_STOCK_COLUMNS
    appended = combined_df.iloc[2:]
    assert (appended["Flujo/Stock"] == "FLUJO").all()
    assert appended["SALDO DEUDOR"].tolist() == [150, 200]


def test_process_flujo_cop_data_discards_ruts_in_stock():
    """Flujo groups whose RUT is already in the stock are discarded."""
    df_stock = create_sample_cop_stock()
    df_flujo = pd.DataFrame(
        {
            "RUT DEUDOR": ["19513991", "33333333", "19513991"],
            "DV": ["1", "3", "1"],
            "FECHA RECEPCIÓN FACTURA": ["2026-08-01", "2026-08-02", "2026-08-01"],
            "NOMBRE DEUDOR": ["Juan Pérez", "José Soto", "Juan Pérez"],
            "SALDO DEUDOR": [100, 200, 100],
            "ESTADO CRM": ["Activo", "", "Activo"],
        }
    )

    combined_df, accepted, discarded = process_flujo_cop_data(df_stock, df_flujo)

    assert accepted == 1
    assert discarded == 1
    assert len(combined_df) == 3
    assert combined_df["RUT COM"].astype(str).tolist() == [
        "19513991",
        "22345678",
        "33333333",
    ]


def test_process_flujo_cop_data_keeps_stock_unchanged():
    """The original stock dataframe is not modified."""
    df_stock = create_sample_cop_stock()
    stock_before = df_stock.copy(deep=True)

    process_flujo_cop_data(df_stock, create_sample_cop_flujo())

    pd.testing.assert_frame_equal(df_stock, stock_before)


def test_process_flujo_cop_data_stock_case_insensitive():
    """Stock columns are matched case-insensitively."""
    df_stock = create_sample_cop_stock()
    df_stock.columns = [col.lower() for col in df_stock.columns]

    combined_df, accepted, discarded = process_flujo_cop_data(
        df_stock, create_sample_cop_flujo()
    )

    assert accepted == 2
    assert discarded == 0
    assert len(combined_df) == 4
    assert list(combined_df.columns) == COP_STOCK_COLUMNS
