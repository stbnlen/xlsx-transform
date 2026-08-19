"""Tests for the Flujo COP tab schema in pages/asig.py."""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.asig import COP_STOCK_COLUMNS, read_cop_stock_file  # noqa: E402
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
