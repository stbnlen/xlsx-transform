"""Tests for the Flujo COP tab schema in pages/asig.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.asig import COP_STOCK_COLUMNS  # noqa: E402
from utils import validate_required_columns  # noqa: E402


def test_cop_stock_columns_schema():
    """The expected COP stock columns are declared."""
    assert COP_STOCK_COLUMNS == [
        "RUT COM",
        "DV",
        "AISNACION",
        "Demandado",
        "SALDO DEUDOR RUT COMPLETO",
        "ESTADO CRM",
        "Flujo/Stock",
        "FF",
    ]


def test_cop_stock_validation_detects_missing_columns():
    """Missing stock columns are reported."""
    df = pd.DataFrame(columns=["RUT COM", "DV", "Demandado"])

    missing, mapping = validate_required_columns(df.columns, COP_STOCK_COLUMNS)

    assert set(missing) == {
        "AISNACION",
        "SALDO DEUDOR RUT COMPLETO",
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
            "aisnacion",
            "DEMANDADO",
            "saldo deudor rut completo",
            "estado crm",
            "flujo/stock",
            "ff",
        ]
    )

    missing, mapping = validate_required_columns(df.columns, COP_STOCK_COLUMNS)

    assert missing == []
    assert mapping["RUT COM"] == "rut com"
    assert mapping["FF"] == "ff"
