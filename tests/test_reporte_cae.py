"""Tests for the Reporte CAE pivot table helpers in pages/reporte_cae.py."""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.reporte_cae import (  # noqa: E402
    PIVOT_SHEET,
    REPORT_SHEET,
    build_cae_pivot,
    write_cae_combined_excel,
    write_cae_nuevos_excel,
)


def create_sample_cae() -> pd.DataFrame:
    """Sample current-report data with Mes, ETAPA SATCHMO and PRODUCTO."""
    return pd.DataFrame(
        {
            "OPERACIÓN": ["1", "2", "3", "4", "5"],
            "ETAPA SATCHMO": ["Etapa1", "Etapa1", "Etapa2", "Etapa2", "Etapa1"],
            "Mes": ["ene", "ene", "ene", "feb", "feb"],
            "PRODUCTO": ["P1", "P2", "P3", "P4", None],
        }
    )


def test_build_cae_pivot_counts_producto():
    """Pivot counts non-null PRODUCTO per ETAPA SATCHMO x Mes."""
    pivot = build_cae_pivot(create_sample_cae())

    assert pivot.loc["Etapa1", "ene"] == 2
    assert pivot.loc["Etapa1", "feb"] == 0  # the only feb Etapa1 PRODUCTO is None
    assert pivot.loc["Etapa2", "ene"] == 1
    assert pivot.loc["Etapa2", "feb"] == 1


def test_build_cae_pivot_missing_column_raises():
    """A missing required column raises KeyError."""
    df = create_sample_cae().drop(columns=["PRODUCTO"])
    with pytest.raises(KeyError):
        build_cae_pivot(df)


def test_build_cae_pivot_case_insensitive():
    """Required columns are matched case-insensitively."""
    df = create_sample_cae().rename(columns={"Mes": "MES", "PRODUCTO": "producto"})
    pivot = build_cae_pivot(df)
    assert pivot.loc["Etapa1", "ene"] == 2


def test_write_cae_combined_excel_creates_both_sheets():
    """The combined download has the new records and the 'Hoja1' pivot."""
    df_actual = create_sample_cae()
    pivot = build_cae_pivot(df_actual)
    df_nuevos = df_actual.head(2)

    data = write_cae_combined_excel(df_nuevos, pivot)
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)

    assert set(sheets.keys()) == {REPORT_SHEET, PIVOT_SHEET}

    nuevos = sheets[REPORT_SHEET]
    assert len(nuevos) == 2
    assert "ETAPA SATCHMO" in nuevos.columns

    pivot_sheet = sheets[PIVOT_SHEET]
    assert pivot_sheet.columns[0] == "ETAPA SATCHMO"
    assert set(pivot_sheet.columns[1:]) == {"ene", "feb"}
    assert len(pivot_sheet) == 2


def test_write_cae_nuevos_excel_single_sheet():
    """The new-records-only download keeps a single REPORT_SHEET sheet."""
    df_actual = create_sample_cae()
    df_nuevos = df_actual.head(3)

    data = write_cae_nuevos_excel(df_nuevos)
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)

    assert set(sheets.keys()) == {REPORT_SHEET}
    assert len(sheets[REPORT_SHEET]) == 3
