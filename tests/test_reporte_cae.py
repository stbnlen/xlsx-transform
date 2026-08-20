"""Tests for the Reporte CAE pivot table helpers in pages/reporte_cae.py."""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.reporte_cae import (  # noqa: E402
    build_cae_pivot,
    write_cae_pivot_excel,
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


def test_write_cae_pivot_excel_creates_hoja1():
    """The download bytes contain a 'Hoja1' sheet with the pivot data."""
    pivot = build_cae_pivot(create_sample_cae())
    data = write_cae_pivot_excel(pivot)

    result = pd.read_excel(io.BytesIO(data), sheet_name="Hoja1")
    assert result.columns[0] == "ETAPA SATCHMO"
    assert set(result.columns[1:]) == {"ene", "feb"}
    assert len(result) == 2
