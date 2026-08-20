"""Tests for the Reporte CAE pivot table helpers in pages/reporte_cae.py."""

import io
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.reporte_cae import (  # noqa: E402
    DATA_SHEET,
    PIVOT_SHEET,
    REPORT_SHEET,
    build_cae_pivot,
    resolve_cae_pivot_columns,
    write_cae_excel_with_pivot,
    write_cae_nuevos_excel,
)


def create_sample_cae() -> pd.DataFrame:
    """Sample current-report data with the pivot fields."""
    return pd.DataFrame(
        {
            "Llave": ["", "", "", "", ""],
            "OPERACIÓN": ["1", "2", "3", "4", "5"],
            "ETAPA SATCHMO": ["Etapa1", "Etapa1", "Etapa2", "Etapa2", "Etapa1"],
            "Mes": ["ene", "ene", "ene", "feb", "feb"],
            "PRODUCTO": ["P1", "P2", "P3", "P4", None],
            "FECHA CREA": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-10",
                    "2026-02-01",
                    "2026-02-15",
                    "2026-02-20",
                ]
            ),
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


def test_resolve_cae_pivot_columns_requires_fecha_crea():
    """FECHA CREA (the pivot filter) is also required."""
    df = create_sample_cae().drop(columns=["FECHA CREA"])
    with pytest.raises(KeyError, match="FECHA CREA"):
        resolve_cae_pivot_columns(df)


def test_build_cae_pivot_case_insensitive():
    """Required columns are matched case-insensitively."""
    df = create_sample_cae().rename(
        columns={"Mes": "MES", "PRODUCTO": "producto", "FECHA CREA": "fecha crea"}
    )
    pivot = build_cae_pivot(df)
    assert pivot.loc["Etapa1", "ene"] == 2


def test_write_cae_nuevos_excel_single_sheet():
    """The new-records-only download keeps a single REPORT_SHEET sheet."""
    df_actual = create_sample_cae()
    df_nuevos = df_actual.head(3)

    data = write_cae_nuevos_excel(df_nuevos)

    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("xl/workbook.xml"))
    sheet_names = [s.get("name") for s in root.iter() if s.tag.endswith("}sheet")]
    assert sheet_names == [REPORT_SHEET]


def test_write_cae_excel_with_pivot_builds_native_pivot():
    """The combined download has 3 sheets and native pivot OOXML parts."""
    df_actual = create_sample_cae()
    df_actual["Llave"] = df_actual["OPERACIÓN"] + df_actual["ETAPA SATCHMO"]
    df_nuevos = df_actual.head(2)

    data = write_cae_excel_with_pivot(df_nuevos, df_actual)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert "xl/pivotTables/pivotTable1.xml" in names
        assert "xl/pivotCache/pivotCacheDefinition1.xml" in names
        assert "xl/pivotCache/pivotCacheRecords1.xml" in names
        assert "xl/pivotTables/_rels/pivotTable1.xml.rels" in names
        assert "xl/pivotCache/_rels/pivotCacheDefinition1.xml.rels" in names

        content_types = zf.read("[Content_Types].xml").decode("utf-8")
        assert "pivotCacheDefinition+xml" in content_types
        assert "pivotCacheRecords+xml" in content_types
        assert "pivotTable+xml" in content_types

        workbook = zf.read("xl/workbook.xml").decode("utf-8")
        assert "<pivotCaches>" in workbook
        assert 'cacheId="100"' in workbook

        root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_names = [s.get("name") for s in root.iter() if s.tag.endswith("}sheet")]
        assert sheet_names == [REPORT_SHEET, DATA_SHEET, PIVOT_SHEET]

        cache_def = zf.read("xl/pivotCache/pivotCacheDefinition1.xml").decode("utf-8")
        ET.fromstring(cache_def)
        assert 'refreshOnLoad="1"' in cache_def
        assert 'sheet="DATOS"' in cache_def
        assert 'recordCount="5"' in cache_def

        cache_records = zf.read("xl/pivotCache/pivotCacheRecords1.xml").decode("utf-8")
        records_root = ET.fromstring(cache_records)
        assert records_root.get("count") == "5"

        pivot_xml = zf.read("xl/pivotTables/pivotTable1.xml").decode("utf-8")
        ET.fromstring(pivot_xml)
        assert 'axis="axisPage"' in pivot_xml  # FECHA CREA filter
        assert 'axis="axisRow"' in pivot_xml  # ETAPA SATCHMO rows
        assert 'axis="axisCol"' in pivot_xml  # Mes columns
        assert 'subtotal="count"' in pivot_xml  # count of PRODUCTO
        assert "Recuento de PRODUCTO" in pivot_xml

        nuevos = pd.read_excel(io.BytesIO(data), sheet_name=REPORT_SHEET)
        assert len(nuevos) == 2
        datos = pd.read_excel(io.BytesIO(data), sheet_name=DATA_SHEET)
        assert len(datos) == 5
        assert "Llave" not in datos.columns


def test_write_cae_excel_with_pivot_cache_matches_data():
    """Cache records reference shared items consistently with the data."""
    df_actual = create_sample_cae()
    df_actual["Llave"] = df_actual["OPERACIÓN"] + df_actual["ETAPA SATCHMO"]

    data = write_cae_excel_with_pivot(df_actual.head(1), df_actual)

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        cache_def = ET.fromstring(zf.read("xl/pivotCache/pivotCacheDefinition1.xml"))
        records = ET.fromstring(zf.read("xl/pivotCache/pivotCacheRecords1.xml"))

    cache_fields = cache_def.find(f"{ns}cacheFields")
    field_names = [f.get("name") for f in cache_fields]
    assert "ETAPA SATCHMO" in field_names
    assert "FECHA CREA" in field_names

    etapa_idx = field_names.index("ETAPA SATCHMO")
    shared = list(cache_fields[etapa_idx])[0]
    etapa_values = [item.get("v") for item in shared]
    assert etapa_values == ["Etapa1", "Etapa2"]

    first_record = records[0]
    etapa_cell = first_record[etapa_idx]
    assert etapa_cell.get("v") == str(etapa_values.index("Etapa1"))
