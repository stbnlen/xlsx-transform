"""Tests for the Flujo FORUM processing functions in pages/asig.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pages.asig import (  # noqa: E402
    FORUM_COLUMN_ORDER,
    process_flujo_file,
    process_flujo_forum_data,
    process_single_file,
)


def create_sample_stock() -> pd.DataFrame:
    """Create a stock dataframe with the standard FORUM column layout."""
    return pd.DataFrame(
        {
            "ORIGEN": ["castigo_data", "vigente_data"],
            "CONTRATO": ["C001", "V001"],
            "RUT": ["19513991", "22345678"],
            "NOMBRE CLIENTE": ["Juan Pérez", "Ana Ruiz"],
            "MONTO CASTIGO": [1500000, 0],
            "ETAPA DEMANDA": ["", ""],
            "FECHA CASTIGO": ["2026-01-15", ""],
            "CIUDAD": ["", ""],
            "CARTERA": ["Comercial", "Consumo"],
            "Tipo gestión ": ["", ""],
            "Año castigo": ["2026", ""],
        }
    )[FORUM_COLUMN_ORDER]


def create_sample_flujo() -> pd.DataFrame:
    """Create a flujo file with the relevant columns."""
    return pd.DataFrame(
        {
            "CONTRATO": ["F001", "F002", "F003"],
            "RUT": ["20123456-2", "19513991-1", "18765432-K"],
            "NOMBRE": ["María González", "Juan Pérez", "Carlos López"],
            "MONTO CASTIGO": [2200000, 1500000, 800000],
        }
    )


def test_process_flujo_file_column_mapping():
    """Flujo columns are mapped to the standard FORUM layout."""
    df_flujo = create_sample_flujo()

    result = process_flujo_file(df_flujo, "flujo_julio.xlsx")

    assert list(result.columns) == FORUM_COLUMN_ORDER
    assert all(result["ORIGEN"] == "flujo_julio")
    assert result["RUT"].tolist() == ["20123456", "19513991", "18765432"]
    assert result["NOMBRE CLIENTE"].tolist() == [
        "María González",
        "Juan Pérez",
        "Carlos López",
    ]
    assert result["MONTO CASTIGO"].tolist() == [2200000, 1500000, 800000]
    assert all(result["ETAPA DEMANDA"] == "")
    assert all(result["CIUDAD"] == "")
    assert all(result["Tipo gestión "] == "")


def test_process_flujo_file_negative_rut():
    """Negative RUT values keep their digits instead of becoming empty."""
    df_flujo = pd.DataFrame(
        {
            "CONTRATO": ["F001", "F002", "F003"],
            "RUT": [-20123456, -19513991.0, "-18765432-5"],
            "NOMBRE": ["A", "B", "C"],
            "MONTO CASTIGO": [1, 2, 3],
        }
    )

    result = process_flujo_file(df_flujo, "flujo.xlsx")

    assert result["RUT"].tolist() == ["20123456", "19513991", "18765432"]
    assert all(result["RUT"] != "")


def test_process_flujo_file_origen_keeps_dots_in_filename():
    """ORIGEN keeps the full filename, stripping only the extension."""
    df_flujo = create_sample_flujo()
    filename = "ASIGNACION JUDUCIAL H. MATTHEI BANTOTAL 19-08-2026.xlsx"

    result = process_flujo_file(df_flujo, filename)

    expected = "ASIGNACION JUDUCIAL H. MATTHEI BANTOTAL 19-08-2026"
    assert all(result["ORIGEN"] == expected)


def test_process_single_file_negative_rut():
    """Negative RUT values in the stock path keep their digits."""
    df_castigo = pd.DataFrame(
        {
            "CONTRATO": ["C001", "C002"],
            "RUT": [-19513991, "-20123456-8"],
            "NOMBRE CLIENTE": ["Juan Pérez", "María González"],
            "MONTO CASTIGO": [1500000, 800000],
            "FECHA CASTIGO": ["2026-01-15", "2026-02-20"],
            "CARTERA": ["Comercial", "Consumo"],
        }
    )

    result = process_single_file(df_castigo, "castigo.xlsx", "Castigo")

    assert result["RUT"].tolist() == ["19513991", "20123456"]
    assert all(result["RUT"] != "")


def test_process_single_file_origen_keeps_dots_in_filename():
    """ORIGEN in the stock path also keeps dots, stripping only the extension."""
    df_castigo = pd.DataFrame(
        {
            "CONTRATO": ["C001"],
            "RUT": ["19513991-1"],
            "NOMBRE CLIENTE": ["Juan Pérez"],
            "MONTO CASTIGO": [1500000],
            "FECHA CASTIGO": ["2026-01-15"],
            "CARTERA": ["Comercial"],
        }
    )
    filename = "ASIGNACION JUDUCIAL H. MATTHEI BANTOTAL 19-08-2026.xlsx"

    result = process_single_file(df_castigo, filename, "Castigo")

    expected = "ASIGNACION JUDUCIAL H. MATTHEI BANTOTAL 19-08-2026"
    assert all(result["ORIGEN"] == expected)


def test_process_flujo_file_case_insensitive():
    """Column matching is case-insensitive and accepts alternate names."""
    df_flujo = pd.DataFrame(
        {
            "contrato": ["F001"],
            "rut comp": ["20123456-2"],
            "Nombre_Cliente": ["María González"],
            "Monto Castigo": [2200000],
            "Fecha Castigo": ["2026-07-01"],
            "cartera": ["Consumo"],
        }
    )

    result = process_flujo_file(df_flujo, "flujo.xlsx")

    assert result["CONTRATO"].tolist() == ["F001"]
    assert result["RUT"].tolist() == ["20123456"]
    assert result["NOMBRE CLIENTE"].tolist() == ["María González"]
    assert result["MONTO CASTIGO"].tolist() == [2200000]
    assert result["FECHA CASTIGO"].tolist() == ["2026-07-01"]
    assert result["CARTERA"].tolist() == ["Consumo"]
    assert result["Año castigo"].tolist() == ["2026"]


def test_process_flujo_file_monto_from_saldo_insoluto():
    """MONTO CASTIGO is resolved from 'SALDO INS', 'SALDO INSOLUTO' and variants."""
    base = {
        "CONTRATO": ["F001"],
        "RUT": ["20123456-2"],
        "NOMBRE": ["María González"],
    }

    df_exact = pd.DataFrame({**base, "SALDO INSOLUTO": [2200000]})
    result = process_flujo_file(df_exact, "flujo.xlsx")
    assert result["MONTO CASTIGO"].tolist() == [2200000]

    df_short = pd.DataFrame({**base, "SALDO INS": [2500000]})
    result = process_flujo_file(df_short, "flujo.xlsx")
    assert result["MONTO CASTIGO"].tolist() == [2500000]

    df_variant = pd.DataFrame({**base, "Saldo Insoluto Final": [3300000]})
    result = process_flujo_file(df_variant, "flujo.xlsx")
    assert result["MONTO CASTIGO"].tolist() == [3300000]


def test_process_flujo_file_real_column_names():
    """Flujo files with 'RUT CLIENTE' and 'CONTRATO RS' columns are mapped."""
    df_flujo = pd.DataFrame(
        {
            "CONTRATO RS": ["F001", "F002"],
            "RUT CLIENTE": ["20123456-2", "19513991-1"],
            "NOMBRE": ["María González", "Juan Pérez"],
            "MONTO CASTIGO": [2200000, 1500000],
        }
    )

    result = process_flujo_file(df_flujo, "flujo.xlsx")

    assert result["CONTRATO"].tolist() == ["F001", "F002"]
    assert result["RUT"].tolist() == ["20123456", "19513991"]

    df_stock = create_sample_stock()
    combined_df, discarded_count = process_flujo_forum_data(
        df_stock, df_flujo, "flujo.xlsx"
    )

    # 19513991 is already in the stock, so only one record is accepted
    assert discarded_count == 1
    assert len(combined_df) == len(df_stock) + 1
    assert combined_df.iloc[-1]["CONTRATO"] == "F001"
    assert combined_df.iloc[-1]["RUT"] == "20123456"


def test_process_flujo_forum_data_appends_new_ruts():
    """Flujo records with new RUTs are appended to the stock."""
    df_stock = create_sample_stock()
    df_flujo = create_sample_flujo()

    combined_df, discarded_count = process_flujo_forum_data(
        df_stock, df_flujo, "flujo_julio.xlsx"
    )

    # 2 stock rows + 2 accepted flujo rows (F002 is a duplicate RUT)
    assert len(combined_df) == 4
    assert discarded_count == 1
    assert list(combined_df.columns) == FORUM_COLUMN_ORDER

    appended = combined_df.iloc[2:]
    assert appended["ORIGEN"].tolist() == ["flujo_julio", "flujo_julio"]
    assert appended["RUT"].tolist() == ["20123456", "18765432"]
    assert appended["CONTRATO"].tolist() == ["F001", "F003"]


def test_process_flujo_forum_data_discards_duplicate_ruts():
    """Flujo records whose RUT is already in the stock are discarded."""
    df_stock = create_sample_stock()
    # All flujo RUTs already exist in the stock (with or without dash)
    df_flujo = pd.DataFrame(
        {
            "CONTRATO": ["F001", "F002"],
            "RUT": ["19513991-1", "22345678"],
            "NOMBRE": ["Juan Pérez", "Ana Ruiz"],
            "MONTO CASTIGO": [100, 200],
        }
    )

    combined_df, discarded_count = process_flujo_forum_data(
        df_stock, df_flujo, "flujo.xlsx"
    )

    assert discarded_count == 2
    assert len(combined_df) == len(df_stock)
    assert combined_df["CONTRATO"].tolist() == ["C001", "V001"]


def test_process_flujo_forum_data_keeps_stock_unchanged():
    """The original stock dataframe is not modified."""
    df_stock = create_sample_stock()
    stock_before = df_stock.copy(deep=True)
    df_flujo = create_sample_flujo()

    process_flujo_forum_data(df_stock, df_flujo, "flujo.xlsx")

    pd.testing.assert_frame_equal(df_stock, stock_before)


def test_process_flujo_forum_data_stock_without_rut_column():
    """A stock file without RUT accepts every flujo record."""
    df_stock = create_sample_stock().drop(columns=["RUT"])
    df_flujo = create_sample_flujo()

    combined_df, discarded_count = process_flujo_forum_data(
        df_stock, df_flujo, "flujo.xlsx"
    )

    assert discarded_count == 0
    assert len(combined_df) == len(df_stock) + len(df_flujo)


def test_process_flujo_forum_data_missing_optional_columns():
    """Flujo files without FECHA CASTIGO/CARTERA get empty values."""
    df_stock = create_sample_stock()
    df_flujo = pd.DataFrame(
        {
            "CONTRATO": ["F001"],
            "RUT": ["20123456-2"],
            "NOMBRE": ["María González"],
            "MONTO CASTIGO": [2200000],
        }
    )

    combined_df, discarded_count = process_flujo_forum_data(
        df_stock, df_flujo, "flujo.xlsx"
    )

    assert discarded_count == 0
    new_row = combined_df.iloc[-1]
    assert new_row["FECHA CASTIGO"] == ""
    assert new_row["CARTERA"] == ""
    assert new_row["Año castigo"] == ""
