#!/usr/bin/env python3
"""
Test script to verify the forum data processing functionality.
This creates sample data similar to what would be uploaded and tests the processing logic.
"""

import os
import sys

import pandas as pd

# Add the current directory to path so we can import from pages.asig
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import the processing functions from asig.py
from pages.asig import process_forum_data, process_single_file


def create_sample_castigo_data():
    """Create sample castigo data similar to what would be in an uploaded file."""
    data = {
        "CONTRATO": ["C001", "C002", "C003"],
        "RUT": ["19513991-1", "20123456-2", "18765432-K"],
        "NOMBRE CLIENTE": ["Juan Pérez", "María González", "Carlos López"],
        "MONTO CASIIGO": [1500000, 2200000, 800000],
        "FECHA CASTIGO": ["2026-01-15", "2026-02-20", "2026-03-10"],
        "CARTERA": ["Comercial", "Consumo", "Hipotecario"],
    }
    return pd.DataFrame(data)


def create_sample_vigente_data():
    """Sample vigente data similar to what would be in an uploaded file."""
    data = {
        "CONTRATO": ["V001", "V002"],
        "RUT": ["22345678-9", "19876543-2"],
        "NOMBRE CLIENTE": ["Ana Ruiz", "Pedro Sánchez"],
        "MONTO CASIIGO": [0, 0],  # Vigente accounts typically have 0 castigo amount
        "FECHA CASTIGO": ["", ""],  # No castigo date for vigente accounts
        "CARTERA": ["Comercial", "Consumo"],
    }
    return pd.DataFrame(data)


def test_process_single_file():
    """Test the process_single_file function."""
    print("Testing process_single_file function...")

    # Test with castigo data
    castigo_df = create_sample_castigo_data()
    processed_castigo = process_single_file(
        castigo_df, "castigo_ejemplo.xlsx", "Castigo"
    )

    print("Original castigo columns:", list(castigo_df.columns))
    print("Processed castigo columns:", list(processed_castigo.columns))
    print("Processed castigo data:")
    print(
        processed_castigo[
            [
                "ORIGEN",
                "CONTRATO",
                "RUT",
                "NOMBRE CLIENTE",
                "MONTO CASIIGO",
                "FECHA CASTIGO",
                "CARTERA",
            ]
        ]
    )
    print()

    # Check that RUT was cleaned (dash removed)
    assert all(
        "-" not in rut for rut in processed_castigo["RUT"]
    ), "RUT should not contain dashes"
    assert (
        processed_castigo["RUT"].iloc[0] == "19513991"
    ), f"Expected '19513991', got '{processed_castigo['RUT'].iloc[0]}'"

    # Check that ORIGEN is set correctly
    assert all(
        origen == "castigo_ejemplo" for origen in processed_castigo["ORIGEN"]
    ), "ORIGEN should be filename without extension"

    # Test with vigente data
    vigente_df = create_sample_vigente_data()
    processed_vigente = process_single_file(
        vigente_df, "vigente_ejemplo.xlsx", "Vigente"
    )

    print("Processed vigente data:")
    print(
        processed_vigente[
            [
                "ORIGEN",
                "CONTRATO",
                "RUT",
                "NOMBRE CLIENTE",
                "MONTO CASIIGO",
                "FECHA CASTIGO",
                "CARTERA",
            ]
        ]
    )
    print()

    # Check that RUT was cleaned
    assert all(
        "-" not in rut for rut in processed_vigente["RUT"]
    ), "RUT should not contain dashes"
    assert (
        processed_vigente["ORIGEN"].iloc[0] == "vigente_ejemplo"
    ), "ORIGEN should be filename without extension"


def test_process_forum_data():
    """Test the complete process_forum_data function."""
    print("Testing process_forum_data function...")

    castigo_df = create_sample_castigo_data()
    vigente_df = create_sample_vigente_data()

    combined_df = process_forum_data(
        castigo_df, vigente_df, "castigo_data.xlsx", "vigente_data.xlsx"
    )

    print("Combined dataframe shape:", combined_df.shape)
    print("Combined dataframe columns:", list(combined_df.columns))
    print("Combined dataframe:")
    print(combined_df)
    print()

    # Check that we have the right number of rows (3 castigo + 2 vigente = 5)
    assert len(combined_df) == 5, f"Expected 5 rows, got {len(combined_df)}"

    # Check that ORIGEN column has the correct values
    origen_values = combined_df["ORIGEN"].tolist()
    expected_origen = ["castigo_data"] * 3 + ["vigente_data"] * 2
    assert origen_values == expected_origen, f"ORIGEN values incorrect: {origen_values}"

    # Check that RUT column is cleaned (no dashes)
    assert all(
        "-" not in rut for rut in combined_df["RUT"]
    ), "RUT column should not contain dashes"

    # Check that empty columns are present and empty
    assert all(
        etapa == "" for etapa in combined_df["ETAPA DEMANDA"]
    ), "ETAPA DEMANDA should be empty"
    assert all(
        ciudad == "" for ciudad in combined_df["CIUDAD"]
    ), "CIUDAD should be empty"
    assert all(
        tipo_gestion == "" for tipo_gestion in combined_df["Tipo gestión"]
    ), "Tipo gestión should be empty"

    # Check column order
    expected_order = [
        "ORIGEN",
        "CONTRATO",
        "RUT",
        "NOMBRE CLIENTE",
        "MONTO CASIIGO",
        "ETAPA DEMANDA",
        "FECHA CASTIGO",
        "CIUDAD",
        "CARTERA",
        "Tipo gestión",
    ]
    actual_order = list(combined_df.columns)
    assert (
        actual_order == expected_order
    ), f"Column order incorrect. Expected: {expected_order}, Got: {actual_order}"

    print("All tests passed!")


if __name__ == "__main__":
    print("Running forum data processing tests...\n")
    test_process_single_file()
    test_process_forum_data()
    print("All forum processing tests completed successfully!")
