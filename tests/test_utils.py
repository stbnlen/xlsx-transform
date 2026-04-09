import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from utils import normalize_column_name, find_matching_column, validate_required_columns


def test_normalize_column_name_basic():
    """Test basic normalization functionality."""
    assert normalize_column_name("RUT") == "rut"
    assert normalize_column_name("nombre_completo") == "nombrecompleto"
    assert normalize_column_name("SALDO_CAPITAL") == "saldocapital"


def test_normalize_column_name_underscores():
    """Test handling of multiple underscores."""
    assert normalize_column_name("SALDO__CAPITAL___") == "saldocapital"
    assert normalize_column_name("___TEST___") == "test"
    assert normalize_column_name("____") == ""


def test_normalize_column_name_special_chars():
    """Test handling of numbers and special characters."""
    assert normalize_column_name("COLUMN_123_TEST!") == "column123test!"
    assert normalize_column_name("TEST@#$%^&*()") == "test@#$%^&*()"
    assert normalize_column_name("TEST123") == "test123"


def test_normalize_column_name_edge_cases():
    """Test edge cases and non-string inputs."""
    assert normalize_column_name("") == ""
    assert normalize_column_name(123) == "123"
    assert normalize_column_name(None) == "none"
    assert normalize_column_name([]) == "[]"
    assert normalize_column_name({}) == "{}"
    assert normalize_column_name(True) == "true"


def test_find_matching_column_exact():
    """Test exact matching functionality."""
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'rut') == 'rut'
    assert find_matching_column(columns, 'nombre') == 'nombre'
    assert find_matching_column(columns, 'edad') == 'edad'


def test_find_matching_column_case_insensitive():
    """Test case insensitive matching."""
    columns = pd.Index(['RUT', 'Nombre', 'Edad'])
    assert find_matching_column(columns, 'rut') == 'RUT'
    assert find_matching_column(columns, 'NOMBRE') == 'Nombre'
    assert find_matching_column(columns, 'EDAD') == 'Edad'


def test_find_matching_column_underscore_insensitive():
    """Test underscore insensitive matching."""
    columns = pd.Index(['nombre_completo', 'edad'])
    assert find_matching_column(columns, 'nombrecompleto') == 'nombre_completo'
    assert find_matching_column(columns, 'NOMBRE_COMPLETO') == 'nombre_completo'
    assert find_matching_column(columns, 'nombre_completo') == 'nombre_completo'


def test_find_matching_column_combined():
    """Test combined case and underscore insensitive matching."""
    columns = pd.Index(['NOMBRE_COMPLETO_CLIENTE', 'edad'])
    assert find_matching_column(columns, 'nombrecompletocliente') == 'NOMBRE_COMPLETO_CLIENTE'
    assert find_matching_column(columns, 'NOMBRE_COMPLETO_CLIENTE') == 'NOMBRE_COMPLETO_CLIENTE'


def test_find_matching_column_not_found():
    """Test behavior when column is not found."""
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'apellido') is None
    assert find_matching_column(columns, '') is None
    assert find_matching_column(columns, 'nonexistent') is None


def test_find_matching_column_edge_cases():
    """Test edge cases for find_matching_column."""
    # Empty columns
    assert find_matching_column(pd.Index([]), 'rut') is None
    
    # Duplicate matches (should return first)
    columns = pd.Index(['rut', 'RUT', 'rut'])
    assert find_matching_column(columns, 'rut') == 'rut'  # First match


def test_validate_required_columns_all_present():
    """Test when all required columns are present."""
    columns = pd.Index(['rut', 'dv', 'nombre_completo', 'saldo_capital'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert missing == []
    assert mapping == {
        'rut': 'rut',
        'dv': 'dv',
        'nombre_completo': 'nombre_completo',
        'saldo_capital': 'saldo_capital'
    }


def test_validate_required_columns_some_missing():
    """Test when some required columns are missing."""
    columns = pd.Index(['rut', 'nombre_completo'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert set(missing) == {'dv', 'saldo_capital'}
    assert mapping == {
        'rut': 'rut',
        'nombre_completo': 'nombre_completo'
    }


def test_validate_required_columns_case_insensitive():
    """Test case and underscore insensitive matching."""
    columns = pd.Index(['RUT', 'DV', 'NOMBRE_COMPLETO', 'SALDO_CAPITAL'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert missing == []
    assert mapping == {
        'rut': 'RUT',
        'dv': 'DV',
        'nombre_completo': 'NOMBRE_COMPLETO',
        'saldo_capital': 'SALDO_CAPITAL'
    }


def test_validate_required_columns_edge_cases():
    """Test edge cases for validate_required_columns."""
    # Empty inputs
    missing, mapping = validate_required_columns(pd.Index([]), [])
    assert missing == []
    assert mapping == {}
    
    # No columns to keep
    missing, mapping = validate_required_columns(pd.Index(['rut', 'dv']), [])
    assert missing == []
    assert mapping == {}
    
    # No matching columns
    missing, mapping = validate_required_columns(pd.Index(['rut', 'dv']), ['nombre', 'edad'])
    assert set(missing) == {'nombre', 'edad'}
    assert mapping == {}


# Parametrized tests for normalize_column_name
@pytest.mark.parametrize("input_val,expected", [
    ("RUT", "rut"),
    ("nombre_completo", "nombrecompleto"),
    ("SALDO_CAPITAL", "saldocapital"),
    ("", ""),
    ("___", ""),
    ("A_B_C", "abc"),
    ("123_TEST", "123test"),
    ("TEST!@#", "test!@#"),
    ("MiXeD_CaSe", "mixedcase"),
])
def test_normalize_column_name_parametrized(input_val, expected):
    """Parametrized test for normalize_column_name."""
    assert normalize_column_name(input_val) == expected


# Parametrized tests for find_matching_column
@pytest.mark.parametrize("columns,search,expected", [
    (pd.Index(['rut', 'nombre', 'edad']), 'rut', 'rut'),
    (pd.Index(['RUT', 'Nombre', 'Edad']), 'rut', 'RUT'),
    (pd.Index(['nombre_completo', 'edad']), 'nombrecompleto', 'nombre_completo'),
    (pd.Index(['NOMBRE_COMPLETO_CLIENTE', 'edad']), 'nombrecompletocliente', 'NOMBRE_COMPLETO_CLIENTE'),
    (pd.Index(['rut', 'nombre', 'edad']), 'apellido', None),
])
def test_find_matching_column_parametrized(columns, search, expected):
    """Parametrized test for find_matching_column."""
    assert find_matching_column(columns, search) == expected


# Parametrized tests for validate_required_columns
@pytest.mark.parametrize("columns,columns_to_keep,expected_missing,expected_mapping_keys", [
    (pd.Index(['rut', 'dv', 'nombre_completo', 'saldo_capital']), 
     ['rut', 'dv', 'nombre_completo', 'saldo_capital'], 
     [], 
     ['rut', 'dv', 'nombre_completo', 'saldo_capital']),
    (pd.Index(['rut', 'nombre_completo']), 
     ['rut', 'dv', 'nombre_completo', 'saldo_capital'], 
     ['dv', 'saldo_capital'], 
     ['rut', 'nombre_completo']),
    (pd.Index(['RUT', 'DV', 'NOMBRE_COMPLETO', 'SALDO_CAPITAL']), 
     ['rut', 'dv', 'nombre_completo', 'saldo_capital'], 
     [], 
     ['rut', 'dv', 'nombre_completo', 'saldo_capital']),
])
def test_validate_required_columns_parametrized(columns, columns_to_keep, expected_missing, expected_mapping_keys):
    """Parametrized test for validate_required_columns."""
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert set(missing) == set(expected_missing)
    assert set(mapping.keys()) == set(expected_mapping_keys)