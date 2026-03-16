import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from utils import normalize_column_name, find_matching_column, validate_required_columns


def test_normalize_column_name():
    """Test the normalize_column_name function."""
    # Test basic normalization
    assert normalize_column_name("RUT") == "rut"
    assert normalize_column_name("nombre_completo") == "nombrecompleto"
    assert normalize_column_name("SALDO_CAPITAL") == "saldocapital"
    
    # Test with multiple underscores
    assert normalize_column_name("SALDO__CAPITAL___") == "saldocapital"
    
    # Test with numbers and special characters (only underscores removed)
    assert normalize_column_name("COLUMN_123_TEST!") == "column123test!"
    
    # Test with non-string input
    assert normalize_column_name(123) == "123"
    assert normalize_column_name(None) == "none"


def test_find_matching_column():
    """Test the find_matching_column function."""
    # Test exact match
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'rut') == 'rut'
    
    # Test case insensitive match
    columns = pd.Index(['RUT', 'Nombre', 'Edad'])
    assert find_matching_column(columns, 'rut') == 'RUT'
    
    # Test underscore insensitive match
    columns = pd.Index(['nombre_completo', 'edad'])
    assert find_matching_column(columns, 'nombrecompleto') == 'nombre_completo'
    
    # Test combined case and underscore insensitive (underscores removed)
    columns = pd.Index(['NOMBRE_COMPLETO_CLIENTE', 'edad'])
    assert find_matching_column(columns, 'nombrecompletocliente') == 'NOMBRE_COMPLETO_CLIENTE'
    
    # Test not found
    columns = pd.Index(['rut', 'nombre', 'edad'])
    assert find_matching_column(columns, 'apellido') is None


def test_validate_required_columns():
    """Test the validate_required_columns function."""
    # Test all columns present
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
    
    # Test some columns missing
    columns = pd.Index(['rut', 'nombre_completo'])
    columns_to_keep = ['rut', 'dv', 'nombre_completo', 'saldo_capital']
    missing, mapping = validate_required_columns(columns, columns_to_keep)
    assert set(missing) == {'dv', 'saldo_capital'}
    assert mapping == {
        'rut': 'rut',
        'nombre_completo': 'nombre_completo'
    }
    
    # Test case and underscore insensitive matching
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