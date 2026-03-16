"""Utility functions for the Excel transformer application."""

import re
from typing import List, Optional, Tuple, Dict


def normalize_column_name(col_name) -> str:
    """Normalize column name for comparison: lowercase and remove underscores.
    
    Args:
        col_name: Column name to normalize
        
    Returns:
        Normalized column name (lowercase, underscores removed)
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return re.sub(r'_+', '', col_name.lower())


def find_matching_column(df_columns, target_col: str) -> Optional[str]:
    """Find actual column name that matches target column (case-insensitive, underscore-insensitive).
    
    Args:
        df_columns: Column names from the dataframe (Index or list-like)
        target_col: Target column name to match
        
    Returns:
        Actual column name from df_columns that matches target_col, or None if not found
    """
    normalized_target = normalize_column_name(target_col)
    for col in df_columns:
        if normalize_column_name(col) == normalized_target:
            return col
    return None


def validate_required_columns(df_columns, 
                            columns_to_keep: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Check if all required columns exist and create mapping.
    
    Args:
        df_columns: Column names from the dataframe (Index or list-like)
        columns_to_keep: List of expected column names
        
    Returns:
        Tuple of (missing_columns, column_mapping)
        missing_columns: List of expected columns not found in df_columns
        column_mapping: Dict mapping expected column names to actual column names
    """
    missing_columns = []
    column_mapping = {}  # Maps expected column name to actual column name in file
    
    for expected_col in columns_to_keep:
        actual_col = find_matching_column(df_columns, expected_col)
        if actual_col is None:
            missing_columns.append(expected_col)
        else:
            column_mapping[expected_col] = actual_col
    
    return missing_columns, column_mapping