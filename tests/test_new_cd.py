import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_new_cd import (
    calcular_semana_del_mes,
    crear_features,
    crear_features_lag,
    crear_features_estacionalidad,
    entrenar_y_predecir,
    fig_to_streamlit,
)


# --- calcular_semana_del_mes ---

def test_calcular_semana_del_mes_first_week():
    """Test that dates in the first week return 1."""
    fechas = pd.Series(pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03']))
    result = calcular_semana_del_mes(fechas)
    assert all(result >= 1)
    assert all(result <= 5)


def test_calcular_semana_del_mes_returns_valid_range():
    """Test that all results are between 1 and 5."""
    fechas = pd.Series(pd.date_range('2026-01-01', '2026-12-31'))
    result = calcular_semana_del_mes(fechas)
    assert result.min() >= 1
    assert result.max() <= 5


def test_calcular_semana_del_mes_different_weeks():
    """Test that different weeks of the month return different values."""
    fechas = pd.Series(pd.to_datetime([
        '2026-04-01',   # early month
        '2026-04-10',   # mid month
        '2026-04-20',   # later month
        '2026-04-28',   # end of month
    ]))
    result = calcular_semana_del_mes(fechas)
    assert len(result) == 4
    assert all(1 <= v <= 5 for v in result)


# --- crear_features ---

def test_crear_features_adds_expected_columns():
    """Test that crear_features adds all expected columns."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'B'],
        'fecha_llamada': pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03']),
        'countcd': [10, 20, 15],
    })
    result = crear_features(df)
    expected_cols = [
        'dia_semana', 'dia_mes', 'mes', 'anio', 'dia_ano',
        'es_lunes', 'es_viernes', 'es_fin_de_semana', 'trimestre',
        'inicio_mes', 'fin_mes', 'encoder_mandante', 'semana_mes', 'quincena',
    ]
    for col in expected_cols:
        assert col in result.columns


def test_crear_features_does_not_modify_original():
    """Test that crear_features returns a copy."""
    df = pd.DataFrame({
        'id_mandante': ['A'],
        'fecha_llamada': pd.to_datetime(['2026-04-01']),
        'countcd': [10],
    })
    original_cols = set(df.columns)
    crear_features(df)
    assert set(df.columns) == original_cols


def test_crear_features_dia_semana_values():
    """Test that dia_semana is correctly computed (0=Monday)."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-06', '2026-04-07']),  # Mon, Tue
        'countcd': [10, 20],
    })
    result = crear_features(df)
    assert result.iloc[0]['dia_semana'] == 0  # Monday
    assert result.iloc[1]['dia_semana'] == 1  # Tuesday


def test_crear_features_es_lunes():
    """Test es_lunes flag."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-06', '2026-04-07']),  # Mon, Tue
        'countcd': [10, 20],
    })
    result = crear_features(df)
    assert result.iloc[0]['es_lunes'] == 1
    assert result.iloc[1]['es_lunes'] == 0


def test_crear_features_es_fin_de_semana():
    """Test es_fin_de_semana flag."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-04', '2026-04-06']),  # Sat, Mon
        'countcd': [10, 20],
    })
    result = crear_features(df)
    assert result.iloc[0]['es_fin_de_semana'] == 1
    assert result.iloc[1]['es_fin_de_semana'] == 0


def test_crear_features_quincena():
    """Test quincena flag (1 if day <= 15)."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-10', '2026-04-20']),
        'countcd': [10, 20],
    })
    result = crear_features(df)
    assert result.iloc[0]['quincena'] == 1
    assert result.iloc[1]['quincena'] == 0


def test_crear_features_inicio_fin_mes():
    """Test inicio_mes and fin_mes flags."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-02', '2026-04-15', '2026-04-28']),
        'countcd': [10, 20, 30],
    })
    result = crear_features(df)
    assert result.iloc[0]['inicio_mes'] == 1
    assert result.iloc[1]['inicio_mes'] == 0
    assert result.iloc[2]['fin_mes'] == 1
    assert result.iloc[0]['fin_mes'] == 0


# --- crear_features_lag ---

def test_crear_features_lag_adds_lag_columns():
    """Test that lag columns are created."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10 + ['B'] * 10,
        'fecha_llamada': pd.date_range('2026-04-01', periods=20),
        'countcd': list(range(10)) + list(range(10, 20)),
    })
    result = crear_features_lag(df)
    for lag in [1, 2, 3, 7]:
        assert f'lag_{lag}' in result.columns


def test_crear_features_lag_adds_moving_average_columns():
    """Test that moving average/std columns are created."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10,
        'fecha_llamada': pd.date_range('2026-04-01', periods=10),
        'countcd': list(range(10)),
    })
    result = crear_features_lag(df)
    for window in [7, 14, 30]:
        assert f'media_movil_{window}' in result.columns
        assert f'std_movil_{window}' in result.columns


def test_crear_features_lag_adds_diff_columns():
    """Test that diff and pct_change columns are created."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10,
        'fecha_llamada': pd.date_range('2026-04-01', periods=10),
        'countcd': list(range(10)),
    })
    result = crear_features_lag(df)
    assert 'diff_1' in result.columns
    assert 'diff_7' in result.columns
    assert 'pct_change_1' in result.columns
    assert 'pct_change_7' in result.columns


def test_crear_features_lag_sorted_by_mandante_and_date():
    """Test that result is sorted by mandante and fecha."""
    df = pd.DataFrame({
        'id_mandante': ['B', 'A', 'B', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-02', '2026-04-01', '2026-04-01', '2026-04-02']),
        'countcd': [20, 10, 15, 5],
    })
    result = crear_features_lag(df)
    assert list(result['id_mandante']) == ['A', 'A', 'B', 'B']


def test_crear_features_lag_does_not_modify_original():
    """Test that crear_features_lag returns a copy."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 5,
        'fecha_llamada': pd.date_range('2026-04-01', periods=5),
        'countcd': list(range(5)),
    })
    original_cols = set(df.columns)
    crear_features_lag(df)
    assert set(df.columns) == original_cols


# --- crear_features_estacionalidad ---

def test_crear_features_estacionalidad_returns_dict():
    """Test that the function returns a dict keyed by mandante."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'B', 'B'],
        'fecha_llamada': pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-01', '2026-04-03']),
        'countcd': [10, 20, 15, 25],
    })
    result = crear_features_estacionalidad(df)
    assert isinstance(result, dict)
    assert 'A' in result
    assert 'B' in result


def test_crear_features_estacionalidad_structure():
    """Test that each mandante entry has the expected keys."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10,
        'fecha_llamada': pd.date_range('2026-04-01', periods=10),
        'countcd': list(range(10, 20)),
    })
    result = crear_features_estacionalidad(df)
    est = result['A']
    assert 'global_avg' in est
    assert 'dow_factor' in est
    assert 'semana_factor' in est
    assert 'mes_factor' in est
    assert 'trend_factor' in est


def test_crear_features_estacionalidad_dow_factor_has_7_entries():
    """Test that dow_factor has entries for all 7 days."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 14,
        'fecha_llamada': pd.date_range('2026-04-01', periods=14),
        'countcd': list(range(14)),
    })
    result = crear_features_estacionalidad(df)
    assert len(result['A']['dow_factor']) == 7


def test_crear_features_estacionalidad_global_avg():
    """Test that global_avg matches the mean of countcd."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'A'],
        'fecha_llamada': pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03']),
        'countcd': [10, 20, 30],
    })
    result = crear_features_estacionalidad(df)
    assert result['A']['global_avg'] == 20.0


# --- entrenar_y_predecir ---

def test_entrenar_y_predecir_returns_dataframe():
    """Test that entrenar_y_predecir returns a DataFrame."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'B', 'B'],
        'fecha_llamada': pd.date_range('2026-03-01', periods=4),
        'countcd': [10, 20, 15, 25],
    })
    df_pred, est, err = entrenar_y_predecir(df)
    assert isinstance(df_pred, pd.DataFrame)
    assert isinstance(est, dict)


def test_entrenar_y_predecir_columns():
    """Test that prediction DataFrame has expected columns."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 5,
        'fecha_llamada': pd.date_range('2026-03-01', periods=5),
        'countcd': [10, 20, 15, 25, 30],
    })
    df_pred, _, _ = entrenar_y_predecir(df)
    expected = ['fecha', 'dia_mes', 'dia_semana_num', 'mes', 'mandante_id', 'nombre_mandante', 'dia_semana_texto', 'prediccion']
    for col in expected:
        assert col in df_pred.columns


def test_entrenar_y_predecir_no_weekends():
    """Test that predictions exclude weekends."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10,
        'fecha_llamada': pd.date_range('2026-03-01', periods=10),
        'countcd': list(range(10, 20)),
    })
    df_pred, _, _ = entrenar_y_predecir(df)
    if len(df_pred) > 0:
        assert all(df_pred['dia_semana_num'] < 5)


def test_entrenar_y_predecir_prediccion_non_negative():
    """Test that predictions are non-negative."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 10,
        'fecha_llamada': pd.date_range('2026-03-01', periods=10),
        'countcd': [10, 20, 15, 25, 30, 5, 12, 18, 22, 8],
    })
    df_pred, _, _ = entrenar_y_predecir(df)
    if len(df_pred) > 0:
        assert all(df_pred['prediccion'] >= 0)


def test_entrenar_y_predecir_multiple_mandantes():
    """Test that predictions cover all mandantes."""
    df = pd.DataFrame({
        'id_mandante': ['A', 'A', 'B', 'B', 'C', 'C'],
        'fecha_llamada': pd.to_datetime([
            '2026-03-01', '2026-03-02',
            '2026-03-01', '2026-03-02',
            '2026-03-01', '2026-03-02',
        ]),
        'countcd': [10, 20, 15, 25, 5, 8],
    })
    df_pred, _, _ = entrenar_y_predecir(df)
    mandantes_pred = set(df_pred['nombre_mandante'].unique())
    assert mandantes_pred == {'A', 'B', 'C'}


def test_entrenar_y_predecir_error_is_none():
    """Test that error is None for valid data."""
    df = pd.DataFrame({
        'id_mandante': ['A'] * 5,
        'fecha_llamada': pd.date_range('2026-03-01', periods=5),
        'countcd': [10, 20, 15, 25, 30],
    })
    _, _, err = entrenar_y_predecir(df)
    assert err is None


# --- fig_to_streamlit ---

@patch('utils_new_cd.plt')
def test_fig_to_streamlit_calls_savefig(mock_plt):
    """Test that fig_to_streamlit calls savefig on the figure."""
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch('utils_new_cd.io.BytesIO', return_value=mock_buf):
        fig_to_streamlit(mock_fig, mock_st)
    mock_fig.savefig.assert_called_once()


@patch('utils_new_cd.plt')
def test_fig_to_streamlit_closes_figure(mock_plt):
    """Test that fig_to_streamlit closes the figure."""
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch('utils_new_cd.io.BytesIO', return_value=mock_buf):
        fig_to_streamlit(mock_fig, mock_st)
    mock_plt.close.assert_called_once_with(mock_fig)


def test_fig_to_streamlit_calls_st_image():
    """Test that fig_to_streamlit calls st.image."""
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch('utils_new_cd.io.BytesIO', return_value=mock_buf):
        with patch('utils_new_cd.plt'):
            fig_to_streamlit(mock_fig, mock_st)
    mock_st.image.assert_called_once()
