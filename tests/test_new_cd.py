from unittest.mock import MagicMock, patch

import pandas as pd

from utils_new_cd import (
    calc_week_of_month,
    create_features,
    create_lag_features,
    create_seasonality_features,
    fig_to_streamlit,
    train_and_predict,
)


def test_calc_week_of_month_first_week():
    fechas = pd.Series(pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]))
    result = calc_week_of_month(fechas)
    assert all(result >= 1)
    assert all(result <= 5)


def test_calc_week_of_month_returns_valid_range():
    fechas = pd.Series(pd.date_range("2026-01-01", "2026-12-31"))
    result = calc_week_of_month(fechas)
    assert result.min() >= 1
    assert result.max() <= 5


def test_calc_week_of_month_different_weeks():
    fechas = pd.Series(
        pd.to_datetime(
            [
                "2026-04-01",
                "2026-04-10",
                "2026-04-20",
                "2026-04-28",
            ]
        )
    )
    result = calc_week_of_month(fechas)
    assert len(result) == 4
    assert all(1 <= v <= 5 for v in result)


def test_create_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "B"],
            "fecha_llamada": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
            "countcd": [10, 20, 15],
        }
    )
    result = create_features(df)
    expected_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "day_of_year",
        "is_monday",
        "is_friday",
        "is_weekend",
        "quarter",
        "month_start",
        "month_end",
        "mandante_encoder",
        "week_of_month",
        "fortnight",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_create_features_does_not_modify_original():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"],
            "fecha_llamada": pd.to_datetime(["2026-04-01"]),
            "countcd": [10],
        }
    )
    original_cols = set(df.columns)
    create_features(df)
    assert set(df.columns) == original_cols


def test_create_features_day_of_week_values():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-06", "2026-04-07"]),
            "countcd": [10, 20],
        }
    )
    result = create_features(df)
    assert result.iloc[0]["day_of_week"] == 0
    assert result.iloc[1]["day_of_week"] == 1


def test_create_features_is_monday():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-06", "2026-04-07"]),
            "countcd": [10, 20],
        }
    )
    result = create_features(df)
    assert result.iloc[0]["is_monday"] == 1
    assert result.iloc[1]["is_monday"] == 0


def test_create_features_is_weekend():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-04", "2026-04-06"]),
            "countcd": [10, 20],
        }
    )
    result = create_features(df)
    assert result.iloc[0]["is_weekend"] == 1
    assert result.iloc[1]["is_weekend"] == 0


def test_create_features_fortnight():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-10", "2026-04-20"]),
            "countcd": [10, 20],
        }
    )
    result = create_features(df)
    assert result.iloc[0]["fortnight"] == 1
    assert result.iloc[1]["fortnight"] == 0


def test_create_features_month_start_end():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-02", "2026-04-15", "2026-04-28"]),
            "countcd": [10, 20, 30],
        }
    )
    result = create_features(df)
    assert result.iloc[0]["month_start"] == 1
    assert result.iloc[1]["month_start"] == 0
    assert result.iloc[2]["month_end"] == 1
    assert result.iloc[0]["month_end"] == 0


def test_create_lag_features_adds_lag_columns():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10 + ["B"] * 10,
            "fecha_llamada": pd.date_range("2026-04-01", periods=20),
            "countcd": list(range(10)) + list(range(10, 20)),
        }
    )
    result = create_lag_features(df)
    for lag in [1, 2, 3, 7]:
        assert f"lag_{lag}" in result.columns


def test_create_lag_features_adds_moving_average_columns():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10,
            "fecha_llamada": pd.date_range("2026-04-01", periods=10),
            "countcd": list(range(10)),
        }
    )
    result = create_lag_features(df)
    for window in [7, 14, 30]:
        assert f"moving_avg_{window}" in result.columns
        assert f"moving_std_{window}" in result.columns


def test_create_lag_features_adds_diff_columns():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10,
            "fecha_llamada": pd.date_range("2026-04-01", periods=10),
            "countcd": list(range(10)),
        }
    )
    result = create_lag_features(df)
    assert "diff_1" in result.columns
    assert "diff_7" in result.columns
    assert "pct_change_1" in result.columns
    assert "pct_change_7" in result.columns


def test_create_lag_features_sorted_by_mandante_and_date():
    df = pd.DataFrame(
        {
            "id_mandante": ["B", "A", "B", "A"],
            "fecha_llamada": pd.to_datetime(
                ["2026-04-02", "2026-04-01", "2026-04-01", "2026-04-02"]
            ),
            "countcd": [20, 10, 15, 5],
        }
    )
    result = create_lag_features(df)
    assert list(result["id_mandante"]) == ["A", "A", "B", "B"]


def test_create_lag_features_does_not_modify_original():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 5,
            "fecha_llamada": pd.date_range("2026-04-01", periods=5),
            "countcd": list(range(5)),
        }
    )
    original_cols = set(df.columns)
    create_lag_features(df)
    assert set(df.columns) == original_cols


def test_create_seasonality_features_returns_dict():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "B", "B"],
            "fecha_llamada": pd.to_datetime(
                ["2026-04-01", "2026-04-02", "2026-04-01", "2026-04-03"]
            ),
            "countcd": [10, 20, 15, 25],
        }
    )
    result = create_seasonality_features(df)
    assert isinstance(result, dict)
    assert "A" in result
    assert "B" in result


def test_create_seasonality_features_structure():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10,
            "fecha_llamada": pd.date_range("2026-04-01", periods=10),
            "countcd": list(range(10, 20)),
        }
    )
    result = create_seasonality_features(df)
    est = result["A"]
    assert "global_avg" in est
    assert "dow_factor" in est
    assert "week_factor" in est
    assert "month_factor" in est
    assert "trend_factor" in est


def test_create_seasonality_features_dow_factor_has_7_entries():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 14,
            "fecha_llamada": pd.date_range("2026-04-01", periods=14),
            "countcd": list(range(14)),
        }
    )
    result = create_seasonality_features(df)
    assert len(result["A"]["dow_factor"]) == 7


def test_create_seasonality_features_global_avg():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "A", "A", "A"],
            "fecha_llamada": pd.to_datetime(
                ["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-02", "2026-04-03"]
            ),
        }
    )
    result = create_seasonality_features(df)
    assert result["A"]["global_avg"] == 5.0 / 3


def test_create_seasonality_features_global_avg_daily():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "A"],
            "fecha_llamada": pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-02"]),
        }
    )
    result = create_seasonality_features(df)
    assert result["A"]["global_avg"] == 1.5


def test_train_and_predict_returns_dataframe():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "B", "B"],
            "fecha_llamada": pd.date_range("2026-03-01", periods=4),
            "countcd": [10, 20, 15, 25],
        }
    )
    df_pred, est, err = train_and_predict(df)
    assert isinstance(df_pred, pd.DataFrame)
    assert isinstance(est, dict)


def test_train_and_predict_columns():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 5,
            "fecha_llamada": pd.date_range("2026-03-01", periods=5),
            "countcd": [10, 20, 15, 25, 30],
        }
    )
    df_pred, _, _ = train_and_predict(df)
    expected = [
        "date",
        "day_of_month",
        "weekday_num",
        "month",
        "mandante_id",
        "mandante_name",
        "weekday_name",
        "prediction",
    ]
    for col in expected:
        assert col in df_pred.columns


def test_train_and_predict_no_weekends():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10,
            "fecha_llamada": pd.date_range("2026-03-01", periods=10),
            "countcd": list(range(10, 20)),
        }
    )
    df_pred, _, _ = train_and_predict(df)
    if len(df_pred) > 0:
        assert all(df_pred["weekday_num"] < 5)


def test_train_and_predict_prediction_non_negative():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 10,
            "fecha_llamada": pd.date_range("2026-03-01", periods=10),
            "countcd": [10, 20, 15, 25, 30, 5, 12, 18, 22, 8],
        }
    )
    df_pred, _, _ = train_and_predict(df)
    if len(df_pred) > 0:
        assert all(df_pred["prediction"] >= 0)


def test_train_and_predict_multiple_mandantes():
    df = pd.DataFrame(
        {
            "id_mandante": ["A", "A", "B", "B", "C", "C"],
            "fecha_llamada": pd.to_datetime(
                [
                    "2026-03-01",
                    "2026-03-02",
                    "2026-03-01",
                    "2026-03-02",
                    "2026-03-01",
                    "2026-03-02",
                ]
            ),
            "countcd": [10, 20, 15, 25, 5, 8],
        }
    )
    df_pred, _, _ = train_and_predict(df)
    mandantes_pred = set(df_pred["mandante_name"].unique())
    assert mandantes_pred == {"A", "B", "C"}


def test_train_and_predict_error_is_none():
    df = pd.DataFrame(
        {
            "id_mandante": ["A"] * 5,
            "fecha_llamada": pd.date_range("2026-03-01", periods=5),
            "countcd": [10, 20, 15, 25, 30],
        }
    )
    _, _, err = train_and_predict(df)
    assert err is None


@patch("utils_new_cd.plt")
def test_fig_to_streamlit_calls_savefig(mock_plt):
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch("utils_new_cd.io.BytesIO", return_value=mock_buf):
        fig_to_streamlit(mock_fig, mock_st)
    mock_fig.savefig.assert_called_once()


@patch("utils_new_cd.plt")
def test_fig_to_streamlit_closes_figure(mock_plt):
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch("utils_new_cd.io.BytesIO", return_value=mock_buf):
        fig_to_streamlit(mock_fig, mock_st)
    mock_plt.close.assert_called_once_with(mock_fig)


def test_fig_to_streamlit_calls_st_image():
    mock_fig = MagicMock()
    mock_st = MagicMock()
    mock_buf = MagicMock()
    mock_buf.seek = MagicMock()
    with patch("utils_new_cd.io.BytesIO", return_value=mock_buf):
        with patch("utils_new_cd.plt"):
            fig_to_streamlit(mock_fig, mock_st)
    mock_st.image.assert_called_once()
