import calendar
import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def calc_week_of_month(dates: pd.Series) -> pd.Series:
    """Calculate the week of the month (1-5) based on the date."""
    days = dates.dt.day
    dow = dates.dt.weekday
    first_dow_of_month = (dates - pd.to_timedelta(days - 1, unit="D")).dt.weekday
    offset = (dow - first_dow_of_month + 7) % 7
    return ((days - 1 + offset) // 7 + 1).clip(1, 5)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and categorical features from the source DataFrame."""
    df = df.copy()
    df["day_of_week"] = df["fecha_llamada"].dt.dayofweek
    df["day_of_month"] = df["fecha_llamada"].dt.day
    df["month"] = df["fecha_llamada"].dt.month
    df["year"] = df["fecha_llamada"].dt.year
    df["day_of_year"] = df["fecha_llamada"].dt.dayofyear
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["fecha_llamada"].dt.quarter
    df["month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["month_end"] = (df["day_of_month"] >= 25).astype(int)
    df["mandante_encoder"] = df["id_mandante"].astype("category").cat.codes
    df["week_of_month"] = calc_week_of_month(df["fecha_llamada"])
    df["fortnight"] = (df["fecha_llamada"].dt.day <= 15).astype(int)

    if "hora_llamada" in df.columns:
        df["hour"] = pd.to_numeric(df["hora_llamada"], errors="coerce")
        df["is_morning"] = (df["hour"] < 12).astype(int)
        df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
        df["is_evening"] = (df["hour"] >= 18).astype(int)

    if "grupo" in df.columns:
        df["group_encoder"] = df["grupo"].astype("category").cat.codes

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and diff features grouped by mandante."""
    df = df.copy()
    df = df.sort_values(["id_mandante", "fecha_llamada"]).reset_index(drop=True)
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df.groupby("id_mandante")["countcd"].shift(lag)
    for window in [7, 14, 30]:
        df[f"moving_avg_{window}"] = df.groupby("id_mandante")["countcd"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"moving_std_{window}"] = df.groupby("id_mandante")["countcd"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    df["diff_1"] = df.groupby("id_mandante")["countcd"].diff(1)
    df["diff_7"] = df.groupby("id_mandante")["countcd"].diff(7)
    df["pct_change_1"] = df.groupby("id_mandante")["countcd"].pct_change(1)
    df["pct_change_7"] = df.groupby("id_mandante")["countcd"].pct_change(7)
    return df


def create_seasonality_features(df: pd.DataFrame) -> dict:
    """Compute seasonality factors (day-of-week, week-of-month, month) per mandante."""
    daily = (
        df.groupby(["id_mandante", "fecha_llamada"]).size().reset_index(name="countcd")
    )

    seasonality = {}
    for mandante in daily["id_mandante"].unique():
        data = daily[daily["id_mandante"] == mandante].copy()
        data = data.sort_values("fecha_llamada").reset_index(drop=True)
        data["day_of_week"] = data["fecha_llamada"].dt.dayofweek
        data["week_of_month"] = calc_week_of_month(data["fecha_llamada"])
        data["month"] = data["fecha_llamada"].dt.month
        data["year"] = data["fecha_llamada"].dt.year

        global_avg = data["countcd"].mean()

        seasonal_dow = data.groupby("day_of_week")["countcd"].mean()
        dow_factor = {}
        for dow in range(7):
            if dow in seasonal_dow.index:
                dow_factor[dow] = seasonal_dow[dow] / global_avg
            else:
                dow_factor[dow] = 0.0

        seasonal_week = data.groupby("week_of_month")["countcd"].mean()
        week_factor = seasonal_week / global_avg

        seasonal_month = data.groupby("month")["countcd"].mean()
        month_factor = seasonal_month / global_avg

        monthly_avg = data.groupby(["year", "month"])["countcd"].mean()
        recent_months = monthly_avg.tail(6)
        trend_factor = (
            recent_months.mean() / global_avg if len(recent_months) > 0 else 1.0
        )

        seasonality[mandante] = {
            "global_avg": global_avg,
            "dow_factor": dow_factor,
            "week_factor": week_factor.to_dict(),
            "month_factor": month_factor.to_dict(),
            "trend_factor": trend_factor,
        }

    return seasonality


def train_and_predict(df: pd.DataFrame) -> tuple:
    """Generate predictions for the current month using seasonal factors."""
    today = datetime.now()
    year = today.year
    month = today.month
    days_in_month = calendar.monthrange(year, month)[1]

    mandante_map = {
        nombre: idx for idx, nombre in enumerate(sorted(df["id_mandante"].unique()))
    }

    seasonality = create_seasonality_features(df)

    WEEKDAY_NAMES = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    prediction_rows = []

    for mandante_name, mandante_id in mandante_map.items():
        est = seasonality[mandante_name]
        global_avg = est["global_avg"]
        dow_factor = est["dow_factor"]
        week_factor = est["week_factor"]
        month_factor = est["month_factor"]
        trend = est["trend_factor"]

        for day in range(1, days_in_month + 1):
            date = datetime(year, month, day)
            weekday = date.weekday()

            if weekday >= 5:
                continue

            first_day_of_month = datetime(year, month, 1)
            week_of_month = min(
                (day - 1 + (weekday - first_day_of_month.weekday() + 7) % 7) // 7 + 1, 5
            )

            base = global_avg * trend
            factor_dow = dow_factor.get(weekday, 1.0)
            factor_week = week_factor.get(week_of_month, 1.0)
            factor_month = month_factor.get(month, 1.0)

            prediction = base * factor_dow * factor_week * factor_month
            prediction = max(prediction, 0)

            prediction_rows.append(
                {
                    "date": date,
                    "day_of_month": day,
                    "weekday_num": weekday,
                    "month": month,
                    "mandante_id": mandante_id,
                    "mandante_name": mandante_name,
                    "weekday_name": WEEKDAY_NAMES[weekday],
                    "prediction": round(prediction, 1),
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)
    if len(prediction_df) > 0:
        prediction_df["date"] = prediction_df["date"].dt.strftime("%Y-%m-%d")

    return prediction_df, seasonality, None


def fig_to_streamlit(fig, st):
    """Render a matplotlib figure as a Streamlit image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)
