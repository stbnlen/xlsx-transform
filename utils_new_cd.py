import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import calendar


def calcular_semana_del_mes(fechas: pd.Series) -> pd.Series:
    """Calcula la semana del mes (1-5) basada en la fecha."""
    dias = fechas.dt.day
    dow = fechas.dt.weekday
    primer_dow_del_mes = (fechas - pd.to_timedelta(dias - 1, unit="D")).dt.weekday
    offset = (dow - primer_dow_del_mes + 7) % 7
    return ((dias - 1 + offset) // 7 + 1).clip(1, 5)


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dia_semana"] = df["fecha_llamada"].dt.dayofweek
    df["dia_mes"] = df["fecha_llamada"].dt.day
    df["mes"] = df["fecha_llamada"].dt.month
    df["anio"] = df["fecha_llamada"].dt.year
    df["dia_ano"] = df["fecha_llamada"].dt.dayofyear
    df["es_lunes"] = (df["dia_semana"] == 0).astype(int)
    df["es_viernes"] = (df["dia_semana"] == 4).astype(int)
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["trimestre"] = df["fecha_llamada"].dt.quarter
    df["inicio_mes"] = (df["dia_mes"] <= 5).astype(int)
    df["fin_mes"] = (df["dia_mes"] >= 25).astype(int)
    df["encoder_mandante"] = df["id_mandante"].astype("category").cat.codes
    df["semana_mes"] = calcular_semana_del_mes(df["fecha_llamada"])
    df["quincena"] = (df["fecha_llamada"].dt.day <= 15).astype(int)

    if "hora_llamada" in df.columns:
        df["hora"] = pd.to_numeric(df["hora_llamada"], errors="coerce")
        df["es_manana"] = (df["hora"] < 12).astype(int)
        df["es_tarde"] = ((df["hora"] >= 12) & (df["hora"] < 18)).astype(int)
        df["es_noche"] = (df["hora"] >= 18).astype(int)

    if "grupo" in df.columns:
        df["encoder_grupo"] = df["grupo"].astype("category").cat.codes

    return df


def crear_features_lag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["id_mandante", "fecha_llamada"]).reset_index(drop=True)
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df.groupby("id_mandante")["countcd"].shift(lag)
    for window in [7, 14, 30]:
        df[f"media_movil_{window}"] = (
            df.groupby("id_mandante")["countcd"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"std_movil_{window}"] = (
            df.groupby("id_mandante")["countcd"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
    df["diff_1"] = df.groupby("id_mandante")["countcd"].diff(1)
    df["diff_7"] = df.groupby("id_mandante")["countcd"].diff(7)
    df["pct_change_1"] = df.groupby("id_mandante")["countcd"].pct_change(1)
    df["pct_change_7"] = df.groupby("id_mandante")["countcd"].pct_change(7)
    return df


def crear_features_estacionalidad(df: pd.DataFrame) -> dict:
    daily = df.groupby(["id_mandante", "fecha_llamada"]).size().reset_index(name="countcd")

    estacionalidad = {}
    for mandante in daily["id_mandante"].unique():
        datos = daily[daily["id_mandante"] == mandante].copy()
        datos = datos.sort_values("fecha_llamada").reset_index(drop=True)
        datos["dia_semana"] = datos["fecha_llamada"].dt.dayofweek
        datos["semana_mes"] = calcular_semana_del_mes(datos["fecha_llamada"])
        datos["mes"] = datos["fecha_llamada"].dt.month
        datos["anio"] = datos["fecha_llamada"].dt.year

        global_avg = datos["countcd"].mean()

        seasonal_dow = datos.groupby("dia_semana")["countcd"].mean()
        dow_factor = {}
        for dow in range(7):
            if dow in seasonal_dow.index:
                dow_factor[dow] = seasonal_dow[dow] / global_avg
            else:
                dow_factor[dow] = 0.0

        seasonal_semana = datos.groupby("semana_mes")["countcd"].mean()
        semana_factor = seasonal_semana / global_avg

        seasonal_mes = datos.groupby("mes")["countcd"].mean()
        mes_factor = seasonal_mes / global_avg

        monthly_avg = datos.groupby(["anio", "mes"])["countcd"].mean()
        recent_months = monthly_avg.tail(6)
        trend_factor = recent_months.mean() / global_avg if len(recent_months) > 0 else 1.0

        estacionalidad[mandante] = {
            "global_avg": global_avg,
            "dow_factor": dow_factor,
            "semana_factor": semana_factor.to_dict(),
            "mes_factor": mes_factor.to_dict(),
            "trend_factor": trend_factor,
        }

    return estacionalidad


def entrenar_y_predecir(df: pd.DataFrame) -> tuple:
    hoy = datetime.now()
    anio = hoy.year
    mes = hoy.month
    dias_en_mes = calendar.monthrange(anio, mes)[1]

    mandante_map = {nombre: idx for idx, nombre in enumerate(sorted(df["id_mandante"].unique()))}

    estacionalidad = crear_features_estacionalidad(df)

    dias_semana_es = {
        0: "Lunes", 1: "Martes", 2: "Miercoles",
        3: "Jueves", 4: "Viernes", 5: "Sabado", 6: "Domingo",
    }

    filas_prediccion = []

    for mandante_nombre, mandante_id in mandante_map.items():
        est = estacionalidad[mandante_nombre]
        global_avg = est["global_avg"]
        dow_factor = est["dow_factor"]
        semana_factor = est["semana_factor"]
        mes_factor = est["mes_factor"]
        trend = est["trend_factor"]

        for dia in range(1, dias_en_mes + 1):
            fecha = datetime(anio, mes, dia)
            dia_semana = fecha.weekday()

            if dia_semana >= 5:
                continue

            primer_dia_mes = datetime(anio, mes, 1)
            semana_mes = min((dia - 1 + (dia_semana - primer_dia_mes.weekday() + 7) % 7) // 7 + 1, 5)

            base = global_avg * trend
            factor_dow = dow_factor.get(dia_semana, 1.0)
            factor_semana = semana_factor.get(semana_mes, 1.0)
            factor_mes = mes_factor.get(mes, 1.0)

            prediccion = base * factor_dow * factor_semana * factor_mes
            prediccion = max(prediccion, 0)

            filas_prediccion.append({
                "fecha": fecha,
                "dia_mes": dia,
                "dia_semana_num": dia_semana,
                "mes": mes,
                "mandante_id": mandante_id,
                "nombre_mandante": mandante_nombre,
                "dia_semana_texto": dias_semana_es[dia_semana],
                "prediccion": round(prediccion, 1),
            })

    df_prediccion = pd.DataFrame(filas_prediccion)
    if len(df_prediccion) > 0:
        df_prediccion["fecha"] = df_prediccion["fecha"].dt.strftime("%Y-%m-%d")

    return df_prediccion, estacionalidad, None


def fig_to_streamlit(fig, st):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)
