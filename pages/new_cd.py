import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime
import calendar

st.title("New CD - Analisis Exploratorio")

uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls", "csv"],
    key="new_cd_uploader",
)

if uploaded_file is None:
    st.markdown("### Sube un archivo para comenzar el analisis")
    st.stop()

# Load data
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

expected_cols = {"id_mandante", "fecha_llamada", "countcd"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"El archivo debe contener las columnas: {expected_cols}")
    st.stop()

# Parse date column
df["fecha_llamada"] = pd.to_datetime(df["fecha_llamada"], errors="coerce")
df["countcd"] = pd.to_numeric(df["countcd"], errors="coerce")
df = df.dropna(subset=["fecha_llamada", "countcd"])

# Set style
sns.set_theme(style="whitegrid")

# Sidebar summary
mandantes = df["id_mandante"].unique()
date_range = f"{df['fecha_llamada'].min().strftime('%Y-%m-%d')} a {df['fecha_llamada'].max().strftime('%Y-%m-%d')}"
total_calls = int(df["countcd"].sum())

st.sidebar.header("Resumen General")
st.sidebar.metric("Total registros", len(df))
st.sidebar.metric("Mandantes", len(mandantes))
st.sidebar.metric("Rango de fechas", date_range)
st.sidebar.metric("Total CountCD", f"{total_calls:,}")


def fig_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


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
    df["semana_mes"] = ((df["fecha_llamada"].dt.day - 1) // 7).clip(0, 3)
    df["quincena"] = (df["fecha_llamada"].dt.day <= 15).astype(int)
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
    estacionalidad = {}
    for mandante in df["id_mandante"].unique():
        datos = df[df["id_mandante"] == mandante].copy()
        datos = datos.sort_values("fecha_llamada").reset_index(drop=True)
        datos["dia_semana"] = datos["fecha_llamada"].dt.dayofweek
        datos["semana_mes"] = ((datos["fecha_llamada"].dt.day - 1) // 7).clip(0, 3)
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

        monthly_avg = datos.groupby(["anio", "mes"])["countcd"].mean()
        recent_months = monthly_avg.tail(6)
        trend_factor = recent_months.mean() / global_avg if len(recent_months) > 0 else 1.0

        estacionalidad[mandante] = {
            "global_avg": global_avg,
            "dow_factor": dow_factor,
            "semana_factor": semana_factor.to_dict(),
            "trend_factor": trend_factor,
        }

    return estacionalidad


def entrenar_y_predecir(df: pd.DataFrame) -> pd.DataFrame:
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
        trend = est["trend_factor"]

        for dia in range(1, dias_en_mes + 1):
            fecha = datetime(anio, mes, dia)
            dia_semana = fecha.weekday()

            if dia_semana >= 5:
                continue

            semana_mes = min((dia - 1) // 7, 3)

            base = global_avg * trend
            factor_dow = dow_factor.get(dia_semana, 1.0)
            factor_semana = semana_factor.get(semana_mes, 1.0)

            prediccion = base * factor_dow * factor_semana
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


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vista General",
    "Por Mandante",
    "Serie Temporal",
    "Distribucion",
    "Estadisticas",
    "Prediccion",
])

# Tab 1: General overview
with tab1:
    st.subheader("Vista General del Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", len(df))
    col2.metric("Columnas", len(df.columns))
    col3.metric("Valores nulos", int(df.isnull().sum().sum()))

    st.write("**Primeras 10 filas:**")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("**Tipos de datos:**")
    st.dataframe(
        pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.astype(str),
            "No nulos": df.count().values,
        }),
        use_container_width=True,
    )

# Tab 2: By mandante
with tab2:
    st.subheader("Analisis por Mandante")

    mandante_agg = df.groupby("id_mandante").agg(
        total_countcd=("countcd", "sum"),
        dias_activos=("fecha_llamada", "nunique"),
        promedio_diario=("countcd", "mean"),
        max_diario=("countcd", "max"),
        min_diario=("countcd", "min"),
        std_diario=("countcd", "std"),
    ).reset_index()
    mandante_agg["std_diario"] = mandante_agg["std_diario"].fillna(0)
    mandante_agg = mandante_agg.sort_values("total_countcd", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            mandante_agg["id_mandante"],
            mandante_agg["total_countcd"],
            color=sns.color_palette("Blues_d", len(mandante_agg)),
        )
        ax.set_title("Total CountCD por Mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Total CountCD")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            mandante_agg["total_countcd"],
            labels=mandante_agg["id_mandante"],
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel", len(mandante_agg)),
        )
        ax.set_title("Distribucion porcentual por Mandante")
        fig_to_streamlit(fig)

    st.write("**Tabla resumen por mandante:**")
    st.dataframe(mandante_agg.style.format({
        "total_countcd": "{:,.0f}",
        "promedio_diario": "{:,.1f}",
        "max_diario": "{:,.0f}",
        "min_diario": "{:,.0f}",
        "std_diario": "{:,.1f}",
    }), use_container_width=True)

# Tab 3: Time series
with tab3:
    st.subheader("Analisis de Serie Temporal")

    daily = df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()
    daily_total = df.groupby("fecha_llamada")["countcd"].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        selected_mandante = st.multiselect(
            "Seleccionar mandantes",
            options=sorted(mandantes),
            default=list(mandantes),
            key="ts_mandante",
        )

    filtered_daily = daily[daily["id_mandante"].isin(selected_mandante)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for mandante in filtered_daily["id_mandante"].unique():
        subset = filtered_daily[filtered_daily["id_mandante"] == mandante]
        ax.plot(subset["fecha_llamada"], subset["countcd"], marker="o", label=mandante)
    ax.set_title("Evolucion diaria de CountCD")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("CountCD")
    ax.legend()
    fig.autofmt_xdate()
    fig_to_streamlit(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        daily_total["fecha_llamada"],
        daily_total["countcd"],
        alpha=0.3,
        color="steelblue",
    )
    ax.plot(daily_total["fecha_llamada"], daily_total["countcd"], color="steelblue")
    ax.set_title("Total diario (todos los mandantes)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Total CountCD")
    fig.autofmt_xdate()
    fig_to_streamlit(fig)

    # Day of week analysis
    df["dia_semana"] = df["fecha_llamada"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_agg = df.groupby("dia_semana")["countcd"].sum().reindex(day_order).dropna().reset_index()
    day_agg["dia_semana"] = pd.Categorical(day_agg["dia_semana"], categories=day_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(day_agg["dia_semana"], day_agg["countcd"], color=sns.color_palette("viridis", len(day_agg)))
    ax.set_title("Total CountCD por dia de la semana")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Total CountCD")
    fig_to_streamlit(fig)

# Tab 4: Distribution
with tab4:
    st.subheader("Analisis de Distribucion")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["countcd"], bins=50, kde=True, ax=ax, color="steelblue")
        ax.set_title("Distribucion de CountCD")
        ax.set_xlabel("CountCD")
        ax.set_ylabel("Frecuencia")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="id_mandante", y="countcd", ax=ax, palette="pastel")
        ax.set_title("Boxplot por Mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("CountCD")
        fig_to_streamlit(fig)

    st.write("**Estadisticas de distribucion por mandante:**")
    dist_stats = df.groupby("id_mandante")["countcd"].describe().reset_index()
    numeric_cols = dist_stats.select_dtypes(include="number").columns
    st.dataframe(dist_stats.style.format({col: "{:.2f}" for col in numeric_cols}), use_container_width=True)

# Tab 5: Statistics
with tab5:
    st.subheader("Estadisticas Descriptivas")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Estadisticas globales de CountCD:**")
        stats_df = pd.DataFrame({
            "Metrica": [
                "Media",
                "Mediana",
                "Desviacion estandar",
                "Minimo",
                "Maximo",
                "Q1 (25%)",
                "Q3 (75%)",
                "IQR",
                "Skewness",
                "Kurtosis",
                "Total",
            ],
            "Valor": [
                f"{df['countcd'].mean():.2f}",
                f"{df['countcd'].median():.2f}",
                f"{df['countcd'].std():.2f}",
                f"{df['countcd'].min():.0f}",
                f"{df['countcd'].max():.0f}",
                f"{df['countcd'].quantile(0.25):.2f}",
                f"{df['countcd'].quantile(0.75):.2f}",
                f"{df['countcd'].quantile(0.75) - df['countcd'].quantile(0.25):.2f}",
                f"{df['countcd'].skew():.4f}",
                f"{df['countcd'].kurtosis():.4f}",
                f"{df['countcd'].sum():,.0f}",
            ],
        })
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.write("**Top 10 fechas con mayor CountCD:**")
        top_dates = (
            df.groupby("fecha_llamada")["countcd"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_dates["fecha_llamada"] = top_dates["fecha_llamada"].dt.strftime("%Y-%m-%d")
        st.dataframe(top_dates, use_container_width=True)

    st.write("**Matriz de correlacion (CountCD vs dia del ano):**")
    df_corr = df.copy()
    df_corr["dia_ano"] = df_corr["fecha_llamada"].dt.dayofyear
    corr_matrix = df_corr[["countcd", "dia_ano"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlacion CountCD vs Dia del Ano")
    fig_to_streamlit(fig)

# Tab 6: Prediccion
with tab6:
    st.subheader("Prediccion de CD - Mes Actual (Descomposicion Estacional)")

    hoy = datetime.now()
    mes_actual = hoy.strftime("%B %Y")
    st.info(f"Mostrando prediccion para dias habiles (Lun-Vie): {mes_actual}")

    with st.spinner("Calculando prediccion..."):
        df_prediccion, estacionalidad, _ = entrenar_y_predecir(df)

    if len(df_prediccion) > 0:
        for mandante in df_prediccion["nombre_mandante"].unique():
            st.write(f"### {mandante}")
            df_mandante = df_prediccion[df_prediccion["nombre_mandante"] == mandante].copy()

            st.dataframe(
                df_mandante[[
                    "fecha", "dia_mes", "dia_semana_num", "mes",
                    "mandante_id", "nombre_mandante", "dia_semana_texto", "prediccion",
                ]].style.format({
                    "prediccion": "{:.1f}",
                }),
                use_container_width=True,
            )

            total_mandante = df_mandante["prediccion"].sum()
            st.metric(
                label=f"Total CD esperado ({mandante})",
                value=f"{total_mandante:,.1f}",
            )

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(
                df_mandante["fecha"],
                df_mandante["prediccion"],
                color=sns.color_palette("Blues_d", len(df_mandante)),
            )
            ax.set_title(f"CD esperados por dia - {mandante} - {mes_actual}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("CD esperados")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig_to_streamlit(fig)

        st.write("### Factores Estacionales por Mandante")
        for mandante, est in estacionalidad.items():
            st.write(f"**{mandante}** (promedio global: {est['global_avg']:.1f}, tendencia: {est['trend_factor']:.2f}x)")

            dow_df = pd.DataFrame({
                "Dia": ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"],
                "Factor": [est["dow_factor"].get(i, 1.0) for i in range(5)],
            })
            st.dataframe(dow_df.style.format({"Factor": "{:.3f}"}), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(dow_df["Dia"], dow_df["Factor"], color=sns.color_palette("viridis", 5))
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor dia de semana - {mandante}")
            ax.set_ylabel("Multiplicador")
            fig.tight_layout()
            fig_to_streamlit(fig)
    else:
        st.warning("No hay datos suficientes para generar la prediccion.")
