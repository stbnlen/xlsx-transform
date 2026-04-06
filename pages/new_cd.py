import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

from utils_new_cd import (
    calcular_semana_del_mes,
    crear_features,
    crear_features_lag,
    crear_features_estacionalidad,
    entrenar_y_predecir,
)

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

expected_cols = {"id_mandante", "fecha_llamada"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"El archivo debe contener las columnas: {expected_cols}")
    st.stop()

# Parse date column
df["fecha_llamada"] = pd.to_datetime(df["fecha_llamada"], errors="coerce")
df = df.dropna(subset=["fecha_llamada"])

# Each row is one contact - derive countcd
if "countcd" not in df.columns:
    df["countcd"] = 1
else:
    df["countcd"] = pd.to_numeric(df["countcd"], errors="coerce")

# Set style
sns.set_theme(style="whitegrid")

# Sidebar summary
mandantes = df["id_mandante"].unique()
date_range = f"{df['fecha_llamada'].min().strftime('%Y-%m-%d')} a {df['fecha_llamada'].max().strftime('%Y-%m-%d')}"
total_contacts = int(df["countcd"].sum())

st.sidebar.header("Resumen General")
st.sidebar.metric("Total registros", len(df))
st.sidebar.metric("Mandantes", len(mandantes))
st.sidebar.metric("Rango de fechas", date_range)
st.sidebar.metric("Total Contactos", f"{total_contacts:,}")


def fig_to_streamlit(fig):
    from utils_new_cd import fig_to_streamlit as _fts
    _fts(fig, st)


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Vista General",
    "Por Mandante",
    "Serie Temporal",
    "Distribucion",
    "Estadisticas",
    "Prediccion",
    "Factores Estacionales",
    "Descargar Metas",
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

    if "grupo" in df.columns:
        st.write("**Distribucion por Grupo:**")
        grupo_counts = df["grupo"].value_counts()
        grupo_df = pd.DataFrame({"Grupo": grupo_counts.index, "Contactos": grupo_counts.values})
        st.dataframe(grupo_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(grupo_df["Grupo"], grupo_df["Contactos"], color=sns.color_palette("Set2", len(grupo_df)))
        ax.set_title("Contactos por Grupo")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("Total Contactos")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

    if "hora_llamada" in df.columns:
        st.write("**Distribucion por Hora del Dia (7:00 - 21:00):**")
        df_temp = df.copy()
        df_temp["hora"] = pd.to_numeric(df_temp["hora_llamada"], errors="coerce")
        df_temp = df_temp[(df_temp["hora"] >= 7) & (df_temp["hora"] <= 21)]
        hora_counts = df_temp["hora"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(hora_counts.index, hora_counts.values, color="steelblue")
        ax.set_title("Contactos por Hora del Dia")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Total Contactos")
        fig.tight_layout()
        fig_to_streamlit(fig)

    if "rut_dv" in df.columns:
        total_ruts = df["rut_dv"].nunique()
        st.metric("RUTs unicos", f"{total_ruts:,}")

# Tab 2: By mandante
with tab2:
    st.subheader("Analisis por Mandante")

    daily_mandante = df.groupby(["id_mandante", "fecha_llamada"]).size().reset_index(name="contactos_diarios")

    mandante_agg = daily_mandante.groupby("id_mandante").agg(
        total_contactos=("contactos_diarios", "sum"),
        dias_activos=("fecha_llamada", "nunique"),
        promedio_diario=("contactos_diarios", "mean"),
        max_diario=("contactos_diarios", "max"),
        min_diario=("contactos_diarios", "min"),
        std_diario=("contactos_diarios", "std"),
    ).reset_index()
    mandante_agg["std_diario"] = mandante_agg["std_diario"].fillna(0)
    mandante_agg = mandante_agg.sort_values("total_contactos", ascending=False)

    if "rut_dv" in df.columns:
        mandante_ruts = df.groupby("id_mandante")["rut_dv"].nunique().reset_index()
        mandante_ruts.columns = ["id_mandante", "ruts_unicos"]
        mandante_agg = mandante_agg.merge(mandante_ruts, on="id_mandante")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            mandante_agg["id_mandante"],
            mandante_agg["total_contactos"],
            color=sns.color_palette("Blues_d", len(mandante_agg)),
        )
        ax.set_title("Total Contactos por Mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Total Contactos")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            mandante_agg["total_contactos"],
            labels=mandante_agg["id_mandante"],
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel", len(mandante_agg)),
        )
        ax.set_title("Distribucion porcentual por Mandante")
        fig_to_streamlit(fig)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Analisis por Mandante y Grupo:**")
        mandante_grupo = df.groupby(["id_mandante", "grupo"])["countcd"].sum().reset_index()
        mandante_grupo.columns = ["Mandante", "Grupo", "Contactos"]
        mandante_grupo_pivot = mandante_grupo.pivot(index="Mandante", columns="Grupo", values="Contactos").fillna(0)
        st.dataframe(mandante_grupo_pivot, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        mandante_grupo_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title("Contactos por Mandante y Grupo")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Total Contactos")
        ax.legend(title="Grupo")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

    st.write("**Tabla resumen por mandante:**")
    st.dataframe(mandante_agg.style.format({
        "total_contactos": "{:,.0f}",
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
    ax.set_title("Evolucion diaria de Contactos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Contactos")
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
    ax.set_ylabel("Total Contactos")
    fig.autofmt_xdate()
    fig_to_streamlit(fig)

    # Day of week analysis
    df["dia_semana"] = df["fecha_llamada"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_agg = df.groupby("dia_semana")["countcd"].sum().reindex(day_order).dropna().reset_index()
    day_agg["dia_semana"] = pd.Categorical(day_agg["dia_semana"], categories=day_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(day_agg["dia_semana"], day_agg["countcd"], color=sns.color_palette("viridis", len(day_agg)))
    ax.set_title("Total Contactos por dia de la semana")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Total Contactos")
    fig_to_streamlit(fig)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Evolucion diaria por Grupo:**")
        daily_grupo = df.groupby(["fecha_llamada", "grupo"])["countcd"].sum().reset_index()
        grupos = sorted(df["grupo"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        for grupo in grupos:
            subset = daily_grupo[daily_grupo["grupo"] == grupo]
            ax.plot(subset["fecha_llamada"], subset["countcd"], marker="o", label=grupo)
        ax.set_title("Evolucion diaria de Contactos por Grupo")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Contactos")
        ax.legend()
        fig.autofmt_xdate()
        fig_to_streamlit(fig)

    if "hora_llamada" in df.columns:
        st.write("---")
        st.write("**Patrones de Hora por Mandante (7:00 - 21:00):**")
        df_temp = df.copy()
        df_temp["hora"] = pd.to_numeric(df_temp["hora_llamada"], errors="coerce")
        df_temp = df_temp[(df_temp["hora"] >= 7) & (df_temp["hora"] <= 21)]
        hora_mandante = df_temp.groupby(["id_mandante", "hora"])["countcd"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        for mandante in sorted(hora_mandante["id_mandante"].unique()):
            subset = hora_mandante[hora_mandante["id_mandante"] == mandante]
            ax.plot(subset["hora"], subset["countcd"], marker="o", label=mandante)
        ax.set_title("Distribucion Horaria por Mandante")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Contactos")
        ax.legend()
        ax.set_xticks(range(7, 22))
        fig.tight_layout()
        fig_to_streamlit(fig)

# Tab 4: Distribution
with tab4:
    st.subheader("Analisis de Distribucion")

    daily_per_mandante = df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(daily_per_mandante["countcd"], bins=50, kde=True, ax=ax, color="steelblue")
        ax.set_title("Distribucion de Contactos Diarios")
        ax.set_xlabel("Contactos por dia")
        ax.set_ylabel("Frecuencia")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=daily_per_mandante, x="id_mandante", y="countcd", ax=ax, palette="pastel")
        ax.set_title("Boxplot por Mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Contactos por dia")
        fig_to_streamlit(fig)

    st.write("**Estadisticas de distribucion por mandante:**")
    dist_stats = daily_per_mandante.groupby("id_mandante")["countcd"].describe().reset_index()
    numeric_cols = dist_stats.select_dtypes(include="number").columns
    st.dataframe(dist_stats.style.format({col: "{:.2f}" for col in numeric_cols}), use_container_width=True)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Distribucion por Grupo:**")
        daily_per_grupo = df.groupby(["fecha_llamada", "grupo"])["countcd"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=daily_per_grupo, x="grupo", y="countcd", ax=ax, palette="pastel")
        ax.set_title("Boxplot por Grupo")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("Contactos por dia")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

# Tab 5: Statistics
with tab5:
    st.subheader("Estadisticas Descriptivas")

    daily_per_mandante = df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Estadisticas globales de Contactos Diarios:**")
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
                "Coef. Variacion",
            ],
            "Valor": [
                f"{daily_per_mandante['countcd'].mean():.2f}",
                f"{daily_per_mandante['countcd'].median():.2f}",
                f"{daily_per_mandante['countcd'].std():.2f}",
                f"{daily_per_mandante['countcd'].min():.0f}",
                f"{daily_per_mandante['countcd'].max():.0f}",
                f"{daily_per_mandante['countcd'].quantile(0.25):.2f}",
                f"{daily_per_mandante['countcd'].quantile(0.75):.2f}",
                f"{daily_per_mandante['countcd'].quantile(0.75) - daily_per_mandante['countcd'].quantile(0.25):.2f}",
                f"{daily_per_mandante['countcd'].skew():.4f}",
                f"{daily_per_mandante['countcd'].kurtosis():.4f}",
                f"{daily_per_mandante['countcd'].sum():,.0f}",
                f"{daily_per_mandante['countcd'].std() / daily_per_mandante['countcd'].mean():.4f}" if daily_per_mandante['countcd'].mean() != 0 else "N/A",
            ],
        })
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.write("**Top 10 fechas con mayor Contactos:**")
        top_dates = (
            df.groupby("fecha_llamada")["countcd"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_dates["fecha_llamada"] = top_dates["fecha_llamada"].dt.strftime("%Y-%m-%d")
        st.dataframe(top_dates, use_container_width=True)

    st.write("**Estadisticas por mandante (diario):**")
    mandante_stats = daily_per_mandante.groupby("id_mandante")["countcd"].agg([
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("count", "count"),
    ]).reset_index()
    mandante_stats.columns = ["Mandante", "Media", "Mediana", "Std", "Min", "Max", "Registros"]
    mandante_stats = mandante_stats.sort_values("Media", ascending=False)
    st.dataframe(mandante_stats.style.format({
        "Media": "{:.2f}",
        "Mediana": "{:.2f}",
        "Std": "{:.2f}",
        "Min": "{:.0f}",
        "Max": "{:.0f}",
        "Registros": "{:.0f}",
    }), use_container_width=True)

    st.write("**Matriz de correlacion (features temporales vs Contactos):**")
    daily_corr = df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contactos")
    daily_corr["dia_ano"] = daily_corr["fecha_llamada"].dt.dayofyear
    daily_corr["dia_mes"] = daily_corr["fecha_llamada"].dt.day
    daily_corr["mes"] = daily_corr["fecha_llamada"].dt.month
    daily_corr["dia_semana_num"] = daily_corr["fecha_llamada"].dt.dayofweek
    daily_corr["semana_mes"] = calcular_semana_del_mes(daily_corr["fecha_llamada"])
    daily_corr["es_lunes"] = (daily_corr["dia_semana_num"] == 0).astype(int)
    daily_corr["es_viernes"] = (daily_corr["dia_semana_num"] == 4).astype(int)

    corr_cols = ["contactos", "dia_ano", "dia_mes", "mes", "dia_semana_num", "semana_mes", "es_lunes", "es_viernes"]
    corr_matrix = daily_corr[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, fmt=".2f", linewidths=0.5)
    ax.set_title("Correlacion: Contactos vs Features Temporales")
    labels = ["Contactos", "Dia del ano", "Dia del mes", "Mes", "Dia semana", "Semana del mes", "Es Lunes", "Es Viernes"]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig_to_streamlit(fig)

    st.write("**Contactos promedio por dia de la semana:**")
    daily_dow = df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contactos")
    daily_dow["dia_semana_num"] = daily_dow["fecha_llamada"].dt.dayofweek
    day_names = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    dow_agg = daily_dow.groupby("dia_semana_num")["contactos"].agg(["mean", "sum", "count"]).reset_index()
    dow_agg["dia_nombre"] = dow_agg["dia_semana_num"].map(lambda x: day_names[x] if x < 7 else "Desconocido")
    dow_agg = dow_agg.sort_values("dia_semana_num")
    st.dataframe(dow_agg[["dia_nombre", "mean", "sum", "count"]].rename(columns={
        "dia_nombre": "Dia",
        "mean": "Promedio",
        "sum": "Total",
        "count": "Registros",
    }).style.format({"Promedio": "{:.2f}", "Total": "{:.0f}", "Registros": "{:.0f}"}), use_container_width=True)

    st.write("**Contactos promedio por mes:**")
    daily_month = df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contactos")
    month_names = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                   "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    daily_month["mes_num"] = daily_month["fecha_llamada"].dt.month
    month_agg = daily_month.groupby("mes_num")["contactos"].agg(["mean", "sum", "count"]).reset_index()
    month_agg["mes_nombre"] = month_agg["mes_num"].map(lambda x: month_names[x - 1] if 1 <= x <= 12 else "Desconocido")
    month_agg = month_agg.sort_values("mes_num")
    st.dataframe(month_agg[["mes_nombre", "mean", "sum", "count"]].rename(columns={
        "mes_nombre": "Mes",
        "mean": "Promedio",
        "sum": "Total",
        "count": "Registros",
    }).style.format({"Promedio": "{:.2f}", "Total": "{:.0f}", "Registros": "{:.0f}"}), use_container_width=True)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Estadisticas por Grupo:**")
        daily_grupo = df.groupby(["fecha_llamada", "id_mandante", "grupo"]).size().reset_index(name="contactos")
        grupo_stats = daily_grupo.groupby("grupo")["contactos"].agg(["sum", "mean", "count"]).reset_index()
        grupo_stats.columns = ["Grupo", "Total Contactos", "Promedio Diario", "Registros"]
        grupo_stats = grupo_stats.sort_values("Total Contactos", ascending=False)
        st.dataframe(grupo_stats.style.format({
            "Total Contactos": "{:,.0f}",
            "Promedio Diario": "{:.2f}",
            "Registros": "{:,.0f}",
        }), use_container_width=True)

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
    else:
        st.warning("No hay datos suficientes para generar la prediccion.")

# Tab 7: Factores Estacionales
with tab7:
    st.subheader("Factores Estacionales por Mandante")

    with st.spinner("Calculando factores estacionales..."):
        df_prediccion, estacionalidad, _ = entrenar_y_predecir(df)

    if len(df_prediccion) > 0:
        for mandante, est in estacionalidad.items():
            st.write(f"**{mandante}** (promedio global: {est['global_avg']:.1f}, tendencia: {est['trend_factor']:.2f}x)")

            dow_df = pd.DataFrame({
                "Dia": ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"],
                "Factor": [est["dow_factor"].get(i, 1.0) for i in range(5)],
            })
            st.write("**Factor por dia de semana:**")
            st.dataframe(dow_df.style.format({"Factor": "{:.3f}"}), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(dow_df["Dia"], dow_df["Factor"], color=sns.color_palette("viridis", 5))
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor dia de semana - {mandante}")
            ax.set_ylabel("Multiplicador")
            fig.tight_layout()
            fig_to_streamlit(fig)

            semana_df = pd.DataFrame({
                "Semana del Mes": [1, 2, 3, 4, 5],
                "Factor": [est["semana_factor"].get(i, 1.0) for i in range(1, 6)],
            })
            st.write("**Factor por semana del mes:**")
            st.dataframe(semana_df.style.format({"Factor": "{:.3f}"}), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(semana_df["Semana del Mes"], semana_df["Factor"], color=sns.color_palette("Blues_d", 5))
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor semana del mes - {mandante}")
            ax.set_xlabel("Semana del Mes")
            ax.set_ylabel("Multiplicador")
            ax.set_xticks([1, 2, 3, 4, 5])
            fig.tight_layout()
            fig_to_streamlit(fig)

            nombres_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                             "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
            mes_df = pd.DataFrame({
                "Mes": nombres_meses,
                "Factor": [est["mes_factor"].get(i, 1.0) for i in range(1, 13)],
            })
            st.write("**Factor por mes (estacionalidad anual):**")
            st.dataframe(mes_df.style.format({"Factor": "{:.3f}"}), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(mes_df["Mes"], mes_df["Factor"], color=sns.color_palette("Greens_d", 12))
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor mensual anual - {mandante}")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Multiplicador")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig_to_streamlit(fig)

        if "hora_llamada" in df.columns:
            st.write("---")
            st.write("**Estacionalidad Horaria por Mandante (7:00 - 21:00):**")

            df_hora = df.copy()
            df_hora["hora"] = pd.to_numeric(df_hora["hora_llamada"], errors="coerce")
            df_hora = df_hora.dropna(subset=["hora"])
            df_hora["hora"] = df_hora["hora"].astype(int)
            df_hora = df_hora[(df_hora["hora"] >= 7) & (df_hora["hora"] <= 21)]

            total_por_mandante_hora = df_hora.groupby("id_mandante").size()

            horas_rango = list(range(7, 22))

            for mandante in sorted(df_hora["id_mandante"].unique()):
                st.write(f"**{mandante}**")

                mandante_data = df_hora[df_hora["id_mandante"] == mandante]
                total_mandante = total_por_mandante_hora[mandante]
                global_avg_hora = total_mandante / len(horas_rango)

                hora_counts = mandante_data.groupby("hora").size()
                hora_factor = {}
                for h in horas_rango:
                    if h in hora_counts.index:
                        hora_factor[h] = hora_counts[h] / global_avg_hora
                    else:
                        hora_factor[h] = 0.0

                hora_df = pd.DataFrame({
                    "Hora": horas_rango,
                    "Factor": [hora_factor.get(h, 0.0) for h in horas_rango],
                    "Contactos": [int(hora_counts.get(h, 0)) for h in horas_rango],
                })

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(hora_df.style.format({"Factor": "{:.3f}", "Contactos": "{:,.0f}"}), use_container_width=True)
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.bar(hora_df["Hora"], hora_df["Factor"], color=sns.color_palette("Oranges_d", len(horas_rango)))
                    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
                    ax.set_title(f"Factor horario - {mandante}")
                    ax.set_xlabel("Hora")
                    ax.set_ylabel("Multiplicador")
                    ax.set_xticks(horas_rango)
                    fig.tight_layout()
                    fig_to_streamlit(fig)
    else:
        st.warning("No hay datos suficientes para generar los factores estacionales.")

# Tab 8: Descargar Metas
with tab8:
    st.subheader("Descargar Metas Diarias")

    hoy = datetime.now()
    mes_actual = hoy.strftime("%B %Y")

    with st.spinner("Calculando metas..."):
        df_prediccion, estacionalidad, _ = entrenar_y_predecir(df)

    if len(df_prediccion) > 0:
        metas_globales = {}
        for mandante in df_prediccion["nombre_mandante"].unique():
            df_mandante = df_prediccion[df_prediccion["nombre_mandante"] == mandante]
            metas_globales[mandante] = df_mandante["prediccion"].sum()

        download_rows = []
        for _, row in df_prediccion.iterrows():
            download_rows.append({
                "dia": row["dia_mes"],
                "numero de mes": row["mes"],
                "año": hoy.year,
                "cartera": row["nombre_mandante"],
                "meta diaria": row["prediccion"],
                "meta global": metas_globales[row["nombre_mandante"]],
            })

        df_download = pd.DataFrame(download_rows)

        st.dataframe(df_download, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_download.to_excel(writer, index=False, sheet_name="Hoja1")

        st.download_button(
            label="Descargar Metas Diarias",
            data=output.getvalue(),
            file_name=f"metas_diarias_{mes_actual.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning("No hay datos suficientes para generar las metas.")
