import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from styles import setup_page
from utils_new_cd import (
    calc_week_of_month,
    train_and_predict,
)

setup_page("New CD", "📞")

st.title("New CD - Análisis Exploratorio")

uploaded_file = st.file_uploader(
    "Cargar archivo Excel",
    type=["xlsx", "xls", "csv"],
    key="new_cd_uploader",
)

if uploaded_file is None:
    st.markdown("### Carga un archivo para comenzar el análisis")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

df.columns = df.columns.str.strip().str.lower()

expected_cols = {"id_mandante", "fecha_llamada"}
if not expected_cols.issubset(set(df.columns)):
    st.error(f"El archivo debe contener las columnas: {expected_cols}")
    st.stop()

df["fecha_llamada"] = pd.to_datetime(df["fecha_llamada"], errors="coerce")
df = df.dropna(subset=["fecha_llamada"])

if "countcd" not in df.columns:
    df["countcd"] = 1
else:
    df["countcd"] = pd.to_numeric(df["countcd"], errors="coerce")

sns.set_theme(style="whitegrid")

mandantes = df["id_mandante"].unique()
date_range = f"{df['fecha_llamada'].min().strftime('%Y-%m-%d')} to {df['fecha_llamada'].max().strftime('%Y-%m-%d')}"
total_contacts = int(df["countcd"].sum())

st.sidebar.header("Resumen general")
st.sidebar.metric("Registros totales", len(df))
st.sidebar.metric("Mandantes", len(mandantes))
st.sidebar.metric("Rango de fechas", date_range)
st.sidebar.metric("Contactos totales", f"{total_contacts:,}")


def fig_to_streamlit(fig):
    from utils_new_cd import fig_to_streamlit as _fts

    _fts(fig, st)


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Vista general",
        "Por mandante",
        "Series temporales",
        "Distribución",
        "Estadísticas",
        "Predicción",
        "Factores estacionales",
        "Descargar objetivos",
    ]
)

with tab1:
    st.subheader("Dataset Vista general")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", len(df))
    col2.metric("Columnas", len(df.columns))
    col3.metric("Valores nulos", int(df.isnull().sum().sum()))

    st.write("**Primeras 10 filas:**")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("**Tipos de datos:**")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-null": df.count().values,
            }
        ),
        use_container_width=True,
    )

    if "grupo" in df.columns:
        st.write("**Distribución by Group:**")
        grupo_counts = df["grupo"].value_counts()
        grupo_df = pd.DataFrame(
            {"Grupo": grupo_counts.index, "Contactos": grupo_counts.values}
        )
        st.dataframe(grupo_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            grupo_df["Grupo"],
            grupo_df["Contactos"],
            color=sns.color_palette("Set2", len(grupo_df)),
        )
        ax.set_title("Contactos por grupo")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("Contactos totales")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

    if "hora_llamada" in df.columns:
        st.write("**Distribución by Hour of Day (7:00 - 21:00):**")
        df_temp = df.copy()
        df_temp["hour"] = pd.to_numeric(df_temp["hora_llamada"], errors="coerce")
        df_temp = df_temp[(df_temp["hour"] >= 7) & (df_temp["hour"] <= 21)]
        hora_counts = df_temp["hour"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(hora_counts.index, hora_counts.values, color="steelblue")
        ax.set_title("Contactos por hora del día")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Contactos totales")
        fig.tight_layout()
        fig_to_streamlit(fig)

    if "rut_dv" in df.columns:
        total_ruts = df["rut_dv"].nunique()
        st.metric("RUTs únicos", f"{total_ruts:,}")

with tab2:
    st.subheader("Análisis por mandante")

    daily_mandante = (
        df.groupby(["id_mandante", "fecha_llamada"])
        .size()
        .reset_index(name="daily_contacts")
    )

    mandante_agg = (
        daily_mandante.groupby("id_mandante")
        .agg(
            total_contacts=("daily_contacts", "sum"),
            active_days=("fecha_llamada", "nunique"),
            daily_average=("daily_contacts", "mean"),
            daily_max=("daily_contacts", "max"),
            daily_min=("daily_contacts", "min"),
            daily_std=("daily_contacts", "std"),
        )
        .reset_index()
    )
    mandante_agg["daily_std"] = mandante_agg["daily_std"].fillna(0)
    mandante_agg = mandante_agg.sort_values("total_contacts", ascending=False)

    if "rut_dv" in df.columns:
        mandante_ruts = df.groupby("id_mandante")["rut_dv"].nunique().reset_index()
        mandante_ruts.columns = ["id_mandante", "unique_ruts"]
        mandante_agg = mandante_agg.merge(mandante_ruts, on="id_mandante")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            mandante_agg["id_mandante"],
            mandante_agg["total_contacts"],
            color=sns.color_palette("Blues_d", len(mandante_agg)),
        )
        ax.set_title("Contactos totales by Mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Contactos totales")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            mandante_agg["total_contacts"],
            labels=mandante_agg["id_mandante"],
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel", len(mandante_agg)),
        )
        ax.set_title("Percentage Distribución by Mandante")
        fig_to_streamlit(fig)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Análisis por mandante and Group:**")
        mandante_grupo = (
            df.groupby(["id_mandante", "grupo"])["countcd"].sum().reset_index()
        )
        mandante_grupo.columns = ["Mandante", "Grupo", "Contactos"]
        mandante_grupo_pivot = mandante_grupo.pivot(
            index="Mandante", columns="Grupo", values="Contactos"
        ).fillna(0)
        st.dataframe(mandante_grupo_pivot, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        mandante_grupo_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title("Contactos por mandante y grupo")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Contactos totales")
        ax.legend(title="Grupo")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

    st.write("**Tabla resumen por mandante:**")
    st.dataframe(
        mandante_agg.style.format(
            {
                "total_contacts": "{:,.0f}",
                "daily_average": "{:,.1f}",
                "daily_max": "{:,.0f}",
                "daily_min": "{:,.0f}",
                "daily_std": "{:,.1f}",
            }
        ),
        use_container_width=True,
    )

with tab3:
    st.subheader("Series temporales Analysis")

    daily = df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()
    daily_total = df.groupby("fecha_llamada")["countcd"].sum().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        selected_mandante = st.multiselect(
            "Select mandantes",
            options=sorted(mandantes),
            default=list(mandantes),
            key="ts_mandante",
        )

    filtered_daily = daily[daily["id_mandante"].isin(selected_mandante)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for mandante in filtered_daily["id_mandante"].unique():
        subset = filtered_daily[filtered_daily["id_mandante"] == mandante]
        ax.plot(subset["fecha_llamada"], subset["countcd"], marker="o", label=mandante)
    ax.set_title("Evolución diaria de contactos")
    ax.set_xlabel("Date")
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
    ax.set_xlabel("Date")
    ax.set_ylabel("Contactos totales")
    fig.autofmt_xdate()
    fig_to_streamlit(fig)

    df["day_of_week"] = df["fecha_llamada"].dt.day_name()
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_agg = (
        df.groupby("day_of_week")["countcd"]
        .sum()
        .reindex(day_order)
        .dropna()
        .reset_index()
    )
    day_agg["day_of_week"] = pd.Categorical(
        day_agg["day_of_week"], categories=day_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        day_agg["day_of_week"],
        day_agg["countcd"],
        color=sns.color_palette("viridis", len(day_agg)),
    )
    ax.set_title("Contactos totales by Day of Week")
    ax.set_xlabel("Día")
    ax.set_ylabel("Contactos totales")
    fig_to_streamlit(fig)

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Evolución diaria por grupo:**")
        daily_grupo = (
            df.groupby(["fecha_llamada", "grupo"])["countcd"].sum().reset_index()
        )
        grupos = sorted(df["grupo"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        for grupo in grupos:
            subset = daily_grupo[daily_grupo["grupo"] == grupo]
            ax.plot(subset["fecha_llamada"], subset["countcd"], marker="o", label=grupo)
        ax.set_title("Daily Evolution of Contactos por grupo")
        ax.set_xlabel("Date")
        ax.set_ylabel("Contactos")
        ax.legend()
        fig.autofmt_xdate()
        fig_to_streamlit(fig)

    if "hora_llamada" in df.columns:
        st.write("---")
        st.write("**Patrones horarios por mandante (7:00 - 21:00):**")
        df_temp = df.copy()
        df_temp["hour"] = pd.to_numeric(df_temp["hora_llamada"], errors="coerce")
        df_temp = df_temp[(df_temp["hour"] >= 7) & (df_temp["hour"] <= 21)]
        hora_mandante = (
            df_temp.groupby(["id_mandante", "hour"])["countcd"].sum().reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for mandante in sorted(hora_mandante["id_mandante"].unique()):
            subset = hora_mandante[hora_mandante["id_mandante"] == mandante]
            ax.plot(subset["hour"], subset["countcd"], marker="o", label=mandante)
        ax.set_title("Hourly Distribución by Mandante")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Contactos")
        ax.legend()
        ax.set_xticks(range(7, 22))
        fig.tight_layout()
        fig_to_streamlit(fig)

with tab4:
    st.subheader("Distribución Analysis")

    daily_per_mandante = (
        df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            daily_per_mandante["countcd"], bins=50, kde=True, ax=ax, color="steelblue"
        )
        ax.set_title("Distribución of Daily Contacts")
        ax.set_xlabel("Contactos por día")
        ax.set_ylabel("Frecuencia")
        fig_to_streamlit(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=daily_per_mandante,
            x="id_mandante",
            y="countcd",
            ax=ax,
            palette="pastel",
        )
        ax.set_title("Diagrama de caja por mandante")
        ax.set_xlabel("Mandante")
        ax.set_ylabel("Contactos por día")
        fig_to_streamlit(fig)

    st.write("**Distribución statistics by mandante:**")
    dist_stats = (
        daily_per_mandante.groupby("id_mandante")["countcd"].describe().reset_index()
    )
    numeric_cols = dist_stats.select_dtypes(include="number").columns
    st.dataframe(
        dist_stats.style.format({col: "{:.2f}" for col in numeric_cols}),
        use_container_width=True,
    )

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Distribución by Group:**")
        daily_per_grupo = (
            df.groupby(["fecha_llamada", "grupo"])["countcd"].sum().reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=daily_per_grupo, x="grupo", y="countcd", ax=ax, palette="pastel"
        )
        ax.set_title("Diagrama de caja por grupo")
        ax.set_xlabel("Grupo")
        ax.set_ylabel("Contactos por día")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig_to_streamlit(fig)

with tab5:
    st.subheader("Descriptive Estadísticas")

    daily_per_mandante = (
        df.groupby(["fecha_llamada", "id_mandante"])["countcd"].sum().reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Global Daily Contact Estadísticas:**")
        stats_df = pd.DataFrame(
            {
                "Métrica": [
                    "Media",
                    "Mediana",
                    "Desviación estándar",
                    "Mínimo",
                    "Máximo",
                    "Q1 (25%)",
                    "Q3 (75%)",
                    "Rango intercuartílico",
                    "Asimetría",
                    "Curtosis",
                    "Total",
                    "Coef. de Variación",
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
                    (
                        f"{daily_per_mandante['countcd'].std() / daily_per_mandante['countcd'].mean():.4f}"
                        if daily_per_mandante["countcd"].mean() != 0
                        else "N/A"
                    ),
                ],
            }
        )
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.write("**Top 10 fechas con más contactos:**")
        top_dates = (
            df.groupby("fecha_llamada")["countcd"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_dates["fecha_llamada"] = top_dates["fecha_llamada"].dt.strftime("%Y-%m-%d")
        st.dataframe(top_dates, use_container_width=True)

    st.write("**Estadísticas by mandante (daily):**")
    mandante_stats = (
        daily_per_mandante.groupby("id_mandante")["countcd"]
        .agg(
            [
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("count", "count"),
            ]
        )
        .reset_index()
    )
    mandante_stats.columns = [
        "Mandante",
        "Media",
        "Mediana",
        "Desv. Est.",
        "Mín.",
        "Máx.",
        "Registros",
    ]
    mandante_stats = mandante_stats.sort_values("Media", ascending=False)
    st.dataframe(
        mandante_stats.style.format(
            {
                "Media": "{:.2f}",
                "Mediana": "{:.2f}",
                "Desv. Est.": "{:.2f}",
                "Mín.": "{:.0f}",
                "Máx.": "{:.0f}",
                "Registros": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    st.write("**Matriz de correlación (variables temporales vs Contactos):**")
    daily_corr = (
        df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contacts")
    )
    daily_corr["day_of_year"] = daily_corr["fecha_llamada"].dt.dayofyear
    daily_corr["day_of_month"] = daily_corr["fecha_llamada"].dt.day
    daily_corr["month"] = daily_corr["fecha_llamada"].dt.month
    daily_corr["weekday_num"] = daily_corr["fecha_llamada"].dt.dayofweek
    daily_corr["week_of_month"] = calc_week_of_month(daily_corr["fecha_llamada"])
    daily_corr["is_monday"] = (daily_corr["weekday_num"] == 0).astype(int)
    daily_corr["is_friday"] = (daily_corr["weekday_num"] == 4).astype(int)

    corr_cols = [
        "contacts",
        "day_of_year",
        "day_of_month",
        "month",
        "weekday_num",
        "week_of_month",
        "is_monday",
        "is_friday",
    ]
    corr_matrix = daily_corr[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        ax=ax,
        fmt=".2f",
        linewidths=0.5,
    )
    ax.set_title("Correlación: Contactos vs variables temporales")
    labels = [
        "Contactos",
        "Día del año",
        "Día del mes",
        "Mes",
        "Día de la semana",
        "Semana del mes",
        "Es lunes",
        "Es viernes",
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig_to_streamlit(fig)

    st.write("**Promedio de contactos por día de la semana:**")
    daily_dow = (
        df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contacts")
    )
    daily_dow["weekday_num"] = daily_dow["fecha_llamada"].dt.dayofweek
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_agg = (
        daily_dow.groupby("weekday_num")["contacts"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    dow_agg["day_name"] = dow_agg["weekday_num"].map(
        lambda x: day_names[x] if x < 7 else "Unknown"
    )
    dow_agg = dow_agg.sort_values("weekday_num")
    st.dataframe(
        dow_agg[["day_name", "mean", "sum", "count"]]
        .rename(
            columns={
                "day_name": "Día",
                "mean": "Promedio",
                "sum": "Total",
                "count": "Registros",
            }
        )
        .style.format({"Promedio": "{:.2f}", "Total": "{:.0f}", "Registros": "{:.0f}"}),
        use_container_width=True,
    )

    st.write("**Promedio de contactos por mes:**")
    daily_month = (
        df.groupby(["fecha_llamada", "id_mandante"]).size().reset_index(name="contacts")
    )
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    daily_month["month_num"] = daily_month["fecha_llamada"].dt.month
    month_agg = (
        daily_month.groupby("month_num")["contacts"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    month_agg["month_name"] = month_agg["month_num"].map(
        lambda x: month_names[x - 1] if 1 <= x <= 12 else "Unknown"
    )
    month_agg = month_agg.sort_values("month_num")
    st.dataframe(
        month_agg[["month_name", "mean", "sum", "count"]]
        .rename(
            columns={
                "month_name": "Mes",
                "mean": "Promedio",
                "sum": "Total",
                "count": "Registros",
            }
        )
        .style.format({"Promedio": "{:.2f}", "Total": "{:.0f}", "Registros": "{:.0f}"}),
        use_container_width=True,
    )

    if "grupo" in df.columns:
        st.write("---")
        st.write("**Estadísticas by Group:**")
        daily_grupo = (
            df.groupby(["fecha_llamada", "id_mandante", "grupo"])
            .size()
            .reset_index(name="contacts")
        )
        grupo_stats = (
            daily_grupo.groupby("grupo")["contacts"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )
        grupo_stats.columns = [
            "Grupo",
            "Contactos totales",
            "Promedio diario",
            "Registros",
        ]
        grupo_stats = grupo_stats.sort_values("Contactos totales", ascending=False)
        st.dataframe(
            grupo_stats.style.format(
                {
                    "Contactos totales": "{:,.0f}",
                    "Promedio diario": "{:.2f}",
                    "Registros": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

with tab6:
    st.subheader("CD Predicción - Current Month (Seasonal Decomposition)")

    hoy = datetime.now()
    mes_actual = hoy.strftime("%B %Y")
    st.info(f"Mostrando predicción para días hábiles (lun-vie): {mes_actual}")

    with st.spinner("Calculating prediction..."):
        df_prediccion, estacionalidad, _ = train_and_predict(df)

    if len(df_prediccion) > 0:
        for mandante in df_prediccion["mandante_name"].unique():
            st.write(f"### {mandante}")
            df_mandante = df_prediccion[
                df_prediccion["mandante_name"] == mandante
            ].copy()

            st.dataframe(
                df_mandante[
                    [
                        "date",
                        "day_of_month",
                        "weekday_num",
                        "month",
                        "mandante_id",
                        "mandante_name",
                        "weekday_name",
                        "prediction",
                    ]
                ].style.format(
                    {
                        "prediction": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

            total_mandante = df_mandante["prediction"].sum()
            st.metric(
                label=f"CD esperados totales ({mandante})",
                value=f"{total_mandante:,.1f}",
            )

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(
                df_mandante["date"],
                df_mandante["prediction"],
                color=sns.color_palette("Blues_d", len(df_mandante)),
            )
            ax.set_title(f"CD esperados por día - {mandante} - {mes_actual}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Expected CDs")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig_to_streamlit(fig)
    else:
        st.warning("No hay suficientes datos para generar la predicción.")

with tab7:
    st.subheader("Factores estacionales by Mandante")

    with st.spinner("Calculando factores estacionales..."):
        df_prediccion, estacionalidad, _ = train_and_predict(df)

    if len(df_prediccion) > 0:
        for mandante, est in estacionalidad.items():
            st.write(
                f"**{mandante}** (promedio global: {est['global_avg']:.1f}, tendencia: {est['trend_factor']:.2f}x)"
            )

            dow_df = pd.DataFrame(
                {
                    "Día": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "Factor": [est["dow_factor"].get(i, 1.0) for i in range(5)],
                }
            )
            st.write("**Factor por día de la semana:**")
            st.dataframe(
                dow_df.style.format({"Factor": "{:.3f}"}), use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(
                dow_df["Día"], dow_df["Factor"], color=sns.color_palette("viridis", 5)
            )
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor por día de la semana - {mandante}")
            ax.set_ylabel("Multiplicador")
            fig.tight_layout()
            fig_to_streamlit(fig)

            semana_df = pd.DataFrame(
                {
                    "Semana del mes": [1, 2, 3, 4, 5],
                    "Factor": [est["week_factor"].get(i, 1.0) for i in range(1, 6)],
                }
            )
            st.write("**Factor por semana del mes:**")
            st.dataframe(
                semana_df.style.format({"Factor": "{:.3f}"}), use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(
                semana_df["Semana del mes"],
                semana_df["Factor"],
                color=sns.color_palette("Blues_d", 5),
            )
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor por semana del mes - {mandante}")
            ax.set_xlabel("Semana del mes")
            ax.set_ylabel("Multiplicador")
            ax.set_xticks([1, 2, 3, 4, 5])
            fig.tight_layout()
            fig_to_streamlit(fig)

            nombres_meses = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            mes_df = pd.DataFrame(
                {
                    "Mes": nombres_meses,
                    "Factor": [est["month_factor"].get(i, 1.0) for i in range(1, 13)],
                }
            )
            st.write("**Factor por mes (estacionalidad anual):**")
            st.dataframe(
                mes_df.style.format({"Factor": "{:.3f}"}), use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(
                mes_df["Mes"],
                mes_df["Factor"],
                color=sns.color_palette("Greens_d", 12),
            )
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Factor mensual anual - {mandante}")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Multiplicador")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            fig_to_streamlit(fig)

        if "hora_llamada" in df.columns:
            st.write("---")
            st.write("**Estacionalidad horaria por mandante (7:00 - 21:00):**")

            df_hora = df.copy()
            df_hora["hour"] = pd.to_numeric(df_hora["hora_llamada"], errors="coerce")
            df_hora = df_hora.dropna(subset=["hour"])
            df_hora["hour"] = df_hora["hour"].astype(int)
            df_hora = df_hora[(df_hora["hour"] >= 7) & (df_hora["hour"] <= 21)]

            total_por_mandante_hora = df_hora.groupby("id_mandante").size()

            horas_rango = list(range(7, 22))

            for mandante in sorted(df_hora["id_mandante"].unique()):
                st.write(f"**{mandante}**")

                mandante_data = df_hora[df_hora["id_mandante"] == mandante]
                total_mandante = total_por_mandante_hora[mandante]
                global_avg_hora = total_mandante / len(horas_rango)

                hora_counts = mandante_data.groupby("hour").size()
                hora_factor = {}
                for h in horas_rango:
                    if h in hora_counts.index:
                        hora_factor[h] = hora_counts[h] / global_avg_hora
                    else:
                        hora_factor[h] = 0.0

                hora_df = pd.DataFrame(
                    {
                        "Hora": horas_rango,
                        "Factor": [hora_factor.get(h, 0.0) for h in horas_rango],
                        "Contactos": [int(hora_counts.get(h, 0)) for h in horas_rango],
                    }
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(
                        hora_df.style.format(
                            {"Factor": "{:.3f}", "Contactos": "{:,.0f}"}
                        ),
                        use_container_width=True,
                    )
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.bar(
                        hora_df["Hora"],
                        hora_df["Factor"],
                        color=sns.color_palette("Oranges_d", len(horas_rango)),
                    )
                    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
                    ax.set_title(f"Factor horario - {mandante}")
                    ax.set_xlabel("Hora")
                    ax.set_ylabel("Multiplicador")
                    ax.set_xticks(horas_rango)
                    fig.tight_layout()
                    fig_to_streamlit(fig)
    else:
        st.warning("Not enough data to generate the seasonal factors.")

with tab8:
    st.subheader("Descargar objetivos diarios")

    hoy = datetime.now()
    mes_actual = hoy.strftime("%B %Y")

    with st.spinner("Calculando objetivos..."):
        df_prediccion, estacionalidad, _ = train_and_predict(df)

    if len(df_prediccion) > 0:
        global_targets = {}
        for mandante in df_prediccion["mandante_name"].unique():
            df_mandante = df_prediccion[df_prediccion["mandante_name"] == mandante]
            global_targets[mandante] = df_mandante["prediction"].sum()

        download_rows = []
        for _, row in df_prediccion.iterrows():
            download_rows.append(
                {
                    "day": row["day_of_month"],
                    "month number": row["month"],
                    "year": hoy.year,
                    "portfolio": row["mandante_name"],
                    "daily target": row["prediction"],
                    "global target": global_targets[row["mandante_name"]],
                }
            )

        df_download = pd.DataFrame(download_rows)

        st.dataframe(df_download, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_download.to_excel(writer, index=False, sheet_name="Hoja1")

        st.download_button(
            label="Descargar objetivos diarios",
            data=output.getvalue(),
            file_name=f"daily_targets_{mes_actual.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning("No hay suficientes datos para generar los objetivos.")
