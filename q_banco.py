import io
from datetime import datetime

import pandas as pd
import streamlit as st

from utils import (
    normalize_column_name,
    validate_required_columns,
)

MESES_ESPANOL = {
    1: "ene",
    2: "feb",
    3: "mar",
    4: "abr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "ago",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dic",
}

PREVIEW_ROWS = 100


def show_q_banco_view() -> None:
    """Display Q_BANCO view for filtering and downloading Excel files."""
    uploaded_file = st.file_uploader(
        "Subir archivo Excel", type=["xlsx", "xls"], key="q_banco_uploader"
    )

    if uploaded_file is not None:
        with st.spinner("Leyendo archivo..."):
            df = pd.read_excel(uploaded_file)

        st.subheader("Vista previa de datos originales:")
        st.dataframe(df.head(PREVIEW_ROWS))
        st.caption(
            f"Mostrando las primeras {min(PREVIEW_ROWS, len(df))} de "
            f"{len(df)} filas."
        )
        st.write(f"Dimensiones originales: {df.shape}")

        columns_to_keep = [
            "rut",
            "dv",
            "n_operacion_principal",
            "origen_core",
            "nombre_completo_cliente",
            "SUCURSAL",
            "CARTERA",
            "ESTADO CRM",
            "ESTADO JUDICIAL",
            "saldo_capital",
            "% DESCUENTO",
            "comuna_particular",
        ]

        missing_columns, column_mapping = validate_required_columns(
            df.columns, columns_to_keep
        )

        if missing_columns:
            st.error(f"Faltan columnas en el archivo cargado: {missing_columns}")
            st.write("Columnas disponibles:", list(df.columns))
            st.write(
                "Columnas disponibles normalizadas:",
                [normalize_column_name(col) for col in df.columns],
            )
        else:
            actual_columns_to_use = [column_mapping[col] for col in columns_to_keep]
            filtered_df = df[actual_columns_to_use].copy()

            # Rename specific columns for Q_BANCO output format
            filtered_df = filtered_df.rename(columns={"saldo_capital": "SALDO CAPITAL"})

            st.subheader("Vista previa de datos filtrados:")
            st.dataframe(filtered_df.head(PREVIEW_ROWS))
            st.write(f"Dimensiones filtradas: {filtered_df.shape}")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="Sheet1")

            # Generar nombre de archivo con formato BFA_{dia}_{mes}_{año}
            ahora = datetime.now()
            dia_actual = ahora.day
            mes_actual = MESES_ESPANOL[ahora.month]
            anio_actual = ahora.year
            nombre_archivo = f"BFA_{dia_actual}_{mes_actual}_{anio_actual}.xlsx"

            st.download_button(
                label="Descargar Excel filtrado",
                data=output.getvalue(),
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
