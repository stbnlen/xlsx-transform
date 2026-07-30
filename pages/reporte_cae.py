import io

import pandas as pd
import streamlit as st
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from styles import setup_page

setup_page("Reporte CAE", "📊")

st.title("Reporte CAE")

col1, col2 = st.columns(2)

with col1:
    st.header("Reporte día anterior")
    file_anterior = st.file_uploader(
        "Subir reporte día anterior",
        type=["xlsx", "xls"],
        key="cae_anterior",
    )
    if file_anterior is not None:
        df_anterior = pd.read_excel(
            file_anterior, sheet_name="REPORTES_GESTIONES_TODAS_ETAPAS"
        )
        st.success(f"Archivo cargado: {file_anterior.name}")
        st.dataframe(df_anterior.head())

with col2:
    st.header("Reporte actual")
    file_actual = st.file_uploader(
        "Subir reporte actual",
        type=["xlsx", "xls"],
        key="cae_actual",
    )
    if file_actual is not None:
        df_actual = pd.read_excel(
            file_actual, sheet_name="REPORTES_GESTIONES_TODAS_ETAPAS"
        )
        key_col = df_actual.columns[0]
        df_actual[key_col] = df_actual["OPERACIÓN"].astype(str) + df_actual[
            "ETAPA SATCHMO"
        ].astype(str)
        st.success(f"Archivo cargado: {file_actual.name}")
        st.dataframe(df_actual.head())

if file_anterior is not None and file_actual is not None:
    st.divider()
    key_anterior_col = df_anterior.columns[0]
    keys_anterior = set(df_anterior[key_anterior_col].astype(str))
    key_actual_col = df_actual.columns[0]
    df_nuevos = df_actual[~df_actual[key_actual_col].astype(str).isin(keys_anterior)]
    df_nuevos = df_nuevos.drop(columns=[key_actual_col])
    df_nuevos["OPERACIÓN"] = df_nuevos["OPERACIÓN"].astype(str)
    df_nuevos["RUT DEUDOR"] = pd.to_numeric(df_nuevos["RUT DEUDOR"], errors="coerce")
    st.subheader("Registros nuevos en reporte actual")
    st.write(f"Total registros nuevos: {len(df_nuevos)}")
    st.dataframe(df_nuevos.head())

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_nuevos.to_excel(
            writer, sheet_name="REPORTES_GESTIONES_TODAS_ETAPAS", index=False
        )
        ws = writer.sheets["REPORTES_GESTIONES_TODAS_ETAPAS"]
        end_col = get_column_letter(len(df_nuevos.columns))
        end_row = len(df_nuevos) + 1
        table_ref = f"A1:{end_col}{end_row}"
        table = Table(displayName="RegistrosNuevos", ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium7",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)
    st.download_button(
        label="Descargar registros nuevos",
        data=output.getvalue(),
        file_name="registros_nuevos_cae.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
