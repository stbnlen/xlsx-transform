import io

import pandas as pd
import streamlit as st
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

REPORT_SHEET = "REPORTES_GESTIONES_TODAS_ETAPAS"

st.title("Reporte CAE")

col1, col2 = st.columns(2)

df_anterior = None
df_actual = None
key_anterior_col = None

with col1:
    st.header("Reporte día anterior")
    file_anterior = st.file_uploader(
        "Subir reporte día anterior",
        type=["xlsx", "xls"],
        key="cae_anterior",
    )
    if file_anterior is not None:
        try:
            with st.spinner("Leyendo reporte día anterior..."):
                df_anterior = pd.read_excel(file_anterior, sheet_name=REPORT_SHEET)
            llave_matches = [
                c for c in df_anterior.columns if str(c).lower() == "llave"
            ]
            if llave_matches:
                candidate_col = llave_matches[0]
                candidate_values = df_anterior[candidate_col]
                valid_values = candidate_values.notna() & candidate_values.astype(
                    str
                ).str.strip().ne("")
                if valid_values.any():
                    key_anterior_col = candidate_col
            if key_anterior_col is None:
                llave_values = df_anterior["OPERACIÓN"].astype(str) + df_anterior[
                    "ETAPA SATCHMO"
                ].astype(str)
                if llave_matches:
                    key_anterior_col = llave_matches[0]
                    df_anterior[key_anterior_col] = llave_values
                else:
                    key_anterior_col = "Llave"
                    df_anterior.insert(0, key_anterior_col, llave_values)
                st.info(
                    "La columna 'Llave' del reporte día anterior estaba vacía o "
                    "no existía; se generó con OPERACIÓN + ETAPA SATCHMO."
                )
            st.success(f"Archivo cargado: {file_anterior.name}")
            st.dataframe(df_anterior.head())
        except Exception as e:
            df_anterior = None
            st.error(f"Error al leer el reporte día anterior: {e}")
            st.info(
                f"El archivo debe contener la hoja '{REPORT_SHEET}' y, si no "
                "incluye la columna 'Llave', las columnas 'OPERACIÓN' y "
                "'ETAPA SATCHMO'."
            )

with col2:
    st.header("Reporte actual")
    file_actual = st.file_uploader(
        "Subir reporte actual",
        type=["xlsx", "xls"],
        key="cae_actual",
    )
    if file_actual is not None:
        try:
            with st.spinner("Leyendo reporte actual..."):
                df_actual = pd.read_excel(file_actual, sheet_name=REPORT_SHEET)
            key_col = df_actual.columns[0]
            df_actual[key_col] = df_actual["OPERACIÓN"].astype(str) + df_actual[
                "ETAPA SATCHMO"
            ].astype(str)
            st.success(f"Archivo cargado: {file_actual.name}")
            st.dataframe(df_actual.head())
        except Exception as e:
            df_actual = None
            st.error(f"Error al leer el reporte actual: {e}")
            st.info(
                f"El archivo debe contener la hoja '{REPORT_SHEET}' y las "
                "columnas 'OPERACIÓN' y 'ETAPA SATCHMO'."
            )

if df_anterior is not None and df_actual is not None:
    st.divider()
    try:
        keys_anterior = set(df_anterior[key_anterior_col].astype(str))
        key_actual_col = df_actual.columns[0]
        df_nuevos = df_actual[
            ~df_actual[key_actual_col].astype(str).isin(keys_anterior)
        ]
        df_nuevos = df_nuevos.drop(columns=[key_actual_col])
        df_nuevos["OPERACIÓN"] = df_nuevos["OPERACIÓN"].astype(str)
        df_nuevos["CÓDIGO TRÁMITE"] = df_nuevos["CÓDIGO TRÁMITE"].astype(str)
        df_nuevos["FECHA SATCHMO"] = pd.to_datetime(
            df_nuevos["FECHA SATCHMO"], errors="coerce"
        ).dt.strftime("%d-%m-%Y")
        df_nuevos["FECHA CREA"] = pd.to_datetime(
            df_nuevos["FECHA CREA"], errors="coerce"
        ).dt.strftime("%d-%m-%Y")
        df_nuevos["RUT DEUDOR"] = pd.to_numeric(
            df_nuevos["RUT DEUDOR"], errors="coerce"
        )
        st.subheader("Registros nuevos en reporte actual")
        st.write(f"Total registros nuevos: {len(df_nuevos)}")
        st.dataframe(df_nuevos.head())

        if df_nuevos.empty:
            st.info("No hay registros nuevos entre ambos reportes.")
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_nuevos.to_excel(writer, sheet_name=REPORT_SHEET, index=False)
                ws = writer.sheets[REPORT_SHEET]
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
                for idx, column in enumerate(df_nuevos.columns, start=1):
                    width = max(
                        len(str(column)),
                        int(df_nuevos[column].astype(str).str.len().max()),
                    )
                    ws.column_dimensions[get_column_letter(idx)].width = min(
                        width + 2, 255
                    )
            st.download_button(
                label="Descargar registros nuevos",
                data=output.getvalue(),
                file_name="registros_nuevos_cae.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Error al comparar los reportes: {e}")
        st.info(
            "Verifica que ambos reportes tengan las columnas 'OPERACIÓN', "
            "'CÓDIGO TRÁMITE', 'FECHA SATCHMO', 'FECHA CREA' y 'RUT DEUDOR'."
        )
elif file_anterior is not None or file_actual is not None:
    st.info("Carga ambos reportes para comparar y detectar registros nuevos.")
