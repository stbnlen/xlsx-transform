import io
from datetime import datetime

import pandas as pd
import streamlit as st

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


def show_pagos_bci_view() -> None:
    """Display PAGOS BCI view for uploading and previewing Excel files."""
    st.header("PAGOS BCI")

    uploaded_file = st.file_uploader(
        "Subir archivo Excel", type=["xlsx", "xls"], key="pagos_bci_uploader"
    )

    if uploaded_file is not None:
        with st.spinner("Leyendo archivo..."):
            df = pd.read_excel(uploaded_file)

        # Show preview
        st.subheader("Vista previa (primeras 5 filas):")
        st.dataframe(df.head())

        # Show basic info
        st.write(f"Dimensiones: {df.shape}")
        st.write(f"Columnas: {list(df.columns)}")
    else:
        st.info(
            "Módulo en construcción: sube un archivo Excel para ver la vista previa."
        )


def show_bci_view() -> None:
    """Display BCI view with 3 file uploaders and merge logic."""
    st.header("BCI")

    maestro_file = st.file_uploader(
        "MAESTRO CLIENTE", type=["xlsx", "xls"], key="bci_maestro_cliente"
    )
    deuda_file = st.file_uploader(
        "DEUDA CASTIGO", type=["xlsx", "xls"], key="bci_deuda_castigo"
    )
    cubo_file = st.file_uploader("CUBO", type=["csv"], key="bci_cubo")

    if maestro_file and deuda_file and cubo_file:
        try:
            with st.spinner("Leyendo y combinando archivos..."):
                df_maestro = pd.read_excel(maestro_file)
                df_deuda = pd.read_excel(deuda_file)
                df_cubo = pd.read_csv(cubo_file, encoding="latin-1", sep=";")

                df_maestro["Source.Name"] = maestro_file.name

                df_maestro["rut cliente 2"] = (
                    df_maestro["rut_cliente"].astype(str).str[-1]
                )

                df_maestro["rut_norm"] = (
                    df_maestro["rut_cliente"]
                    .astype(str)
                    .str.replace("-", "")
                    .str.lstrip("0")
                )
                df_cubo["rut_norm"] = (
                    df_cubo["rut_cli"].astype(str).str.replace("-", "").str.lstrip("0")
                )
                df_deuda["rut_norm"] = (
                    df_deuda["fld_rut_deudor"]
                    .astype(str)
                    .str.replace("-", "")
                    .str.lstrip("0")
                )

                df_merged = df_maestro.merge(
                    df_deuda[["rut_norm", "fld_saldo"]],
                    on="rut_norm",
                    how="left",
                )

                df_merged = df_merged.merge(
                    df_cubo[["rut_norm", "mto_sdo_act"]],
                    on="rut_norm",
                    how="left",
                )

                df_result = (
                    df_merged[
                        [
                            "Source.Name",
                            "rut_cliente",
                            "rut cliente 2",
                            "ap_paterno",
                            "ap_materno",
                            "nombres",
                            "fld_saldo",
                            "mto_sdo_act",
                        ]
                    ]
                    .rename(
                        columns={
                            "rut_cliente": "rut_cliente.1",
                            "rut cliente 2": "rut_cliente.2",
                            "fld_saldo": "ARCHIVO DEUDA ASIG.fld_saldo",
                            "mto_sdo_act": "CUBO.SALDO ACTUAL RUT",
                        }
                    )
                    .copy()
                )

                df_result["ORIGEN"] = "STOCK"

                df_result["rut_cliente.1"] = (
                    df_result["rut_cliente.1"].astype(str).str[:-1].str.lstrip("0")
                )

            st.subheader("Vista previa (primeras 5 filas):")
            st.dataframe(df_result.head())

            st.write(f"Dimensiones: {df_result.shape}")
            st.write(f"Columnas: {list(df_result.columns)}")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_result.to_excel(writer, index=False)
            output.seek(0)

            # Generar nombre de archivo con formato BCI_{dia}_{mes}_{año}
            ahora = datetime.now()
            dia_actual = ahora.day
            mes_actual = MESES_ESPANOL[ahora.month]
            anio_actual = ahora.year
            nombre_archivo = f"BCI_{dia_actual}_{mes_actual}_{anio_actual}.xlsx"

            st.download_button(
                label="Descargar Excel combinado",
                data=output.getvalue(),
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Error al combinar los archivos: {e}")
            st.info(
                "Verifica que MAESTRO CLIENTE tenga las columnas rut_cliente, "
                "ap_paterno, ap_materno y nombres; DEUDA CASTIGO la columna "
                "fld_rut_deudor y fld_saldo; y CUBO las columnas rut_cli y "
                "mto_sdo_act."
            )
    else:
        st.info(
            "Carga los 3 archivos (MAESTRO CLIENTE, DEUDA CASTIGO y CUBO) para ver el resultado."
        )
