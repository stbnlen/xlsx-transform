import streamlit as st
import pandas as pd
import io


def show_pagos_bci_view():
    """Display PAGOS BCI view for uploading and previewing Excel files."""
    st.header("PAGOS BCI")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader")

    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Show preview
        st.subheader("Data Preview (First 5 rows):")
        st.dataframe(df.head())

        # Show basic info
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
    else:
        st.write("En construcción - Por favor sube un archivo Excel para ver la vista previa")


def show_bci_view():
    """Display BCI view with 3 file uploaders and merge logic."""
    st.header("BCI")

    maestro_file = st.file_uploader("MAESTRO CLIENTE", type=["xlsx", "xls"], key="bci_maestro_cliente")
    deuda_file = st.file_uploader("DEUDA CASTIGO", type=["xlsx", "xls"], key="bci_deuda_castigo")
    cubo_file = st.file_uploader("CUBO", type=["csv"], key="bci_cubo")

    if maestro_file and deuda_file and cubo_file:
        df_maestro = pd.read_excel(maestro_file)
        df_deuda = pd.read_excel(deuda_file)
        df_cubo = pd.read_csv(cubo_file, encoding="latin-1", sep=";")

        df_maestro["Source.Name"] = maestro_file.name

        df_maestro["rut cliente 2"] = df_maestro["rut_cliente"].astype(str).str[-1]

        df_maestro["rut_norm"] = df_maestro["rut_cliente"].astype(str).str.replace("-", "").str.lstrip("0")
        df_cubo["rut_norm"] = df_cubo["rut_cli"].astype(str).str.replace("-", "").str.lstrip("0")
        df_deuda["rut_norm"] = df_deuda["fld_rut_deudor"].astype(str).str.replace("-", "").str.lstrip("0")

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

        df_result = df_merged[
            ["Source.Name", "rut_cliente", "rut cliente 2", "ap_paterno", "ap_materno", "nombres", "fld_saldo", "mto_sdo_act"]
        ].rename(columns={
            "rut_cliente": "rut_cliente.1",
            "rut cliente 2": "rut_cliente.2",
            "fld_saldo": "ARCHIVO DEUDA ASIG.fld_saldo",
            "mto_sdo_act": "CUBO.SALDO ACTUAL RUT",
        }).copy()

        df_result["ORIGEN"] = "STOCK"

        df_result["rut_cliente.1"] = df_result["rut_cliente.1"].astype(str).str[:-1].str.lstrip("0")

        st.subheader("Data Preview (First 5 rows):")
        st.dataframe(df_result.head())

        st.write(f"Shape: {df_result.shape}")
        st.write(f"Columns: {list(df_result.columns)}")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_result.to_excel(writer, index=False)
        output.seek(0)

        st.download_button(
            label="Download merged Excel file",
            data=output,
            file_name="bci_merged.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.write("Sube los 3 archivos para ver el resultado")
