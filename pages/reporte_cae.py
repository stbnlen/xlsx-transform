import streamlit as st

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
        st.success(f"Archivo cargado: {file_anterior.name}")

with col2:
    st.header("Reporte actual")
    file_actual = st.file_uploader(
        "Subir reporte actual",
        type=["xlsx", "xls"],
        key="cae_actual",
    )
    if file_actual is not None:
        st.success(f"Archivo cargado: {file_actual.name}")

if file_anterior is not None and file_actual is not None:
    st.divider()
    st.info("Ambos archivos cargados. Procesamiento en construcción...")
