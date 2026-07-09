import pandas as pd
import streamlit as st

from styles import setup_page

setup_page("Compromisos", "")

st.title("Compromisos")

uploaded_file = st.file_uploader(
    "Cargar archivo Excel",
    type=["xlsx", "xls"],
    key="compromisos_uploader",
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, dtype=str)

        # Eliminar columnas L, M, N, O (posiciones 11, 12, 13, 14)
        columnas_a_eliminar = [11, 12, 13, 14]
        df = df.drop(df.columns[columnas_a_eliminar], axis=1)

        st.success("Archivo cargado correctamente")
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
        st.write(f"Total de filas: {len(df)}")
        st.write(f"Total de columnas: {len(df.columns)}")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Carga un archivo Excel para comenzar")
