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

        # Mapeo de columnas (clave: inicio del nombre en minúsculas, valor: nombre final)
        mapeo_columnas = {
            "nombre empresa": "EMPRESA",
            "cobra": "Sigla_Cobra",
            "rut clte": "Rut_cliente",
            "operacion": "Operación",
            "monto a pago": "Monto",
            "tipo pago": "Tipo_de_Pago",
            "forma de pago": "Modo",
        }

        # Construir diccionario de rename por coincidencia parcial
        rename_dict = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            for clave, nuevo_nombre in mapeo_columnas.items():
                if col_lower.startswith(clave):
                    rename_dict[col] = nuevo_nombre
                    break

        df_renombrado = df.rename(columns=rename_dict)

        # Filtrar solo las columnas mapeadas
        columnas_finales = list(rename_dict.values())
        df_final = df_renombrado[columnas_finales]

        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa de los datos originales")
        st.dataframe(df.head())
        st.write(f"Total de filas: {len(df)}")
        st.write(f"Total de columnas: {len(df.columns)}")

        st.subheader("Dataframe final (solo columnas seleccionadas)")
        st.dataframe(df_final.head())
        st.write(f"Total de filas: {len(df_final)}")
        st.write(f"Total de columnas: {len(df_final.columns)}")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Carga un archivo Excel para comenzar")
