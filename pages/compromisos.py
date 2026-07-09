import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

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
            "fecha de compromiso": "FECHA DE COMPR",
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

        # Calcular próximo día hábil
        hoy = datetime.now().date()
        proximo_dia = hoy + timedelta(days=1)
        # Si cae en sábado (5), saltar al lunes (7)
        if proximo_dia.weekday() == 5:
            proximo_dia += timedelta(days=2)
        # Si cae en domingo (6), saltar al lunes (8)
        elif proximo_dia.weekday() == 6:
            proximo_dia += timedelta(days=1)

        # Convertir columna de fecha y filtrar
        df_final["FECHA DE COMPR"] = pd.to_datetime(
            df_final["FECHA DE COMPR"], errors="coerce", dayfirst=True
        )
        df_filtrado = df_final[df_final["FECHA DE COMPR"].dt.date == proximo_dia].copy()

        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa de los datos originales")
        st.dataframe(df.head())
        st.write(f"Total de filas: {len(df)}")
        st.write(f"Total de columnas: {len(df.columns)}")

        st.subheader(f"Compromisos para el {proximo_dia.strftime('%d/%m/%Y')}")
        st.dataframe(df_filtrado)
        st.write(
            f"Total de registros para el {proximo_dia.strftime('%d/%m/%Y')}: {len(df_filtrado)}"
        )
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Carga un archivo Excel para comenzar")
