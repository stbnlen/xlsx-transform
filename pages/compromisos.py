import io

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from styles import setup_page

setup_page("Compromisos", "")

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

st.title("Compromisos")

uploaded_file = st.file_uploader(
    "Cargar archivo Excel",
    type=["xlsx", "xls"],
    key="compromisos_uploader",
)

if uploaded_file is not None:
    try:
        # Leer el archivo sin dtype=str para que las fechas se parseen automáticamente
        df = pd.read_excel(uploaded_file)

        # Eliminar columnas L, M, N, O (posiciones 11, 12, 13, 14)
        columnas_a_eliminar = [11, 12, 13, 14]
        df = df.drop(df.columns[columnas_a_eliminar], axis=1)

        # Mapeo de columnas (clave: inicio del nombre en minúsculas, valor: nombre final exacto)
        mapeo_columnas = {
            "nombre empresa": "EMPRESA",
            "cobra": "Sigla_Cobra",
            "rut clte": "  Rut_Cliente",
            "operacion": "Operación ",
            "monto a pago": "Monto  ",
            "tipo pago": "   Tipo_de_Pago ",
            "forma de pago": "Modo : presencial/boton de pago ",
            "fecha de compromiso": "FECHA DE COMPR.",
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

        # Filtrar solo las columnas mapeadas, con FECHA DE COMPR. al final
        columnas_finales = [
            col for col in rename_dict.values() if col != "FECHA DE COMPR."
        ]
        columnas_finales.append("FECHA DE COMPR.")
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

        # La columna FECHA DE COMPR. ya viene como datetime64[ns]
        fecha_col = "FECHA DE COMPR."

        # Filtrar por fecha
        df_filtrado = df_final[df_final[fecha_col].dt.date == proximo_dia].copy()

        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa de los datos originales")
        st.dataframe(df.head())
        st.write(f"Total de filas: {len(df)}")
        st.write(f"Total de columnas: {len(df.columns)}")

        st.subheader(f"Compromisos para el {proximo_dia.strftime('%d/%m/%Y')}")
        if len(df_filtrado) > 0:
            st.dataframe(df_filtrado)
            st.write(
                f"Total de registros para el {proximo_dia.strftime('%d/%m/%Y')}: {len(df_filtrado)}"
            )

            # Botón de descarga
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_filtrado.to_excel(writer, index=False, sheet_name="Hoja1")

                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

                worksheet = writer.sheets["Hoja1"]

                # Formato para encabezados: fondo rojo, letras blancas, negrita, Calibri 10
                header_fill = PatternFill(
                    start_color="FF0000", end_color="FF0000", fill_type="solid"
                )
                header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")

                # Bordes para todas las celdas
                thin_border = Border(
                    left=Side(style="thin"),
                    right=Side(style="thin"),
                    top=Side(style="thin"),
                    bottom=Side(style="thin"),
                )

                # Fuente para Tipo_de_Pago: negrita y rojo
                tipo_pago_font = Font(
                    name="Aptos Narrow", size=11, bold=True, color="FF0000"
                )

                # Fuente para Modo: negrita y negro
                modo_font = Font(
                    name="Aptos Narrow", size=11, bold=True, color="000000"
                )

                # Fuente para Fecha: negrita y rojo
                fecha_font = Font(
                    name="Aptos Narrow", size=11, bold=True, color="FF0000"
                )

                # Fondo celeste para Modo
                modo_fill = PatternFill(
                    start_color="CCFFFF", end_color="CCFFFF", fill_type="solid"
                )

                # Fuente normal para datos
                data_font = Font(name="Aptos Narrow", size=11)

                # Aplicar formato a la primera fila (encabezados)
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin_border

                # Alto de fila de encabezados
                worksheet.row_dimensions[1].height = 21.60

                # Encontrar índice de columnas especiales
                col_operacion = None
                col_monto = None
                col_tipo_pago = None
                col_modo = None
                col_fecha = None

                for idx, cell in enumerate(worksheet[1], 1):
                    col_name = cell.value.strip().lower() if cell.value else ""
                    if "operación" in col_name:
                        col_operacion = idx
                    elif "monto" in col_name:
                        col_monto = idx
                    elif "tipo_de_pago" in col_name:
                        col_tipo_pago = idx
                    elif "modo" in col_name:
                        col_modo = idx
                        # Aplicar formato especial al encabezado de Modo
                        cell.fill = modo_fill
                        cell.font = modo_font
                    elif "fecha" in col_name:
                        col_fecha = idx

                # Aplicar formato a las filas de datos
                for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
                    for cell in row:
                        cell.font = data_font
                        cell.border = thin_border

                        # Formatear columna Operación como número sin separador de miles
                        if col_operacion and cell.column == col_operacion:
                            cell.number_format = "0"

                        # Formatear columna Monto con separador de miles
                        if col_monto and cell.column == col_monto:
                            cell.number_format = "#,##0"

                        # Formatear columna Tipo_de_Pago: negrita y rojo
                        if col_tipo_pago and cell.column == col_tipo_pago:
                            cell.font = tipo_pago_font

                        # Formatear fecha sin hora
                        if col_fecha and cell.column == col_fecha:
                            cell.number_format = "d-mmm"
                            cell.font = fecha_font
                            cell.alignment = Alignment(
                                horizontal="center", vertical="center"
                            )

                # Ajustar ancho de columnas
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except Exception:
                            pass
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            excel_data = output.getvalue()

            dia_actual = proximo_dia.day
            mes_actual = MESES_ESPANOL[proximo_dia.month]
            anio_actual = proximo_dia.year
            nombre_archivo = f"compromisos_{dia_actual}_{mes_actual}_{anio_actual}.xlsx"

            st.download_button(
                label="Descargar archivo de compromisos",
                data=excel_data,
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning(f"No hay compromisos para el {proximo_dia.strftime('%d/%m/%Y')}")
            st.write("Fechas disponibles en el archivo:")
            fechas_unicas = df_final[fecha_col].dropna().dt.date.unique()
            fechas_ordenadas = sorted(fechas_unicas)
            for fecha in fechas_ordenadas[:10]:  # Mostrar primeras 10 fechas
                st.write(f"- {fecha.strftime('%d/%m/%Y')}")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
else:
    st.info("Carga un archivo Excel para comenzar")
