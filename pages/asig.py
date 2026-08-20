import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from pagos_bci import show_bci_view
from q_banco import show_q_banco_view
from q_cmr import show_q_cmr_view
from utils import validate_required_columns

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


def _find_column_insensitive(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column matching candidates case-insensitively."""
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _find_column_by_prefix(df: pd.DataFrame, prefix: str) -> str | None:
    """Find the first column whose name starts with prefix (case-insensitive)."""
    key = prefix.strip().lower()
    for col in df.columns:
        if str(col).strip().lower().startswith(key):
            return col
    return None


FORUM_COLUMN_ORDER = [
    "ORIGEN",
    "CONTRATO",
    "RUT",
    "NOMBRE CLIENTE",
    "MONTO CASTIGO",
    "ETAPA DEMANDA",
    "FECHA CASTIGO",
    "CIUDAD",
    "CARTERA",
    "Tipo gestión ",
    "Año castigo",
]

COP_STOCK_COLUMNS = [
    "RUT COM",
    "DV",
    "AISGNACION",
    "Demandado",
    "SALDO DEUDOR",
    "RUT COMPLETO",
    "ESTADO CRM",
    "Flujo/Stock",
    "FF",
]


def _normalize_rut_series(series: pd.Series) -> pd.Series:
    """Normalize RUT values: drop a leading negative sign, keep the digits
    before the verifier dash, and remove the float '.0' suffix."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"^-", "", regex=True)
        .str.split("-")
        .str[0]
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _year_from_date_series(series: pd.Series) -> pd.Series:
    """Extract clean year strings (e.g. '2026') from a date series."""
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y").fillna("")


def process_forum_data(
    df_castigo: pd.DataFrame,
    df_vigente: pd.DataFrame,
    filename1: str,
    filename2: str,
) -> pd.DataFrame:
    """Process and combine the two dataframes according to requirements."""

    # Process Castigo dataframe
    df_castigo_processed = process_single_file(df_castigo, filename1, "Castigo")

    # Process Vigente dataframe
    df_vigente_processed = process_single_file(df_vigente, filename2, "Vigente")

    # Combine the dataframes
    combined_df = pd.concat(
        [df_castigo_processed, df_vigente_processed], ignore_index=True
    )

    # Reorder columns to match the requested order
    # Ensure all required columns exist
    for col in FORUM_COLUMN_ORDER:
        if col not in combined_df.columns:
            combined_df[col] = ""

    # Fill NaN with empty string in columns that should be empty
    for col in ["ETAPA DEMANDA", "CIUDAD", "Tipo gestión "]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna("")

    # Return dataframe with columns in the specified order
    return combined_df[FORUM_COLUMN_ORDER]


def process_single_file(
    df: pd.DataFrame, filename: str, file_type: str
) -> pd.DataFrame:
    """Process a single file to extract and format required data."""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Clean RUT column (drop negative sign, dash and everything after it)
    if "RUT" in processed_df.columns:
        processed_df["RUT"] = _normalize_rut_series(processed_df["RUT"])
    elif "RUT COMP" in processed_df.columns:
        processed_df["RUT"] = _normalize_rut_series(processed_df["RUT COMP"])
    else:
        # If RUT column doesn't exist, create empty column
        processed_df["RUT"] = ""

    # Add ORIGEN column from filename (without extension)
    origen_name = os.path.splitext(filename)[0]  # Remove only the extension
    processed_df["ORIGEN"] = origen_name

    # Handle column mapping based on file type
    if file_type == "Castigo":
        # Map Castigo file columns (case-insensitive, e.g. "contrato"/"CONTRATO")
        if "CONTRATO" not in processed_df.columns:
            source_col = _find_column_insensitive(df, ["CONTRATO"])
            processed_df["CONTRATO"] = df[source_col] if source_col else ""

        if "NOMBRE CLIENTE" not in processed_df.columns:
            if "NOMBRE CLIENTE" in df.columns:
                processed_df["NOMBRE CLIENTE"] = df["NOMBRE CLIENTE"]
            elif "Demandado" in df.columns:
                processed_df["NOMBRE CLIENTE"] = df["Demandado"]
            else:
                processed_df["NOMBRE CLIENTE"] = ""

        # For MONTO CASTIGO, check multiple possible source column names
        if "MONTO CASTIGO" not in processed_df.columns:
            if "MONTO CASTIGO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["MONTO CASTIGO"]
            elif "Monto Castigo" in df.columns:
                processed_df["MONTO CASTIGO"] = df["Monto Castigo"]
            elif "SALDO CAPITAL SEMAFORO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["SALDO CAPITAL SEMAFORO"]
            elif "MONTO CASIIGO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["MONTO CASIIGO"]
            elif "FSALDOINSOLUTO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["FSALDOINSOLUTO"]
            else:
                processed_df["MONTO CASTIGO"] = ""

        if (
            "FECHA CASTIGO" not in processed_df.columns
            and "FECHA CASTIGO" in df.columns
        ):
            processed_df["FECHA CASTIGO"] = df["FECHA CASTIGO"]
        elif "FECHA CASTIGO" not in processed_df.columns:
            processed_df["FECHA CASTIGO"] = ""

        # For CARTERA, handle case sensitivity
        if "CARTERA" not in processed_df.columns:
            if "cartera" in df.columns:
                processed_df["CARTERA"] = df["cartera"]
            elif "CARTERA" in df.columns:
                processed_df["CARTERA"] = df["CARTERA"]
            elif "Marca Cartera" in df.columns:
                processed_df["CARTERA"] = df["Marca Cartera"]
            else:
                processed_df["CARTERA"] = ""
        # For Tipo gestión, map from Tipo de gestión column
        if "Tipo gestión " not in processed_df.columns:
            if "Tipo de gestión" in df.columns:
                processed_df["Tipo gestión "] = df["Tipo de gestión"]
            elif "Tipo gestión" in df.columns:
                processed_df["Tipo gestión "] = df["Tipo gestión"]
            else:
                processed_df["Tipo gestión "] = ""

        # Add Año castigo column based on FECHA CASTIGO
        if (
            "Año castigo" not in processed_df.columns
            and "FECHA CASTIGO" in processed_df.columns
        ):
            processed_df["Año castigo"] = _year_from_date_series(
                processed_df["FECHA CASTIGO"]
            )
        elif "Año castigo" not in processed_df.columns:
            processed_df["Año castigo"] = ""

    elif file_type == "Vigente":
        # Map Vigente file columns (case-insensitive, e.g. "contrato"/"CONTRATO")
        if "CONTRATO" not in processed_df.columns:
            source_col = _find_column_insensitive(df, ["NumContrato", "CONTRATO"])
            processed_df["CONTRATO"] = df[source_col] if source_col else ""

        if "NOMBRE CLIENTE" not in processed_df.columns:
            if "Nombre_Cliente" in df.columns:
                processed_df["NOMBRE CLIENTE"] = df["Nombre_Cliente"]
            elif "NOMBRE CLIENTE" in df.columns:
                processed_df["NOMBRE CLIENTE"] = df["NOMBRE CLIENTE"]
            elif "Demandado" in df.columns:
                processed_df["NOMBRE CLIENTE"] = df["Demandado"]
            else:
                processed_df["NOMBRE CLIENTE"] = ""

        # For vigente files, MONTO CASTIGO checks multiple possible source columns
        if "MONTO CASTIGO" not in processed_df.columns:
            if "fSaldoInsoluto" in df.columns:
                processed_df["MONTO CASTIGO"] = df["fSaldoInsoluto"]
            elif "FSALDOINSOLUTO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["FSALDOINSOLUTO"]
            elif "Monto Castigo" in df.columns:
                processed_df["MONTO CASTIGO"] = df["Monto Castigo"]
            elif "SALDO CAPITAL SEMAFORO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["SALDO CAPITAL SEMAFORO"]
            elif "MONTO CASIIGO" in df.columns:
                processed_df["MONTO CASTIGO"] = df["MONTO CASIIGO"]
            else:
                processed_df["MONTO CASTIGO"] = 0

        if (
            "FECHA CASTIGO" not in processed_df.columns
            and "Fecha Castigo" in df.columns
        ):
            processed_df["FECHA CASTIGO"] = df["Fecha Castigo"]
        elif "FECHA CASTIGO" not in processed_df.columns:
            processed_df["FECHA CASTIGO"] = ""

        # For CARTERA, handle case sensitivity - if not found, use "Dual vigente" for vigente files
        if "CARTERA" not in processed_df.columns:
            if "cartera" in df.columns:
                processed_df["CARTERA"] = df["cartera"]
            elif "CARTERA" in df.columns:
                processed_df["CARTERA"] = df["CARTERA"]
            elif "Marca Cartera" in df.columns:
                processed_df["CARTERA"] = df["Marca Cartera"]
            else:
                processed_df["CARTERA"] = (
                    "Dual vigente"  # Default for vigente files when column not found
                )

        # Add empty columns for the fields that should be empty initially
        processed_df["ETAPA DEMANDA"] = ""
        processed_df["CIUDAD"] = ""
        # Note: Tipo gestión is handled above for both file types

        # Add Año castigo column based on FECHA CASTIGO
        if (
            "Año castigo" not in processed_df.columns
            and "FECHA CASTIGO" in processed_df.columns
        ):
            processed_df["Año castigo"] = _year_from_date_series(
                processed_df["FECHA CASTIGO"]
            )
        elif "Año castigo" not in processed_df.columns:
            processed_df["Año castigo"] = ""

    return processed_df


def process_flujo_file(df_flujo: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Process a flujo file into the standard FORUM column layout."""
    processed_df = pd.DataFrame(index=df_flujo.index)

    origen_name = os.path.splitext(filename)[0]
    processed_df["ORIGEN"] = origen_name

    contrato_col = _find_column_insensitive(df_flujo, ["CONTRATO", "CONTRATO RS"])
    processed_df["CONTRATO"] = df_flujo[contrato_col] if contrato_col else ""

    rut_col = _find_column_insensitive(df_flujo, ["RUT", "RUT CLIENTE", "RUT COMP"])
    if rut_col:
        processed_df["RUT"] = _normalize_rut_series(df_flujo[rut_col])
    else:
        processed_df["RUT"] = ""

    nombre_col = _find_column_insensitive(
        df_flujo, ["NOMBRE CLIENTE", "NOMBRE", "Nombre_Cliente", "Demandado"]
    )
    processed_df["NOMBRE CLIENTE"] = df_flujo[nombre_col] if nombre_col else ""

    monto_col = _find_column_insensitive(
        df_flujo,
        [
            "MONTO CASTIGO",
            "SALDO CAPITAL SEMAFORO",
            "FSALDOINSOLUTO",
            "SALDO INSOLUTO",
            "SALDO INS",
        ],
    )
    if monto_col is None:
        monto_col = _find_column_by_prefix(df_flujo, "saldo ins")
    processed_df["MONTO CASTIGO"] = df_flujo[monto_col] if monto_col else ""

    processed_df["ETAPA DEMANDA"] = ""

    fecha_col = _find_column_insensitive(df_flujo, ["FECHA CASTIGO", "Fecha Castigo"])
    processed_df["FECHA CASTIGO"] = df_flujo[fecha_col] if fecha_col else ""

    processed_df["CIUDAD"] = ""

    cartera_col = _find_column_insensitive(
        df_flujo, ["CARTERA", "cartera", "Marca Cartera"]
    )
    processed_df["CARTERA"] = df_flujo[cartera_col] if cartera_col else ""

    processed_df["Tipo gestión "] = ""

    processed_df["Año castigo"] = _year_from_date_series(processed_df["FECHA CASTIGO"])

    return processed_df


def process_flujo_forum_data(
    df_stock: pd.DataFrame, df_flujo: pd.DataFrame, filename_flujo: str
) -> tuple[pd.DataFrame, int]:
    """Append flujo records to stock, skipping RUTs already present in stock.

    Returns the combined dataframe and the count of discarded duplicates.
    """
    df_flujo_processed = process_flujo_file(df_flujo, filename_flujo)

    stock_ruts: set[str] = set()
    if "RUT" in df_stock.columns:
        stock_ruts = set(_normalize_rut_series(df_stock["RUT"]))
        stock_ruts.discard("")
        stock_ruts.discard("nan")

    mask_new = ~df_flujo_processed["RUT"].isin(stock_ruts)
    discarded_count = int((~mask_new).sum())

    combined_df = pd.concat(
        [df_stock.copy(), df_flujo_processed[mask_new]], ignore_index=True
    )

    for col in FORUM_COLUMN_ORDER:
        if col not in combined_df.columns:
            combined_df[col] = ""

    for col in ["ETAPA DEMANDA", "CIUDAD", "Tipo gestión ", "Año castigo"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna("")

    return combined_df[FORUM_COLUMN_ORDER], discarded_count


COP_STOCK_SHEET = "Hoja2"


def read_cop_stock_file(file: io.BytesIO) -> pd.DataFrame:
    """Read COP stock data from the 'Hoja2' sheet of the Excel file."""
    try:
        return pd.read_excel(file, sheet_name=COP_STOCK_SHEET)
    except ValueError as e:
        raise ValueError(
            f"El archivo Stock no contiene la hoja '{COP_STOCK_SHEET}'"
        ) from e


st.title("Asignaciones")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Q_BANCO", "Q_CMR", "FORUM", "Flujo FORUM", "Flujo COP", "BCI"]
)

with tab1:
    st.header("Q_BANCO")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR")
    show_q_cmr_view()

with tab3:
    st.header("FORUM")

    # First file uploader
    st.subheader("Castigo")
    uploaded_file1 = st.file_uploader(
        "Selecciona el archivo Castigo", type=["xlsx", "xls"], key="forum_uploader1"
    )

    # Second file uploader
    st.subheader("Vigente")
    uploaded_file2 = st.file_uploader(
        "Selecciona el archivo Vigente", type=["xlsx", "xls"], key="forum_uploader2"
    )

    # Process files when both are uploaded
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("Ambos archivos se cargaron correctamente.")

        try:
            # Read both Excel files
            with st.spinner("Leyendo archivos..."):
                df_castigo = pd.read_excel(uploaded_file1)
                df_vigente = pd.read_excel(uploaded_file2)

            # Show previews with type handling to avoid Arrow conversion errors
            st.write("Vista previa del archivo Castigo:")
            display_castigo = df_castigo.head().astype(str)
            st.dataframe(display_castigo)

            st.write("Vista previa del archivo Vigente:")
            display_vigente = df_vigente.head().astype(str)
            st.dataframe(display_vigente)

            # Process and combine the dataframes
            combined_df = process_forum_data(
                df_castigo, df_vigente, uploaded_file1.name, uploaded_file2.name
            )

            # Show the combined result with type handling to avoid Arrow conversion errors
            st.write("Datos combinados (listos para descargar):")
            display_combined = combined_df.copy()
            for col in display_combined.columns:
                if display_combined[col].dtype == "object":
                    display_combined[col] = display_combined[col].astype(str)
            st.dataframe(display_combined)

            # Provide download button for XLSX
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                combined_df.to_excel(writer, sheet_name="Anexar1", index=False)
            excel_data = output.getvalue()

            ahora = datetime.now()
            dia_actual = ahora.day
            mes_actual = MESES_ESPANOL[ahora.month]
            anio_actual = ahora.year
            nombre_archivo = f"FRM_{dia_actual}_{mes_actual}_{anio_actual}.xlsx"

            st.download_button(
                label="Descargar datos combinados como XLSX",
                data=excel_data,
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")
            st.info(
                "Asegúrate de que los archivos tengan las columnas requeridas: CONTRATO, RUT, NOMBRE CLIENTE, MONTO CASTIGO, FECHA CASTIGO, CARTERA"
            )
    elif uploaded_file1 is not None:
        st.info("Carga el archivo Vigente para continuar.")
        try:
            with st.spinner("Leyendo archivo..."):
                df1 = pd.read_excel(uploaded_file1)
            st.write("Vista previa del archivo Castigo:")
            st.dataframe(df1.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    elif uploaded_file2 is not None:
        st.info("Carga el archivo Castigo para continuar.")
        try:
            with st.spinner("Leyendo archivo..."):
                df2 = pd.read_excel(uploaded_file2)
            st.write("Vista previa del archivo Vigente:")
            st.dataframe(df2.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    else:
        st.info("Carga ambos archivos, Castigo y Vigente, para continuar.")

with tab4:
    st.header("Flujo FORUM")
    st.write(
        "Une los registros de un archivo de flujo al stock generado en la pestaña "
        "FORUM. Los registros cuyo RUT ya existe en el stock se descartan."
    )

    stock_file = st.file_uploader(
        "Selecciona el archivo Stock (resultado de la pestaña FORUM)",
        type=["xlsx", "xls"],
        key="flujo_forum_stock",
    )
    flujo_file = st.file_uploader(
        "Selecciona el archivo Flujo",
        type=["xlsx", "xls"],
        key="flujo_forum_flujo",
    )

    if stock_file is not None and flujo_file is not None:
        st.success("Ambos archivos se cargaron correctamente.")

        try:
            df_stock = pd.read_excel(stock_file)
            df_flujo = pd.read_excel(flujo_file)

            st.write("Vista previa del archivo Stock:")
            st.dataframe(df_stock.head().astype(str))

            st.write("Vista previa del archivo Flujo:")
            st.dataframe(df_flujo.head().astype(str))

            combined_df, discarded_count = process_flujo_forum_data(
                df_stock, df_flujo, flujo_file.name
            )
            accepted_count = len(df_flujo) - discarded_count

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Registros en el flujo", len(df_flujo))
            col_b.metric("Aceptados (RUT nuevo)", accepted_count)
            col_c.metric("Descartados (RUT en stock)", discarded_count)

            st.write(
                f"Stock final: {len(combined_df)} registros "
                f"(stock original: {len(df_stock)})."
            )

            display_combined = combined_df.copy()
            for col in display_combined.columns:
                if display_combined[col].dtype == "object":
                    display_combined[col] = display_combined[col].astype(str)
            st.dataframe(display_combined)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                combined_df.to_excel(writer, sheet_name="Anexar1", index=False)
            excel_data = output.getvalue()

            ahora = datetime.now()
            dia_actual = ahora.day
            mes_actual = MESES_ESPANOL[ahora.month]
            anio_actual = ahora.year
            nombre_archivo = f"FRM_{dia_actual}_{mes_actual}_{anio_actual}.xlsx"

            st.download_button(
                label="Descargar stock actualizado como XLSX",
                data=excel_data,
                file_name=nombre_archivo,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error al procesar los archivos: {e}")
            st.info(
                "Asegúrate de que el archivo de flujo tenga las columnas requeridas: "
                "CONTRATO, RUT, NOMBRE, MONTO CASTIGO"
            )
    elif stock_file is not None:
        st.info("Carga el archivo Flujo para continuar.")
        try:
            df_stock_preview = pd.read_excel(stock_file)
            st.write("Vista previa del archivo Stock:")
            st.dataframe(df_stock_preview.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    elif flujo_file is not None:
        st.info("Carga el archivo Stock para continuar.")
        try:
            df_flujo_preview = pd.read_excel(flujo_file)
            st.write("Vista previa del archivo Flujo:")
            st.dataframe(df_flujo_preview.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    else:
        st.info("Carga ambos archivos, Stock y Flujo, para continuar.")

with tab5:
    st.header("Flujo COP")
    st.write(
        "Une los registros de un archivo de flujo al stock de COP. "
        "El stock se lee desde la hoja 'Hoja2' del archivo. "
        "Carga ambos archivos para ver sus vistas previas."
    )

    cop_stock_file = st.file_uploader(
        "Selecciona el archivo Stock",
        type=["xlsx", "xls"],
        key="flujo_cop_stock",
    )
    cop_flujo_file = st.file_uploader(
        "Selecciona el archivo Flujo",
        type=["xlsx", "xls"],
        key="flujo_cop_flujo",
    )

    if cop_stock_file is not None and cop_flujo_file is not None:
        st.success("Ambos archivos se cargaron correctamente.")

    if cop_stock_file is not None:
        try:
            df_cop_stock = read_cop_stock_file(cop_stock_file)
            st.write("Vista previa del archivo Stock (hoja 'Hoja2'):")
            st.dataframe(df_cop_stock.head().astype(str))

            missing_columns, _ = validate_required_columns(
                df_cop_stock.columns, COP_STOCK_COLUMNS
            )
            if missing_columns:
                st.warning(
                    "Columnas esperadas no encontradas en el stock: "
                    + ", ".join(missing_columns)
                )
            else:
                st.success("El archivo Stock tiene todas las columnas esperadas.")
        except Exception as e:
            st.error(f"Error al leer el archivo Stock: {e}")

    if cop_flujo_file is not None:
        try:
            df_cop_flujo = pd.read_excel(cop_flujo_file)
            st.write("Vista previa del archivo Flujo:")
            st.dataframe(df_cop_flujo.head().astype(str))
        except Exception as e:
            st.error(f"Error al leer el archivo Flujo: {e}")

    if cop_stock_file is None and cop_flujo_file is None:
        st.info("Carga ambos archivos, Stock y Flujo, para continuar.")

with tab6:
    show_bci_view()
