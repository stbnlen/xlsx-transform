import streamlit as st
import pandas as pd
import io

from utils import (
    normalize_column_name,
    find_matching_column,
    validate_required_columns,
)


def show_q_cmr_view():
    """Display Q_CMR view for filtering and downloading Excel files."""
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="q_cmr_uploader")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Original Data Preview:")
        st.dataframe(df)
        st.write(f"Original shape: {df.shape}")
        
        columns_to_keep = ['rut', 'n_operacion_principal', 'dv', 'nombre_completo_cliente', 
                          'CARTERA', 'CATEGORIA', 'SUCURSAL', 'EJECUTIVA ASIGNADA', 
                          'ESTADO JUDICIAL', 'DESCUENTO CAMPAÑA', 'SALDO_DEUDA', 'ESTADO INICIAL', 'TRAMO', 
                          'estado_cuenta']
            
        missing_columns, column_mapping = validate_required_columns(df.columns, columns_to_keep)
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            st.write("Normalized available columns:", [normalize_column_name(col) for col in df.columns])
        else:
            actual_columns_to_use = [column_mapping[col] for col in columns_to_keep]
            filtered_df = df[actual_columns_to_use].copy()
            
            rename_dict = {actual: expected for expected, actual in column_mapping.items()}
            filtered_df = filtered_df.rename(columns=rename_dict)
            
            st.subheader("Filtered Data Preview:")
            st.dataframe(filtered_df)
            st.write(f"Filtered shape: {filtered_df.shape}")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="Hoja1")
            
            st.download_button(
                label="Download Filtered Excel",
                data=output.getvalue(),
                file_name="q_cmr_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )