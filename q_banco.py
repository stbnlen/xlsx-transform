import streamlit as st
import pandas as pd
import io

from utils import (
    normalize_column_name,
    find_matching_column,
    validate_required_columns,
)


def show_q_banco_view():
    """Display Q_BANCO view for filtering and downloading Excel files."""
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Original Data Preview:")
        st.dataframe(df)
        st.write(f"Original shape: {df.shape}")
        
        columns_to_keep = ['rut', 'dv', 'n_operacion_principal', 'origen_core', 
                          'nombre_completo_cliente', 'SUCURSAL', 'CARTERA', 
                          'ESTADO CRM', 'ESTADO JUDICIAL', 'saldo_capital', 
                          '% DESCUENTO', 'comuna_particular']
           
        missing_columns, column_mapping = validate_required_columns(df.columns, columns_to_keep)
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            st.write("Normalized available columns:", [normalize_column_name(col) for col in df.columns])
        else:
            actual_columns_to_use = [column_mapping[col] for col in columns_to_keep]
            filtered_df = df[actual_columns_to_use].copy()
            
            # Rename specific columns for Q_BANCO output format
            filtered_df = filtered_df.rename(
                columns={'n_operacion_principal': 'n_operacion', 'saldo_capital': 'SALDO CAPITAL'}
            )
            
            st.subheader("Filtered Data Preview:")
            st.dataframe(filtered_df)
            st.write(f"Filtered shape: {filtered_df.shape}")
            
            # Create Excel file in memory - using BytesIO with ExcelWriter is standard practice
            output: io.BytesIO = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                filtered_df.to_excel(writer, index=False)
            # Reset buffer position for reading
            output.seek(0)
            
            st.download_button(
                label="Download Filtered Excel",
                data=output.getvalue(),
                file_name="q_banco_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )