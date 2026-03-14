import streamlit as st
import pandas as pd
import io
import re

def normalize_column_name(col_name):
    """Normalize column name for comparison: lowercase and remove underscores"""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return re.sub(r'_+', '', col_name.lower())

def find_matching_column(df_columns, target_col):
    """Find actual column name that matches target column (case-insensitive, underscore-insensitive)"""
    normalized_target = normalize_column_name(target_col)
    for col in df_columns:
        if normalize_column_name(col) == normalized_target:
            return col
    return None

def show_q_banco_view():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Original Data Preview:")
        st.dataframe(df)
        st.write(f"Original shape: {df.shape}")
        
        # Define the columns to keep for Q_BANCO
        columns_to_keep = ['rut', 'dv', 'n_operacion_principal', 'origen_core', 
                          'nombre_completo_cliente', 'SUCURSAL', 'CARTERA', 
                          'ESTADO CRM', 'ESTADO JUDICIAL', 'saldo_capital', 
                          '% DESCUENTO', 'comuna_particular']
        
        # Check if all required columns exist (case-insensitive, underscore-insensitive)
        missing_columns = []
        column_mapping = {}  # Maps expected column name to actual column name in file
        
        for expected_col in columns_to_keep:
            actual_col = find_matching_column(df.columns, expected_col)
            if actual_col is None:
                missing_columns.append(expected_col)
            else:
                column_mapping[expected_col] = actual_col
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            st.write("Normalized available columns:", [normalize_column_name(col) for col in df.columns])
        else:
            # Filter the dataframe using actual column names
            actual_columns_to_use = [column_mapping[col] for col in columns_to_keep]
            filtered_df = df[actual_columns_to_use].copy()
            
            # Rename columns to match the requested output names
            filtered_df = filtered_df.rename(columns={
                'n_operacion_principal': 'n_operacion',
                'saldo_capital': 'SALDO CAPITAL'
            })
            
            st.subheader("Filtered Data Preview:")
            st.dataframe(filtered_df)
            st.write(f"Filtered shape: {filtered_df.shape}")
            
            # Prepare filtered file for download
            output = io.BytesIO()  # type: ignore
            with pd.ExcelWriter(output, engine="openpyxl") as writer:  # type: ignore
                filtered_df.to_excel(writer, index=False)
            
            st.download_button(
                label="Download Filtered Excel",
                data=output.getvalue(),
                file_name=f"{st.session_state.current_view}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# Function to handle Q_CMR view logic
def show_q_cmr_view():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)
        
        st.subheader("Original Data Preview:")
        st.dataframe(df)
        st.write(f"Original shape: {df.shape}")
        
        # Define the columns to keep for Q_CMR
        columns_to_keep = ['rut', 'n_operacion_principal', 'dv', 'nombre_completo_cliente', 
                          'CARTERA', 'CATEGORIA', 'SUCURSAL', 'EJECUTIVA ASIGNADA', 
                          'ESTADO JUDICIAL', 'DESCUENTO CAMPAÑA', 'SALDO_DEUDA', 'TRAMO', 
                          'estado_cuenta']
        
        # Check if all required columns exist (case-insensitive, underscore-insensitive)
        missing_columns = []
        column_mapping = {}  # Maps expected column name to actual column name in file
        
        for expected_col in columns_to_keep:
            actual_col = find_matching_column(df.columns, expected_col)
            if actual_col is None:
                missing_columns.append(expected_col)
            else:
                column_mapping[expected_col] = actual_col
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
            st.write("Normalized available columns:", [normalize_column_name(col) for col in df.columns])
        else:
            # Filter the dataframe using actual column names
            actual_columns_to_use = [column_mapping[col] for col in columns_to_keep]
            filtered_df = df[actual_columns_to_use].copy()
            
            # Rename columns to match the expected names (for consistency)
            rename_dict: dict[str, str] = {actual: expected for expected, actual in column_mapping.items()}
            filtered_df = filtered_df.rename(columns=rename_dict)
            
            st.subheader("Filtered Data Preview:")
            st.dataframe(filtered_df)
            st.write(f"Filtered shape: {filtered_df.shape}")
            
            # Prepare filtered file for download
            output = io.BytesIO()  # type: ignore
            with pd.ExcelWriter(output, engine="openpyxl") as writer:  # type: ignore
                filtered_df.to_excel(writer, index=False)
            
            st.download_button(
                label="Download Filtered Excel",
                data=output.getvalue(),
                file_name=f"{st.session_state.current_view}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )