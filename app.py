import streamlit as st
import pandas as pd
import io
from utils import normalize_column_name, find_matching_column, validate_required_columns

def show_q_banco_view():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="q_banco_uploader")
    
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
        missing_columns, column_mapping = validate_required_columns(df.columns, columns_to_keep)
        
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
                file_name="q_banco_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# Function to handle Q_CMR view logic
def show_q_cmr_view():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="q_cmr_uploader")
    
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
        missing_columns, column_mapping = validate_required_columns(df.columns, columns_to_keep)
        
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
                file_name="q_cmr_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Main app with tabs
tab1, tab2 = st.tabs(["Q_BANCO", "Q_CMR"])

with tab1:
    st.header("Q_BANCO View")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR View")
    show_q_cmr_view()