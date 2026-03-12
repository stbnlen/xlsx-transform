import streamlit as st
import pandas as pd
import io

st.title("Excel Transformer - Column Filter")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_excel(uploaded_file)
    
    st.subheader("Original Data Preview:")
    st.dataframe(df)
    st.write(f"Original shape: {df.shape}")
    
    # Define the columns to keep
    columns_to_keep = ['rut', 'dv', 'n_operacion_principal', 'origen_core', 
                      'nombre_completo_cliente', 'SUCURSAL', 'CARTERA', 
                      'ESTADO CRM', 'ESTADO JUDICIAL', 'saldo_capital', 
                      '% DESCUENTO', 'comuna_particular']
    
    # Check if all required columns exist
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing columns in the uploaded file: {missing_columns}")
        st.write("Available columns:", list(df.columns))
    else:
        # Filter the dataframe
        filtered_df = df[columns_to_keep].copy()
        
        # Rename columns to match the requested output names
        # Create a new dataframe with renamed columns to avoid potential issues
        filtered_df = filtered_df.rename(
            columns={'saldo_capital': 'SALDO CAPITAL'}
        )
        
        st.subheader("Filtered Data Preview:")
        st.dataframe(filtered_df)
        st.write(f"Filtered shape: {filtered_df.shape}")
        
        # Prepare filtered file for download
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                filtered_df.to_excel(writer, index=False)
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
            st.stop()
        
        st.download_button(
            label="Download Filtered Excel",
            data=output.getvalue(),
            file_name="filtered_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        