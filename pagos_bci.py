import streamlit as st
import pandas as pd
import io


def show_pagos_bci_view():
    """Display PAGOS BCI view for uploading and previewing Excel files."""
    st.header("PAGOS BCI")
    
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="pagos_bci_uploader")
    
    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Show preview
        st.subheader("Data Preview (First 5 rows):")
        st.dataframe(df.head())
        
        # Show basic info
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
    else:
        st.write("En construcción - Por favor sube un archivo Excel para ver la vista previa")