import streamlit as st
import pandas as pd
import io

st.title("Excel Transformer - Column Filter")

# Initialize session state for view if not present
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'Q_BANCO'

# Show active view title
st.markdown(f"**Vista activa:** {st.session_state.current_view}")

# Function to handle Q_BANCO view logic
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
        
        # Check if all required columns exist
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
        else:
            # Filter the dataframe
            filtered_df = df[columns_to_keep].copy()
            
            # Rename columns to match the requested output names
            filtered_df = filtered_df.rename(columns={
                'n_operacion_principal': 'n_operacion',
                'saldo_capital': 'SALDO CAPITAL'
            })  # type: ignore
            
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
        
        # Check if all required columns exist
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing columns in the uploaded file: {missing_columns}")
            st.write("Available columns:", list(df.columns))
        else:
            # Filter the dataframe
            filtered_df = df[columns_to_keep].copy()
            
            # Rename columns if needed (for Q_CMR, we're keeping the original names based on user request)
            # The user requested: rut, n_operacion_principal, dv, nombre_completo, CARTERA, CATEGORIA, SUCURSAL, 
            # EJECUTIVA ASIGNADA, ESTADO JUDICIAL, DESCUENTO CAMPAÑA, SALDO_DEUDA, TRAMO, estado_cuenta
            # But the file has 'nombre_completo_cliente', not 'nombre_completo'
            # Since the user said "nombre_completo" but the file has 'nombre_completo_cliente', 
            # I'll keep the original name from the file to avoid errors
            # If the user really wants it renamed, they should clarify
            
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

# Buttons to switch view with explicit rerun to ensure proper state handling
col1, col2 = st.columns(2)
with col1:
    # Always show Q_BANCO button, but handle logic in callback
    if st.button('Q_BANCO'):
        # Only change view if not already in Q_BANCO mode
        if st.session_state.current_view != 'Q_BANCO':
            st.session_state.current_view = 'Q_BANCO'
            st.rerun()
with col2:
    # Always show Q_CMR button, but handle logic in callback
    if st.button('Q_CMR'):
        # Only change view if not already in Q_CMR mode
        if st.session_state.current_view != 'Q_CMR':
            st.session_state.current_view = 'Q_CMR'
            st.rerun()

# Add visual indicator of active view by styling the buttons
# We'll use markdown to show which view is active with a visual cue
st.markdown("---")
if st.session_state.current_view == 'Q_BANCO':
    st.markdown("🔹 **Modo Q_BANCO activo** - Mostrando lógica completa de filtrado")
else:
    st.markdown("🔹 **Modo Q_CMR activo** - Mostrando lógica completa de filtrado")

# Conditionally show the view
if st.session_state.current_view == 'Q_BANCO':
    show_q_banco_view()
else:
    show_q_cmr_view()