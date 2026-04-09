import streamlit as st
import pandas as pd
import io
from q_banco import show_q_banco_view
from q_cmr import show_q_cmr_view
from pagos_bci import show_bci_view


def process_forum_data(df_castigo, df_vigente, filename1, filename2):
    """Process and combine the two dataframes according to requirements."""
    
    # Process Castigo dataframe
    df_castigo_processed = process_single_file(df_castigo, filename1, 'Castigo')
    
    # Process Vigente dataframe
    df_vigente_processed = process_single_file(df_vigente, filename2, 'Vigente')
    
    # Combine the dataframes
    combined_df = pd.concat([df_castigo_processed, df_vigente_processed], ignore_index=True)
    
    # Reorder columns to match the requested order
    column_order = [
        'ORIGEN',           # From filename
        'CONTRATO',         # From data
        'RUT',              # From data (cleaned)
        'NOMBRE CLIENTE',   # From data
        'MONTO CASIIGO',    # From data
        'ETAPA DEMANDA',    # Empty column
        'FECHA CASTIGO',    # From data
        'CIUDAD',           # Empty column
        'CARTERA',          # From data
        'Tipo gestión',     # Empty column
        'Año castigo'       # Year from FECHA CASTIGO (last column)
    ]
    
    # Ensure all required columns exist
    for col in column_order:
        if col not in combined_df.columns:
            combined_df[col] = ''
    
    # Return dataframe with columns in the specified order
    return combined_df[column_order]


def process_single_file(df, filename, file_type):
    """Process a single file to extract and format required data."""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean RUT column (remove dash and everything after it)
    if 'RUT' in processed_df.columns:
        processed_df['RUT'] = processed_df['RUT'].astype(str).str.split('-').str[0]
    else:
        # If RUT column doesn't exist, create empty column
        processed_df['RUT'] = ''
    
    # Add ORIGEN column from filename (without extension)
    origen_name = filename.split('.')[0]  # Remove file extension
    processed_df['ORIGEN'] = origen_name
    
    # Handle column mapping based on file type
    if file_type == 'Castigo':
        # Map Castigo file columns
        if 'CONTRATO' not in processed_df.columns and 'CONTRATO' in df.columns:
            processed_df['CONTRATO'] = df['CONTRATO']
        elif 'CONTRATO' not in processed_df.columns:
            processed_df['CONTRATO'] = ''
            
        if 'NOMBRE CLIENTE' not in processed_df.columns and 'NOMBRE CLIENTE' in df.columns:
            processed_df['NOMBRE CLIENTE'] = df['NOMBRE CLIENTE']
        elif 'NOMBRE CLIENTE' not in processed_df.columns:
            processed_df['NOMBRE CLIENTE'] = ''
            
        # For MONTO CASIIGO, use MONTO CASTIGO from castigo file
        if 'MONTO CASIIGO' not in processed_df.columns:
            if 'MONTO CASTIGO' in df.columns:
                processed_df['MONTO CASIIGO'] = df['MONTO CASTIGO']
            else:
                processed_df['MONTO CASIIGO'] = ''
                
        if 'FECHA CASTIGO' not in processed_df.columns and 'FECHA CASTIGO' in df.columns:
            processed_df['FECHA CASTIGO'] = df['FECHA CASTIGO']
        elif 'FECHA CASTIGO' not in processed_df.columns:
            processed_df['FECHA CASTIGO'] = ''
            
        # For CARTERA, handle case sensitivity
        if 'CARTERA' not in processed_df.columns:
            if 'cartera' in df.columns:
                processed_df['CARTERA'] = df['cartera']
            elif 'CARTERA' in df.columns:
                processed_df['CARTERA'] = df['CARTERA']
            else:
                processed_df['CARTERA'] = ''
                 
        # For Tipo gestión, map from Tipo de gestión column
        if 'Tipo gestión' not in processed_df.columns:
            if 'Tipo de gestión' in df.columns:
                processed_df['Tipo gestión'] = df['Tipo de gestión']
            elif 'Tipo gestión' in df.columns:
                processed_df['Tipo gestión'] = df['Tipo gestión']
            else:
                processed_df['Tipo gestión'] = ''
                 
        # Add Año castigo column based on FECHA CASTIGO
        if 'Año castigo' not in processed_df.columns and 'FECHA CASTIGO' in processed_df.columns:
            # Convert to datetime and extract year
            processed_df['Año castigo'] = pd.to_datetime(processed_df['FECHA CASTIGO'], errors='coerce').dt.year
            # Replace NaN values with empty string
            processed_df['Año castigo'] = processed_df['Año castigo'].fillna('').astype(str)
        elif 'Año castigo' not in processed_df.columns:
            processed_df['Año castigo'] = ''
             
    elif file_type == 'Vigente':
        # Map Vigente file columns
        if 'CONTRATO' not in processed_df.columns and 'NumContrato' in df.columns:
            processed_df['CONTRATO'] = df['NumContrato']
        elif 'CONTRATO' not in processed_df.columns and 'CONTRATO' in df.columns:
            processed_df['CONTRATO'] = df['CONTRATO']
        elif 'CONTRATO' not in processed_df.columns:
            processed_df['CONTRATO'] = ''
            
        if 'NOMBRE CLIENTE' not in processed_df.columns and 'Nombre_Cliente' in df.columns:
            processed_df['NOMBRE CLIENTE'] = df['Nombre_Cliente']
        elif 'NOMBRE CLIENTE' not in processed_df.columns and 'NOMBRE CLIENTE' in df.columns:
            processed_df['NOMBRE CLIENTE'] = df['NOMBRE CLIENTE']
        elif 'NOMBRE CLIENTE' not in processed_df.columns:
            processed_df['NOMBRE CLIENTE'] = ''
            
        # For vigente files, MONTO CASIIGO is typically 0 or empty since they're current accounts
        if 'MONTO CASIIGO' not in processed_df.columns:
            # Check if there's a balance column that might represent castigo amount
            if 'fSaldoInsoluto' in df.columns:
                processed_df['MONTO CASIIGO'] = df['fSaldoInsoluto']
            else:
                processed_df['MONTO CASIIGO'] = 0  # Default to 0 for vigente accounts
                
        if 'FECHA CASTIGO' not in processed_df.columns and 'Fecha Castigo' in df.columns:
            processed_df['FECHA CASTIGO'] = df['Fecha Castigo']
        elif 'FECHA CASTIGO' not in processed_df.columns:
            processed_df['FECHA CASTIGO'] = ''
            
        # For CARTERA, handle case sensitivity - if not found, use "Dual vigente" for vigente files
        if 'CARTERA' not in processed_df.columns:
            if 'cartera' in df.columns:
                processed_df['CARTERA'] = df['cartera']
            elif 'CARTERA' in df.columns:
                processed_df['CARTERA'] = df['CARTERA']
            else:
                processed_df['CARTERA'] = 'Dual vigente'  # Default for vigente files when column not found
        
        # Add empty columns for the fields that should be empty initially
        processed_df['ETAPA DEMANDA'] = ''
        processed_df['CIUDAD'] = ''
        # Note: Tipo gestión is handled above for both file types
        
        # Add Año castigo column based on FECHA CASTIGO
        if 'Año castigo' not in processed_df.columns and 'FECHA CASTIGO' in processed_df.columns:
            # Convert to datetime and extract year
            processed_df['Año castigo'] = pd.to_datetime(processed_df['FECHA CASTIGO'], errors='coerce').dt.year
            # Replace NaN values with empty string
            processed_df['Año castigo'] = processed_df['Año castigo'].fillna('').astype(str)
        elif 'Año castigo' not in processed_df.columns:
            processed_df['Año castigo'] = ''
            
    return processed_df


st.title("Asignaciones")

tab1, tab2, tab3, tab4 = st.tabs(["Q_BANCO", "Q_CMR", "FORUM", "BCI"])

with tab1:
    st.header("Q_BANCO View")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR View")
    show_q_cmr_view()

with tab3:
    st.header("FORUM Module")
    
    # First file uploader
    st.subheader("Castigo")
    uploaded_file1 = st.file_uploader("Choose first XLSX file", type=["xlsx", "xls"], key="forum_uploader1")
    
    # Second file uploader
    st.subheader("Vigente")
    uploaded_file2 = st.file_uploader("Choose second XLSX file", type=["xlsx", "xls"], key="forum_uploader2")
    
    # Process files when both are uploaded
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("Both files uploaded successfully!")
        
        try:
            # Read both Excel files
            df_castigo = pd.read_excel(uploaded_file1)
            df_vigente = pd.read_excel(uploaded_file2)
            
            # Show previews with type handling to avoid Arrow conversion errors
            st.write("Preview of Castigo file:")
            # Convert all columns to string for display to avoid Arrow errors
            display_castigo = df_castigo.head().astype(str)
            st.dataframe(display_castigo)
            
            st.write("Preview of Vigente file:")
            # Convert all columns to string for display to avoid Arrow errors
            display_vigente = df_vigente.head().astype(str)
            st.dataframe(display_vigente)
            
            # Process and combine the dataframes
            combined_df = process_forum_data(df_castigo, df_vigente, uploaded_file1.name, uploaded_file2.name)
            
            # Show the combined result with type handling to avoid Arrow conversion errors
            st.write("Combined Data (Ready for Download):")
            # Convert problematic columns to string for display to avoid Arrow errors
            display_combined = combined_df.copy()
            for col in display_combined.columns:
                if display_combined[col].dtype == 'object':
                    display_combined[col] = display_combined[col].astype(str)
            st.dataframe(display_combined)
            
            # Provide download button for XLSX
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Anexar1', index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download Combined Data as XLSX",
                data=excel_data,
                file_name="combined_forum_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.info("Please make sure your files have the required columns: CONTRATO, RUT, NOMBRE CLIENTE, MONTO CASIIGO, FECHA CASTIGO, CARTERA")
    elif uploaded_file1 is not None:
        st.info("Please upload the Vigente file to proceed.")
        try:
            import pandas as pd
            df1 = pd.read_excel(uploaded_file1)
            st.write("Preview of castigo file:")
            st.dataframe(df1.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif uploaded_file2 is not None:
        st.info("Please upload the Castigo file to proceed.")
        try:
            import pandas as pd
            df2 = pd.read_excel(uploaded_file2)
            st.write("Preview of vigente file:")
            st.dataframe(df2.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload both Castigo and Vigente files to proceed.")

with tab4:
    show_bci_view()