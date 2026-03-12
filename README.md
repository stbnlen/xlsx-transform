# Excel Transformer - Column Filter

A Streamlit application that allows users to upload an Excel file, preview its contents, filter to specific columns based on the selected mode (Q_BANCO or Q_CMR), and download the filtered result.

## Features

- Switch between two modes: Q_BANCO and Q_CMR
- Visual indicator showing which mode is currently active
- Disabled buttons for the currently active mode (cannot re-select same mode)
- Upload Excel files (.xlsx, .xls)
- Preview both original and filtered data in interactive tables
- Automatic column filtering based on selected mode:
  - **Q_BANCO mode**: rut, dv, n_operacion, origen_core, nombre_completo_cliente, SUCURSAL, CARTERA, ESTADO CRM, ESTADO JUDICIAL, SALDO CAPITAL, % DESCUENTO, comuna_particular
  - **Q_CMR mode**: rut, n_operacion_principal, dv, nombre_completo_cliente, CARTERA, CATEGORIA, SUCURSAL, EJECUTIVA ASIGNADA, ESTADO JUDICIAL, DESCUENTO CAMPAÑA, SALDO_DEUDA, TRAMO, estado_cuenta
- Download the filtered Excel file
- Built with Streamlit, pandas, and openpyxl

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the application locally:

```bash
streamlit run app.py
```

The application will open in your default web browser (usually at http://localhost:8501).

### Steps to use the app:

1. Observe the "Vista activa:" indicator showing which mode is currently selected
2. Click either the Q_BANCO or Q_CMR button to switch modes (the currently active button will be disabled)
3. Click "Browse files" or drag and drop an Excel file (.xlsx or .xls) into the upload area
4. Once uploaded, you'll see:
   - A preview of the original data with its dimensions
   - If all required columns are present for the selected mode, a preview of the filtered data with its dimensions
   - A "Download Filtered Excel" button to download the result
5. If required columns are missing, the app will show an error with the list of available columns

## Modes Explained

### Q_BANCO Mode
Filters to keep these specific columns:
- rut, dv, n_operacion (from n_operacion_principal), origen_core, nombre_completo_cliente, 
- SUCURSAL, CARTERA, ESTADO CRM, ESTADO JUDICIAL, SALDO CAPITAL (from saldo_capital), 
- % DESCUENTO, comuna_particular

### Q_CMR Mode  
Filters to keep these specific columns:
- rut, n_operacion_principal, dv, nombre_completo_cliente, CARTERA, CATEGORIA, 
- SUCURSAL, EJECUTIVA ASIGNADA, ESTADO JUDICIAL, DESCUENTO CAMPAÑA, SALDO_DEUDA, TRAMO, estado_cuenta

## Project Structure

- `app.py` - Main Streamlit application with dual mode functionality
- `requirements.txt` - Python dependencies
- `AGENTS.md` - Guidelines for AI agents working on this project
- `README.md` - This file

## Dependencies

- streamlit
- pandas
- openpyxl

## How It Works

The application follows these steps:

1. User selects a mode (Q_BANCO or Q_CMR) via buttons
2. The active mode is displayed and the corresponding button is disabled
3. User uploads an Excel file via `st.file_uploader`
4. The file is read into a pandas DataFrame using `pd.read_excel`
5. The original DataFrame is displayed using `st.dataframe`
6. The application checks for required columns based on the selected mode and shows an error if any are missing
7. If all columns are present, it filters the DataFrame to keep only the mode-specific columns
8. For Q_BANCO mode, specific columns are renamed:
   - 'n_operacion_principal' → 'n_operacion'
   - 'saldo_capital' → 'SALDO CAPITAL'
9. The filtered DataFrame is displayed using `st.dataframe`
10. The filtered DataFrame is written to an in-memory buffer using `pd.ExcelWriter` with openpyxl engine
11. The buffer contents are made available for download via `st.download_button`

## Notes

- The app validates that all required columns exist in the uploaded file before processing for the selected mode
- Column name transformations are applied only in Q_BANCO mode to match the exact output format requested
- The app uses in-memory operations, so no temporary files are saved to disk
- Button states are managed using Streamlit's session state to prevent re-selecting the active mode

## Development

To modify the application:

1. Edit `app.py` to change the filtering logic, column selection, or mode definitions
2. Test changes locally with `streamlit run app.py`
3. Ensure dependencies are up to date in `requirements.txt`

## License

This project is open source and available for modification and distribution.