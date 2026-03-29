# Excel Transformer - Column Filter

A Streamlit application that allows users to upload an Excel file, preview its contents, filter to specific columns based on the selected module, and download the filtered result.

## Features

- Navigate between modules using Streamlit's native page navigation
- **Asignaciones module**: Contains Q_BANCO and Q_CMR filtering options
- **Pagos module**: Contains PAGOS_FRM and PAGOS BCI options for financial data analysis
- Upload Excel files (.xlsx, .xls)
- Preview both original and processed data in interactive tables
- Automatic column filtering/processing based on selected mode:
  - **Q_BANCO mode**: Extracts and renames specific columns for banking operations
  - **Q_CMR mode**: Filters to specific columns for commercial portfolio management
  - **PAGOS_FRM mode**: Comprehensive financial data analysis including:
    - Monthly aggregation and trend analysis
    - Descriptive statistics and outlier detection
    - Exploratory data analysis charts
    - Seasonal decomposition and trend analysis
    - Yearly and monthly pattern analysis
    - Correlation analysis
    - Executive performance analysis
    - Predictive modeling capabilities
  - **PAGOS BCI mode**: Specific column filtering for BCI processing (defined in pagos_bci.py)
- Download processed Excel files
- Built with Streamlit, pandas, numpy, scipy, scikit-learn, XGBoost, LightGBM, matplotlib, and seaborn

## Project Structure

- `app.py` - Main Streamlit application (landing page)
- `pages/asig.py` - Asignaciones module with Q_BANCO and Q_CMR tabs
- `pages/pagos.py` - Pagos module with PAGOS_FRM and PAGOS BCI tabs
- `q_banco.py` - Q_BANCO filtering logic
- `q_cmr.py` - Q_CMR filtering logic
- `pagos_frm.py` - PAGOS_FRM filtering logic
- `pagos_bci.py` - PAGOS BCI filtering logic
- `utils.py` - Utility functions for column normalization and validation
- `tests/` - Directory containing unit tests
- `requirements.txt` - Python dependencies
- `AGENTS.md` - Guidelines for AI agents working on this project
- `README.md` - This file

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

### Navigation

Use the sidebar to navigate between:
- **Asignaciones**: Contains Q_BANCO and Q_CMR modules
- **Pagos**: Contains PAGOS_FRM and PAGOS BCI modules

### Steps to use each module:

1. Select the desired module from the sidebar navigation
2. Within each module, use the tabs to select the specific view (e.g., Q_BANCO or Q_CMR in Asignaciones module)
3. Click "Browse files" or drag and drop an Excel file (.xlsx or .xls) into the upload area
4. Once uploaded, you'll see:
   - A preview of the original data with its dimensions
   - If all required columns are present for the selected mode, a preview of the filtered data with its dimensions
   - A "Download Filtered Excel" button to download the result
5. If required columns are missing, the app will show an error with the list of available columns

## Modules Explained

### Asignaciones Module

#### Q_BANCO Tab
Filters to keep these specific columns:
- rut, dv, n_operacion (from n_operacion_principal), origen_core, nombre_completo_cliente, 
- SUCURSAL, CARTERA, ESTADO CRM, ESTADO JUDICIAL, SALDO CAPITAL (from saldo_capital), 
- % DESCUENTO, comuna_particular

#### Q_CMR Tab  
Filters to keep these specific columns:
- rut, n_operacion_principal, dv, nombre_completo_cliente, CARTERA, CATEGORIA, 
- SUCURSAL, EJECUTIVA ASIGNADA, ESTADO JUDICIAL, DESCUENTO CAMPAÑA, SALDO_DEUDA, TRAMO, estado_cuenta

### Pagos Module

#### PAGOS_FRM Tab
Specific column filtering for financial processing (see pagos_frm.py for exact column list)

#### PAGOS BCI Tab
Specific column filtering for BCI processing (see pagos_bci.py for exact column list)

## Dependencies

- streamlit
- pandas
- openpyxl

## How It Works

The application follows these steps:

1. User selects a module (Asignaciones or Pagos) via the sidebar navigation
2. Within the selected module, user chooses a specific view via tabs
3. User uploads an Excel file via `st.file_uploader`
4. The file is read into a pandas DataFrame using `pd.read_excel`
5. The original DataFrame is displayed using `st.dataframe`
6. The application checks for required columns based on the selected view and shows an error if any are missing
7. If all columns are present, it filters the DataFrame to keep only the view-specific columns
8. For Q_BANCO mode, specific columns are renamed:
   - 'n_operacion_principal' → 'n_operacion'
   - 'saldo_capital' → 'SALDO CAPITAL'
9. The filtered DataFrame is displayed using `st.dataframe`
10. The filtered DataFrame is written to an in-memory buffer using `pd.ExcelWriter` with openpyxl engine
11. The buffer contents are made available for download via `st.download_button`

## Notes

- The app validates that all required columns exist in the uploaded file before processing for the selected view
- Column name transformations are applied only in Q_BANCO mode to match the exact output format requested
- The app uses in-memory operations, so no temporary files are saved to disk

## Development

To modify the application:

1. Edit the respective view files (`q_banco.py`, `q_cmr.py`, `pagos_frm.py`, `pagos_bci.py`) to change the filtering logic, column selection, or mode definitions
2. For structural changes, modify the page files in the `pages/` directory
3. Test changes locally with `streamlit run app.py`
4. Ensure dependencies are up to date in `requirements.txt`

## License

This project is open source and available for modification and distribution.