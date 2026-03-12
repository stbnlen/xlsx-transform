# Excel Transformer - Column Filter

A Streamlit application that allows users to upload an Excel file, preview its contents, filter to specific columns, and download the filtered result.

## Features

- Upload Excel files (.xlsx, .xls)
- Preview both original and filtered data in interactive tables
- Automatically filters to keep only specified columns:
  - rut, dv, n_operacion, origen_core, nombre_completo_cliente, 
  - SUCURSAL, CARTERA, ESTADO CRM, ESTADO JUDICIAL, SALDO CAPITAL, 
  - % DESCUENTO, comuna_particular
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

1. Click "Browse files" or drag and drop an Excel file (.xlsx or .xls) into the upload area
2. Once uploaded, you'll see:
   - A preview of the original data with its dimensions
   - If all required columns are present, a preview of the filtered data with its dimensions
   - A "Download Filtered Excel" button to download the result
3. If required columns are missing, the app will show an error with the list of available columns

## Project Structure

- `app.py` - Main Streamlit application with column filtering logic
- `requirements.txt` - Python dependencies
- `AGENTS.md` - Guidelines for AI agents working on this project
- `README.md` - This file

## Dependencies

- streamlit
- pandas
- openpyxl

## How It Works

The application follows these steps:

1. User uploads an Excel file via `st.file_uploader`
2. The file is read into a pandas DataFrame using `pd.read_excel`
3. The original DataFrame is displayed using `st.dataframe`
4. The application checks for required columns and shows an error if any are missing
5. If all columns are present, it filters the DataFrame to keep only the specified columns
6. Specific columns are renamed to match the expected output format:
   - 'n_operacion_principal' → 'n_operacion'
   - 'saldo_capital' → 'SALDO CAPITAL'
7. The filtered DataFrame is displayed using `st.dataframe`
8. The filtered DataFrame is written to an in-memory buffer using `pd.ExcelWriter` with openpyxl engine
9. The buffer contents are made available for download via `st.download_button`

## Notes

- The app validates that all required columns exist in the uploaded file before processing
- Column name transformations are applied to match the exact output format requested
- The app uses in-memory operations, so no temporary files are saved to disk
- Column filtering reduces the dataset from 146 columns to 12 columns (based on the sample data)

## Development

To modify the application:

1. Edit `app.py` to change the filtering logic or column selection
2. Test changes locally with `streamlit run app.py`
3. Ensure dependencies are up to date in `requirements.txt`

## License

This project is open source and available for modification and distribution.