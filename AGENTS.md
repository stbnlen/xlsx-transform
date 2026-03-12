# Agent Guidelines for xlsx-transform

## Project Overview

This is a simple Streamlit application that allows users to upload an Excel file, preview it, and download it. The project consists of a single `app.py` file using Streamlit, pandas, and openpyxl.

## Commands

### Running the Application

```bash
streamlit run app.py
```

### Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Testing

This project has no tests configured. To add tests:

```bash
pip install pytest
pytest                    # Run all tests
pytest tests/             # Run tests in specific directory
pytest tests/test_file.py::test_function_name  # Run single test
```

### Linting

Install and run ruff (recommended for Python):

```bash
pip install ruff
ruff check .              # Lint all files
ruff check app.py         # Lint specific file
ruff check --fix .        # Auto-fix issues
```

Or use pylint:

```bash
pip install pylint
pylint app.py
```

## Code Style Guidelines

### General

- Python 3.x compatible code
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Add type hints where beneficial for clarity

### Imports

```python
# Standard library first
import io
import os

# Third-party libraries
import streamlit as st
import pandas as pd

# Local imports (if any)
# from . import module
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `process_data`, `output_buffer`)
- **Classes**: `PascalCase` (e.g., `DataProcessor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ROWS`)
- **Private methods**: prefix with underscore (e.g., `_private_method`)

### Type Annotations

Use type hints for function parameters and return values:

```python
def process_file(file: io.BytesIO) -> pd.DataFrame:
    """Process uploaded Excel file and return DataFrame."""
    return pd.read_excel(file)
```

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Let Streamlit handle UI errors gracefully with `st.error()`

```python
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
```

### Streamlit Patterns

- Use `st.file_uploader` for file inputs
- Use `st.dataframe` for table preview
- Use `st.download_button` for file downloads
- Keep interactive elements in the main flow
- Use `st.write` for debug/info output

### Formatting

- Use Black formatter for consistent formatting: `pip install black && black app.py`
- Sort imports with isort: `pip install isort && isort app.py`
- Run both: `black app.py && isort app.py`

### File Structure

For a project of this size, a single `app.py` is appropriate. As the project grows, consider:

```
xlsx-transform/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── utils/
│   └── __init__.py
├── tests/
│   └── test_app.py
└── .streamlit/
    └── config.toml     # Streamlit configuration (optional)
```

### Security Considerations

- Never hardcode secrets or API keys
- Use Streamlit's session state for user-specific data
- Validate file types before processing (already done with `type=["xlsx", "xls"]`)

### Best Practices

1. Keep the application responsive - avoid heavy computations in the main thread
2. Use `io.BytesIO` for in-memory file operations
3. Add docstrings to functions explaining purpose and parameters
4. Use descriptive variable names
5. Comment complex logic, not obvious code
