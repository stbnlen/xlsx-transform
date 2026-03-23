import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Excel Transformer",
    page_icon="📊",
    layout="wide"
)

# Main title and description
st.title("Excel Transformer")
st.markdown("### Select a module from the sidebar navigation")

# Optional: Add some information about the app
st.info("""
Use the sidebar navigation to access different modules:
- **Asignaciones**: Contains Q_BANCO and Q_CMR modules
- **Pagos**: Contains PAGOS_FRM and PAGOS BCI modules
""")