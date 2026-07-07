import streamlit as st

from styles import inject_custom_css, render_card

st.set_page_config(
    page_title="Excel Transformer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_custom_css()

st.title("Excel Transformer")
st.markdown(
    "<h3 style='color: #94a3b8; font-weight: 400;'>"
    "Selecciona un módulo para comenzar"
    "</h3>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown("### Módulos")

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    render_card(
        title="Asignaciones",
        description="Procesa archivos de asignación Q_BANCO, Q_CMR, FORUM y BCI.",
        icon="📋",
        page="pages/asig.py",
    )

with row1_col2:
    render_card(
        title="Pagos",
        description="Analiza datos de pagos PAGOS_FRM y PAGOS BCI.",
        icon="💰",
        page="pages/pagos.py",
    )

with row1_col3:
    render_card(
        title="New CD",
        description="Análisis y predicción del centro de contacto.",
        icon="📞",
        page="pages/new_cd.py",
    )

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    render_card(
        title="Reporte CAE",
        description="Módulo de reportes CAE.",
        icon="📊",
        page="pages/reporte_cae.py",
    )

with row2_col2:
    render_card(
        title="Compromisos",
        description="Módulo de seguimiento de compromisos.",
        icon="🤝",
        page="pages/compromisos.py",
    )

with row2_col3:
    st.empty()
