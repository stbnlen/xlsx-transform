import streamlit as st

from styles import inject_custom_css

st.set_page_config(
    page_title="Excel Transformer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_custom_css()

PAGES = [
    st.Page("pages/home.py", title="Inicio", icon="🏠", default=True),
    st.Page("pages/asig.py", title="Asignaciones", icon="📋"),
    st.Page("pages/pagos.py", title="Pagos", icon="💰"),
    st.Page("pages/new_cd.py", title="New CD", icon="📞"),
    st.Page("pages/reporte_cae.py", title="Reporte CAE", icon="📊"),
    st.Page("pages/compromisos.py", title="Compromisos", icon="🤝"),
]

pg = st.navigation(PAGES)
pg.run()
