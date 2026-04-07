import streamlit as st
from q_banco import show_q_banco_view
from q_cmr import show_q_cmr_view
from pagos_bci import show_bci_view

st.title("Asignaciones")

tab1, tab2, tab3, tab4 = st.tabs(["Q_BANCO", "Q_CMR", "FORUM", "BCI"])

with tab1:
    st.header("Q_BANCO View")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR View")
    show_q_cmr_view()

with tab3:
    st.markdown("<h1>en construccion</h1>", unsafe_allow_html=True)

with tab4:
    show_bci_view()