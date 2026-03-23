import streamlit as st
from q_banco import show_q_banco_view
from q_cmr import show_q_cmr_view

st.title("Asignaciones")

tab1, tab2 = st.tabs(["Q_BANCO", "Q_CMR"])

with tab1:
    st.header("Q_BANCO View")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR View")
    show_q_cmr_view()