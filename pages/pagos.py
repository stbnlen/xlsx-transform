import streamlit as st
from pagos_frm import show_pagos_frm_view
from pagos_bci import show_pagos_bci_view

st.title("Pagos")

tab1, tab2 = st.tabs(["PAGOS_FRM", "PAGOS BCI"])

with tab1:
    st.header("PAGOS_FRM View")
    show_pagos_frm_view()

with tab2:
    st.header("PAGOS BCI")
    show_pagos_bci_view()