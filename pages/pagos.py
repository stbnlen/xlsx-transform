import streamlit as st

from pagos_bci import show_pagos_bci_view
from pagos_frm import show_pagos_frm_view

st.title("Payments")

tab1, tab2 = st.tabs(["PAGOS_FRM", "PAGOS BCI"])

with tab1:
    st.header("PAGOS_FRM")
    show_pagos_frm_view()

with tab2:
    st.header("PAGOS BCI")
    show_pagos_bci_view()
