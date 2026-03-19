import streamlit as st

# Import view functions from separate files
from q_banco import show_q_banco_view
from q_cmr import show_q_cmr_view
from pagos_frm import show_pagos_frm_view
from pagos_bci import show_pagos_bci_view


# Main app with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Q_BANCO", "Q_CMR", "PAGOS_FRM", "PAGOS BCI"])

with tab1:
    st.header("Q_BANCO View")
    show_q_banco_view()

with tab2:
    st.header("Q_CMR View")
    show_q_cmr_view()

with tab3:
    st.header("PAGOS_FRM View")
    show_pagos_frm_view()

with tab4:
    st.header("PAGOS BCI")
    show_pagos_bci_view()