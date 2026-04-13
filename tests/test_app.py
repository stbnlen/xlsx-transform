import os


def test_app_tabs():
    with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), "r") as f:
        content = f.read()

    assert "st.set_page_config" in content
    assert 'st.title("Excel Transformer")' in content
    assert "Select a module from the sidebar navigation" in content


def test_app_imports():
    with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), "r") as f:
        content = f.read()

    assert "import streamlit as st" in content
    assert "from q_banco import show_q_banco_view" not in content
    assert "from q_cmr import show_q_cmr_view" not in content
    assert "from pagos_frm import show_pagos_frm_view" not in content
    assert "from pagos_bci import show_pagos_bci_view" not in content


def test_asig_page_tabs():
    asig_path = os.path.join(os.path.dirname(__file__), "..", "pages", "asig.py")
    assert os.path.exists(asig_path), "asig.py should exist in pages directory"

    with open(asig_path, "r") as f:
        content = f.read()

    assert 'st.tabs(["Q_BANCO", "Q_CMR", "FORUM", "BCI"])' in content

    assert "with tab1:" in content
    assert "show_q_banco_view()" in content
    assert "with tab2:" in content
    assert "show_q_cmr_view()" in content
    assert "with tab3:" in content
    assert 'st.header("FORUM")' in content
    assert 'st.subheader("Castigo (Charge-off)")' in content
    assert 'st.subheader("Vigente (Current)")' in content
    assert "def process_forum_data(" in content
    assert "def process_single_file(" in content
    assert "with tab4:" in content
    assert "show_bci_view()" in content


def test_pagos_page_tabs():
    pagos_path = os.path.join(os.path.dirname(__file__), "..", "pages", "pagos.py")
    assert os.path.exists(pagos_path), "pagos.py should exist in pages directory"

    with open(pagos_path, "r") as f:
        content = f.read()

    assert 'st.tabs(["PAGOS_FRM", "PAGOS BCI"])' in content

    assert "with tab1:" in content
    assert "show_pagos_frm_view()" in content
    assert "with tab2:" in content
    assert "show_pagos_bci_view()" in content
