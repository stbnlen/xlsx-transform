import os


def test_app_router():
    with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), "r") as f:
        content = f.read()

    assert "st.set_page_config" in content
    assert "st.navigation" in content
    assert "st.Page" in content
    assert "Asignaciones" in content
    assert "Pagos" in content
    assert "New CD" in content
    assert "Reporte CAE" in content
    assert "Compromisos" in content


def test_home_dashboard():
    home_path = os.path.join(os.path.dirname(__file__), "..", "pages", "home.py")
    assert os.path.exists(home_path), "home.py should exist in pages directory"

    with open(home_path, "r") as f:
        content = f.read()

    assert 'st.title("Excel Transformer")' in content
    assert "render_card" in content
    assert "Asignaciones" in content
    assert "Pagos" in content
    assert "New CD" in content
    assert "Reporte CAE" in content
    assert "Compromisos" in content
    assert "Selecciona un módulo para comenzar" in content


def test_styles_provides_navigation():
    styles_path = os.path.join(os.path.dirname(__file__), "..", "styles.py")
    assert os.path.exists(styles_path), "styles.py should exist"

    with open(styles_path, "r") as f:
        content = f.read()

    assert "st.page_link" in content
    assert "inject_custom_css" in content
    assert "render_card" in content


def test_app_imports():
    with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), "r") as f:
        content = f.read()

    assert "import streamlit as st" in content
    assert "from q_banco import show_q_banco_view" not in content
    assert "from q_cmr import show_q_cmr_view" not in content
    assert "from pagos_frm import show_pagos_frm_view" not in content
    assert "from pagos_bci import show_pagos_bci_view" not in content


def test_navigation_smoke():
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Excel Transformer"]

    at.switch_page("pages/asig.py")
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Asignaciones"]


def test_asig_page_tabs():
    asig_path = os.path.join(os.path.dirname(__file__), "..", "pages", "asig.py")
    assert os.path.exists(asig_path), "asig.py should exist in pages directory"

    with open(asig_path, "r") as f:
        content = f.read()

    assert '["Q_BANCO", "Q_CMR", "FORUM", "Flujo FORUM", "Flujo COP", "BCI"]' in content

    assert "with tab1:" in content
    assert "show_q_banco_view()" in content
    assert "with tab2:" in content
    assert "show_q_cmr_view()" in content
    assert "with tab3:" in content
    assert 'st.header("FORUM")' in content
    assert 'st.subheader("Castigo")' in content
    assert 'st.subheader("Vigente")' in content
    assert "def process_forum_data(" in content
    assert "def process_single_file(" in content
    assert "with tab4:" in content
    assert 'st.header("Flujo FORUM")' in content
    assert "def process_flujo_forum_data(" in content
    assert "with tab5:" in content
    assert 'st.header("Flujo COP")' in content
    assert "COP_STOCK_COLUMNS" in content
    assert "with tab6:" in content
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
