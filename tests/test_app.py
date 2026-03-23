import sys
import os

# Add the parent directory to sys.path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_app_tabs():
    """Test that the app uses page navigation correctly."""
    # Read the app.py file and check that it sets up navigation correctly
    with open(os.path.join(os.path.dirname(__file__), '..', 'app.py'), 'r') as f:
        content = f.read()
    
    # Check that the app uses Streamlit's native page structure
    assert 'st.set_page_config' in content
    assert 'st.title("Excel Transformer")' in content
    assert 'Select a module from the sidebar navigation' in content


def test_app_imports():
    """Test that the app imports are correct for the new structure."""
    # Read the app.py file and check imports
    with open(os.path.join(os.path.dirname(__file__), '..', 'app.py'), 'r') as f:
        content = f.read()
    
    # The main app.py should only import streamlit now
    assert 'import streamlit as st' in content
    # Should not import the view functions directly anymore
    assert 'from q_banco import show_q_banco_view' not in content
    assert 'from q_cmr import show_q_cmr_view' not in content
    assert 'from pagos_frm import show_pagos_frm_view' not in content
    assert 'from pagos_bci import show_pagos_bci_view' not in content


def test_asig_page_tabs():
    """Test that the Asignaciones page uses the tabs correctly."""
    # Read the asig.py file and check the tab structure
    asig_path = os.path.join(os.path.dirname(__file__), '..', 'pages', 'asig.py')
    assert os.path.exists(asig_path), "asig.py should exist in pages directory"
    
    with open(asig_path, 'r') as f:
        content = f.read()
    
    # Check that the tabs are created with the correct names
    assert 'st.tabs(["Q_BANCO", "Q_CMR"])' in content
    
    # Check that each tab context uses the correct view function
    assert 'with tab1:' in content
    assert 'show_q_banco_view()' in content
    assert 'with tab2:' in content
    assert 'show_q_cmr_view()' in content


def test_pagos_page_tabs():
    """Test that the Pagos page uses the tabs correctly."""
    # Read the pagos.py file and check the tab structure
    pagos_path = os.path.join(os.path.dirname(__file__), '..', 'pages', 'pagos.py')
    assert os.path.exists(pagos_path), "pagos.py should exist in pages directory"
    
    with open(pagos_path, 'r') as f:
        content = f.read()
    
    # Check that the tabs are created with the correct names
    assert 'st.tabs(["PAGOS_FRM", "PAGOS BCI"])' in content
    
    # Check that each tab context uses the correct view function
    assert 'with tab1:' in content
    assert 'show_pagos_frm_view()' in content
    assert 'with tab2:' in content
    assert 'show_pagos_bci_view()' in content