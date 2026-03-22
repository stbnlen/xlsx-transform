import sys
import os

# Add the parent directory to sys.path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_app_tabs():
    """Test that the app creates the correct tabs by checking the source."""
    # Read the app.py file and check that it creates the correct tabs
    with open(os.path.join(os.path.dirname(__file__), '..', 'app.py'), 'r') as f:
        content = f.read()
    
    # Check that the tabs are created with the correct names
    assert 'st.tabs(["Q_BANCO", "Q_CMR", "PAGOS_FRM", "PAGOS BCI"])' in content


def test_app_imports():
    """Test that the app imports the correct view functions."""
    # Read the app.py file and check imports
    with open(os.path.join(os.path.dirname(__file__), '..', 'app.py'), 'r') as f:
        content = f.read()
    
    # Check that the correct imports are present
    assert 'from q_banco import show_q_banco_view' in content
    assert 'from q_cmr import show_q_cmr_view' in content
    assert 'from pagos_frm import show_pagos_frm_view' in content
    assert 'from pagos_bci import show_pagos_bci_view' in content


def test_app_tab_structure():
    """Test that the app uses the tabs correctly."""
    # Read the app.py file and check the tab structure
    with open(os.path.join(os.path.dirname(__file__), '..', 'app.py'), 'r') as f:
        content = f.read()
    
    # Check that each tab context uses the correct view function
    assert 'with tab1:' in content
    assert 'show_q_banco_view()' in content
    assert 'with tab2:' in content
    assert 'show_q_cmr_view()' in content
    assert 'with tab3:' in content
    assert 'show_pagos_frm_view()' in content
    assert 'with tab4:' in content
    assert 'show_pagos_bci_view()' in content