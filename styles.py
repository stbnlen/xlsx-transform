"""Custom styles and UI helpers for the Streamlit application."""

import streamlit as st


def inject_custom_css() -> None:
    """Inject global custom CSS to override default Streamlit looks."""
    css = """
    <style>
    /* Hide Streamlit default chrome - more specific selectors */
    [data-testid="stToolbar"],
    footer,
    #MainMenu {
        visibility: hidden;
        display: none;
    }

    /* Hide only the deploy button, not the entire header */
    [data-testid="stHeader"] [data-testid="stDeployButton"] {
        display: none;
    }

    /* Base layout spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Dashboard cards */
    .dashboard-card {
        position: relative;
        margin-bottom: 1rem;
        border-radius: 16px;
        border: 1px solid #334155;
        background-color: #1e293b;
        padding: 1.5rem;
        transition: all 0.2s ease-in-out;
        height: 100%;
        min-height: 180px;
        overflow: hidden;
    }
    
    .dashboard-card:hover {
        border-color: #6366f1;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.25);
        transform: translateY(-4px);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.5rem;
        text-decoration: none;
        transition: color 0.2s ease-in-out;
    }
    
    .card-title:hover {
        color: #818cf8;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Primary button override */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    /* Headers */
    h1, h2, h3 {
        border-bottom: none !important;
    }
    
    h1 a, h2 a, h3 a {
        display: none !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #94a3b8;
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }

    /* File uploader */
    .stUploadButton > button {
        border-radius: 10px;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_card(title: str, description: str, icon: str, page: str) -> None:
    """Render a clickable dashboard navigation card."""
    from pathlib import Path

    slug = Path(page).stem
    card_html = f"""
    <div class="dashboard-card">
        <div class="card-icon">{icon}</div>
        <a href="/{slug}" class="card-title">{title}</a>
        <div class="card-description">{description}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def setup_page(title: str, icon: str = "📊") -> None:
    """Configure a standard internal page with custom styling.

    This must be called before any other Streamlit command.
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_custom_css()
    st.page_link("app.py", label="⬅ Volver al inicio", icon="🏠")
