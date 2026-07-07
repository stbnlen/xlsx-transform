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
    .stVerticalBlock {
        margin-bottom: 1rem;
        border-radius: 16px;
        border: 1px solid #334155;
        background-color: #1e293b;
        padding: 1.5rem;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    
    .stVerticalBlock:hover {
        border-color: #6366f1;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.25);
        transform: translateY(-4px);
    }
    
    .stPageLinkContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stPageLinkContainer a {
        text-decoration: none !important;
        color: #f8fafc !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }
    
    .stCaption {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        margin-top: 0.5rem !important;
    }

    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .card-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.4;
        margin-bottom: 1rem;
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
    """Render a clickable dashboard navigation card using Streamlit's native page_link."""
    with st.container():
        st.page_link(
            page,
            label=f"{icon} {title}",
            use_container_width=True,
        )
        st.caption(description)


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
