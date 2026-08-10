"""Custom styles and UI helpers for the Streamlit application.

The CSS below avoids hardcoded theme colors: surfaces and secondary texts
are derived from ``currentColor`` via ``color-mix`` so they adapt to both
dark and light client themes. Only the brand accent is fixed, matching
``primaryColor`` in ``.streamlit/config.toml``.

Selectors target ``data-testid`` attributes verified against
Streamlit 1.61 (tabs no longer expose ``data-baseweb`` attributes).
"""

from pathlib import Path

import streamlit as st

BRAND_CSS_VARS = """
:root {
    --brand-primary: #6366f1;
    --brand-primary-hover: #818cf8;
    --brand-shadow: rgba(99, 102, 241, 0.25);
}
"""


def inject_custom_css() -> None:
    """Inject global custom CSS to override default Streamlit looks."""
    css = f"""
    <style>
    {BRAND_CSS_VARS}

    /* Hide Streamlit default chrome */
    [data-testid="stToolbar"],
    footer,
    #MainMenu {{
        visibility: hidden;
        display: none;
    }}

    /* Base layout spacing */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Dashboard cards: any container holding a .card-marker */
    div[data-testid="stVerticalBlock"]:has(.card-marker) {{
        border-radius: 16px;
        border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
        background-color: color-mix(in srgb, currentColor 5%, transparent);
        padding: 1.5rem;
        min-height: 180px;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker):hover {{
        border-color: var(--brand-primary);
        box-shadow: 0 10px 25px -5px var(--brand-shadow);
        transform: translateY(-4px);
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker) .card-marker {{
        display: none;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker) .card-icon {{
        font-size: 2.5rem;
        line-height: 1;
        margin-bottom: 0.75rem;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink"] {{
        margin-bottom: 0.5rem;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink-NavLink"] {{
        padding: 0;
        margin: 0;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink-NavLink"] span,
    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink-NavLink"] p {{
        color: var(--brand-primary) !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink-NavLink"]:hover span,
    div[data-testid="stVerticalBlock"]:has(.card-marker)
    [data-testid="stPageLink-NavLink"]:hover p {{
        color: var(--brand-primary-hover) !important;
    }}

    div[data-testid="stVerticalBlock"]:has(.card-marker)
    .card-description {{
        font-size: 0.95rem;
        color: color-mix(in srgb, currentColor 62%, transparent);
        line-height: 1.4;
    }}

    /* Primary button override */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }}

    /* Tabs (Streamlit >= 1.56 selectors) */
    div[data-testid="stTabs"] {{
        margin-bottom: 2rem;
    }}

    div[data-testid="stTabs"] [role="tablist"] {{
        gap: 8px !important;
        background-color: color-mix(
            in srgb, currentColor 6%, transparent
        ) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        display: flex !important;
    }}

    div[data-testid="stTabs"] [data-testid="stTab"] {{
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        color: color-mix(
            in srgb, currentColor 62%, transparent
        ) !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
    }}

    div[data-testid="stTabs"] [data-testid="stTab"]:hover {{
        background-color: color-mix(
            in srgb, currentColor 12%, transparent
        ) !important;
        color: currentColor !important;
    }}

    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
        background-color: var(--brand-primary) !important;
        color: #ffffff !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] button {{
        border-radius: 10px;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_card(
    title: str,
    description: str,
    icon: str,
    page: str,
    badge: str | None = None,
) -> None:
    """Render a clickable dashboard navigation card.

    The card is a Streamlit container styled via CSS (see the
    ``.card-marker`` rules). Navigation uses ``st.page_link`` so routing
    works with ``st.navigation`` regardless of the deployment base path.
    """
    slug = Path(page).stem
    card = st.container()
    with card:
        st.markdown(
            f'<span class="card-marker card-marker-{slug}"></span>'
            f'<div class="card-icon">{icon}</div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label=title)
        st.markdown(
            f'<div class="card-description">{description}</div>',
            unsafe_allow_html=True,
        )
        if badge:
            st.badge(badge, icon="🚧", color="orange")
