"""
Trackside Labs Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import logging
from pathlib import Path

import streamlit as st

from src.utils.env_file import load_env_file

# `streamlit run` reads no .env file. Fill the env before importing anything else under
# `src`: the persistence layer validates USE_DB_STORAGE / SUPABASE_* at import time.
# Render has no .env.local; its real environment variables always win.
if Path(".env.local").exists():
    load_env_file(Path(".env.local"))

from src.dashboard import (  # noqa: E402
    BRAND_NAME,
    configure_page,
    render_analytics_scripts,
    render_global_styles,
    render_header,
    render_page,
    render_sidebar,
)
from src.dashboard.analytics import track_page_view  # noqa: E402

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1.api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

configure_page()
render_global_styles()
render_analytics_scripts()
render_header()

page, enable_logging = render_sidebar()
track_page_view(page)
render_page(page, enable_logging)

st.markdown(
    (f'<div class="brand-footer">{BRAND_NAME} | independent motorsport forecasting project</div>'),
    unsafe_allow_html=True,
)
