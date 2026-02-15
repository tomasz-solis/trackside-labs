"""
Trackside Labs Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import logging

import streamlit as st

from src.dashboard import (
    BRAND_NAME,
    configure_page,
    enable_fastf1_cache,
    render_global_styles,
    render_header,
    render_page,
    render_sidebar,
)

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1.api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

enable_fastf1_cache()
configure_page()
render_global_styles()
render_header()

page, enable_logging = render_sidebar()
render_page(page, enable_logging)

st.markdown(
    (f'<div class="brand-footer">{BRAND_NAME} | independent motorsport forecasting project</div>'),
    unsafe_allow_html=True,
)
