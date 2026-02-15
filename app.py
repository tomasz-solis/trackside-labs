"""
Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import logging

import streamlit as st

from src.dashboard import (
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

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color: rgba(237,239,243,0.55); padding: 0.75rem 0;">Built with ❤️ for racing fans and data nerds</div>',
    unsafe_allow_html=True,
)
