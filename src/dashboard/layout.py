"""Dashboard layout, global styling, and navigation sidebar."""

import base64
from collections import deque
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import streamlit as st

try:
    from PIL import Image
except Exception:  # pragma: no cover - fallback for environments without Pillow
    Image = None

_CUSTOM_CSS = """
<style>
/* ---- App background ---- */
[data-testid="stAppViewContainer"] {
    background: #0F1115;
}
[data-testid="stHeader"] {
    background: rgba(15,17,21,0.0);
}
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ---- Typography ---- */
.main-header {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  text-align: center;
  margin: 0.2rem 0 0.1rem 0;
  color: #EDEFF3;
}
.sub-header {
  font-size: 0.95rem;
  text-align: center;
  color: rgba(237,239,243,0.72);
  margin: 0 0 1.1rem 0;
}
.micro-disclaimer {
  font-size: 0.78rem;
  text-align: center;
  color: rgba(237,239,243,0.58);
  margin: 0 0 1.25rem 0;
}
.brand-row {
  display: flex;
  justify-content: center;
  margin: 0.05rem 0 0.25rem 0;
}
.brand-logo {
  width: 520px;
  max-width: 72vw;
  height: auto;
}

@media (max-width: 960px) {
  .brand-logo {
    width: 360px;
    max-width: 88vw;
  }
}

/* Stronger global text defaults (Streamlit DOM-safe) */
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] *,
[data-testid="stHeader"] *,
[data-testid="stToolbar"] * {
  color: rgba(237,239,243,0.88);
}

/* Headings */
h1, h2, h3, h4 {
  color: #EDEFF3 !important;
  letter-spacing: 0.2px;
}
h1, h2 {
  text-shadow: 0 0 14px rgba(255, 46, 99, 0.10);
}

/* ---- Sidebar controls contrast ---- */
[data-testid="stSidebar"] * {
  color: rgba(237,239,243,0.86) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* ---- Inputs (readable values + placeholders) ---- */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: #10141c !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
[data-baseweb="select"] * {
  color: rgba(237,239,243,0.92) !important;
}
[data-baseweb="select"] input::placeholder {
  color: rgba(237,239,243,0.45) !important;
}
label {
  color: rgba(237,239,243,0.78) !important;
}


/* ---- Tables ---- */
[data-testid="stDataFrame"] {
  background: rgba(21,25,34,0.72);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 0.35rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* The grid itself */
[data-testid="stDataFrame"] [role="grid"] {
  background: #10141c !important;
  color: rgba(237,239,243,0.88) !important;
  border-radius: 12px;
}

/* ---- HTML tables (our controlled dark tables) ---- */
.rc-table {
  background: rgba(21,25,34,0.72);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 0.6rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  overflow: hidden;
}

.rc-table table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.92rem;
  color: rgba(237,239,243,0.88);
}

.rc-table thead th {
  background: #151922;
  color: rgba(237,239,243,0.92);
  text-align: left;
  font-weight: 700;
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid rgba(255,255,255,0.10);
}

.rc-table tbody tr:nth-child(even) td {
  background: rgba(16,20,28,0.92);
}

.rc-table table th:first-child,
.rc-table table td:first-child {
  width: 64px;
  text-align: center;
  font-weight: 800;
  color: rgba(237,239,243,0.92);
}

.rc-table table td:nth-child(2) {
  font-weight: 700;
  letter-spacing: 0.2px;
}

.rc-table tbody tr:hover td {
  background: rgba(255,255,255,0.03);
}

.rc-table { overflow: auto; }

.rc-table table th:first-child,
.rc-table table td:first-child {
  position: sticky;
  left: 0;
  z-index: 3;
  background: #10141c;
}

.rc-table table th:nth-child(2),
.rc-table table td:nth-child(2) {
  position: sticky;
  left: 64px; /* same as Pos width */
  z-index: 2;
  background: #10141c;
}

/* Header row */
[data-testid="stDataFrame"] [role="columnheader"] {
  background: #151922 !important;
  color: rgba(237,239,243,0.92) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}

/* Body cells */
[data-testid="stDataFrame"] [role="gridcell"] {
  background: #10141c !important;
  color: rgba(237,239,243,0.88) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Hover */
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
  background: rgba(255,255,255,0.03) !important;
}

/* Hide/neutralize Streamlit spinner/status blocks that show as white bars */
[data-testid="stSpinner"] {
  background: transparent !important;
}
[data-testid="stSpinner"] > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.stCacheStatus, [data-testid="stStatusWidget"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* hide Streamlit footer/menu noise */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Cache + status widgets */
.stCacheStatus, [data-testid="stStatusWidget"] { display: none !important; }

/* Footer variants */
footer, [data-testid="stFooter"] { display: none !important; }
#MainMenu { visibility: hidden !important; }

[data-testid="stSpinner"] { display: none !important; }

</style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title="Racecraft Labs",
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_global_styles() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@lru_cache(maxsize=1)
def _build_logo_data_uri(path_str: str) -> str:
    """Load logo, remove edge-connected dark background, and crop to content."""
    logo_path = Path(path_str)
    if Image is None:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    with Image.open(logo_path) as img:
        rgba = img.convert("RGBA")
        px = rgba.load()
        width, height = rgba.size

        dark_threshold = 18

        def is_dark(x: int, y: int) -> bool:
            r, g, b, a = px[x, y]
            return a > 0 and r <= dark_threshold and g <= dark_threshold and b <= dark_threshold

        # Flood-fill dark pixels connected to image edges (background canvas only).
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        for x in range(width):
            if is_dark(x, 0):
                queue.append((x, 0))
            if is_dark(x, height - 1):
                queue.append((x, height - 1))
        for y in range(height):
            if is_dark(0, y):
                queue.append((0, y))
            if is_dark(width - 1, y):
                queue.append((width - 1, y))

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            if not is_dark(x, y):
                continue
            visited.add((x, y))

            if x > 0:
                queue.append((x - 1, y))
            if x < width - 1:
                queue.append((x + 1, y))
            if y > 0:
                queue.append((x, y - 1))
            if y < height - 1:
                queue.append((x, y + 1))

        for x, y in visited:
            r, g, b, _a = px[x, y]
            px[x, y] = (r, g, b, 0)

        alpha_bbox = rgba.getchannel("A").getbbox()
        cropped = rgba.crop(alpha_bbox) if alpha_bbox else rgba

        buffer = BytesIO()
        cropped.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def render_header() -> None:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        logo_data_uri = _build_logo_data_uri(str(logo_path))
        st.markdown(
            (
                '<div class="brand-row">'
                f'<img class="brand-logo" src="{logo_data_uri}" alt="Racecraft Labs logo" />'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Race Weekend Prediction Engine</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="main-header">Racecraft Labs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Race Weekend Prediction Engine</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="micro-disclaimer">Independent fan project • not affiliated with any racing series, teams, or governing bodies</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, bool]:
    page = st.radio(
        "Navigation",
        ["Live Prediction", "Model Insights", "Prediction Accuracy", "About"],
        horizontal=True,
    )

    with st.expander("⚙️ Settings", expanded=False):
        enable_logging = st.checkbox(
            "Save Predictions for Accuracy Tracking",
            value=False,
            help=(
                "When enabled, predictions are saved after each session (FP1/FP2/FP3/SQ) "
                "for later accuracy analysis. Max 1 prediction per session."
            ),
        )
        st.markdown("**Model Version:** v1.0")
        st.markdown("**Last Updated:** 2026-02-01")

    return page, enable_logging
