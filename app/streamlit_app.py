"""
streamlit_app.py - CellGuard Battery SOH, Impedance, and RUL Diagnostic Suite.
Main application entry point with st.navigation, dark laboratory control room theme,
and full Leave-One-Cell-Out certified inference models.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Add src and app to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

for p in [SRC_DIR, APP_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import BatteryPredictor
from ui.components import inject_theme, render_topbar, render_footer
from ui.pages.overview import render_overview
from ui.pages.predict import render_predict
from ui.pages.life_rul import render_life_rul
from ui.pages.gap_filler import render_gap_filler
from ui.pages.performance import render_performance
from ui.pages.ee_background import render_ee_background

# Configure Streamlit page
st.set_page_config(
    page_title="CellGuard — Battery SOH & RUL Prognostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_predictor():
    """Load inference pipelines and model pickles once."""
    return BatteryPredictor(MODELS_DIR)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load verification metrics and recorded LOCO results."""
    p = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_schema():
    """Load feature schema definitions and valid ranges."""
    p = os.path.join(MODELS_DIR, "feature_schema.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_feature_table():
    """Load per-cycle feature table across all 4 cells."""
    p = os.path.join(MODELS_DIR, "feature_table.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()


# Load cached assets
predictor = get_predictor()
metrics = load_metrics()
schema = load_schema()
feature_table = load_feature_table()


# Define Page Callables
def render_sidebar_header():
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;">
          <div style="width: 28px; height: 28px; background: rgba(34,211,238,0.12); border: 1px solid rgba(34,211,238,0.35); border-radius: 6px; display: flex; align-items: center; justify-content: center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="2.5"><rect x="2" y="7" width="16" height="12" rx="2"></rect><path d="M20 11v4"></path></svg>
          </div>
          <div>
            <div style="font-weight: 700; font-size: 0.95rem; color: #FFFFFF; letter-spacing: 0.04em;">CELLGUARD PRO</div>
            <div style="font-size: 0.68rem; color: #22D3EE; font-family: 'JetBrains Mono', monospace;">BMS DIAGNOSTICS</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def page_overview_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("Overview")
    render_overview(predictor, metrics, feature_table)
    render_footer(metrics)


def page_predict_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("Predict Cycle")
    render_predict(predictor, schema, metrics)
    render_footer(metrics)


def page_life_rul_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("Battery Life & RUL")
    render_life_rul(predictor, feature_table, metrics)
    render_footer(metrics)


def page_gap_filler_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("Gap Filler")
    render_gap_filler(predictor, feature_table, metrics)
    render_footer(metrics)


def page_performance_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("Model Performance")
    render_performance(metrics)
    render_footer(metrics)


def page_ee_bg_view():
    inject_theme()
    render_sidebar_header()
    render_topbar("EE Background")
    render_ee_background()
    render_footer(metrics)


# Streamlit 1.30+ Navigation Setup
page_overview = st.Page(page_overview_view, title="Overview", icon="🏠", default=True)
page_predict = st.Page(page_predict_view, title="Predict Cycle", icon="⚡")
page_life_rul = st.Page(page_life_rul_view, title="Battery Life and RUL", icon="📈")
page_gap_filler = st.Page(page_gap_filler_view, title="Gap Filler", icon="🔬")
page_performance = st.Page(page_performance_view, title="Model Performance", icon="📊")
page_ee_bg = st.Page(page_ee_bg_view, title="EE Background", icon="🔌")

pages = [
    page_overview,
    page_predict,
    page_life_rul,
    page_gap_filler,
    page_performance,
    page_ee_bg,
]

pg = st.navigation(pages)

# Handle programmatic jump from Overview action buttons if set
if "nav_selection" in st.session_state:
    target = st.session_state.pop("nav_selection")
    if target == "Predict Cycle":
        st.switch_page(page_predict)
    elif target == "Model Performance":
        st.switch_page(page_performance)

pg.run()
