"""
streamlit_app.py - CellGuard Intelligent Battery Diagnostics & Prognostics Platform.
Unified entry point with Streamlit navigation, clean industrial design system,
and Leave-One-Cell-Out certified inference models.
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
from ui.components import inject_theme, render_sidebar_header, render_sidebar_footer, render_footer
from ui.pages.overview import render_overview
from ui.pages.predict import render_predict
from ui.pages.soc_soh import render_soc_soh
from ui.pages.life_rul import render_life_rul
from ui.pages.impedance import render_impedance
from ui.pages.performance import render_performance
from ui.pages.about_project import render_about_project

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="CellGuard — Battery Diagnostics & Prognostics",
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
def page_overview_view():
    inject_theme()
    render_sidebar_header()
    render_overview(predictor, metrics, feature_table)
    render_sidebar_footer()
    render_footer(metrics)


def page_predict_view():
    inject_theme()
    render_sidebar_header()
    render_predict(predictor, schema, metrics)
    render_sidebar_footer()
    render_footer(metrics)


def page_soc_soh_view():
    inject_theme()
    render_sidebar_header()
    render_soc_soh(predictor, feature_table, metrics)
    render_sidebar_footer()
    render_footer(metrics)


def page_life_rul_view():
    inject_theme()
    render_sidebar_header()
    render_life_rul(predictor, feature_table, metrics)
    render_sidebar_footer()
    render_footer(metrics)


def page_impedance_view():
    inject_theme()
    render_sidebar_header()
    render_impedance(predictor, feature_table, metrics)
    render_sidebar_footer()
    render_footer(metrics)


def page_performance_view():
    inject_theme()
    render_sidebar_header()
    render_performance(metrics)
    render_sidebar_footer()
    render_footer(metrics)


def page_about_project_view():
    inject_theme()
    render_sidebar_header()
    render_about_project(metrics)
    render_sidebar_footer()
    render_footer(metrics)


# Streamlit Navigation Setup with exact requested titles and icons
page_overview = st.Page(page_overview_view, title="Overview", icon="📊", default=True)
page_predict = st.Page(page_predict_view, title="Battery Health Diagnostics", icon="🩺")
page_soc_soh = st.Page(page_soc_soh_view, title="SOC & SOH Monitoring", icon="⚡")
page_life_rul = st.Page(page_life_rul_view, title="RUL Forecasting", icon="⏳")
page_impedance = st.Page(page_impedance_view, title="Impedance Analysis", icon="🔬")
page_performance = st.Page(page_performance_view, title="Model Performance", icon="🎯")
page_about_project = st.Page(page_about_project_view, title="About the Project", icon="📘")

pages = [
    page_overview,
    page_predict,
    page_soc_soh,
    page_life_rul,
    page_impedance,
    page_performance,
    page_about_project,
]

pg = st.navigation(pages)

# Handle programmatic jump between pages if requested
if "nav_selection" in st.session_state:
    target = st.session_state.pop("nav_selection")
    if target in ["Diagnostics", "Predict Cycle", "Battery Health Diagnostics"]:
        st.switch_page(page_predict)
    elif target in ["SOC & SOH", "SOC & SOH Monitoring"]:
        st.switch_page(page_soc_soh)
    elif target in ["RUL", "RUL Forecasting", "Battery Life & RUL"]:
        st.switch_page(page_life_rul)
    elif target in ["Impedance", "Impedance Analysis"]:
        st.switch_page(page_impedance)
    elif target in ["Performance", "Model Performance"]:
        st.switch_page(page_performance)
    elif target in ["About", "About the Project"]:
        st.switch_page(page_about_project)

pg.run()
