"""CellGuard multi-page battery health workspace."""

import json
import os
import sys

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for path in (SRC_DIR, APP_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from inference import BatteryPredictor
from ui.components import inject_theme, render_footer, render_sidebar_footer, render_sidebar_header
from ui.pages.about_project import render_about_project
from ui.pages.calculation_metrics import render_calculation_metrics
from ui.pages.ee_background import render_ee_background
from ui.pages.gap_filler import render_gap_filler
from ui.pages.impedance import render_impedance
from ui.pages.life_rul import render_life_rul
from ui.pages.overview import render_overview
from ui.pages.performance import render_performance
from ui.pages.predict import render_predict
from ui.pages.soc_soh import render_soc_soh

st.set_page_config(
    page_title="CellGuard · Battery Health Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_predictor():
    return BatteryPredictor(MODELS_DIR)


@st.cache_data(show_spinner=False)
def load_json(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


@st.cache_data(show_spinner=False)
def load_feature_table(source_mtime: float = 0.0):
    path = os.path.join(MODELS_DIR, "feature_table.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


predictor = get_predictor()
metrics = load_json("metrics.json")
schema = load_json("feature_schema.json")
feature_path = os.path.join(MODELS_DIR, "feature_table.csv")
feature_table = load_feature_table(os.path.getmtime(feature_path) if os.path.exists(feature_path) else 0.0)


def _page(render_fn, *args):
    inject_theme()
    render_sidebar_header()
    render_fn(*args)
    render_sidebar_footer()
    render_footer()


def overview_page():
    _page(render_overview, predictor, metrics, feature_table)


def calculation_page():
    _page(render_calculation_metrics, feature_table)


def accuracy_page():
    _page(render_performance, metrics)


def diagnostics_page():
    _page(render_predict, predictor, schema, metrics)


def soh_page():
    _page(render_soc_soh, predictor, feature_table, metrics)


def rul_page():
    _page(render_life_rul, predictor, feature_table, metrics)


def impedance_page():
    _page(render_impedance, predictor, feature_table, metrics)


def gap_page():
    _page(render_gap_filler, predictor, feature_table, metrics)


def ee_page():
    _page(render_ee_background)


def about_page():
    _page(render_about_project, metrics)


pages = [
    st.Page(overview_page, title="Overview", icon="📊", default=True),
    st.Page(calculation_page, title="Calculation Metrics", icon="🧮"),
    st.Page(diagnostics_page, title="Battery Health Check", icon="🩺"),
    st.Page(soh_page, title="SOH & SOC", icon="⚡"),
    st.Page(rul_page, title="Battery Life", icon="⏳"),
    st.Page(impedance_page, title="Impedance History", icon="🔬"),
    st.Page(accuracy_page, title="Accuracy Snapshot", icon="🎯"),
    st.Page(gap_page, title="Fill Missing Readings", icon="🧩"),
    st.Page(ee_page, title="Electrical Engineering", icon="🔋"),
    st.Page(about_page, title="About CellGuard", icon="ℹ️"),
]

st.navigation(pages).run()
