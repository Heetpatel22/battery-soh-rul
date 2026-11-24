"""
about_project.py - Project Methodology, NASA Ames Dataset & System Architecture.
Technical documentation and specifications for the 4th-Year EEE Final Year Project.
"""

import os
import streamlit as st
import pandas as pd

from ui.components import (
    render_compact_header,
    render_category_legend,
    render_thevenin_circuit_svg,
    render_html,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render_about_project(metrics: dict):
    render_compact_header(
        page_title="About CellGuard & Project Methodology",
        selected_battery="Project Specifications",
        data_source="NASA Ames Prognostics Center of Excellence (PCoE)",
        model_status="4th-Year EEE Final Year Project · CellGuard",
    )

    render_category_legend()

    # HERO BANNER
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-left:4px solid #2563EB; border-radius:6px; padding:18px 22px; margin-bottom:18px; box-shadow:0 1px 3px rgba(15,23,42,0.04);">
      <h3 style="font-size:1.15rem; font-weight:800; color:#0F172A; margin:0 0 6px 0;">
        CellGuard – Intelligent Battery Diagnostics &amp; Impedance Estimation
      </h3>
      <div style="font-size:0.84rem; color:#475569; line-height:1.6;">
        A Machine Learning-based Battery Management System (BMS) intelligence platform focused on predicting internal electrochemical impedance parameters:
        <strong>Ohmic Resistance (Re)</strong> and <strong>Charge-Transfer Resistance (Rct)</strong> from non-invasive, observable partial-window discharge parameters.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. SPECIFICATIONS TABLE
    st.markdown('<div class="cg-section-label">BATTERY SPECIFICATIONS & EXPERIMENTAL PROTOCOL</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Cell Form Factor", "18650 Cylindrical", help="Standard cylindrical cell (18mm diameter, 65mm length)")
    with s2:
        st.metric("Cathode / Anode", "LiCoO₂ / Graphite", help="Lithium Cobalt Oxide cathode with synthetic graphite anode")
    with s3:
        st.metric("Nominal Rated Capacity", "2.00 Ah (7.4 Wh)", help="Manufacturer nominal capacity at fresh state")
    with s4:
        st.metric("Operating Chamber Temp", "24°C Ambient", help="Constant controlled laboratory environmental chamber")

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:16px 20px; margin:12px 0;">
      <h4 style="font-size:0.90rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
        NASA Laboratory Cycling Protocols
      </h4>
      <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
        <ul>
          <li><strong>Charging Protocol:</strong> Constant Current - Constant Voltage (CC-CV). Charged at a constant current of 1.5 A until terminal voltage reached 4.2 V, followed by constant voltage hold until charge current tapered down to 20 mA.</li>
          <li><strong>Discharge Cycling:</strong> Constant Current (CC) discharge at 2.0 A (1C rate) until reaching cell-specific cutoff voltages: B0005 (2.7 V), B0006 (2.5 V), B0007 (2.2 V), and B0018 (2.5 V).</li>
          <li><strong>EIS Frequency Sweeps:</strong> Periodic AC impedance spectroscopy from 0.1 Hz to 5 kHz performed at rest, fitting equivalent circuit semi-circles to isolate <em>Re</em> and <em>Rct</em>.</li>
          <li><strong>End of Life (EOL) Standard:</strong> Defined by NASA researchers as a 30% reduction from rated capacity, equivalent to <strong>1.40 Ah (70% SOH)</strong>.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. FEATURE EXTRACTION SUMMARY
    st.markdown('<div class="cg-section-label">PARTIAL-WINDOW OBSERVABLE FEATURE EXTRACTION (4.0 V → 3.4 V)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:16px 20px; margin-bottom:14px;">
      <h4 style="font-size:0.90rem; font-weight:800; color:#0F172A; text-transform:uppercase; margin-top:0;">
        Why a Partial Window (4.0 V down to 3.4 V) is Essential for Real-World BMS
      </h4>
      <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
        In real electric vehicles and grid storage systems, batteries are almost never fully discharged to 0% in normal operation.
        CellGuard inspects only the first 10–20 minutes of discharge between <strong>4.0 V and 3.4 V</strong>, extracting 18 deterministic electrical and thermal features:
        <ol>
          <li><strong>Voltage Crossing Durations:</strong> <code>T_40_38</code>, <code>T_38_36</code>, <code>T_36_34</code>, <code>T_40_36</code>, <code>T_38_34</code> (seconds).</li>
          <li><strong>Partial Window Charge:</strong> <code>Q_40_36</code> = &int; |I| dt / 3600 (Ah transferred between 4.0V and 3.6V).</li>
          <li><strong>Timed Voltage Samples:</strong> <code>V_at_60s</code>, <code>V_at_300s</code>, <code>V_at_600s</code>, <code>V_at_1200s</code>.</li>
          <li><strong>Curve Shape Parameters:</strong> <code>V_start</code>, <code>V_mean_first600s</code>, <code>V_slope_mid</code> (dV/dt).</li>
          <li><strong>Thermal Features:</strong> <code>T_start</code>, <code>T_max</code>, <code>T_rise</code> (heat generated during window), <code>T_mean</code>.</li>
          <li><strong>Ohmic Load Step Proxy:</strong> <code>R_dc_proxy</code> = (V_rest - V_loaded) / ΔI at current turn-on.</li>
          <li><strong>Aging Progression:</strong> <code>disc_idx</code> (discharge cycle number).</li>
        </ol>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. VERIFICATION STATEMENT
    st.markdown('<div class="cg-section-label">SCIENTIFIC INTEGRITY & DATA LEAKAGE VERIFICATION</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px; padding:16px 20px; margin-bottom:16px;">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
        <span style="font-size:1.1rem;">🛡️</span>
        <span style="font-size:0.90rem; font-weight:800; color:#166534; text-transform:uppercase;">
          Verified Absence of Target Leakage
        </span>
      </div>
      <div style="font-size:0.82rem; color:#14532D; line-height:1.6;">
        <strong>Strict Architectural Isolation:</strong>
        In this project, <strong>neither Ohmic Resistance (Re) nor Charge-Transfer Resistance (Rct) is ever provided as an input feature</strong> to any model.
        The models are trained strictly on the 19 observable discharge parameters and evaluated using <strong>Leave-One-Cell-Out (LOCO)</strong> cross-validation, guaranteeing that all performance metrics reflect true out-of-sample prediction capability on unseen battery hardware.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. SYSTEM WORKFLOW DIAGRAM
    st.markdown('<div class="cg-section-label">SYSTEM ARCHITECTURE PIPELINE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:18px; text-align:center;">
      <div style="display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:12px; font-size:0.80rem;">
        <div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:12px; width:180px;">
          <div style="font-weight:800; color:#1D4ED8;">1. Raw Discharge Telemetry</div>
          <div style="color:#475569; font-size:0.72rem; margin-top:4px;">V(t), I(t), Temp(t), Cycle No.</div>
        </div>
        <div style="font-size:1.2rem; color:#94A3B8;">&rarr;</div>
        <div style="background:#EFF6FF; border:1px solid #93C5FD; border-radius:6px; padding:12px; width:180px;">
          <div style="font-weight:800; color:#1D4ED8;">2. Feature Extraction</div>
          <div style="color:#475569; font-size:0.72rem; margin-top:4px;">19 Observable Parameters (4.0V&rarr;3.4V)</div>
        </div>
        <div style="font-size:1.2rem; color:#94A3B8;">&rarr;</div>
        <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:6px; padding:12px; width:180px;">
          <div style="font-weight:800; color:#0F766E;">3. ML Predictors (LOCO)</div>
          <div style="color:#475569; font-size:0.72rem; margin-top:4px;">Extra Trees / RF / XGBoost / Ridge</div>
        </div>
        <div style="font-size:1.2rem; color:#94A3B8;">&rarr;</div>
        <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:6px; padding:12px; width:180px;">
          <div style="font-weight:800; color:#15803D;">4. Battery Diagnostics</div>
          <div style="color:#475569; font-size:0.72rem; margin-top:4px;">Re, Rct, R_int, SOH, Power Loss</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
