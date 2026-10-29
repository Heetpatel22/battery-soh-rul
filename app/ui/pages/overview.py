"""
overview.py - Executive Engineering Overview.
Presents mission-critical BMS telemetry, error bounds, prognostic workflow, and dataset degradation overview.
"""

import streamlit as st
import pandas as pd
from ui.components import (
    render_kpi,
    tachometer_soh_gauge_svg,
    prismatic_battery_svg,
    render_html,
)
from ui.charts import plot_capacity_fade_all_cells


def render_overview(predictor, metrics: dict, feature_table: pd.DataFrame):
    """Render the executive Overview page."""
    # Top Telemetry Header
    render_html("""
    <div style="display: flex; align-items: center; justify-content: space-between; background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 16px; margin-bottom: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8;">
      <div><span style="color: #64748B;">SYSTEM:</span> <span style="color: #E2E8F0; font-weight: 600;">CELLGUARD PRO v2.4</span></div>
      <div><span style="color: #64748B;">CALIBRATION:</span> <span style="color: #E2E8F0;">24°C AMBIENT &bull; 2.0 A DISCHARGE</span></div>
      <div><span style="color: #64748B;">FEATURE EXTRACTION:</span> <span style="color: #22D3EE;">4.0 V &rarr; 3.4 V (PARTIAL WINDOW)</span></div>
      <div><span style="color: #64748B;">PRIMARY BENCHMARK:</span> <span style="color: #10B981;">B0018 UNSEEN</span></div>
    </div>
    """)

    # Hero Row: Executive Narrative & Live Diagnostic Instrument
    col_hero, col_instrument = st.columns([1.35, 1], gap="large")

    with col_hero:
        render_html("""
        <div>
          <h2 style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; color: #FFFFFF; line-height: 1.15; margin: 0 0 12px 0;">
            Non-Destructive SOH &amp; RUL Prognosis from Early Terminal Dynamics
          </h2>
          <p style="font-size: 0.95rem; color: #94A3B8; line-height: 1.6; margin: 0 0 20px 0;">
            Standard automotive BMS systems cannot perform full-discharge Coulomb counting or high-frequency Electrochemical Impedance Spectroscopy (EIS) during operational driving. 
            CellGuard uses regularized linear Ridge and tree ensembles to infer internal electrochemical states strictly from early plateau traversal ($4.0\text{ V} \to 3.4\text{ V}$).
          </p>
        </div>
        """)

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("⚡ Test Discharge Cycle", type="primary", use_container_width=True):
                st.session_state["nav_selection"] = "Predict Cycle"
                st.session_state["selected_sample"] = "B0018_cycle_050_middle.csv"
                st.rerun()
        with btn_c2:
            if st.button("📊 View Model Benchmarks", use_container_width=True):
                st.session_state["nav_selection"] = "Model Performance"
                st.rerun()

    with col_instrument:
        # Live sample instrument card (B0018 Cycle 50)
        gauge_svg = tachometer_soh_gauge_svg(83.3, capacity_ah=1.666, size=190)
        bat_svg = prismatic_battery_svg(83.3, width=95, height=180)
        render_html(f"""
        <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-around;">
          <div>{gauge_svg}</div>
          <div style="border-left: 1px solid rgba(255,255,255,0.06); padding-left: 16px;">{bat_svg}</div>
        </div>
        """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 4 Certified LOCO Error Band Tiles
    err_bands = metrics.get("error_bands", {})
    cap_mae = err_bands.get("capacity_mae_loco", 0.046)
    re_mae = err_bands.get("re_mae_loco", 0.0066)
    rct_mae = err_bands.get("rct_mae_loco", 0.0055)
    rul_mae = err_bands.get("rul_mae_loco", 15.2)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Capacity Error", f"{cap_mae:.3f}", "Ah", error_band=f"{cap_mae:.3f}", subtext="LOCO MAE across all 4 cells", status="Healthy")
    with k2:
        render_kpi("Ohmic Drift (Re)", f"{re_mae * 1000:.1f}", "mΩ", error_band=f"{re_mae * 1000:.1f}", subtext="Electrolyte series impedance", status="Healthy")
    with k3:
        render_kpi("Kinetics (Rct)", f"{rct_mae * 1000:.1f}", "mΩ", error_band=f"{rct_mae * 1000:.1f}", subtext="Charge-transfer polarization", status="Healthy")
    with k4:
        render_kpi("RUL Precision", f"{rul_mae:.1f}", "cyc", error_band=f"{rul_mae:.1f}", subtext="Retirement at 1.40 Ah EOL", status="Healthy")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 4 Technical Pipeline Stages
    render_html("""
    <div style="margin-bottom: 8px;">
      <span style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #22D3EE; text-transform: uppercase; letter-spacing: 0.08em;">
        ARCHITECTURE WORKFLOW
      </span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
      <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; font-weight: 700; margin-bottom: 4px;">01 / SENSING</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;">Terminal V, I, T</div>
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">Standard onboard telemetry sampled non-invasively at 0.1 to 1 Hz.</div>
      </div>
      <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; font-weight: 700; margin-bottom: 4px;">02 / WINDOWING</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;">4.0 V &rarr; 3.4 V Features</div>
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">18 deterministic timings, partial charge, and initial load-step proxy.</div>
      </div>
      <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; font-weight: 700; margin-bottom: 4px;">03 / PIPELINE</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;">Ridge &amp; Forest Models</div>
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">Regularized linear extrapolation for capacity; tree regressors for RUL.</div>
      </div>
      <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; font-weight: 700; margin-bottom: 4px;">04 / TELEMETRY</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;">Certified Diagnostics</div>
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">SOH %, ECM parameters (R0, R1), and retirement cycles with error bands.</div>
      </div>
    </div>
    """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Full Dataset Capacity Fade Chart
    if not feature_table.empty:
        fig_fade = plot_capacity_fade_all_cells(feature_table)
        st.plotly_chart(fig_fade, use_container_width=True)
